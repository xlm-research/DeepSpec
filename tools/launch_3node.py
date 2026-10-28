"""Start/reuse the three-node Ray cluster, then run a Qwen3.8-27B recipe."""

import concurrent.futures
import ipaddress
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "dflash2": "examples/qwen3.8-27b-dflash2-3node/run.sh",
    "dspark": "examples/qwen3.8-27b-3node/run.sh",
}
HELP = """Usage: bash run.sh [dflash2|dspark] [--dry-run] [key=value ...]

Default: dflash2, head/training 10.120.0.211, inference 10.120.0.218,10.120.0.187.
Starts missing Ray nodes over SSH, reuses live nodes, then starts training.
Run from any of the three nodes; all must share this checkout and h800conda-tmp.sh.
SSH must work without a password (known host keys, BatchMode=yes).

  bash run.sh
  bash run.sh dspark training.num_train_steps=10
  bash run.sh --dry-run
  LOAD_PATH=/path/to/checkpoints bash run.sh

Environment: TRAIN_NODE_IPS (one IPv4 address), INFERENCE_NODE_IPS (two IPv4
addresses), RAY_PORT (default 6379), SSH_USER (default current user), SSH_PORT
(default 22). Existing recipe variables such as LOAD_PATH, OUTPUT_DIR and
TRAIN_DATA_PATH are forwarded to the head. --dry-run makes no SSH/Ray calls;
--print-config-only prints the recipe config without starting Ray.
Ray stays running after training, ready for the next run. No nodes are stopped.
"""

# Preserve recipe options and network settings when forwarding to the head.
FORWARD_KEYS = {
    "TRAIN_DATA_PATH",
    "LOAD_PATH",
    "CKPT_STEP",
    "OUTPUT_DIR",
    "CACHE_DIR",
    "LOG_DIR",
    "TARGET_MODEL_PATH",
    "MOONCAKE_PROTOCOL",
    "CUDA_VISIBLE_DEVICES",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "TORCHSPEC_LOG_LEVEL",
    "WANDB_NAME",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_MODE",
    "HF_HOME",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "SSH_USER",
    "SSH_PORT",
    "RAY_TEMP_DIR",
}
NODE_QUERY = """import json, sys
import ray
ray.init(address=sys.argv[1], logging_level='ERROR', log_to_driver=False)
try:
    nodes = {n['NodeManagerAddress']: n['Resources'].get('GPU', 0)
             for n in ray.nodes() if n['Alive']}
    print('TORCHSPEC_NODES=' + json.dumps(nodes))
finally:
    ray.shutdown()
"""


class Launcher:
    def __init__(self):
        self.head = os.environ.get("TRAIN_NODE_IPS", "10.120.0.211")
        self.workers = os.environ.get("INFERENCE_NODE_IPS", "10.120.0.218,10.120.0.187").split(",")
        if len(self.workers) != 2:
            raise ValueError("INFERENCE_NODE_IPS must contain two comma-separated IPv4 addresses")
        for node in [self.head, *self.workers]:
            ipaddress.IPv4Address(node)
        if len(set([self.head, *self.workers])) != 3:
            raise ValueError("Training and inference nodes must be three different addresses")
        address = os.environ.get("RAY_ADDRESS", "auto")
        port = os.environ.get("RAY_PORT", address.rsplit(":", 1)[-1] if ":" in address else "6379")
        self.port = self.valid_port(port)
        self.address = f"{self.head}:{self.port}"
        if address not in ("", "auto", self.address):
            raise ValueError(f"RAY_ADDRESS must point to the configured head: {self.address}")
        self.ssh_port = self.valid_port(os.environ.get("SSH_PORT", "22"))
        self.env = {key: os.environ[key] for key in FORWARD_KEYS if key in os.environ}
        self.env.update(
            TRAIN_NODE_IPS=self.head,
            INFERENCE_NODE_IPS=",".join(self.workers),
            RAY_ADDRESS=self.address,
            RAY_PORT=self.port,
            VLLM_USE_V2_MODEL_RUNNER="0",
        )
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
            self.env.setdefault(key, "eth0")
        no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
        self.env["no_proxy"] = ",".join(
            filter(None, [no_proxy, "localhost", "127.0.0.1", self.head, *self.workers])
        )
        self.env["NO_PROXY"] = self.env["no_proxy"]
        self.local_ips = subprocess.check_output(["hostname", "-I"], text=True).split()

    @staticmethod
    def valid_port(value):
        if not value.isdecimal() or not 1 <= int(value) <= 65535:
            raise ValueError(f"Invalid port: {value}")
        return str(int(value))

    def run_on(self, node, command, *, capture_output=False, timeout=None, check=True):
        # Send the script on stdin so SSH's remote shell cannot reparse its arguments.
        script = "\n".join(
            [
                "set -euo pipefail",
                f"cd -- {shlex.quote(str(ROOT))}",
                "source ./h800conda-tmp.sh",
                "export " + shlex.join([f"{key}={value}" for key, value in self.env.items()]),
                "exec " + shlex.join(command),
            ]
        )
        if node in self.local_ips:
            args = ["bash", "-c", script]
            stdin = None
        else:
            user = os.environ.get("SSH_USER")
            target = f"{user}@{node}" if user else node
            args = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=10",
                "-o",
                "ServerAliveCountMax=3",
                "-p",
                self.ssh_port,
                "--",
                target,
                "bash -s",
            ]
            stdin = script + "\n"
        try:
            return subprocess.run(
                args,
                input=stdin,
                text=True,
                capture_output=capture_output,
                timeout=timeout,
                check=check,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{command[0]} on {node} timed out after {timeout}s") from None

    def nodes(self, timeout=30):
        result = self.run_on(
            self.head,
            ["python", "-c", NODE_QUERY, self.address],
            capture_output=True,
            timeout=timeout,
        )
        for line in result.stdout.splitlines():
            if line.startswith("TORCHSPEC_NODES="):
                return json.loads(line.split("=", 1)[1])
        raise RuntimeError("Ray returned no node list")

    def check_gpus(self, nodes):
        for node in [self.head, *self.workers]:
            if node in nodes and nodes[node] < 8:
                raise RuntimeError(
                    f"Ray node {node} advertises {nodes[node]} GPUs; this recipe needs 8"
                )

    def ensure_cluster(self):
        print(f"Checking Ray at {self.address}", flush=True)
        # Avoid Ray's long connection retries when the head has not been started.
        try:
            with socket.create_connection((self.head, int(self.port)), timeout=2):
                head_running = True
        except OSError:
            head_running = False
        if not head_running:
            print(f"Starting Ray head: {self.head}", flush=True)
            command = [
                "ray",
                "start",
                "--head",
                "--port",
                self.port,
                "--node-ip-address",
                self.head,
                "--disable-usage-stats",
            ]
            if self.env.get("RAY_TEMP_DIR"):
                command.extend(["--temp-dir", self.env["RAY_TEMP_DIR"]])
            self.run_on(self.head, command, timeout=120)
        nodes = self.nodes()
        self.check_gpus(nodes)
        missing = [node for node in self.workers if node not in nodes]
        if missing:
            # The two workers are independent once the head has started.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = []
                for node in missing:
                    print(f"Starting Ray worker: {node}", flush=True)
                    futures.append(
                        pool.submit(
                            self.run_on,
                            node,
                            [
                                "ray",
                                "start",
                                "--address",
                                self.address,
                                "--node-ip-address",
                                node,
                                "--disable-usage-stats",
                            ],
                            timeout=120,
                        )
                    )
                for future in futures:
                    future.result()
        deadline = time.monotonic() + 120
        while True:
            self.check_gpus(nodes)
            missing = [node for node in [self.head, *self.workers] if node not in nodes]
            if not missing:
                print("Ray ready: 3 nodes, at least 8 GPUs each. Reusing the cluster.", flush=True)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Timed out waiting for Ray nodes: {', '.join(missing)}")
            print(f"Waiting for Ray nodes: {', '.join(missing)}", flush=True)
            time.sleep(min(2, remaining))
            nodes = self.nodes(timeout=max(1, min(30, deadline - time.monotonic())))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(HELP)
        return 0
    recipe = args.pop(0) if args and args[0] in RECIPES else "dflash2"
    if args and not args[0].startswith("--") and "=" not in args[0]:
        raise ValueError(f"Unknown recipe: {args[0]}; choose dflash2 or dspark")
    launcher = Launcher()
    command = ["bash", str(ROOT / RECIPES[recipe]), *args]
    if "--dry-run" in args:
        args.remove("--dry-run")
        print(
            f"Recipe: {recipe}; Ray head: {launcher.address}; workers: {', '.join(launcher.workers)}"
        )
        print(
            "Plan: activate h800conda-tmp.sh on each node; reuse live Ray nodes; start missing nodes;"
        )
        print(
            "      wait for 3 x 8 GPUs; run training on the head. No SSH/Ray calls made.",
            flush=True,
        )
        env = {**os.environ, **launcher.env}
        return subprocess.run(
            ["bash", str(ROOT / RECIPES[recipe]), "--dry-run", *args],
            env=env,
            check=True,
        ).returncode
    if "--print-config-only" in args:
        # Configuration inspection remains local and does not need a live cluster.
        return launcher.run_on(launcher.local_ips[0], command).returncode
    if launcher.head not in launcher.local_ips:
        print(f"Forwarding launch to training/head node {launcher.head}", flush=True)
        return launcher.run_on(
            launcher.head, ["bash", str(ROOT / "run.sh"), recipe, *args]
        ).returncode
    # Catch malformed checkpoint/recipe arguments before changing cluster state.
    subprocess.run(
        ["bash", str(ROOT / RECIPES[recipe]), "--dry-run", *args],
        env={**os.environ, **launcher.env},
        stdout=subprocess.DEVNULL,
        check=True,
    )
    launcher.ensure_cluster()
    return launcher.run_on(launcher.head, command).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            # Shell commands contain forwarded environment values; don't dump them.
            print(f"ERROR: launch command failed (exit {exc.returncode})", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            sys.exit(exc.returncode)
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
