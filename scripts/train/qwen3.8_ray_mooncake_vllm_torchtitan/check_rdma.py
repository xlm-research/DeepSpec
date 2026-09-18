"""Read-only, two-node IPv4 RoCE preflight; this is not a transfer test."""

import argparse
import datetime
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "error": str(error)}


def inspect_node(device, port, gid_index):
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    network = command(["ip", "-j", "address", "show"])
    interfaces = json.loads(network["stdout"]) if network["returncode"] == 0 else []
    root = Path("/sys/class/infiniband") / device
    port_path = root / "ports" / str(port)
    fields = {
        "state": read(port_path / "state"),
        "link_layer": read(port_path / "link_layer"),
        "gid": read(port_path / "gids" / str(gid_index)),
        "gid_type": read(port_path / "gid_attrs/types" / str(gid_index)),
        "netdev": read(port_path / "gid_attrs/ndevs" / str(gid_index)),
    }
    try:
        mapped = ipaddress.IPv6Address(fields["gid"]).ipv4_mapped
        fields["roce_ipv4"] = str(mapped) if mapped is not None else None
    except (ipaddress.AddressValueError, TypeError):
        fields["roce_ipv4"] = None
    interface = next((i for i in interfaces if i["ifname"] == fields["netdev"]), None)
    errors = []
    if not root.exists():
        errors.append(f"RDMA device {device} is absent")
    if fields["link_layer"] != "Ethernet":
        errors.append("Selected port is not an Ethernet/RoCE port")
    if fields["state"] != "4: ACTIVE":
        errors.append("Selected RDMA port is not ACTIVE")
    if fields["gid_type"] != "RoCE v2" or not fields["roce_ipv4"]:
        errors.append(f"GID {gid_index} is not a readable IPv4 RoCE v2 GID")
    if interface is None:
        errors.append("GID has no netdev visible in this container network namespace")
    else:
        if "UP" not in interface["flags"] or "LOWER_UP" not in interface["flags"]:
            errors.append(f"RoCE netdev {fields['netdev']} is not UP/LOWER_UP")
        ips = [a["local"] for a in interface["addr_info"] if a["family"] == "inet"]
        if fields["roce_ipv4"] not in ips:
            errors.append("GID IPv4 does not match the associated netdev address")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
    )
    caps = int(status["CapEff"].strip(), 16)
    return {
        "hostname": socket.gethostname(),
        "network_namespace": os.readlink("/proc/self/ns/net"),
        "device": device,
        "pci_device": str((root / "device").resolve()),
        "port": port,
        "gid_index": gid_index,
        **fields,
        "interfaces": interfaces,
        "network_command": network,
        "rdma_link": command(["rdma", "link", "show", f"{device}/{port}"]),
        "rdma_namespace_mode": command(["rdma", "system", "show"]),
        "cap_net_admin": bool(caps & (1 << 12)),
        "cap_sys_admin": bool(caps & (1 << 21)),
        "device_allocations": {
            k: v for k, v in os.environ.items() if k.startswith("PCIDEVICE_")
        },
        "gpu_processes": command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_HEAD_ADDRESS"))
    parser.add_argument("--producer-node", default=os.environ.get("PRODUCER_NODE"))
    parser.add_argument("--consumer-node", default=os.environ.get("CONSUMER_NODE"))
    parser.add_argument("--device", default="mlx5_10")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all((args.ray_address, args.producer_node, args.consumer_node)):
        parser.error(
            "Set Ray address and both node selectors via arguments or environment"
        )
    if args.output.exists():
        parser.error("Output must not exist; preserve previous diagnostics")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "network_preflight_pass": False,
        "cross_node_transfer_tested": False,
        "nodes": [],
    }
    try:
        ray.init(
            address=args.ray_address,
            namespace="dspark-rdma-preflight",
            log_to_driver=False,
        )
        live = [n for n in ray.nodes() if n["Alive"]]
        selected = []
        for selector in (args.producer_node, args.consumer_node):
            matches = [
                n for n in live if selector in (n["NodeID"], n["NodeManagerAddress"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one live Ray node for {selector}; got {len(matches)}"
                )
            selected.append(matches[0])
        if selected[0]["NodeID"] == selected[1]["NodeID"]:
            raise ValueError("Producer and consumer must be different nodes")
        options = [
            {
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    n["NodeID"], soft=False
                )
            }
            for n in selected
        ]
        inspect = ray.remote(num_cpus=0, num_gpus=0, max_retries=0)(inspect_node)
        report["nodes"] = ray.get(
            [
                inspect.options(**o).remote(args.device, args.port, args.gid_index)
                for o in options
            ],
            timeout=60,
        )
        for i, role in enumerate(("producer", "consumer")):
            report["nodes"][i].update(
                role=role, ray_ip=selected[i]["NodeManagerAddress"]
            )
        route = ray.remote(num_cpus=0, num_gpus=0, max_retries=0)(command)
        for i, node in enumerate(report["nodes"]):
            peer_ip = report["nodes"][1 - i]["roce_ipv4"]
            if peer_ip is None:
                node["route_check_skipped"] = "Peer has no readable IPv4 RoCE GID"
                continue
            result = ray.get(
                route.options(**options[i]).remote(
                    ["ip", "-j", "route", "get", peer_ip]
                ),
                timeout=15,
            )
            node["route_to_peer"] = result
            routes = json.loads(result["stdout"]) if result["returncode"] == 0 else []
            if (
                not routes
                or routes[0].get("dev") != node["netdev"]
                or routes[0].get("prefsrc") != node["roce_ipv4"]
            ):
                node["errors"].append(
                    "Peer RoCE route does not use the selected netdev and GID IPv4"
                )
        report["network_preflight_pass"] = all(not n["errors"] for n in report["nodes"])
    except Exception as error:
        report["error"] = repr(error)
    finally:
        ray.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    for node in report["nodes"]:
        print(
            json.dumps(
                {
                    k: node[k]
                    for k in ("role", "hostname", "netdev", "roce_ipv4", "errors")
                }
            )
        )
    print(
        json.dumps(
            {
                "network_preflight_pass": report["network_preflight_pass"],
                "report": str(args.output),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["network_preflight_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
