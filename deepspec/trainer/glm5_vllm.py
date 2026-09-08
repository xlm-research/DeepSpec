"""Offline vLLM extraction for one bounded GLM training partition.

Only the child process imports vLLM. Training ranks retain CPU request metadata
while a node-local TP group owns the GPUs, and resume after that process exits.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

import torch

from deepspec.trainer.glm5_partitioned_swap import (
    Glm5PartitionCache,
    Glm5TrainingPartition,
    atomic_write_json,
    load_json,
)


@dataclass(frozen=True)
class VllmPartitionConfig:
    python_executable: str = sys.executable
    source_dir: str | None = None
    tensor_parallel_size: int = 4
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.8
    load_format: str = "instanttensor"
    timeout_seconds: int = 86400
    raw_cache_dir: str | None = None

    def __post_init__(self):
        for name in (
            "tensor_parallel_size",
            "max_num_batched_tokens",
            "timeout_seconds",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"vLLM {name} must be positive.")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError(
                "vLLM gpu_memory_utilization must be between zero and one."
            )


def rank_group(*, global_rank, local_rank, local_world_size, tp_size, devices):
    """Map contiguous local draft ranks to one independent vLLM TP replica."""
    if local_world_size < tp_size or local_world_size % tp_size:
        raise ValueError("vLLM TP size must divide LOCAL_WORLD_SIZE.")
    if not 0 <= local_rank < local_world_size or len(devices) < local_world_size:
        raise ValueError("vLLM requires one visible GPU per local training rank.")
    start = local_rank // tp_size * tp_size
    node_start = global_rank - local_rank
    return list(range(node_start + start, node_start + start + tp_size)), devices[
        start : start + tp_size
    ]


def child_environment(devices, source_dir=None):
    env = os.environ.copy()
    # Neither vLLM's own workers nor their TCP store may inherit torchrun ranks.
    for name in list(env):
        if name in {
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "GROUP_WORLD_SIZE",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "CUDA_VISIBLE_DEVICES",
            # vLLM's hidden-state connector requires stable KV allocations.
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
            # Launcher options have already been serialized in the job file.
            "VLLM_PYTHON_BIN",
            "VLLM_SOURCE_DIR",
            "VLLM_RAW_CACHE_DIR",
            "VLLM_MAX_NUM_BATCHED_TOKENS",
            "VLLM_GPU_MEMORY_UTILIZATION",
            "VLLM_LOAD_FORMAT",
            "VLLM_TIMEOUT_SECONDS",
        } or name.startswith("TORCHELASTIC_"):
            env.pop(name)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    # This is a fresh exec, with no inherited training CUDA context. Let vLLM
    # choose its default method and switch to spawn if CUDA is initialized.
    env["TOKENIZERS_PARALLELISM"] = "false"
    root = Path(__file__).resolve().parents[2]
    checkout = Path(source_dir) if source_dir else root / "vllm"
    paths = [str(root)]
    # The outer vllm/ checkout is a namespace package and otherwise shadows
    # editable installs when the worker starts from the DeepSpec repository.
    if (checkout / "vllm" / "__init__.py").is_file():
        paths.insert(0, str(checkout.resolve()))
    elif source_dir:
        raise ValueError(f"No vLLM package under source_dir={source_dir}.")
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    return env


def runtime_identity(config):
    """Fingerprint the actual worker interpreter's extraction implementation."""
    probe = """
import hashlib, importlib.metadata, importlib.util, json
from pathlib import Path
root = Path(importlib.util.find_spec('vllm').origin).parent
names = ['models/glm5next/nvidia/model.py', 'v1/core/kv_cache_utils.py',
         'distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py']
print(json.dumps({'version': importlib.metadata.version('vllm'),
                  'extraction_code': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                      for name in names}}))
"""
    result = subprocess.run(
        [config.python_executable, "-c", probe],
        env=child_environment([], config.source_dir),
        text=True,
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            f"Cannot inspect the vLLM extraction runtime: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def teacher_identity(*, model_path, layer_ids, config):
    root = Path(model_path)
    model_config = load_json(str(root / "config.json"))
    text_config = model_config.get("text_config", model_config)
    depth = int(text_config["num_hidden_layers"])
    if (
        layer_ids != sorted(set(layer_ids))
        or not layer_ids
        or any(layer < 0 or layer >= depth - 1 for layer in layer_ids)
    ):
        raise ValueError(
            "vLLM target_layer_ids must be sorted, unique, and precede the final layer."
        )
    index = load_json(str(root / "model.safetensors.index.json"))
    weights = []
    for name in sorted(set(index["weight_map"].values())):
        stat = (root / name).stat()
        weights.append((name, stat.st_size, stat.st_mtime_ns))
    return {
        "backend": "vllm",
        "format_version": 1,
        "runtime": runtime_identity(config),
        "config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()
        ).hexdigest(),
        "weights_sha256": hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
        "index_sha256": hashlib.sha256(
            (root / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "quantization": model_config.get(
            "quantization_config", text_config.get("quantization_config")
        ),
        "target_layer_ids": layer_ids,
        "aux_layer_ids": [layer + 1 for layer in layer_ids] + [depth],
        "final_hidden_source": "full_model_final_norm",
        "activation_dtype": "bfloat16",
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "load_format": config.load_format,
        "enable_prefix_caching": False,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def write_request(
    directory, batch, *, logical_sample_id, dataset_index, stream_micro_step
):
    ids, mask = batch["input_ids"], batch["loss_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError(
            "vLLM partition inputs require input_ids/loss_mask shaped [1, T]."
        )
    attention = batch.get("attention_mask")
    if attention is not None and not bool(attention.bool().all()):
        raise ValueError("vLLM partition extraction requires unpadded text inputs.")
    if any(name in batch for name in ("pixel_values", "images", "videos", "mm_inputs")):
        raise ValueError("vLLM partition extraction currently supports text only.")
    name = f"input_{logical_sample_id:012d}.pt"
    path = os.path.join(directory, name)
    with open(path, "xb") as handle:
        torch.save({"input_ids": ids.cpu(), "loss_mask": mask.cpu()}, handle)
    return {
        "input_file": name,
        "logical_sample_id": logical_sample_id,
        "dataset_index": dataset_index,
        "stream_micro_step": stream_micro_step,
    }


def convert_hidden_states(tensors, batch, *, norm_weight, norm_eps, num_layers):
    """Decode mHC auxiliary states; the last auxiliary state is pre-final-norm."""
    ids = batch["input_ids"][0]
    hidden = tensors["hidden_states"]
    expected = (ids.numel(), num_layers + 1, norm_weight.numel())
    if not torch.equal(tensors["token_ids"], ids):
        raise ValueError("vLLM output token IDs do not match the planned sample.")
    if tuple(hidden.shape) != expected or hidden.dtype != torch.bfloat16:
        raise ValueError(
            f"Expected BF16 hidden states {expected}, got {hidden.shape}/{hidden.dtype}."
        )
    last = torch.empty_like(hidden[:, -1])
    for start in range(0, len(ids), 2048):
        chunk = hidden[start : start + 2048]
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("vLLM hidden states contain non-finite values.")
        normalized = chunk[:, -1].float()
        normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + norm_eps)
        last[start : start + 2048] = (normalized * norm_weight.float()).to(hidden.dtype)
    return {
        **batch,
        "target_hidden_states": hidden[:, :-1].flatten(1).contiguous().unsqueeze(0),
        "target_last_hidden_states": last.unsqueeze(0),
        "context_start": torch.tensor([0]),
        "context_len": torch.tensor([len(ids)]),
        "seq_len": torch.tensor([len(ids)]),
    }


def run_worker_process(*, job_path, config, devices):
    """Wait for the child and terminate its entire process group before returning."""
    entrypoint = (
        Path(__file__).resolve().parents[2]
        / "scripts/data/generate_glm5_vllm_partition.py"
    )
    command = [config.python_executable, "-u", str(entrypoint), job_path]
    log_path = job_path + ".log"
    print(
        f"[deepspec-vllm] devices={devices} job={job_path} log={log_path}", flush=True
    )
    env = child_environment(devices, config.source_dir)
    env["DEEPSPEC_VLLM_PARENT_PID"] = str(os.getpid())
    with open(log_path, "a", buffering=1) as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=config.timeout_seconds)
            if code:
                raise RuntimeError(
                    f"vLLM extraction exited with status {code}; see {log_path}."
                )
        finally:
            # vLLM grandchildren can outlive a failed engine or a successful driver.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            process.wait()
            deadline = time.monotonic() + 30
            while live_process_group(process.pid):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"vLLM process group {process.pid} did not exit; draft loading is blocked."
                    )
                time.sleep(0.1)


def live_process_group(group_id):
    """Include orphaned workers; zombies no longer retain CUDA allocations."""
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group_id and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def generate_job(job, *, extract):
    """Write each output to its planned owner, without depending on prompt equality."""
    partition = Glm5TrainingPartition(**job["partition"])
    for rank in job["owner_ranks"]:
        cache = Glm5PartitionCache(root=job["cache_root"], global_rank=rank)
        incomplete, _ = cache.partition_paths(partition)
        request_manifest = load_json(os.path.join(incomplete, "vllm_requests.json"))
        if request_manifest["teacher"] != job["teacher"]:
            raise ValueError("vLLM request teacher identity changed.")
        samples = []
        for request in request_manifest["requests"]:
            batch = torch.load(
                os.path.join(incomplete, request["input_file"]), weights_only=True
            )
            features = extract(batch)
            samples.append(
                cache.write_sample(
                    partition=partition,
                    batch=features,
                    **{
                        name: request[name]
                        for name in (
                            "logical_sample_id",
                            "dataset_index",
                            "stream_micro_step",
                        )
                    },
                )
            )
            print(
                f"[deepspec-vllm] rank={rank} sample={request['logical_sample_id']} tokens={batch['input_ids'].numel()}",
                flush=True,
            )
        cache.write_local_manifest(
            partition=partition,
            samples=samples,
            target_shard_layout=request_manifest["target_shard_layout"],
            state="LOCAL_COMPLETE",
        )


def load_hidden_states_with_retry(path, *, loader, timeout_seconds):
    """Wait for the writer on filesystems that return EAGAIN for busy flock."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            # Keep the connector's shared lock: file existence alone does not
            # mean that its asynchronous safetensors write has completed.
            return loader(path)
        except BlockingIOError as error:
            if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for vLLM hidden-state writer: {path}"
                ) from error
            time.sleep(min(0.1, remaining))


def worker_main(job_path):
    # A killed torchrun parent must not leave an engine holding its devices.
    def terminate(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    parent_pid = int(os.environ.get("DEEPSPEC_VLLM_PARENT_PID", os.getppid()))

    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGKILL)

    if "DEEPSPEC_VLLM_PARENT_PID" in os.environ:
        threading.Thread(target=watch_parent, daemon=True).start()
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector as connector,
    )

    job = load_json(job_path)
    config = VllmPartitionConfig(**job["config"])
    actual_teacher = teacher_identity(
        model_path=job["model_path"],
        layer_ids=job["teacher"]["target_layer_ids"],
        config=config,
    )
    if actual_teacher != job["teacher"]:
        raise ValueError(
            "vLLM teacher or adapter changed after planning this partition."
        )
    root = Path(job["model_path"])
    index = load_json(str(root / "model.safetensors.index.json"))
    model_config = load_json(str(root / "config.json"))
    text_config = model_config.get("text_config", model_config)
    norm_name = "model.language_model.norm.weight"
    with safe_open(root / index["weight_map"][norm_name], framework="pt") as handle:
        norm_weight = handle.get_tensor(norm_name)
    # Keep at most one request's raw tensors. The final partition lives on disk.
    raw_root = config.raw_cache_dir or str(Path(job_path).parent)
    os.makedirs(raw_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="glm5-vllm-raw-", dir=raw_root) as raw_dir:
        llm = LLM(
            model=str(root),
            dtype="bfloat16",
            load_format=config.load_format,
            seed=0,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=job["max_length"] + 1,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": job["teacher"][
                            "aux_layer_ids"
                        ],
                    }
                },
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": raw_dir},
            ),
        )

        def extract(batch):
            (output,) = llm.generate(
                [{"prompt_token_ids": batch["input_ids"][0].tolist()}],
                SamplingParams(temperature=0, max_tokens=1),
                use_tqdm=False,
            )
            path = output.kv_transfer_params["hidden_states_path"]
            try:
                # generate() can finish before the asynchronous connector write.
                tensors = load_hidden_states_with_retry(
                    path,
                    loader=connector.load_hidden_states,
                    timeout_seconds=config.timeout_seconds,
                )
                return convert_hidden_states(
                    tensors,
                    batch,
                    norm_weight=norm_weight,
                    norm_eps=text_config["rms_norm_eps"],
                    num_layers=len(job["teacher"]["target_layer_ids"]),
                )
            finally:
                connector.cleanup_hidden_states(path)

        generate_job(job, extract=extract)
    atomic_write_json(
        job_path + ".complete",
        {"partition": job["partition"], "teacher": job["teacher"]},
    )


if __name__ == "__main__":
    worker_main(sys.argv[1])
