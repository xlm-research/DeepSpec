"""Full Qwen3.8 teacher extraction for bounded DSpark training partitions."""

from __future__ import annotations

from dataclasses import dataclass
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

from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.trainer.glm5_vllm import (
    child_environment,
    live_process_group,
    load_hidden_states_with_retry,
)


@dataclass(frozen=True)
class QwenVllmConfig:
    python_executable: str = sys.executable
    source_dir: str | None = None
    tensor_parallel_size: int = 4
    max_num_batched_tokens: int = 8192
    # The draft and its accumulated gradients remain resident between partitions.
    gpu_memory_utilization: float = 0.45
    load_format: str = "auto"
    timeout_seconds: int = 86400
    verify_logits: bool = False
    logprob_atol: float = 0.1

    def __post_init__(self):
        for name in (
            "tensor_parallel_size",
            "max_num_batched_tokens",
            "timeout_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"Qwen vLLM {name} must be positive.")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("Qwen vLLM gpu_memory_utilization must be in (0, 1).")
        if self.logprob_atol <= 0:
            raise ValueError("Qwen vLLM logprob_atol must be positive.")


def teacher_identity(model_path, layer_ids):
    root = Path(model_path)
    config = load_json(str(root / "config.json"))
    text = config.get("text_config", {})
    if (
        config.get("model_type") != "qwen3_5"
        or text.get("num_hidden_layers") != 64
        or text.get("hidden_size") != 5120
    ):
        raise ValueError("The vLLM Qwen3.8 trainer requires a Qwen3.8-27B checkpoint.")
    if not layer_ids or layer_ids != sorted(set(layer_ids)):
        raise ValueError("target_layer_ids must be nonempty, sorted, and unique.")
    if any(layer < 0 or layer >= 63 for layer in layer_ids):
        raise ValueError("Qwen auxiliary decoder layer IDs must be in [0, 62].")
    index = load_json(str(root / "model.safetensors.index.json"))
    weights = []
    for name in sorted(set(index["weight_map"].values())):
        stat = (root / name).stat()
        weights.append((name, stat.st_size, stat.st_mtime_ns))
    return {
        "backend": "vllm",
        "model_path": str(root.resolve()),
        "config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()
        ).hexdigest(),
        "index_sha256": hashlib.sha256(
            (root / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "weights_sha256": hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
        "target_layer_ids": list(layer_ids),
        "aux_layer_ids": [layer + 1 for layer in layer_ids] + [64],
        "target_execution_num_hidden_layers": 64,
        "target_final_hidden_source": "full_model_final_norm_output",
        "hidden_size": 5120,
        "activation_dtype": "bfloat16",
    }


def context_indices(sequence_length, cp_size, cp_rank):
    """Return the global token positions expected by Qwen's native CP draft."""
    if sequence_length < 1 or cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("Invalid sequence length or context-parallel coordinate.")
    if cp_size == 1:
        return torch.arange(sequence_length)
    chunk = (sequence_length + 2 * cp_size - 1) // (2 * cp_size)
    offsets = torch.arange(chunk)
    return torch.cat(
        (offsets + cp_rank * chunk, offsets + (2 * cp_size - cp_rank - 1) * chunk)
    )


def convert_hidden_states(
    tensors, batch, *, hidden_size, num_layers, cp_size=1, cp_rank=0
):
    """Preserve decoder outputs and the teacher's actual final normalized state."""
    ids = batch["input_ids"]
    mask = batch["loss_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError(
            "Qwen teacher inputs must have input_ids/loss_mask shaped [1, T]."
        )
    hidden = tensors["hidden_states"]
    length = ids.shape[1]
    expected = (length, num_layers + 1, hidden_size)
    if not torch.equal(tensors["token_ids"], ids[0]):
        raise ValueError("Extracted token IDs do not match the training input.")
    if hidden.shape != expected or hidden.dtype != torch.bfloat16:
        raise ValueError(
            f"Expected BF16 features {expected}, got {hidden.shape}/{hidden.dtype}."
        )
    indices = context_indices(length, cp_size, cp_rank)
    local = hidden.index_select(0, indices.clamp_max(length - 1))
    local[indices >= length] = 0
    for start in range(0, len(local), 2048):
        chunk = local[start : start + 2048]
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Qwen teacher features contain non-finite values.")
    return {
        "input_ids": ids,
        "loss_mask": mask,
        "target_hidden_states": local[:, :-1].flatten(1).contiguous().unsqueeze(0),
        "target_last_hidden_states": local[:, -1].contiguous().unsqueeze(0),
        "context_chunk_len": torch.tensor([len(indices)]),
        "seq_len": torch.tensor([length]),
    }


def run_worker_process(job_path, config, devices):
    """Run a fresh interpreter and release every worker before draft training."""
    entrypoint = (
        Path(__file__).resolve().parents[2]
        / "scripts/data/generate_qwen3_8_vllm_partition.py"
    )
    env = child_environment(devices, config.source_dir)
    env["DEEPSPEC_VLLM_PARENT_PID"] = str(os.getpid())
    with open(str(job_path) + ".log", "a", buffering=1) as log:
        process = subprocess.Popen(
            [config.python_executable, "-u", str(entrypoint), str(job_path)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=config.timeout_seconds)
            if code:
                raise RuntimeError(
                    f"Qwen vLLM exited with status {code}; see {job_path}.log"
                )
        finally:
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
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Qwen vLLM workers still own GPUs; draft training is blocked."
                    )
                time.sleep(0.1)


def _watch_parent():
    parent_pid = int(os.environ.get("DEEPSPEC_VLLM_PARENT_PID", os.getppid()))
    if "DEEPSPEC_VLLM_PARENT_PID" not in os.environ:
        return

    def watch():
        while True:
            time.sleep(1)
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()


def worker_main(job_path):
    _watch_parent()
    from safetensors import safe_open
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector as connector,
    )

    job = load_json(job_path)
    config = QwenVllmConfig(**job["config"])
    identity = teacher_identity(job["model_path"], job["teacher"]["target_layer_ids"])
    if identity != job["teacher"]:
        raise ValueError(
            "Qwen teacher identity changed after this partition was planned."
        )
    root = Path(job["model_path"])
    index = load_json(str(root / "model.safetensors.index.json"))
    with tempfile.TemporaryDirectory(
        prefix="qwen38-raw-", dir=Path(job_path).parent
    ) as raw_dir:
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
            language_model_only=True,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": identity["aux_layer_ids"],
                        "extract_final_hidden_state": True,
                    }
                },
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "shared_storage_path": raw_dir,
                    "separate_hidden_state_pages": True,
                },
            ),
        )
        head = None
        if config.verify_logits:
            with safe_open(
                root / index["weight_map"]["lm_head.weight"], framework="pt"
            ) as handle:
                head = handle.get_tensor("lm_head.weight").cuda()
        results = []
        for request in job["requests"]:
            batch = torch.load(request["input_path"], weights_only=True)
            (output,) = llm.generate(
                [{"prompt_token_ids": batch["input_ids"][0].tolist()}],
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    logprobs=20 if head is not None else None,
                ),
                use_tqdm=False,
            )
            path = output.kv_transfer_params["hidden_states_path"]
            try:
                tensors = load_hidden_states_with_retry(
                    path,
                    loader=connector.load_hidden_states,
                    timeout_seconds=config.timeout_seconds,
                )
                error = None
                for cp_rank, output_path in enumerate(request["output_paths"]):
                    features = convert_hidden_states(
                        tensors,
                        batch,
                        hidden_size=identity["hidden_size"],
                        num_layers=len(identity["target_layer_ids"]),
                        cp_size=len(request["output_paths"]),
                        cp_rank=cp_rank,
                    )
                    if head is not None and cp_rank == 0:
                        final = tensors["hidden_states"][-1:, -1].cuda()
                        logprobs = (
                            torch.nn.functional.linear(final, head)
                            .float()[0]
                            .log_softmax(-1)
                        )
                        error = max(
                            abs(logprobs[token].item() - prob.logprob)
                            for token, prob in output.outputs[0].logprobs[0].items()
                        )
                        if error > config.logprob_atol:
                            raise ValueError(
                                f"Qwen final-state logprob error {error} exceeds {config.logprob_atol}."
                            )
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(features, str(output_path) + ".tmp")
                    os.replace(str(output_path) + ".tmp", output_path)
                    del features
                results.append(
                    {"tokens": batch["input_ids"].numel(), "logprob_max_abs": error}
                )
                print(f"[qwen38-vllm] {results[-1]}", flush=True)
            finally:
                connector.cleanup_hidden_states(path)
    atomic_write_json(
        str(job_path) + ".complete", {"teacher": identity, "samples": results}
    )
