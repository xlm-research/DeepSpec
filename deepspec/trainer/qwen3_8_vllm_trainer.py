"""Qwen-only vLLM teacher backend, preserving the existing DSpark update loop."""

from dataclasses import asdict
from datetime import timedelta
import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist

from deepspec.data.cuda_prefetcher import move_batch_to_device
from deepspec.trainer.dspark_trainer import Qwen3_8DSparkTrainer
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json, load_json
from deepspec.trainer.qwen3_8_vllm import (
    QwenVllmConfig,
    run_worker_process,
    teacher_identity,
)


def replica_layout(
    *, global_rank, local_rank, local_size, cp_size, tp_size, vllm_tp, devices
):
    """One vLLM replica serves one draft CP x TP group on the same node."""
    group_size = cp_size * tp_size
    if local_size % group_size or vllm_tp > group_size or group_size % vllm_tp:
        raise ValueError(
            "Draft CP x TP groups must fit on one node and be divisible by vLLM TP."
        )
    if len(devices) != local_size or not 0 <= local_rank < local_size:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one device per local rank.")
    start = local_rank // group_size * group_size
    owner = global_rank - local_rank + start
    return (
        owner,
        [owner + rank * tp_size for rank in range(cp_size)],
        devices[start : start + vllm_tp],
    )


class Qwen3_8VllmDSparkTrainer(Qwen3_8DSparkTrainer):
    """Extract a partition in a subprocess, then train with its immutable features.

    The draft, optimizer, RNG and in-flight accumulated gradients stay resident.
    This retains Qwen's exact per-epoch partition boundaries, including boundaries
    inside an optimizer accumulation window. No GLM model-swap code is changed.
    """

    def __init__(self, local_rank, args):
        if (
            args.data.get("online_target", False)
            or not args.data.get("offline_target_data_batches", False)
            or args.data.get("multimodal", False)
            or (args.train.get("partitioned_model_swap") or {}).get("enabled", False)
        ):
            raise ValueError(
                "Qwen vLLM requires text-only bounded offline target batches."
            )
        self.qwen_vllm_config = QwenVllmConfig(**args.train.qwen_vllm)
        super().__init__(local_rank, args)
        if self.heterogeneous_target_data_batches:
            raise ValueError(
                "Qwen vLLM requires matching draft and offline-target rank layouts."
            )
        self._vllm_control_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=self.qwen_vllm_config.timeout_seconds + 300),
        )
        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = devices.split(",") if devices else [str(i) for i in range(local_size)]
        self._vllm_owner, self._cp_cache_owners, self._vllm_devices = replica_layout(
            global_rank=self.global_rank,
            local_rank=local_rank,
            local_size=local_size,
            cp_size=self.context_parallel_size,
            tp_size=self.parallel_config.tp,
            vllm_tp=self.qwen_vllm_config.tensor_parallel_size,
            devices=devices,
        )
        if self._vllm_owner != self.parallel.model_parallel_src_rank:
            raise ValueError(
                "Qwen vLLM owner does not match the draft model-parallel mesh."
            )
        print(
            f"[qwen38-vllm] rank={self.global_rank} owner={self._vllm_owner} "
            f"teacher_tp={self.qwen_vllm_config.tensor_parallel_size} "
            f"draft_cp={self.context_parallel_size} draft_tp={self.parallel_config.tp} "
            "teacher_layers=64 final_state=full_model_final_norm_output",
            flush=True,
        )

    def _build_draft_model(self, *, target_config, model_args):
        self._teacher_identity = teacher_identity(
            model_args.target_model_name_or_path, list(model_args.target_layer_ids)
        )
        identity_path = Path(self.checkpoint_dir_root) / "qwen38_vllm_teacher.json"
        error = None
        if self.global_rank == 0:
            try:
                previous = load_json(identity_path)
                if previous is not None and previous != self._teacher_identity:
                    raise ValueError(
                        "The Qwen vLLM checkpoint directory belongs to a different teacher."
                    )
                if self.resume_checkpoint_dir is not None and previous is None:
                    raise ValueError(
                        "A Qwen vLLM resume requires its teacher identity sidecar."
                    )
                atomic_write_json(identity_path, self._teacher_identity)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise ValueError(errors[0])
        draft = super()._build_draft_model(
            target_config=target_config, model_args=model_args
        )
        draft.config.deepspec_target_backend = "vllm"
        draft.config.deepspec_target_execution_num_hidden_layers = 64
        draft.config.deepspec_target_final_hidden_source = self._teacher_identity[
            "target_final_hidden_source"
        ]
        return draft

    def build_online_target(self):
        # The subprocess owns the entire teacher; no native teacher is allocated.
        return None

    def prepare_online_target_batch(self, batch):
        raise RuntimeError(
            "Qwen vLLM extraction runs only at a target partition boundary."
        )

    def _collective_stage(self, action):
        error = None
        result = None
        try:
            result = action()
        except Exception as exc:
            error = f"rank {self.global_rank}: {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error, group=self._vllm_control_group)
        if any(errors):
            raise RuntimeError(
                "Qwen vLLM partition failed: " + "; ".join(e for e in errors if e)
            )
        return result

    def iter_training_batches(self, batches):
        if self.data_batch_micro_batches is None:
            raise RuntimeError("Qwen vLLM requires a bounded data partition schedule.")
        batches = iter(batches)
        cp_rank = self.parallel.context_parallel_rank
        cache_owner = self._cp_cache_owners[cp_rank]
        for partition_index, count in enumerate(self.data_batch_micro_batches, start=1):
            self._data_batch_phase = "target_inference"
            paths = [
                self._data_batch_cache_file_path(
                    batch_index=partition_index, sample_index=i, cache_rank=cache_owner
                )
                for i in range(count)
            ]
            self._active_data_batch_cache = paths
            job_dir = None

            def prepare():
                nonlocal job_dir
                if self.global_rank == self._vllm_owner:
                    job_dir = Path(
                        tempfile.mkdtemp(
                            prefix="qwen38-job-", dir=self.data_batch_rank_cache_dir
                        )
                    )
                requests = []
                for index in range(count):
                    batch = next(batches)
                    if self.global_rank == self._vllm_owner:
                        if batch["input_ids"].shape[0] != 1 or not bool(
                            batch["attention_mask"].bool().all()
                        ):
                            raise ValueError(
                                "Qwen vLLM requires one unpadded text sequence per request."
                            )
                        input_path = job_dir / f"input_{index:08d}.pt"
                        torch.save(
                            {
                                name: batch[name].cpu()
                                for name in ("input_ids", "loss_mask")
                            },
                            input_path,
                        )
                        requests.append(
                            {
                                "input_path": str(input_path),
                                "output_paths": [
                                    self._data_batch_cache_file_path(
                                        batch_index=partition_index,
                                        sample_index=index,
                                        cache_rank=rank,
                                    )
                                    for rank in self._cp_cache_owners
                                ],
                            }
                        )
                    batch.clear()
                if self.global_rank == self._vllm_owner:
                    atomic_write_json(
                        job_dir / "job.json",
                        {
                            "model_path": self.args.model.target_model_name_or_path,
                            "teacher": self._teacher_identity,
                            "config": asdict(self.qwen_vllm_config),
                            "max_length": int(self.args.data.max_length),
                            "requests": requests,
                        },
                    )
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()

            self._collective_stage(prepare)

            def generate():
                if self.global_rank != self._vllm_owner:
                    return
                job_path = job_dir / "job.json"
                print(
                    f"[qwen38-vllm] partition={partition_index} samples={count} job={job_path}",
                    flush=True,
                )
                run_worker_process(job_path, self.qwen_vllm_config, self._vllm_devices)
                complete = load_json(str(job_path) + ".complete")
                if (
                    complete is None
                    or complete.get("teacher") != self._teacher_identity
                    or len(complete.get("samples", [])) != count
                ):
                    raise RuntimeError(
                        "Qwen vLLM partition completion record is missing or inconsistent."
                    )
                # Keep verification evidence without retaining request tensors.
                atomic_write_json(
                    Path(self.checkpoint_dir_root)
                    / f"vllm_rank{self._vllm_owner}_partition{partition_index}.json",
                    complete,
                )
                shutil.rmtree(job_dir)

            self._collective_stage(generate)
            self._data_batch_phase = "draft_training"
            for index, path in enumerate(paths):
                cpu_batch = torch.load(path, map_location="cpu", weights_only=True)
                gpu_batch = move_batch_to_device(cpu_batch, self.device)
                del cpu_batch
                self._data_batch_end_after_current = index + 1 == len(paths)
                yield gpu_batch
                self._data_batch_end_after_current = False
                del gpu_batch

            torch.cuda.synchronize(self.device)
            dist.barrier(group=self._vllm_control_group)
            self._collective_stage(
                lambda: (
                    self._delete_data_batch_cache(paths)
                    if self.global_rank == cache_owner
                    else None
                )
            )
            self._active_data_batch_cache = None
            self._data_batch_phase = None
