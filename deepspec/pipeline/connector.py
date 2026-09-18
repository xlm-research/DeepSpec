"""Use vLLM's existing hidden-state extraction with a Mooncake writer."""

import json
import os
import threading
import time
from pathlib import Path

import ray
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)

from deepspec.trainer.qwen3_8_vllm import convert_hidden_states

from .schema import normalize_pipeline_config
from .store import TensorStore, describe_tensors
from .topology import sample_producer


class MooncakeHiddenStatesConnector(ExampleHiddenStatesConnector):
    def __init__(self, vllm_config, role, *args, **kwargs):
        super().__init__(vllm_config, role, *args, **kwargs)
        path = self._kv_transfer_config.get_from_extra_config("pipeline_config", "")
        self.pipeline = json.loads(Path(path).read_text())
        normalize_pipeline_config(self.pipeline)
        # Dense EngineCore resets data_parallel_rank, retaining this index.
        self.producer_rank = vllm_config.parallel_config.data_parallel_index
        self.buffer = None
        self.store = None
        limit = self.pipeline.get("writer_inflight")
        self.write_slots = threading.BoundedSemaphore(limit) if limit else None

    def _submit_async_write(self, pending):
        # Acquire before the parent's pinned D2H allocation, not inside its
        # executor: otherwise queued requests can retain the entire pool in RAM.
        if not self._is_tp_rank_zero:
            return
        if self.write_slots is not None and not self.write_slots.acquire(
            timeout=self.pipeline["timeout_seconds"]
        ):
            raise TimeoutError("Hidden-state writer staging timed out")
        try:
            super()._submit_async_write(pending)
        except BaseException:
            if self.write_slots is not None:
                self.write_slots.release()
            raise

    def _on_write_done(self, req_id, future):
        try:
            super()._on_write_done(req_id, future)
        finally:
            if self.write_slots is not None:
                self.write_slots.release()

    def register_kv_caches(self, kv_caches):
        from vllm.distributed import get_tensor_model_parallel_rank

        super().register_kv_caches(kv_caches)
        self.buffer = ray.get_actor(
            self.pipeline["buffer_name"], namespace=self.pipeline["namespace"]
        )
        context = ray.get_runtime_context()
        self.node_id = context.get_node_id()
        ray.get(
            self.buffer.event.remote(
                "producer_worker",
                pid=os.getpid(),
                node_id=context.get_node_id(),
                ray_gpu_ids=context.get_accelerator_ids().get("GPU", []),
                cuda_device=torch.cuda.current_device(),
                producer_rank=self.producer_rank,
                tp_rank=get_tensor_model_parallel_rank(),
            )
        )
        if self._is_tp_rank_zero:
            writer_slots = max(1, int(self.pipeline["transport"]["async_put_pool_size"]))
            self.store = TensorStore(
                self.pipeline["store"],
                async_put_pool_size=writer_slots,
                host_buffer_size=int(self.pipeline.get("max_sample_nbytes", 0)),
            )

    def _write_tensors(self, tensors, event, filename, lock_fd):
        started = time.monotonic()
        position = int(Path(filename).name)
        sample = self.pipeline["samples"][position]
        try:
            if not self._is_tp_rank_zero or self.producer_rank != sample_producer(
                self.pipeline, position
            ):
                raise ValueError("Feature reached the wrong producer DP/TP worker")
            ray.get(
                self.buffer.begin_write.remote(
                    position, self.producer_rank, self.node_id
                )
            )
            event.synchronize()
            synchronized = time.monotonic()
            batch = torch.load(
                sample["input_path"], weights_only=True, map_location="cpu"
            )
            input_loaded = time.monotonic()
            features = convert_hidden_states(
                tensors,
                batch,
                hidden_size=self.pipeline["teacher"]["hidden_size"],
                num_layers=len(self.pipeline["teacher"]["target_layer_ids"]),
            )
            converted = time.monotonic()
            fields = describe_tensors(
                f"dspark/{self.pipeline['run_id']}/{position}",
                features,
                chunk_bytes=self.pipeline["transport"]["chunk_bytes"],
            )
            described = time.monotonic()
            transfer = self.store.put_async(
                fields,
                features,
                timeout=self.pipeline["timeout_seconds"],
            )
            write_metrics = transfer.wait(timeout=self.pipeline["timeout_seconds"])
            # The handle result is authoritative when multiple writer slots
            # complete out of order; retaining it here also keeps the legacy
            # StoreProbe/diagnostic interface intact.
            self.store.last_write = write_metrics
            descriptor = {
                key: sample[key]
                for key in ("position", "sample_id", "input_identity", "length")
            }
            descriptor["fields"] = fields
            ray.get(
                self.buffer.publish.remote(
                    position,
                    descriptor,
                    self.producer_rank,
                    {
                        "seconds": time.monotonic() - started,
                        "synchronize_seconds": synchronized - started,
                        "convert_seconds": converted - synchronized,
                        "input_load_seconds": input_loaded - synchronized,
                        "feature_convert_seconds": converted - input_loaded,
                        "describe_seconds": described - converted,
                        "store": write_metrics,
                    },
                )
            )
        except BaseException as error:
            ray.get(
                self.buffer.fail.remote(f"Feature write {position} failed: {error!r}")
            )
            raise
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
