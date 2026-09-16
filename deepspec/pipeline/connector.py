"""Use vLLM's existing hidden-state extraction with a Mooncake writer."""

import json
import os
import time
from pathlib import Path

import ray
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
)

from deepspec.trainer.qwen3_8_vllm import convert_hidden_states

from .store import TensorStore, describe_tensors


class MooncakeHiddenStatesConnector(ExampleHiddenStatesConnector):
    def __init__(self, vllm_config, role, *args, **kwargs):
        super().__init__(vllm_config, role, *args, **kwargs)
        path = self._kv_transfer_config.get_from_extra_config("pipeline_config", "")
        self.pipeline = json.loads(Path(path).read_text())
        self.buffer = None
        self.store = None

    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        self.buffer = ray.get_actor(
            self.pipeline["buffer_name"], namespace=self.pipeline["namespace"]
        )
        context = ray.get_runtime_context()
        ray.get(
            self.buffer.event.remote(
                "producer_worker",
                pid=os.getpid(),
                node_id=context.get_node_id(),
                ray_gpu_ids=context.get_accelerator_ids().get("GPU", []),
                cuda_device=torch.cuda.current_device(),
            )
        )
        if self._is_tp_rank_zero:
            self.store = TensorStore(self.pipeline["store"])

    def _write_tensors(self, tensors, event, filename, lock_fd):
        started = time.monotonic()
        position = int(Path(filename).name)
        sample = self.pipeline["samples"][position]
        try:
            event.synchronize()
            batch = torch.load(
                sample["input_path"], weights_only=True, map_location="cpu"
            )
            features = convert_hidden_states(
                tensors,
                batch,
                hidden_size=self.pipeline["teacher"]["hidden_size"],
                num_layers=len(self.pipeline["teacher"]["target_layer_ids"]),
            )
            fields = describe_tensors(
                f"dspark/{self.pipeline['run_id']}/{position}", features
            )
            self.store.put(fields, features)
            descriptor = {
                key: sample[key]
                for key in ("position", "sample_id", "input_identity", "length")
            }
            descriptor["fields"] = fields
            ray.get(self.buffer.publish.remote(position, descriptor))
            ray.get(
                self.buffer.event.remote(
                    "write_complete",
                    position=position,
                    seconds=time.monotonic() - started,
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
