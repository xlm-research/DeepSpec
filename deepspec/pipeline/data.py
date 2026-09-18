"""Keep Titan's input plan while resolving ready feature objects from Mooncake."""

import json
import os
from dataclasses import dataclass, replace

import ray
import torch
import torch.distributed as dist
from torchtitan.models.dspark_draft.data import FeatureLoader

from .prefetch import FeaturePrefetch
from .schema import normalize_pipeline_config
from .store import FIELDS, TensorStore
from .topology import consumer_dp


class MooncakeFeatureLoader(FeatureLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(FeatureLoader.Config):
        pipeline_config: str = ""

    def __init__(self, config, **kwargs):
        from pathlib import Path

        self.pipeline = json.loads(Path(config.pipeline_config).read_text())
        normalize_pipeline_config(self.pipeline)
        dp = consumer_dp(self.pipeline)
        tp = self.pipeline["consumer_world_size"] // dp
        if (
            kwargs["dp_world_size"] != dp
            or dist.get_world_size() != self.pipeline["consumer_world_size"]
            or kwargs["dp_rank"] != dist.get_rank() // tp
        ):
            raise ValueError("Native consumer DP/TP ranks differ from the stream plan")
        self.dp_rank = kwargs["dp_rank"]
        if not config.plan_path:
            raise ValueError("Streaming features require the native input plan")
        super().__init__(replace(config, require_producer_manifest=False), **kwargs)
        teacher = self.pipeline["teacher"]
        requirements = json.loads(Path(config.plan_path).read_text())[
            "producer_requirements"
        ]
        if any(teacher.get(key) != value for key, value in requirements.items()):
            raise ValueError(
                "Mooncake target identity differs from the native input plan"
            )
        if (
            config.target_layer_ids != teacher["target_layer_ids"]
            or config.hidden_size != teacher["hidden_size"]
            or teacher["activation_dtype"] != "bfloat16"
            or teacher["target_final_hidden_source"] != "full_model_final_norm_output"
        ):
            raise ValueError("Mooncake teacher identity differs from the DSpark recipe")
        if self.pipeline["run_id"] != self.plan_run_id:
            raise ValueError("Mooncake buffer belongs to a different input plan")
        if not ray.is_initialized():
            ray.init(
                address=os.environ["RAY_ADDRESS"],
                namespace=self.pipeline["namespace"],
                log_to_driver=False,
            )
        self.buffer = ray.get_actor(
            self.pipeline["buffer_name"], namespace=self.pipeline["namespace"]
        )
        self.store = TensorStore(self.pipeline["store"])
        self.reader = dist.get_rank()
        self.descriptors = {}
        self.prefetch = None
        if self.pipeline["receive_device"] == "cpu":
            self.prefetch = FeaturePrefetch(
                self.pipeline["store"],
                depth=int(self.pipeline["transport"]["prefetch_depth"]),
                max_bytes=self.pipeline["transport"]["prefetch_bytes"],
                timeout=self.pipeline["timeout_seconds"],
                device=torch.cuda.current_device()
                if torch.cuda.is_available()
                else None,
                event=self._transfer_event,
            )

    def _transfer_event(self, event, **fields):
        ray.get(self.buffer.event.remote(event, reader=self.reader, **fields))

    def _fill_prefetch(self):
        if self.prefetch is not None:
            for position, descriptor in self.descriptors.items():
                if len(self.prefetch.pending) >= self.prefetch.depth:
                    break
                if position not in self.prefetch.pending:
                    self.prefetch.submit(descriptor)

    def take_features(self, descriptor, device):
        position = descriptor["position"]
        if self.prefetch is None:
            from .store import FEATURE_FIELDS

            result = self.store.get(descriptor["fields"], FEATURE_FIELDS, device=device)
        else:
            result = self.prefetch.take(position)
        del self.descriptors[position]
        self._fill_prefetch()
        return result

    def read_entry(self, entry):
        expected = self.expected_samples[entry["id"]]
        descriptor = ray.get(
            self.buffer.claim.remote(expected["position"], self.reader)
        )
        for key in ("position", "sample_id", "input_identity", "length"):
            if descriptor[key] != expected[key]:
                raise ValueError(f"Streamed {key} differs from the input plan")
        length = expected["length"]
        hidden = self.pipeline["teacher"]["hidden_size"]
        layers = len(self.pipeline["teacher"]["target_layer_ids"])
        for name, width in (
            ("target_hidden_states", hidden * layers),
            ("target_last_hidden_states", hidden),
        ):
            spec = descriptor["fields"][name]
            if spec["shape"] != [1, length, width] or spec["dtype"] != "bfloat16":
                raise ValueError(
                    f"Streamed {name} does not match the DSpark input contract"
                )
        batch = self.store.get(descriptor["fields"], FIELDS[:-2])
        self.descriptors[descriptor["position"]] = descriptor
        self._fill_prefetch()
        batch["_mooncake_features"] = descriptor
        return batch

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()
        self.store.close()
        ray.shutdown()
