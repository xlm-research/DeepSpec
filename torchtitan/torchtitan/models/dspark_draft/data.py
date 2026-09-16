"""Read prepared feature batches through Titan's stateful data interface."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer
from .planning import input_identity
from .features import ProducerFeatures


class PreparedTokens(BaseTokenizer):
    """Token IDs are already fixed by feature production; text is not retokenized."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int

    def __init__(self, config, *, tokenizer_path):
        super().__init__()
        self.vocab_size = config.vocab_size

    def encode(self, tokens, **kwargs):
        if not isinstance(tokens, list) or any(
            not isinstance(token, int) or not 0 <= token < self.vocab_size
            for token in tokens
        ):
            raise ValueError("Prepared feature inputs require valid token IDs")
        return list(tokens)

    def decode(self, tokens, **kwargs):
        raise ValueError("Text decoding requires the target tokenizer assets")

    def get_vocab_size(self):
        return self.vocab_size


class FeatureLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        manifest: str
        global_microbatch_start: int = 0
        plan_path: str = ""
        target_layer_ids: list[int] = field(default_factory=list)
        hidden_size: int = 0
        require_producer_manifest: bool = False

    def __init__(
        self,
        config,
        *,
        dp_world_size,
        dp_rank,
        tokenizer,
        max_context_length,
        num_tokens_per_batch,
    ):
        self.manifest_path = Path(config.manifest).resolve()
        raw = self.manifest_path.read_bytes()
        manifest = json.loads(raw)
        entries = manifest["batches"]
        if not entries or len(entries) % dp_world_size:
            raise ValueError("Feature partition must contain complete DP microbatches")
        self.entries = entries[dp_rank::dp_world_size]
        self.identity = hashlib.sha256(raw).hexdigest()
        self.plan_identity = ""
        self.plan_run_id = ""
        self.expected_inputs = {}
        self.producer = None
        if config.plan_path:
            plan_bytes = Path(config.plan_path).read_bytes()
            plan = json.loads(plan_bytes)
            if plan["version"] != 1:
                raise ValueError("Unsupported DSpark input plan version")
            self.plan_identity = hashlib.sha256(plan_bytes).hexdigest()
            self.plan_run_id = plan["run_id"]
            start = config.global_microbatch_start * dp_world_size
            planned = plan["batches"][start : start + len(entries)]
            if [entry["id"] for entry in entries] != [entry["id"] for entry in planned]:
                raise ValueError(
                    "Feature partition differs from the whole-run input plan"
                )
            self.expected_inputs = {
                entry["id"]: entry["input_identity"] for entry in planned
            }
            self.expected_samples = {entry["id"]: entry for entry in planned}
        if "producer_manifest" in manifest:
            if not self.expected_inputs:
                raise ValueError("Producer features require a whole-run input plan")
            self.producer = ProducerFeatures(
                self.manifest_path.parent / manifest["producer_manifest"],
                manifest["producer_sha256"],
                layer_ids=config.target_layer_ids,
                hidden_size=config.hidden_size,
                vocab_size=tokenizer.get_vocab_size(),
            )
            for entry in entries:
                expected = self.expected_samples[entry["id"]]
                if expected["sample_id"] not in self.producer.samples:
                    raise ValueError(
                        f"Producer is missing planned sample {expected['sample_id']}"
                    )
                actual = self.producer.samples[expected["sample_id"]]
                if (
                    actual["position"] != expected["position"]
                    or actual["length"] != expected["length"]
                ):
                    raise ValueError(
                        "Producer sample order or length differs from the plan"
                    )
        elif config.require_producer_manifest:
            raise ValueError("This recipe requires verified producer feature facts")
        self.cursor = 0
        self.global_microbatch_start = config.global_microbatch_start
        self.num_tokens_per_batch = num_tokens_per_batch
        self.max_context_length = max_context_length

    def __iter__(self):
        while self.cursor < len(self.entries):
            entry = self.entries[self.cursor]
            batch = self.read_entry(entry)
            tokens = batch["input_ids"]
            if (
                self.expected_inputs
                and input_identity(batch) != self.expected_inputs[entry["id"]]
            ):
                raise ValueError(
                    "Feature tokens or loss mask differ from the input plan"
                )
            if (
                tokens.ndim != 2
                or tokens.shape[0] * self.max_context_length
                != self.num_tokens_per_batch
            ):
                raise ValueError(
                    "Feature batch does not match the training token budget"
                )
            if tokens.shape[1] > self.max_context_length:
                raise ValueError("Feature batch exceeds the training context limit")
            if batch["loss_mask"].shape != tokens.shape:
                raise ValueError("Feature tokens and supervision mask do not align")
            batch["num_valid_tokens"] = int(batch["loss_mask"].count_nonzero())
            self.cursor += 1
            yield batch, tokens

    def read_entry(self, entry):
        if self.producer is not None:
            expected = self.expected_samples[entry["id"]]
            return self.producer.read(
                expected["sample_id"], expected["input_identity"]
            )
        return torch.load(
            self.manifest_path.parent / entry["path"],
            map_location="cpu",
            weights_only=True,
        )

    def state_dict(self):
        return {
            "feature_identity": self.identity,
            "cursor": self.cursor,
            "next_global_microbatch": self.next_global_microbatch,
        }

    @property
    def next_global_microbatch(self):
        return self.global_microbatch_start + self.cursor

    def load_state_dict(self, state):
        if state["feature_identity"] != self.identity:
            if state["next_global_microbatch"] != self.global_microbatch_start:
                raise ValueError("The next partition does not continue the checkpoint")
            self.cursor = 0
            return
        cursor = int(state["cursor"])
        if not 0 <= cursor <= len(self.entries):
            raise ValueError("Checkpoint feature cursor is outside this partition")
        self.cursor = cursor
