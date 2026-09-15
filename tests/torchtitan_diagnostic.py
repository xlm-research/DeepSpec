"""Capture actual first-update projections for TP numerical diagnosis."""

from dataclasses import dataclass, fields
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tests.torchtitan_phase_fixtures import fixed_features, ObservedTrainer


class TraceTrainer(ObservedTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(ObservedTrainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.projections = {}
        for name, module in self.model_parts[0].named_modules():
            if name.startswith("layers.") and (
                isinstance(module, torch.nn.Linear)
                or type(module).__name__ in ("DraftNorm", "Qwen3RMSNorm")
            ):
                module.register_forward_hook(
                    lambda module, args, output, name=name: self.capture(
                        name, module, args, output
                    )
                )

    def capture(self, name, module, args, output):
        entries = self.projections.setdefault(name, [])
        if len(entries) == 2:
            return
        weight = module.weight.detach()
        if hasattr(weight, "to_local"):
            weight = weight.to_local()
        item = {
            "input": args[0].detach().cpu().clone(),
            "output": output.detach().cpu().clone(),
            "weight": weight.cpu().clone(),
        }
        entries.append(item)
        output.register_hook(
            lambda grad: item.__setitem__("grad_output", grad.detach().cpu().clone())
        )

    def close(self):
        torch.save(
            self.projections,
            Path(os.environ["DEEPSPEC_PHASE_TEST_ROOT"])
            / f"projections-rank{dist.get_rank()}.pt",
        )
        super().close()


def traced_features():
    config = fixed_features()
    return TraceTrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config)}
    )
