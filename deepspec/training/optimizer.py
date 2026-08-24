from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor, distribute_tensor

from deepspec.utils.optim import CosineAnnealingWarmupLR


class MasterWeightAdamW(Optimizer):
    """AdamW over model parameters with FP32 master weights and moments.

    Unlike the former detached-parameter wrapper, optimizer param groups refer
    to the real model Parameters/DTensors. This makes optimizer state
    reshardable through ``torch.distributed.checkpoint.state_dict`` while
    preserving the project's FP32-master update semantics.
    """

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Invalid AdamW hyperparameters.")
        defaults = dict(
            lr=float(lr),
            betas=tuple(float(value) for value in betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        # Eager state allocation gives distributed checkpoint load a complete
        # sharded tensor template even before the first optimizer step.
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad:
                    self._initialize_parameter_state(parameter)

    def _initialize_parameter_state(self, parameter) -> None:
        state = self.state[parameter]
        if state:
            return
        state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
        state["master_param"] = parameter.detach().clone().to(torch.float32)
        state["exp_avg"] = torch.zeros_like(state["master_param"])
        state["exp_avg_sq"] = torch.zeros_like(state["master_param"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MasterWeightAdamW does not support sparse gradients.")
                self._initialize_parameter_state(parameter)
                state = self.state[parameter]
                state["step"].add_(1)
                step = int(state["step"].item())
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_float = gradient.to(torch.float32)
                if weight_decay:
                    master.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad_float, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad_float,
                    grad_float,
                    value=1.0 - beta2,
                )
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                master.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-lr / bias_correction1,
                )
                parameter.copy_(master.to(parameter.dtype))
        return loss


class BF16Optimizer:
    """Trainer-facing optimizer/scheduler bundle with legacy-state loading."""

    def __init__(self, model, lr, total_steps, warmup_ratio, weight_decay=0.0):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.optimizer = MasterWeightAdamW(
            parameters,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=int(total_steps),
            warmup_steps=int(float(warmup_ratio) * int(total_steps)),
        )

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        optimizer_state = state_dict["optimizer_state_dict"]
        if "fp32_params" in state_dict:
            optimizer_state = self._convert_legacy_optimizer_state(
                optimizer_state,
                state_dict["fp32_params"],
            )
        self.optimizer.load_state_dict(optimizer_state)
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self._copy_master_weights_to_model()

    def _convert_legacy_optimizer_state(self, optimizer_state, fp32_params):
        converted = self.optimizer.state_dict()
        converted["param_groups"] = optimizer_state["param_groups"]
        new_ids = [
            parameter_id
            for group in converted["param_groups"]
            for parameter_id in group["params"]
        ]
        if len(new_ids) != len(fp32_params):
            raise ValueError(
                "Legacy checkpoint optimizer parameter count does not match the model: "
                f"{len(fp32_params)} != {len(new_ids)}."
            )
        old_ids = [
            parameter_id
            for group in optimizer_state["param_groups"]
            for parameter_id in group["params"]
        ]
        converted_state = {}
        for index, (new_id, old_id, master) in enumerate(zip(new_ids, old_ids, fp32_params)):
            current = converted["state"][new_id]
            old = optimizer_state["state"].get(old_id, {})
            self._copy_legacy_tensor(
                current["master_param"],
                master,
                name=f"fp32_params[{index}]",
            )
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if name in old:
                    value = old[name]
                    if torch.is_tensor(value) and torch.is_tensor(current[name]):
                        self._copy_legacy_tensor(
                            current[name],
                            value,
                            name=f"optimizer.state[{index}].{name}",
                        )
                    else:
                        current[name] = value
            converted_state[new_id] = current
        converted["state"] = converted_state
        return converted

    @staticmethod
    def _copy_legacy_tensor(destination, source, *, name: str) -> None:
        if isinstance(destination, DTensor):
            local = destination.to_local()
            if tuple(source.shape) == tuple(local.shape):
                local.copy_(source.to(device=local.device, dtype=local.dtype))
                return
            if tuple(source.shape) == tuple(destination.shape):
                distributed = distribute_tensor(
                    source.to(device=local.device, dtype=destination.dtype),
                    destination.device_mesh,
                    destination.placements,
                )
                destination.copy_(distributed)
                return
            raise ValueError(
                "Legacy optimizer shard layout is incompatible with the new "
                f"FSDP2 mesh for {name}: saved={tuple(source.shape)}, "
                f"local={tuple(local.shape)}, global={tuple(destination.shape)}. "
                "Keep the old checkpoint and warm-start from its HF model export."
            )
        if tuple(source.shape) != tuple(destination.shape):
            raise ValueError(
                f"Legacy optimizer tensor shape mismatch for {name}: "
                f"{tuple(source.shape)} != {tuple(destination.shape)}."
            )
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    @torch.no_grad()
    def _copy_master_weights_to_model(self):
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter.copy_(
                    self.optimizer.state[parameter]["master_param"].to(parameter.dtype)
                )

    def get_learning_rate(self):
        return float(self.optimizer.param_groups[0]["lr"])


__all__ = ["BF16Optimizer", "MasterWeightAdamW"]
