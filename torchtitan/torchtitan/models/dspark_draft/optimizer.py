# Adapted from the DeepSpec DSpark implementation (MIT).
# The original training semantics and third-party notices are retained.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torchtitan.components.optimizer import OptimizersContainer


_OPTIMIZER_DTYPE = torch.float32


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
            group.setdefault("step", 0)
            for parameter in group["params"]:
                if parameter.requires_grad:
                    self._initialize_parameter_state(parameter)

    def _initialize_parameter_state(self, parameter) -> None:
        state = self.state[parameter]
        if state:
            return
        state["step"] = torch.zeros((), dtype=_OPTIMIZER_DTYPE, device=parameter.device)
        state["master_param"] = parameter.detach().clone().to(_OPTIMIZER_DTYPE)
        state["exp_avg"] = torch.zeros_like(state["master_param"])
        state["exp_avg_sq"] = torch.zeros_like(state["master_param"])

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        # The standard loader casts state to the parameter dtype. Recover from
        # the original FP32 checkpoint tensors; promoting the cast values would
        # retain BF16 rounding and change every subsequent optimizer update.
        for saved_group, group in zip(
            state_dict["param_groups"], self.param_groups, strict=True
        ):
            for saved_id, parameter in zip(
                saved_group["params"], group["params"], strict=True
            ):
                saved = state_dict["state"].get(saved_id, {})
                for name in ("step", "master_param", "exp_avg", "exp_avg_sq"):
                    if name in saved:
                        self.state[parameter][name] = saved[name].to(
                            device=parameter.device, dtype=_OPTIMIZER_DTYPE
                        )
        # Older checkpoints stored the common Adam step only in each parameter
        # state.  Recover it once while loading instead of reading one CUDA
        # scalar per parameter on every optimizer step.
        for group in self.param_groups:
            if "step" in group:
                group["step"] = int(group["step"])
                continue
            group_step = 0
            for parameter in group["params"]:
                state_step = self.state[parameter].get("step")
                if state_step is not None:
                    group_step = int(state_step.item())
                    break
            group["step"] = group_step

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
            step = int(group.get("step", 0)) + 1
            group["step"] = step
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError(
                        "MasterWeightAdamW does not support sparse gradients."
                    )
                self._initialize_parameter_state(parameter)
                state = self.state[parameter]
                state["step"].fill_(step)
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                # BF16-reduced gradients are promoted before every optimizer
                # update; master weights and both moment accumulators therefore
                # remain FP32 throughout training.
                grad_float = gradient.to(_OPTIMIZER_DTYPE)
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
                denominator = (
                    exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                )
                master.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-lr / bias_correction1,
                )
                parameter.copy_(master.to(parameter.dtype))
        return loss


class DraftOptimizers(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    @staticmethod
    def _resolve_optimizer_factory(name):
        if name == "MasterWeightAdamW":
            # Titan puts optimizer hyperparameters in each parameter group.
            # The required default is taken from the recipe, never redefined.
            return lambda groups, **kwargs: MasterWeightAdamW(
                groups, lr=groups[0]["lr"], **kwargs
            )
        return OptimizersContainer._resolve_optimizer_factory(name)

    @staticmethod
    def _build_impl_kwargs(config):
        if config.implementation != "for-loop":
            raise ValueError(
                "DSpark master-weight Adam requires for-loop implementation"
            )
        return {}
