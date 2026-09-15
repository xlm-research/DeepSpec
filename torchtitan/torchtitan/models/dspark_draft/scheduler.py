# Adapted from the DeepSpec DSpark implementation (MIT).
# The original training semantics and third-party notices are retained.

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from dataclasses import dataclass
from torchtitan.components.optimizer import LRSchedulersContainer


class TwoStageScheduler(_LRScheduler):
    def __init__(self, optimizer, after_scheduler: _LRScheduler, last_epoch=-1):
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class WarmupScheduler(TwoStageScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        super().__init__(optimizer, after_scheduler, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_last_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)


class CosineAnnealingWarmupLR(WarmupScheduler):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


class DraftSchedulers(LRSchedulersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        def build(self, *, optimizers, training_steps):
            if self.decay_type != "cosine" or self.decay_ratio is not None:
                raise ValueError("DSpark uses its warmup-then-cosine schedule")
            if self.min_lr_factor != 0.0:
                raise ValueError("DSpark's retained schedule ends at zero")
            return DraftSchedulers(
                optimizers, self.total_steps or training_steps, self.warmup_steps
            )

    def __init__(self, optimizers, total_steps, warmup_steps):
        self.schedulers = [
            CosineAnnealingWarmupLR(optimizer, total_steps, warmup_steps)
            for optimizer in optimizers
        ]

    def state_dict(self):
        return {
            str(index): scheduler.state_dict()
            for index, scheduler in enumerate(self.schedulers)
        }

    def load_state_dict(self, state):
        if set(state) != {str(index) for index in range(len(self.schedulers))}:
            raise ValueError("Checkpoint scheduler topology differs from this run")
        for index, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state[str(index)])
            for group, lr in zip(
                scheduler.optimizer.param_groups, scheduler.get_last_lr()
            ):
                group["lr"] = lr
