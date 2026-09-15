"""Accumulate DSpark diagnostic numerators within one training update."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_active_metrics: ContextVar[dict | None] = ContextVar("dspark_metrics", default=None)


@contextmanager
def collect_metrics():
    values = {}
    token = _active_metrics.set(values)
    try:
        yield values
    finally:
        _active_metrics.reset(token)


def add_metric(name, value, *, den=None, tag="train", reduction="mean"):
    values = _active_metrics.get()
    if values is None:
        return
    numerator = value.detach().float()
    denominator = torch.ones_like(numerator) if den is None else den.detach().float()
    key = f"{tag}/{name}"
    pair = torch.stack((numerator, denominator))
    values[key] = values[key] + pair if key in values else pair
