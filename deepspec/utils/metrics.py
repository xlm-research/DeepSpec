import re

import torch
import torch.distributed as dist


_REDUCTION_PATTERN = re.compile(r"^(dp_)?(mean|sum|max|min|last)$")
_DEFAULT_RATIO_REDUCTION = "dp_sum"
_metrics = {}
_reduction_group = None
_validated_schema = None


def configure_reduction_group(group) -> None:
    global _reduction_group, _validated_schema
    _reduction_group = group
    _validated_schema = None


def _group_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(_reduction_group)


def _detach_scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        assert value.numel() == 1, "metrics only support scalar values"
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32)


def _clone_to_reduce_device(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().clone().to(torch.float32)
    if dist.get_backend() == "nccl" and not tensor.is_cuda:
        tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))
    return tensor


def _reduce_dp_value(value: torch.Tensor, op_name: str) -> torch.Tensor:
    if op_name == "sum" or op_name == "mean":
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=_reduction_group)
        if op_name == "mean":
            value = value / _group_world_size()
        return value
    if op_name == "max":
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=_reduction_group)
        return value
    if op_name == "min":
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=_reduction_group)
        return value
    if op_name == "last":
        gathered = [torch.empty_like(value) for _ in range(_group_world_size())]
        dist.all_gather(gathered, value, group=_reduction_group)
        return gathered[-1]
    raise AssertionError(f"unsupported reduction: {op_name}")


def _local_reduce(values, reduction: str) -> torch.Tensor:
    assert values, "cannot reduce an empty metric buffer"
    if reduction == "last":
        return values[-1]
    stacked = torch.stack([_clone_to_reduce_device(value) for value in values])
    if reduction == "max":
        return stacked.max()
    if reduction == "min":
        return stacked.min()
    if reduction in ("mean", "sum"):
        # Sum-style metrics report the average per emit in the logging window.
        return stacked.mean()
    raise AssertionError(f"unsupported reduction: {reduction}")


def _schema():
    items = []
    for name, entry in sorted(_metrics.items()):
        if entry["kind"] == "ratio":
            count = len(entry["num"])
        else:
            count = len(entry["values"])
        items.append((name, entry["kind"], entry["reduction"], count))
    return items


def _assert_schema_consistent():
    global _validated_schema
    local_schema = _schema()
    if local_schema == _validated_schema:
        return
    if _group_world_size() == 1:
        _validated_schema = local_schema
        return
    gathered = [None for _ in range(_group_world_size())]
    dist.all_gather_object(gathered, local_schema, group=_reduction_group)
    reference = gathered[0]
    for rank, schema in enumerate(gathered[1:], start=1):
        assert schema == reference, (
            "metric schema mismatch across ranks: "
            f"rank0={reference}, rank{rank}={schema}"
        )
    _validated_schema = local_schema


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    if denominator.item() == 0:
        return 0.0
    return (numerator / denominator).item()


@torch.compiler.disable(recursive=False)
def add_metric(
    name,
    value,
    *,
    den=None,
    reduction: str = _DEFAULT_RATIO_REDUCTION,
    tag: str = "train",
):
    """Record one scalar metric for the next logging-window flush.

    Tensor inputs are detached at the API boundary to prevent metric logging
    from retaining autograd graphs. Ratios must pass their denominator through
    ``den=``; callers should not pre-divide locally because ``flush`` computes
    the global ratio as ``sum(num) / sum(den)``.
    """

    assert _REDUCTION_PATTERN.match(reduction), f"unsupported reduction: {reduction}"
    metric_name = f"{tag}/{name}"
    if den is not None:
        assert reduction == _DEFAULT_RATIO_REDUCTION, (
            "ratio metrics must use the default dp_sum reduction"
        )
        value = _detach_scalar(value)
        den = _detach_scalar(den)
        entry = _metrics.setdefault(
            metric_name,
            {"kind": "ratio", "reduction": reduction, "num": [], "den": []},
        )
        assert entry["kind"] == "ratio", f"metric kind changed for {metric_name}"
        assert entry["reduction"] == reduction, (
            f"metric reduction changed for {metric_name}: "
            f"{entry['reduction']} != {reduction}"
        )
        entry["num"].append(value)
        entry["den"].append(den)
        return

    value = _detach_scalar(value)
    entry = _metrics.setdefault(
        metric_name,
        {"kind": "scalar", "reduction": reduction, "values": []},
    )
    assert entry["kind"] == "scalar", f"metric kind changed for {metric_name}"
    assert entry["reduction"] == reduction, (
        f"metric reduction changed for {metric_name}: "
        f"{entry['reduction']} != {reduction}"
    )
    entry["values"].append(value)


class PendingMetricReduction:
    """Packed asynchronous metric reductions finalized after optimizer compute."""

    def __init__(
        self,
        *,
        local_values,
        packed_reductions,
        last_reduction,
    ):
        self.local_values = local_values
        self.packed_reductions = packed_reductions
        self.last_reduction = last_reduction

    def wait(self, *, materialize: bool = True) -> dict[str, float]:
        summary = (
            {
                name: float(value.item())
                for name, value in self.local_values.items()
            }
            if materialize
            else {}
        )
        for tensor, work, specs in self.packed_reductions:
            if work is not None:
                work.wait()
            if not materialize:
                continue
            values = tensor.detach().cpu().tolist()
            for spec in specs:
                kind, name, first, second = spec
                if kind == "ratio":
                    denominator = float(values[second])
                    summary[name] = (
                        0.0 if denominator == 0.0 else float(values[first]) / denominator
                    )
                else:
                    summary[name] = float(values[first]) / float(second)
        if self.last_reduction is not None:
            gathered, work, names = self.last_reduction
            if work is not None:
                work.wait()
            if materialize:
                values = gathered[-1].detach().cpu().tolist()
                summary.update(
                    {name: float(value) for name, value in zip(names, values)}
                )
        return summary


def flush_async() -> PendingMetricReduction:
    """Pack scalar metrics and launch at most one collective per reduction op.

    DSpark records many numerator/denominator pairs. Reducing each scalar
    separately produced more than forty tiny NCCL calls per optimizer step at
    128 GPUs. Packing them turns that latency-bound tail into one SUM, plus an
    optional MAX/MIN/LAST collective, and ``async_op`` lets the SUM overlap the
    FP32 Adam update.
    """

    _assert_schema_consistent()
    try:
        local_values = {}
        sum_values = []
        sum_specs = []
        max_values = []
        max_specs = []
        min_values = []
        min_specs = []
        last_values = []
        last_names = []
        world_size = _group_world_size()
        for name, entry in sorted(_metrics.items()):
            reduction = entry["reduction"]
            if entry["kind"] == "ratio":
                num = torch.stack(
                    [_clone_to_reduce_device(value) for value in entry["num"]]
                ).sum()
                den = torch.stack(
                    [_clone_to_reduce_device(value) for value in entry["den"]]
                ).sum()
                first = len(sum_values)
                sum_values.extend((num, den))
                sum_specs.append(("ratio", name, first, first + 1))
                continue

            local_reduction = reduction[3:] if reduction.startswith("dp_") else reduction
            value = _local_reduce(entry["values"], local_reduction)
            if not reduction.startswith("dp_"):
                local_values[name] = value
            elif local_reduction in ("sum", "mean"):
                index = len(sum_values)
                sum_values.append(value)
                divisor = world_size if local_reduction == "mean" else 1
                sum_specs.append(("scalar", name, index, divisor))
            elif local_reduction == "max":
                index = len(max_values)
                max_values.append(value)
                max_specs.append(("scalar", name, index, 1))
            elif local_reduction == "min":
                index = len(min_values)
                min_values.append(value)
                min_specs.append(("scalar", name, index, 1))
            elif local_reduction == "last":
                last_values.append(value)
                last_names.append(name)
            else:
                raise AssertionError(f"unsupported reduction: {local_reduction}")

        packed_reductions = []
        for values, specs, op in (
            (sum_values, sum_specs, dist.ReduceOp.SUM),
            (max_values, max_specs, dist.ReduceOp.MAX),
            (min_values, min_specs, dist.ReduceOp.MIN),
        ):
            if not values:
                continue
            tensor = torch.stack(values)
            work = None
            if world_size > 1:
                work = dist.all_reduce(
                    tensor,
                    op=op,
                    group=_reduction_group,
                    async_op=True,
                )
            packed_reductions.append((tensor, work, specs))

        last_reduction = None
        if last_values:
            tensor = torch.stack(last_values)
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
            work = None
            if world_size > 1:
                work = dist.all_gather(
                    gathered,
                    tensor,
                    group=_reduction_group,
                    async_op=True,
                )
            else:
                gathered[0].copy_(tensor)
            last_reduction = (gathered, work, last_names)

        return PendingMetricReduction(
            local_values=local_values,
            packed_reductions=packed_reductions,
            last_reduction=last_reduction,
        )
    finally:
        reset()


def flush() -> dict[str, float]:
    return flush_async().wait()


def reset() -> None:
    _metrics.clear()


__all__ = [
    "PendingMetricReduction",
    "add_metric",
    "configure_reduction_group",
    "flush",
    "flush_async",
    "reset",
]
