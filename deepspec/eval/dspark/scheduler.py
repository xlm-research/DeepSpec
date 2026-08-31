from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch


DEFAULT_TEMPERATURE_GRID = tuple(index / 20.0 for index in range(1, 101))


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_binary_targets(
    targets: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
) -> None:
    valid_targets = targets[valid_mask]
    if valid_targets.numel() == 0:
        raise ValueError("At least one calibration target must be valid.")
    if not bool(torch.isfinite(valid_targets).all().item()):
        raise ValueError("Calibration targets must be finite at valid positions.")
    is_binary = (valid_targets == 0) | (valid_targets == 1)
    if not bool(is_binary.all().item()):
        raise ValueError("Calibration targets must be binary prefix labels.")


def _ece_from_valid(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int,
) -> torch.Tensor:
    probabilities = probabilities.to(torch.float64)
    targets = targets.to(torch.float64)
    bin_indices = (probabilities * num_bins).long().clamp_(0, num_bins - 1)
    counts = torch.bincount(bin_indices, minlength=num_bins).to(torch.float64)
    predicted_sum = torch.zeros_like(counts).scatter_add_(
        0,
        bin_indices,
        probabilities,
    )
    target_sum = torch.zeros_like(counts).scatter_add_(
        0,
        bin_indices,
        targets,
    )
    nonempty = counts > 0
    calibration_gap = (
        predicted_sum[nonempty] / counts[nonempty]
        - target_sum[nonempty] / counts[nonempty]
    ).abs()
    return (calibration_gap * counts[nonempty]).sum() / counts.sum()


@torch.no_grad()
def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = 15,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute equal-width-bin ECE for binary labels."""
    num_bins = _positive_int(num_bins, name="num_bins")
    if not isinstance(probabilities, torch.Tensor) or not probabilities.is_floating_point():
        raise TypeError("probabilities must be a floating-point torch.Tensor.")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor.")
    if probabilities.shape != targets.shape:
        raise ValueError(
            "probabilities and targets must have the same shape, got "
            f"{tuple(probabilities.shape)} and {tuple(targets.shape)}."
        )
    if probabilities.device != targets.device:
        raise ValueError("probabilities and targets must be on the same device.")
    if probabilities.numel() == 0:
        raise ValueError("probabilities must be non-empty.")
    if valid_mask is None:
        valid_mask = torch.ones_like(probabilities, dtype=torch.bool)
    elif (
        not isinstance(valid_mask, torch.Tensor)
        or valid_mask.dtype != torch.bool
        or valid_mask.shape != probabilities.shape
        or valid_mask.device != probabilities.device
    ):
        raise ValueError("valid_mask must be a bool tensor matching probabilities.")
    valid_probabilities = probabilities[valid_mask]
    if valid_probabilities.numel() == 0:
        raise ValueError("At least one probability must be valid.")
    if not bool(torch.isfinite(valid_probabilities).all().item()):
        raise ValueError("Probabilities must be finite at valid positions.")
    if not bool(
        ((valid_probabilities >= 0.0) & (valid_probabilities <= 1.0)).all().item()
    ):
        raise ValueError("Probabilities must lie in [0, 1].")
    _validate_binary_targets(targets, valid_mask=valid_mask)
    return _ece_from_valid(
        valid_probabilities,
        targets[valid_mask],
        num_bins=num_bins,
    )


@dataclass(frozen=True)
class SequentialTemperatureScaler:
    """Per-position temperature scaling for cumulative prefix confidence."""

    temperatures: torch.Tensor
    num_bins: int = 15

    def __post_init__(self) -> None:
        _positive_int(self.num_bins, name="num_bins")
        if (
            not isinstance(self.temperatures, torch.Tensor)
            or not self.temperatures.is_floating_point()
            or self.temperatures.ndim != 1
            or self.temperatures.numel() == 0
        ):
            raise ValueError("temperatures must be a non-empty floating 1D tensor.")
        temperatures = self.temperatures.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(temperatures).all().item()) or not bool(
            (temperatures > 0.0).all().item()
        ):
            raise ValueError("temperatures must contain finite positive values.")
        object.__setattr__(self, "temperatures", temperatures.clone())
        object.__setattr__(self, "num_bins", int(self.num_bins))

    @property
    def block_size(self) -> int:
        return int(self.temperatures.numel())

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        confidence_logits: torch.Tensor,
        prefix_targets: torch.Tensor,
        *,
        temperature_grid: Sequence[float] | torch.Tensor = DEFAULT_TEMPERATURE_GRID,
        num_bins: int = 15,
        valid_mask: torch.Tensor | None = None,
    ) -> SequentialTemperatureScaler:
        """Fit temperatures left-to-right against cumulative-prefix ECE."""
        num_bins = _positive_int(num_bins, name="num_bins")
        if (
            not isinstance(confidence_logits, torch.Tensor)
            or not confidence_logits.is_floating_point()
            or confidence_logits.ndim != 2
            or confidence_logits.shape[0] == 0
            or confidence_logits.shape[1] == 0
        ):
            raise ValueError(
                "confidence_logits must be a non-empty floating tensor [N, gamma]."
            )
        if (
            not isinstance(prefix_targets, torch.Tensor)
            or prefix_targets.shape != confidence_logits.shape
            or prefix_targets.device != confidence_logits.device
        ):
            raise ValueError(
                "prefix_targets must match confidence_logits shape and device."
            )
        if valid_mask is None:
            valid_mask = torch.ones_like(confidence_logits, dtype=torch.bool)
        elif (
            not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != confidence_logits.shape
            or valid_mask.device != confidence_logits.device
        ):
            raise ValueError(
                "valid_mask must be a bool tensor matching confidence_logits."
            )
        if confidence_logits.shape[1] > 1 and bool(
            (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any().item()
        ):
            raise ValueError("valid_mask must mark a contiguous prefix in every row.")
        for position in range(confidence_logits.shape[1]):
            if not bool(valid_mask[:, position].any().item()):
                raise ValueError(
                    f"No valid calibration examples for position {position}."
                )
        valid_logits = confidence_logits[valid_mask]
        if not bool(torch.isfinite(valid_logits).all().item()):
            raise ValueError("confidence_logits must be finite at valid positions.")
        _validate_binary_targets(prefix_targets, valid_mask=valid_mask)
        if confidence_logits.shape[1] > 1:
            comparable = valid_mask[:, :-1] & valid_mask[:, 1:]
            if bool(
                (
                    prefix_targets[:, 1:][comparable]
                    > prefix_targets[:, :-1][comparable]
                ).any().item()
            ):
                raise ValueError("prefix_targets must be non-increasing by position.")

        grid = torch.as_tensor(
            temperature_grid,
            dtype=torch.float64,
            device=confidence_logits.device,
        )
        if grid.ndim != 1 or grid.numel() == 0:
            raise ValueError("temperature_grid must be a non-empty 1D sequence.")
        if not bool(torch.isfinite(grid).all().item()) or not bool(
            (grid > 0.0).all().item()
        ):
            raise ValueError("temperature_grid must contain finite positive values.")
        if torch.unique(grid).numel() != grid.numel():
            raise ValueError("temperature_grid must not contain duplicates.")

        logits = torch.where(
            valid_mask,
            confidence_logits,
            torch.zeros_like(confidence_logits),
        ).to(torch.float64)
        targets = prefix_targets.to(torch.float64)
        calibrated_prefix = torch.ones(
            logits.shape[0],
            dtype=torch.float64,
            device=logits.device,
        )
        selected_temperatures = torch.empty(
            logits.shape[1],
            dtype=torch.float64,
            device=logits.device,
        )
        for position in range(logits.shape[1]):
            position_mask = valid_mask[:, position]
            best_temperature = None
            best_tie_key = None
            for temperature in grid:
                candidate_prefix = calibrated_prefix * torch.sigmoid(
                    logits[:, position] / temperature
                )
                candidate_ece = _ece_from_valid(
                    candidate_prefix[position_mask],
                    targets[:, position][position_mask],
                    num_bins=num_bins,
                )
                tie_key = (
                    float(candidate_ece.item()),
                    abs(math.log(float(temperature.item()))),
                    float(temperature.item()),
                )
                if best_tie_key is None or tie_key < best_tie_key:
                    best_temperature = temperature
                    best_tie_key = tie_key
            assert best_temperature is not None
            selected_temperatures[position] = best_temperature
            calibrated_prefix = calibrated_prefix * torch.sigmoid(
                logits[:, position] / best_temperature
            )
        return cls(
            temperatures=selected_temperatures,
            num_bins=num_bins,
        )

    def calibrate_logits(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(confidence_logits, torch.Tensor)
            or not confidence_logits.is_floating_point()
            or confidence_logits.ndim == 0
            or confidence_logits.shape[-1] != self.block_size
        ):
            raise ValueError(
                "confidence_logits must be floating with final dimension "
                f"{self.block_size}."
            )
        if not bool(torch.isfinite(confidence_logits).all().item()):
            raise ValueError("confidence_logits must be finite.")
        dtype = (
            torch.float64
            if confidence_logits.dtype == torch.float64
            else torch.float32
        )
        temperatures = self.temperatures.to(
            device=confidence_logits.device,
            dtype=dtype,
        )
        return torch.sigmoid(confidence_logits.to(dtype) / temperatures)

    def calibrate_probabilities(
        self,
        confidence_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(confidence_probabilities, torch.Tensor)
            or not confidence_probabilities.is_floating_point()
            or confidence_probabilities.ndim == 0
            or confidence_probabilities.shape[-1] != self.block_size
        ):
            raise ValueError(
                "confidence_probabilities must be floating with final dimension "
                f"{self.block_size}."
            )
        probabilities = confidence_probabilities.to(
            torch.float64
            if confidence_probabilities.dtype == torch.float64
            else torch.float32
        )
        if not bool(torch.isfinite(probabilities).all().item()) or not bool(
            ((probabilities >= 0.0) & (probabilities <= 1.0)).all().item()
        ):
            raise ValueError("confidence_probabilities must be finite and in [0, 1].")
        epsilon = torch.finfo(probabilities.dtype).eps
        logits = torch.logit(probabilities.clamp(epsilon, 1.0 - epsilon))
        return self.calibrate_logits(logits)


@dataclass(frozen=True)
class SPSProfile:
    """Exact integer-batch lookup table for profiled engine steps/second."""

    batch_sizes: tuple[int, ...]
    steps_per_second: tuple[float, ...]

    def __post_init__(self) -> None:
        batch_sizes = tuple(self.batch_sizes)
        steps_per_second = tuple(self.steps_per_second)
        if not batch_sizes or len(batch_sizes) != len(steps_per_second):
            raise ValueError(
                "batch_sizes and steps_per_second must be non-empty and equal-length."
            )
        normalized_batch_sizes = []
        normalized_rates = []
        for batch_size in batch_sizes:
            normalized_batch_sizes.append(
                _positive_int(batch_size, name="batch size")
            )
        if any(
            right <= left
            for left, right in zip(
                normalized_batch_sizes,
                normalized_batch_sizes[1:],
            )
        ):
            raise ValueError("batch_sizes must be strictly increasing.")
        for rate in steps_per_second:
            if isinstance(rate, bool) or not isinstance(rate, Real):
                raise ValueError(f"SPS values must be real numbers, got {rate!r}.")
            normalized_rate = float(rate)
            if not math.isfinite(normalized_rate) or normalized_rate <= 0.0:
                raise ValueError("SPS values must be finite and strictly positive.")
            normalized_rates.append(normalized_rate)
        object.__setattr__(self, "batch_sizes", tuple(normalized_batch_sizes))
        object.__setattr__(self, "steps_per_second", tuple(normalized_rates))

    @classmethod
    def from_mapping(cls, profile: Mapping[int, float]) -> SPSProfile:
        if not isinstance(profile, Mapping) or not profile:
            raise ValueError("profile must be a non-empty batch-size-to-SPS mapping.")
        for batch_size in profile:
            _positive_int(batch_size, name="batch size")
        items = sorted(profile.items())
        return cls(
            batch_sizes=tuple(batch_size for batch_size, _ in items),
            steps_per_second=tuple(rate for _, rate in items),
        )

    @classmethod
    def from_dense(
        cls,
        steps_per_second: Sequence[float],
        *,
        first_batch_size: int = 1,
    ) -> SPSProfile:
        first_batch_size = _positive_int(
            first_batch_size,
            name="first_batch_size",
        )
        rates = tuple(steps_per_second)
        return cls(
            batch_sizes=tuple(
                range(first_batch_size, first_batch_size + len(rates))
            ),
            steps_per_second=rates,
        )

    def lookup(self, batch_size: int) -> float:
        batch_size = _positive_int(batch_size, name="batch_size")
        index = bisect_left(self.batch_sizes, batch_size)
        if index == len(self.batch_sizes) or self.batch_sizes[index] != batch_size:
            raise ValueError(f"SPS profile is missing batch size {batch_size}.")
        return self.steps_per_second[index]

    def require_range(self, start: int, stop: int) -> None:
        start = _positive_int(start, name="start")
        stop = _positive_int(stop, name="stop")
        if stop < start:
            raise ValueError("stop must be greater than or equal to start.")
        missing = [
            batch_size
            for batch_size in range(start, stop + 1)
            if batch_size not in self.batch_sizes
        ]
        if missing:
            raise ValueError(
                "SPS profile must cover every candidate batch size; missing "
                + ", ".join(str(batch_size) for batch_size in missing)
                + "."
            )


@dataclass(frozen=True)
class PrefixSchedule:
    prefix_lengths: torch.Tensor
    prefix_survival_probabilities: torch.Tensor
    expected_accepted_tokens: float
    target_batch_size: int
    expected_throughput: float

    @property
    def admitted_draft_tokens(self) -> int:
        return int(self.prefix_lengths.sum().item())


class HardwareAwarePrefixScheduler:
    """Algorithm 1 greedy scheduler with first non-improvement stopping."""

    def __init__(self, sps_profile: SPSProfile | Mapping[int, float]):
        self.sps_profile = (
            sps_profile
            if isinstance(sps_profile, SPSProfile)
            else SPSProfile.from_mapping(sps_profile)
        )

    @torch.no_grad()
    def schedule(self, confidence_probabilities: torch.Tensor) -> PrefixSchedule:
        if (
            not isinstance(confidence_probabilities, torch.Tensor)
            or not confidence_probabilities.is_floating_point()
            or confidence_probabilities.ndim != 2
            or confidence_probabilities.shape[0] == 0
        ):
            raise ValueError(
                "confidence_probabilities must be floating with shape [R, gamma] "
                "and R > 0."
            )
        if not bool(torch.isfinite(confidence_probabilities).all().item()) or not bool(
            (
                (confidence_probabilities >= 0.0)
                & (confidence_probabilities <= 1.0)
            ).all().item()
        ):
            raise ValueError("confidence_probabilities must be finite and in [0, 1].")

        request_count, block_size = confidence_probabilities.shape
        prefix_survival = confidence_probabilities.to(torch.float64).cumprod(dim=1)
        # Position-major flattening makes equal-confidence ties deterministic:
        # shallower extensions are considered first, then request order.
        flat_survival = prefix_survival.transpose(0, 1).contiguous().reshape(-1)
        positive_indices = torch.nonzero(
            flat_survival > 0.0,
            as_tuple=False,
        ).flatten()
        candidate_count = int(positive_indices.numel())
        self.sps_profile.require_range(
            request_count,
            request_count + candidate_count,
        )

        baseline_throughput = request_count * self.sps_profile.lookup(request_count)
        prefix_lengths = torch.zeros(
            request_count,
            dtype=torch.long,
            device=confidence_probabilities.device,
        )
        if candidate_count == 0:
            return PrefixSchedule(
                prefix_lengths=prefix_lengths,
                prefix_survival_probabilities=prefix_survival,
                expected_accepted_tokens=float(request_count),
                target_batch_size=request_count,
                expected_throughput=float(baseline_throughput),
            )

        candidate_survival = flat_survival[positive_indices]
        order = torch.argsort(candidate_survival, descending=True, stable=True)
        sorted_indices = positive_indices[order]
        sorted_survival = candidate_survival[order]
        candidate_batch_sizes = range(
            request_count + 1,
            request_count + candidate_count + 1,
        )
        candidate_sps = torch.tensor(
            [self.sps_profile.lookup(batch_size) for batch_size in candidate_batch_sizes],
            dtype=torch.float64,
            device=confidence_probabilities.device,
        )
        candidate_expected_accepts = (
            sorted_survival.cumsum(dim=0) + request_count
        )
        candidate_throughput = candidate_expected_accepts * candidate_sps
        previous_throughput = torch.cat(
            [
                candidate_throughput.new_tensor([baseline_throughput]),
                candidate_throughput[:-1],
            ]
        )
        improves = candidate_throughput > previous_throughput
        first_non_improvement = torch.nonzero(
            ~improves,
            as_tuple=False,
        ).flatten()
        admitted_count = (
            int(first_non_improvement[0].item())
            if first_non_improvement.numel() > 0
            else candidate_count
        )

        if admitted_count > 0:
            admitted_indices = sorted_indices[:admitted_count]
            admitted_positions = torch.div(
                admitted_indices,
                request_count,
                rounding_mode="floor",
            ) + 1
            admitted_requests = admitted_indices.remainder(request_count)
            prefix_lengths.scatter_reduce_(
                0,
                admitted_requests,
                admitted_positions,
                reduce="amax",
                include_self=True,
            )
            expected_accepted_tokens = float(
                candidate_expected_accepts[admitted_count - 1].item()
            )
            expected_throughput = float(
                candidate_throughput[admitted_count - 1].item()
            )
        else:
            expected_accepted_tokens = float(request_count)
            expected_throughput = float(baseline_throughput)
        return PrefixSchedule(
            prefix_lengths=prefix_lengths,
            prefix_survival_probabilities=prefix_survival,
            expected_accepted_tokens=expected_accepted_tokens,
            target_batch_size=request_count + admitted_count,
            expected_throughput=expected_throughput,
        )

    def select_prefix_lengths(
        self,
        confidence_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        return self.schedule(confidence_probabilities).prefix_lengths


__all__ = [
    "DEFAULT_TEMPERATURE_GRID",
    "HardwareAwarePrefixScheduler",
    "PrefixSchedule",
    "SPSProfile",
    "SequentialTemperatureScaler",
    "expected_calibration_error",
]
