"""Exact local DSpark optimizer metrics and exportable loss curves."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMPONENTS = ("ce_loss", "l1_loss", "confidence_loss")


def estimated_training_acceptance(record: dict[str, Any]) -> float:
    """Distribution overlap under the training contexts and loss-position weights."""
    if (
        not math.isfinite(record["loss_weights"]["l1_loss"])
        or record["loss_weights"]["l1_loss"] <= 0
    ):
        raise ValueError("Acceptance estimation requires a measured L1 loss with positive weight")
    l1 = float(record["l1_loss"])
    if not math.isfinite(l1) or not -1e-6 <= l1 <= 2.0 + 1e-6:
        raise ValueError(f"Invalid distribution L1 for acceptance estimation: {l1}")
    return min(1.0, max(0.0, 1.0 - l1 / 2.0))


def plot_acceptance_curve(records: list[dict[str, Any]], output: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    if not records:
        raise ValueError("Cannot plot acceptance without optimizer metrics")
    steps = [record["optimizer_step"] for record in records]
    if steps != sorted(set(steps)):
        raise ValueError("Optimizer metric steps must be unique and ordered")
    values = [estimated_training_acceptance(record) for record in records]
    figure = Figure(figsize=(10, 5), dpi=140)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(steps, values, marker=".", linewidth=1.4)
    axis.set_title(
        "Estimated training acceptance probability\nTeacher-forced, loss-position weighted"
    )
    axis.set_xlabel("Optimizer step (pre-update training forward)")
    axis.set_ylabel("Estimated probability")
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png")
    temporary.replace(output)


def plot_loss_curve(records: list[dict[str, Any]], output: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    if not records:
        raise ValueError("Cannot plot a loss curve without optimizer metrics")
    steps = [record["optimizer_step"] for record in records]
    if steps != sorted(set(steps)):
        raise ValueError("Optimizer metric steps must be unique and ordered")
    figure = Figure(figsize=(11, 7), dpi=140)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True).flat
    for axis, key, label in zip(
        axes,
        ("total_loss", *COMPONENTS),
        ("Weighted total DSpark loss", "CE loss", "L1 loss", "Confidence loss"),
        strict=True,
    ):
        axis.plot(steps, [record[key] for record in records], marker=".", linewidth=1.4)
        axis.set_title(label)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Loss")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png")
    temporary.replace(output)


class LocalDSparkMetrics:
    def __init__(
        self,
        output_dir: str,
        ce_alpha: float = 0.1,
        l1_alpha: float = 0.9,
        confidence_alpha: float = 1.0,
    ) -> None:
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output / "metrics.jsonl"
        if self.metrics_path.exists():
            raise ValueError("Fresh full-data training requires a new optimizer-metrics file")
        self.weights = dict(zip(COMPONENTS, (ce_alpha, l1_alpha, confidence_alpha), strict=True))
        self.records: list[dict[str, Any]] = []

    def record(self, metrics: dict[str, Any], step: int) -> None:
        if self.records and step <= self.records[-1]["optimizer_step"]:
            raise ValueError(f"Repeated or unordered optimizer metric step {step}")
        components = {key: float(metrics[f"train/{key}"]) for key in COMPONENTS}
        if not all(math.isfinite(value) for value in components.values()):
            raise ValueError(f"Nonfinite DSpark loss at optimizer metric step {step}")
        record = {
            "optimizer_step": step,
            **components,
            "total_loss": sum(self.weights[key] * components[key] for key in COMPONENTS),
            "loss_weights": self.weights,
            "metrics": {
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            },
        }
        if self.weights["l1_loss"] > 0:
            record["estimated_training_acceptance"] = estimated_training_acceptance(record)
        with self.metrics_path.open("a") as destination:
            destination.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.append(record)
        plot_loss_curve(self.records, self.output / "loss.png")
        if self.weights["l1_loss"] > 0:
            plot_acceptance_curve(self.records, self.output / "acceptance.png")

    def finalize(self, optimizer_steps: int) -> None:
        if [record["optimizer_step"] for record in self.records] != list(
            range(1, optimizer_steps + 1)
        ):
            raise RuntimeError("Full-data optimizer metrics do not cover every completed step")
        plot_loss_curve(self.records, self.output / "loss.png")
        if self.weights["l1_loss"] > 0:
            plot_acceptance_curve(self.records, self.output / "acceptance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a local DSpark metric PNG")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("loss", "acceptance"), default="loss")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    plot = plot_acceptance_curve if args.kind == "acceptance" else plot_loss_curve
    plot(records, args.output)


if __name__ == "__main__":
    main()
