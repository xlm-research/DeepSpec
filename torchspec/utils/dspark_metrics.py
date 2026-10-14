"""Exact local DSpark optimizer metrics and exportable loss curves."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMPONENTS = ("ce_loss", "l1_loss", "confidence_loss")


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
        with self.metrics_path.open("a") as destination:
            destination.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.append(record)
        plot_loss_curve(self.records, self.output / "loss.png")

    def finalize(self, optimizer_steps: int) -> None:
        if [record["optimizer_step"] for record in self.records] != list(
            range(1, optimizer_steps + 1)
        ):
            raise RuntimeError("Full-data optimizer metrics do not cover every completed step")
        plot_loss_curve(self.records, self.output / "loss.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a local DSpark loss PNG")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    plot_loss_curve(records, args.output)


if __name__ == "__main__":
    main()
