import unittest

import torch
import torch.distributed as dist

from deepspec.utils.metrics import (
    add_metric,
    configure_reduction_group,
    flush_async,
)
from tests.distributed_test_utils import require_torchrun


class PackedMetricReductionTest(unittest.TestCase):
    def test_async_packed_reductions_preserve_metric_semantics(self):
        runtime = require_torchrun(self, world_size=2)
        configure_reduction_group(None)
        rank_value = torch.tensor(float(dist.get_rank() + 1), device=runtime.device)
        add_metric("ratio", rank_value, den=rank_value.new_tensor(2.0))
        add_metric("mean", rank_value, reduction="dp_mean")
        add_metric("local", rank_value, reduction="last")

        pending = flush_async()
        # The returned work is genuinely asynchronous: independent CUDA work
        # may be submitted before the host asks for the reduced summary.
        independent = (rank_value + 1.0).square()
        summary = pending.wait()

        self.assertAlmostEqual(summary["train/ratio"], 0.75)
        self.assertAlmostEqual(summary["train/mean"], 1.5)
        self.assertAlmostEqual(summary["train/local"], float(dist.get_rank() + 1))
        self.assertTrue(torch.isfinite(independent))


if __name__ == "__main__":
    unittest.main()


def test_pipeline_metrics_never_subtract_different_node_clocks(tmp_path):
    from deepspec.pipeline.metrics import summarize_metrics
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.runtime import EventWriter
    from tests.pipeline_topology_fixtures import task_config, node_facts, input_plan

    config = task_config("M3")
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="metrics", now=100
    ).to_dict()
    events = []
    for index, node in enumerate(plan["node_budgets"]):
        with EventWriter(
            tmp_path / f"{index}.jsonl",
            run_id=plan["run_id"],
            plan_hash=plan["plan_hash"],
            sender_identity={"component": "sampler", "node_id": node},
        ) as writer:
            event = writer.emit(
                "resource_sample",
                {
                    "memory": {"headroom_bytes": 300 - index},
                    "processes": {"rss_bytes": 100 + index},
                    "gpu_processes": [],
                },
                basis="observed",
            )
            event["local_monotonic"] = index * 1000000
            events.append(event)
            if index == 0:
                events.append(
                    writer.emit(
                        "feature_produced",
                        {
                            "position": 0,
                            "nbytes": 20,
                            "tokens": 40,
                            "duration_seconds": 2,
                        },
                        basis="observed",
                    )
                )
                events.append(
                    writer.emit(
                        "phase_duration",
                        {"phase": "production", "duration_seconds": 4},
                        basis="observed",
                    )
                )
                events.append(
                    writer.emit(
                        "wait",
                        {"reason": "source_capacity", "duration_seconds": 3},
                        basis="observed",
                    )
                )
    result = summarize_metrics(plan, events)
    assert result["feature_tokens_per_second"] == 10
    assert result["writer_seconds"] == 2 and result["wait_seconds"] == 3
    assert result["training_rank_seconds"] is None
    assert result["waits_by_reason"] == {"source_capacity": 3}
    assert sorted(n["peak_run_rss_bytes"] for n in result["nodes"].values()) == [
        100,
        101,
        102,
    ]


def test_pipeline_metrics_preserve_missing_measurements(tmp_path):
    from deepspec.pipeline.metrics import summarize_metrics
    from deepspec.pipeline.planning import build_plan
    from deepspec.pipeline.runtime import EventWriter
    from tests.pipeline_topology_fixtures import task_config, node_facts, input_plan

    config = task_config("M0")
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="metrics", now=100
    ).to_dict()
    with EventWriter(
        tmp_path / "events.jsonl",
        run_id=plan["run_id"],
        plan_hash=plan["plan_hash"],
        sender_identity={"component": "reader"},
    ) as writer:
        event = writer.emit(
            "feature_read",
            {
                "position": 0,
                "reader_rank": 0,
                "nbytes": 20,
                "verified": True,
                "duration_seconds": None,
            },
            basis="verified",
            missing={"duration_seconds": "reader clock unavailable"},
        )
    result = summarize_metrics(plan, [event])
    assert (
        result["feature_tokens_per_second"] is None
        and result["read_wait_seconds"] is None
    )
    assert result["complete_reads"] == 1
    assert any(m["reason"] == "reader clock unavailable" for m in result["missing"])
