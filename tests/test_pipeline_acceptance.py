import json

import pytest
from jsonschema import Draft202012Validator

from tests.pipeline_topology_fixtures import (
    CONTRACTS,
    LAYOUTS,
    input_plan,
    node_facts,
    task_config,
)
from tests.run_pipeline_acceptance import NORMAL_CASES, record_result


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_fixtures_satisfy_structure_and_use_exact_sample_counts(layout, steps):
    config = task_config(layout, steps=steps)
    Draft202012Validator(
        json.loads((CONTRACTS / "task-config.schema.json").read_text())
    ).validate(config)
    samples = input_plan(config)["batches"]
    assert len(samples) == 4 * steps
    assert [s["position"] for s in samples] == list(range(4 * steps))
    assert node_facts(config)[0]["gpus"][1]["index"] == "3"


def test_inventory_has_thirteen_normal_cases_and_append_only_runs(tmp_path):
    assert len(NORMAL_CASES) == len(set(NORMAL_CASES)) == 13
    one = record_result(
        tmp_path,
        "m3-4k",
        status="blocked",
        evidence_level="training",
        reason="No cluster",
    )
    two = record_result(
        tmp_path,
        "m3-4k",
        status="not_run",
        evidence_level="inventory",
        reason="Not executed",
    )
    assert one["record_id"] != two["record_id"]
    assert len(json.loads((tmp_path / "results.json").read_text())["runs"]) == 2
    assert (tmp_path / "m3-4k" / one["record_id"] / "result.json").is_file()


@pytest.mark.parametrize("evidence", [{}, {"executed": False}, {"executed": True}])
def test_missing_evidence_cannot_be_marked_passed(tmp_path, evidence):
    with pytest.raises(ValueError):
        record_result(
            tmp_path,
            "m3-4k",
            status="passed",
            evidence_level="training",
            evidence=evidence,
        )
    assert not list(tmp_path.iterdir())


def test_cpu_suite_has_test_identity_without_fabricating_a_topology_plan(tmp_path):
    artifact = tmp_path / "junit.xml"
    artifact.write_text("<testsuites />")
    evidence = {
        "executed": True,
        "run_id": "probe",
        "test_suite_hash": "source-digest",
        "started_at": "2026-09-20T00:00:00Z",
        "ended_at": "2026-09-20T00:01:00Z",
        "checks": {"tests_passed": True},
        "artifacts": [str(artifact)],
    }
    result = record_result(
        tmp_path,
        "capacity-contracts",
        status="passed",
        evidence_level="cpu_contract",
        evidence=evidence,
    )
    assert result["plan_hash"] is None
    assert result["test_suite_hash"] == "source-digest"
    with pytest.raises(ValueError, match="identities"):
        record_result(
            tmp_path,
            "m0-4k",
            status="passed",
            evidence_level="training",
            evidence=evidence,
        )


@pytest.mark.parametrize(
    "layout,case", [("M1-12", "m1-21-4k"), ("M2-DP1", "m2-dp2-4k")]
)
def test_training_acceptance_rejects_mismatched_dp_before_launch(
    tmp_path, monkeypatch, layout, case
):
    from deepspec.pipeline import cli, execution
    from deepspec.pipeline.planning import build_plan
    from tests.run_pipeline_acceptance import execute_training_plan

    config = task_config(layout, output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="case", now=100
    ).to_dict()
    monkeypatch.setattr(cli, "load_run", lambda _: (tmp_path, plan, {}))
    monkeypatch.setattr(
        execution, "run_plan", lambda _: pytest.fail("training launched")
    )
    with pytest.raises(ValueError, match="topology/context"):
        execute_training_plan(tmp_path, case, tmp_path / "plan.json")
