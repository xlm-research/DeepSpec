import copy

import pytest

from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.runtime import PipelineError
from deepspec.pipeline.schema import normalize_task_config
from tests.pipeline_topology_fixtures import (
    LAYOUTS,
    input_plan,
    node_facts,
    task_config,
)


def make_plan(config, *, facts=None, inputs=None, now=100):
    config = normalize_task_config(config).to_dict()
    return build_plan(
        config,
        node_facts(config) if facts is None else facts,
        input_plan(config) if inputs is None else inputs,
        run_id="fixture-run",
        now=now,
    ).to_dict()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_supported_layout_has_complete_cpu_bundles_ranks_and_readers(layout):
    config = task_config(layout)
    plan = make_plan(config)
    assert len(plan["replicas"]) == config["inference"]["dp"]
    assert (
        sum(
            b["resources"].get("GPU", 0)
            for pg in plan["placement_groups"]
            for b in pg["bundles"]
        )
        == config["inference"]["tp"] * config["inference"]["dp"]
        + 4 * config["training"]["dp"]
    )
    assert len(plan["training_ranks"]) == 4 * config["training"]["dp"]
    assert sum(len(s["reader_ranks"]) for s in plan["samples"]) == 48
    assert plan["counts"]["native_cursor"] == 12 // config["training"]["dp"]
    for billing in plan["cpu_budgets"].values():
        assert (
            sum(billing["components"].values()) == billing["total"] <= billing["limit"]
        )


def test_m2_capacity_uses_each_node_and_does_not_hardcode_dp_two():
    config = task_config("M2-DP2")
    config["inference"]["dp"] = 3
    for allocation in config["inference"]["nodes"]:
        allocation["gpus"] = 13
    plan = make_plan(config, facts=node_facts(config, gpu_count=16))
    assert len(plan["replicas"]) == 3
    for replica in plan["replicas"]:
        assert replica["worker_nodes"] == ["node-a"] * 4 + ["node-b"] * 4
        assert replica["core_node"] == "node-a" and replica["writer_slot"] == 0
    config["inference"]["nodes"][1]["gpus"] = 11
    with pytest.raises(PipelineError, match="capacity"):
        make_plan(config, facts=node_facts(config, gpu_count=16))


def test_m3_counts_local_readers_and_pool_once():
    plan = make_plan(task_config("M3"))
    assert plan["node_budgets"]["node-a"]["writers"] == 1
    assert plan["node_budgets"]["node-b"]["readers"] == 4
    assert plan["node_budgets"]["node-c"]["readers"] == 4
    assert plan["node_budgets"]["node-b"]["pool_bytes"] > 0
    assert plan["node_budgets"]["node-c"]["pool_bytes"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("heartbeat", 30),
        ("run", 0),
        ("run", float("inf")),
        ("run", float("nan")),
        ("run", True),
    ],
)
def test_invalid_timeouts_have_field_paths(field, value):
    config = task_config()
    config["timeouts_seconds"][field] = value
    with pytest.raises(PipelineError) as error:
        normalize_task_config(config)
    assert "timeouts_seconds" in error.value.to_dict()["field_path"]


def test_normalization_materializes_only_missing_defaults():
    config = task_config()
    del config["transport"]["rdma_devices"]
    del config["timeouts_seconds"]["budget_snapshot"]
    normalized = normalize_task_config(config)
    assert "rdma_devices" not in config["transport"]
    assert normalized.to_dict()["transport"]["rdma_devices"] == ""
    assert normalized.to_dict()["timeouts_seconds"]["budget_snapshot"] == 5
    config["timeouts_seconds"]["budget_snapshot"] = 6
    assert normalize_task_config(config).config_hash != normalized.config_hash


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_alias",
        "duplicate_node",
        "unknown",
        "bad_uuid",
        "overlap",
        "cpu",
        "memory",
        "identity",
        "stale",
        "loopback",
        "world",
        "samples",
        "capability",
    ],
)
def test_invalid_facts_or_layout_fail_before_execution(mutation):
    config = task_config("M0" if mutation == "overlap" else "M3")
    facts = node_facts(config)
    inputs = input_plan(config)
    if mutation == "duplicate_alias":
        config["nodes"][1]["alias"] = config["nodes"][0]["alias"]
    elif mutation == "duplicate_node":
        config["nodes"][1]["selector"] = copy.deepcopy(config["nodes"][0]["selector"])
    elif mutation == "unknown":
        config["training"]["typo"] = 1
    elif mutation == "bad_uuid":
        config["inference"]["nodes"][0]["allowed_gpu_uuids"] = ["GPU-missing"] * 4
    elif mutation == "overlap":
        for role in ("inference", "training"):
            config[role]["nodes"][0]["allowed_gpu_uuids"] = [
                f"GPU-a-{i}" for i in range(4)
            ]
    elif mutation == "cpu":
        config["nodes"][1]["cpu_limit"] = 1
    elif mutation == "memory":
        facts[1]["memory"]["headroom_bytes"] = 1
    elif mutation == "identity":
        facts[1]["identities"]["model"] = "other-model"
    elif mutation == "stale":
        facts[0]["request_sent_at"] = 95
    elif mutation == "loopback":
        config["store"]["master"] = {"mode": "external", "endpoint": "127.0.0.1:5000"}
    elif mutation == "world":
        config["training"]["nodes"][0]["gpus"] = 8
    elif mutation == "samples":
        inputs["batches"].pop()
    elif mutation == "capability":
        facts[0]["capabilities"]["allocation_gate"] = False
    with pytest.raises(PipelineError) as error:
        make_plan(config, facts=facts, inputs=inputs)
    assert error.value.to_dict()["field_path"]


def test_selector_ambiguity_and_explicit_uuid_inventory():
    config = task_config()
    facts = node_facts(config)
    config["nodes"][0]["selector"] = {"ip": facts[0]["ip"]}
    config["inference"]["nodes"][0]["allowed_gpu_uuids"] = [
        f"GPU-a-{i}" for i in (1, 3, 5, 7)
    ]
    assert (
        make_plan(config, facts=facts)["replicas"][0]["worker_nodes"] == ["node-a"] * 4
    )
    facts.append({**facts[0], "node_id": "different-node"})
    with pytest.raises(PipelineError):
        make_plan(config, facts=facts)


@pytest.mark.parametrize("length", [1, 7])
def test_prepared_contract_accepts_native_converted_features(
    tmp_path, monkeypatch, length
):
    import json

    import torch
    from safetensors.torch import save_file

    from deepspec.pipeline import run as legacy_run
    from deepspec.pipeline.buffer import BufferLedger
    from deepspec.pipeline.planning import Run, prepare_input
    from deepspec.pipeline.store import describe_tensors
    from deepspec.trainer.qwen3_8_vllm import convert_hidden_states

    model = tmp_path / "model"
    model.mkdir()
    save_file({"weight": torch.zeros(8, 8)}, model / "model.safetensors")
    config = normalize_task_config(
        task_config("M0", output_dir=tmp_path / "run", steps=1)
    ).to_dict()
    config["model_path"] = str(model)
    config["data"]["context_length"] = length
    run = Run.create(config["output_dir"])
    ids = torch.arange(length).reshape(1, -1)
    batch = {"input_ids": ids, "loss_mask": torch.ones_like(ids)}
    features = convert_hidden_states(
        {
            "token_ids": ids[0],
            "hidden_states": torch.zeros(length, 3, 8, dtype=torch.bfloat16),
        },
        batch,
        hidden_size=8,
        num_layers=2,
    )
    nbytes = sum(t.numel() * t.element_size() for t in features.values())

    def prepare(legacy, path, **kwargs):
        from pathlib import Path

        inputs = Path(run.output_dir) / "inputs"
        inputs.mkdir()
        samples = []
        for position in range(4):
            target = inputs / f"sample-{position}.pt"
            torch.save(batch, target)
            samples.append(
                {
                    "position": position,
                    "id": target.name,
                    "sample_id": f"sample-{position}",
                    "input_identity": f"identity-{position}",
                    "input_path": str(target),
                    "length": length,
                    "nbytes": nbytes,
                }
            )
        legacy.update(
            samples=samples, teacher={"hidden_size": 8, "target_layer_ids": [0, 1]}
        )
        (inputs / "input-plan.json").write_text(json.dumps({"batches": samples}))

    # The expensive tokenizer/model lookup is replaced; the production planner,
    # native feature converter, serializer and strict publication gate are real.
    monkeypatch.setattr(legacy_run, "prepare", prepare)
    inputs, _ = prepare_input(config, run)
    sample = inputs["batches"][0]
    descriptor = {**sample, "fields": describe_tensors("native/sample-0", features)}
    ledger = BufferLedger(
        inputs["batches"],
        capacity=4 * nbytes,
        window=4,
        readers=range(4),
        samples_per_update=4,
    )
    assert ledger.reserve(0)
    ledger.start_write(0)
    ledger.ready(0, descriptor)
    assert ledger.claim(0, 0) == descriptor
    for field in ("seq_len", "context_chunk_len"):
        ledger = BufferLedger(
            inputs["batches"],
            capacity=4 * nbytes,
            window=4,
            readers=range(4),
            samples_per_update=4,
        )
        assert ledger.reserve(0)
        ledger.start_write(0)
        malformed = copy.deepcopy(descriptor)
        malformed["fields"][field]["shape"] = []
        with pytest.raises(ValueError, match=f"Published {field} shape/dtype"):
            ledger.ready(0, malformed)
