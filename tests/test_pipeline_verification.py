"""Independent verification uses real, tiny CPU DCP files, never success flags."""

import copy
import hashlib
import json
import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

from deepspec.pipeline.planning import build_plan
from deepspec.pipeline.schema import content_hash
from deepspec.pipeline.verification import (
    checkpoint_expectation,
    load_checkpoint_expectations,
    save_checkpoint_expectation,
    verify_checkpoint,
    verify_execution,
    verify_progress,
    verify_release,
)
from tests.pipeline_topology_fixtures import input_plan, node_facts, task_config


def make_checkpoint(tmp_path, dp=1, steps=3, mutate=None):
    config = task_config(f"M1-1{dp}", steps=steps, output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="verify", now=100
    ).to_dict()
    state = {
        "model": {"fc.weight": torch.zeros(2, 3)},
        "optimizer": {
            "state": {
                "fc.weight": {
                    "step": torch.tensor(0.0),
                    "master_param": torch.zeros(2, 3),
                    "exp_avg": torch.zeros(2, 3),
                    "exp_avg_sq": torch.zeros(2, 3),
                }
            },
            "param_groups": [{"params": ["fc.weight"], "step": 0, "lr": 0.01}],
        },
        "lr_scheduler": {"0": {"last_epoch": 0}},
        "dataloader": {
            "cursor": 0,
            "next_global_microbatch": 0,
            "feature_identity": "manifest",
        },
        "train_state": {
            "step": 0,
            "ntokens_seen": 0,
            "run_id": "verify",
            "training_identity": "native-recipe",
        },
    }
    expectations = []
    for rank in plan["training_ranks"]:
        local = copy.deepcopy(state)
        local["train_state"][f"rank_{rank['global_rank']}"] = {
            "cpu_rng": torch.arange(8, dtype=torch.uint8),
            "cuda_rng": torch.arange(8, dtype=torch.uint8),
            "python_rng": (3, (1, 2), None),
            "numpy_rng": ("MT19937", [1, 2], 2),
        }
        expectations.append(
            checkpoint_expectation(
                plan, rank["global_rank"], local, changed_parameters=("fc.weight",)
            )
        )
        state["train_state"].update(
            {k: v for k, v in local["train_state"].items() if k.startswith("rank_")}
        )
    state["model"]["fc.weight"].fill_(0.25)
    optim = state["optimizer"]["state"]["fc.weight"]
    optim["step"].fill_(steps)
    optim["master_param"].fill_(0.25)
    optim["exp_avg"].fill_(0.1)
    optim["exp_avg_sq"].fill_(0.01)
    state["optimizer"]["param_groups"][0]["step"] = steps
    state["lr_scheduler"]["0"]["last_epoch"] = steps
    state["dataloader"].update(
        cursor=4 * steps // dp, next_global_microbatch=4 * steps // dp
    )
    state["train_state"]["step"] = steps
    if mutate:
        mutate(state)
    path = tmp_path / "checkpoints" / f"step-{steps}"
    dcp.save(state, checkpoint_id=path, no_dist=True)
    commit = {
        "format_version": 1,
        "checkpoint": str(path),
        "run_id": plan["run_id"],
        "training_identity": "native-recipe",
        "completed_updates": steps,
        "next_global_microbatch": 4 * steps // dp,
        "partition_identity": "manifest",
        "world_size": 4 * dp,
        "input_plan_identity": plan["input_plan_hash"],
        "metadata_sha256": hashlib.sha256(
            (path / ".metadata").read_bytes()
        ).hexdigest(),
    }
    (path / "commit.json").write_text(json.dumps(commit))
    return plan, path, expectations


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_complete_native_state_uses_plan_counts_and_cpu_only(tmp_path, dp, steps):
    plan, path, expectations = make_checkpoint(tmp_path, dp, steps)
    result = verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)
    assert result["verified"]
    assert result["counts"]["native_cursor"] == steps * 4 // dp
    assert result["counts"]["reader_count"] == steps * 16
    assert result["changed_parameters"] == ["fc.weight"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s: s["model"]["fc.weight"].fill_(float("nan")), "non-finite"),
        (lambda s: s["model"]["fc.weight"].zero_(), "unchanged"),
        (lambda s: s["optimizer"]["state"]["fc.weight"]["step"].fill_(2), "optimizer"),
        (lambda s: s["dataloader"].update(next_global_microbatch=6), "cursor"),
        (lambda s: s["train_state"].update(step=2), "step"),
        (lambda s: s["train_state"].update(training_identity="wrong"), "identity"),
        (lambda s: s.pop("lr_scheduler"), "coverage"),
        (lambda s: s["optimizer"]["state"]["fc.weight"].pop("exp_avg"), "coverage"),
        (lambda s: s["train_state"].pop("rank_3"), "coverage"),
    ],
)
def test_saved_state_cannot_self_certify_success(tmp_path, mutate, match):
    plan, path, expectations = make_checkpoint(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match=match):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)


@pytest.mark.parametrize(
    "field", ["input_plan_identity", "run_id", "training_identity"]
)
def test_wrong_native_commit_identity_is_rejected(tmp_path, field):
    plan, path, expectations = make_checkpoint(tmp_path)
    commit = json.loads((path / "commit.json").read_text())
    commit[field] = "different"
    (path / "commit.json").write_text(json.dumps(commit))
    with pytest.raises(ValueError, match="identity"):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)


def test_wrong_teacher_in_initial_expectation_is_rejected(tmp_path):
    plan, path, expectations = make_checkpoint(tmp_path)
    expectations[0]["model_identity"] = "wrong-teacher"
    with pytest.raises(ValueError, match="identity"):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)


@pytest.mark.parametrize("fault", ["missing_file", "truncated", "tensor_hole"])
def test_missing_dcp_storage_or_tensor_range_is_rejected(tmp_path, fault):
    plan, path, expectations = make_checkpoint(tmp_path)
    storage = next(path.glob("*.distcp"))
    if fault == "missing_file":
        storage.unlink()
    elif fault == "truncated":
        storage.write_bytes(b"partial")
    else:
        metadata = pickle.loads((path / ".metadata").read_bytes())
        metadata.state_dict_metadata["model.fc.weight"].chunks[0].sizes = torch.Size(
            [1, 3]
        )
        (path / ".metadata").write_bytes(pickle.dumps(metadata))
        commit = json.loads((path / "commit.json").read_text())
        commit["metadata_sha256"] = hashlib.sha256(
            (path / ".metadata").read_bytes()
        ).hexdigest()
        (path / "commit.json").write_text(json.dumps(commit))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)


def test_cpu_budget_is_checked_before_loading_any_tensor(tmp_path, monkeypatch):
    plan, path, expectations = make_checkpoint(tmp_path)
    monkeypatch.setattr(
        "deepspec.pipeline.verification._load_keys",
        lambda *a, **kw: pytest.fail(
            "Insufficient CPU budget must reject before DCP load"
        ),
    )
    with pytest.raises(ValueError, match="budget"):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=1)


@pytest.mark.parametrize("dp,cursor", [(1, 12), (2, 6)])
def test_five_updates_reject_historical_three_update_cursor(tmp_path, dp, cursor):
    plan, path, expectations = make_checkpoint(
        tmp_path,
        dp=dp,
        steps=5,
        mutate=lambda s: s["dataloader"].update(next_global_microbatch=cursor),
    )
    with pytest.raises(ValueError, match="cursor"):
        verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)


@pytest.mark.parametrize("fault", [None, "tamper", "missing_rank", "late"])
def test_initial_expectations_are_bound_to_pre_update_rank_events(tmp_path, fault):
    from deepspec.pipeline.runtime import atomic_json, message_envelope

    plan, _, expectations = make_checkpoint(tmp_path)
    events = []
    for value in expectations:
        rank = value["rank"]
        path = tmp_path / "initial-state" / f"rank-{rank}.json"
        atomic_json(path, value)
        events.append(
            message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"global_rank": rank},
                event="checkpoint_expectation",
                basis="observed",
                data={
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            )
        )
    if fault == "tamper":
        value["initial_parameters"]["model.fc.weight"][0]["sha256"] = "rewritten"
        atomic_json(path, value)
    elif fault == "missing_rank":
        events.pop()
    elif fault == "late":
        events.insert(
            0,
            message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"global_rank": 0},
                event="rank_update_completed",
                basis="observed",
                data={},
            ),
        )
    if fault:
        with pytest.raises(ValueError):
            load_checkpoint_expectations(plan, events)
    else:
        assert load_checkpoint_expectations(plan, events) == expectations


def test_training_initial_schema_is_written_once_before_any_update(tmp_path):
    from types import SimpleNamespace

    config = task_config("M0", output_dir=tmp_path)
    plan = build_plan(
        config, node_facts(config), input_plan(config), run_id="initial", now=100
    ).to_dict()
    raw = {
        "model": {
            name: torch.zeros(2, 3)
            for name in (
                "fc.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
            )
        },
        "dataloader": {"next_global_microbatch": 0, "feature_identity": "manifest"},
        "train_state": {"step": 0, "training_identity": "recipe"},
    }
    trainer = SimpleNamespace(
        step=0,
        completed_updates=0,
        checkpointer=SimpleNamespace(
            states={
                k: SimpleNamespace(state_dict=lambda v=v: v) for k, v in raw.items()
            }
        ),
    )
    saved = save_checkpoint_expectation(plan, 0, trainer)
    path = tmp_path / "initial-state/rank-0.json"
    original = path.read_bytes()
    assert saved["sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        save_checkpoint_expectation(plan, 0, trainer)
    assert path.read_bytes() == original
    trainer.step = 1
    with pytest.raises(ValueError, match="precede"):
        save_checkpoint_expectation(plan, 1, trainer)
    assert not (tmp_path / "initial-state/rank-1.json").exists()


def test_initial_model_can_have_multiple_native_local_checkpoint_chunks(tmp_path):
    from torch.distributed.tensor._shards_wrapper import LocalShardsWrapper

    plan, path, expectations = make_checkpoint(tmp_path)
    local = LocalShardsWrapper([torch.zeros(2, 1), torch.zeros(2, 2)], [(0, 0), (0, 1)])
    for rank, expectation in enumerate(expectations):
        captured = checkpoint_expectation(
            plan,
            rank,
            {
                "model": {"fc.weight": local},
                "train_state": {"step": 0, "training_identity": "native-recipe"},
                "dataloader": {
                    "next_global_microbatch": 0,
                    "feature_identity": "manifest",
                },
            },
            changed_parameters=("fc.weight",),
        )
        expectation["initial_parameters"] = captured["initial_parameters"]
    assert verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)[
        "verified"
    ]


def progress_evidence(plan):
    from deepspec.pipeline.runtime import message_envelope

    events = []
    for rank in plan["training_ranks"]:
        for step in range(1, plan["counts"]["optimizer_steps"] + 1):
            events.append(
                message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "training_rank", **rank},
                    event="rank_update_completed",
                    basis="observed",
                    data={
                        "optimizer_step": step,
                        "native_cursor": step * plan["counts"]["gas"],
                        "sample_cursor": step * 4,
                        "loss": 0.5,
                    },
                )
            )
    for sample in plan["samples"]:
        for rank in sample["reader_ranks"]:
            events.append(
                message_envelope(
                    plan["run_id"],
                    plan["plan_hash"],
                    {"component": "feature_buffer"},
                    event="feature_read",
                    basis="verified",
                    data={
                        "position": sample["position"],
                        "reader_rank": rank,
                        "verified": True,
                        "nbytes": sample["nbytes"],
                    },
                )
            )
    return events


@pytest.mark.parametrize("dp", [1, 2])
def test_read_ack_does_not_replace_optimizer_progress(tmp_path, dp):
    plan, _, _ = make_checkpoint(tmp_path, dp=dp, steps=5)
    events = progress_evidence(plan)
    assert verify_progress(plan, events)["reader_count"] == 80
    with pytest.raises(ValueError, match="updates"):
        verify_progress(plan, [e for e in events if e["event"] == "feature_read"])
    with pytest.raises(ValueError, match="reader"):
        verify_progress(plan, events[:-1])
    events[0]["data"]["loss"] = float("nan")
    with pytest.raises(ValueError):
        verify_progress(plan, events)


def test_duplicate_conflicting_or_foreign_progress_is_rejected(tmp_path):
    plan, _, _ = make_checkpoint(tmp_path)
    events = progress_evidence(plan)
    with pytest.raises(ValueError, match="duplicate"):
        verify_progress(plan, events + [events[-1]])
    events[0]["plan_hash"] = content_hash({"different": True})
    with pytest.raises(ValueError, match="identity|another"):
        verify_progress(plan, events)


def test_native_optimizer_flat_state_and_scheduler_survive_independent_cpu_load(
    tmp_path,
):
    from torchtitan.components.optimizer import ParamGroupConfig
    from torchtitan.models.dspark_draft.optimizer import DraftOptimizers
    from torchtitan.models.dspark_draft.scheduler import DraftSchedulers

    model = torch.nn.Module()
    model.fc = torch.nn.Linear(3, 2, bias=False)
    optimizer = DraftOptimizers.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=".*",
                optimizer_name="MasterWeightAdamW",
                optimizer_kwargs={"lr": 0.01, "weight_decay": 0.0},
            )
        ],
    ).build(model_parts=[model])
    scheduler = DraftSchedulers.Config(
        warmup_steps=1, total_steps=4, decay_type="cosine"
    ).build(optimizers=optimizer, training_steps=3)
    plan, path, expectations = make_checkpoint(tmp_path)
    # The production expectation is captured before updates through the same
    # native state_dict methods as PhaseCheckpointer, with no new optimizer.
    raw = _native_state(model, optimizer, scheduler, plan, step=0)
    expectations = [
        checkpoint_expectation(
            plan, r["global_rank"], raw, changed_parameters=("fc.weight",)
        )
        for r in plan["training_ranks"]
    ]
    for _ in range(3):
        model.fc.weight.grad = torch.ones_like(model.fc.weight)
        optimizer.step()
        scheduler.step()
    state = _native_state(model, optimizer, scheduler, plan, step=3)
    dcp.save(state, checkpoint_id=path, no_dist=True)
    commit = json.loads((path / "commit.json").read_text())
    commit["metadata_sha256"] = hashlib.sha256(
        (path / ".metadata").read_bytes()
    ).hexdigest()
    (path / "commit.json").write_text(json.dumps(commit))
    report = verify_checkpoint(plan, path, expectations, memory_budget_bytes=2**20)
    assert report["verified"] and report["changed_parameters"] == ["fc.weight"]


def _native_state(model, optimizer, scheduler, plan, *, step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "train_state": {
            "step": step,
            "run_id": plan["run_id"],
            "training_identity": "native-recipe",
            **{
                f"rank_{r['global_rank']}": {
                    "cpu_rng": torch.arange(8, dtype=torch.uint8),
                    "cuda_rng": torch.arange(8, dtype=torch.uint8),
                    "python_rng": (3, (1, 2), None),
                    "numpy_rng": ("MT19937", [1, 2], 2),
                }
                for r in plan["training_ranks"]
            },
        },
        "dataloader": {
            "cursor": step * 4,
            "next_global_microbatch": step * 4,
            "feature_identity": "manifest",
        },
    }


@pytest.mark.parametrize(
    "fault", [None, "source", "reader", "resource", "missing", "foreign"]
)
def test_success_flag_cannot_hide_source_or_resource_residue(tmp_path, fault):
    plan, _, _ = make_checkpoint(tmp_path)
    source, registry, cleanup = released_evidence(plan)
    if fault == "source":
        source["records"][0]["delete_confirmed"] = False
    elif fault == "reader":
        source["records"][0]["acked"].pop()
    elif fault == "resource":
        registry["resources"][0]["release_state"] = "unknown"
    elif fault == "missing":
        cleanup.clear()
    elif fault == "foreign":
        cleanup[0]["ray_id"] = "other-worker"
    if fault is None:
        assert (
            verify_release(plan, source=source, registries=[registry], cleanup=cleanup)[
                "sources_released"
            ]
            == 12
        )
    else:
        with pytest.raises(ValueError):
            verify_release(plan, source=source, registries=[registry], cleanup=cleanup)


def released_evidence(plan):
    from deepspec.pipeline.runtime import message_envelope

    def envelope(**payload):
        return message_envelope(
            plan["run_id"], plan["plan_hash"], {"component": "probe"}, **payload
        )

    source = envelope(
        reserved_bytes=0,
        resident_bytes=0,
        records=[
            {
                "position": s["position"],
                "state": "released",
                "delete_confirmed": True,
                "acked": list(s["reader_ranks"]),
            }
            for s in plan["samples"]
        ],
    )
    resource = {
        "allocation_id": "worker",
        "ray_id": "actual-worker",
        "release_state": "released",
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
    }
    registry = envelope(resources=[resource], cleanup_complete=True, state="succeeded")
    cleanup = [
        envelope(resource_id="worker", ray_id="actual-worker", release_state="released")
    ]
    return source, registry, cleanup


@pytest.mark.parametrize("fault", [None, "missing_commit", "foreign_commit"])
def test_complete_verification_requires_every_rank_native_commit(tmp_path, fault):
    from deepspec.pipeline.runtime import atomic_json, message_envelope

    plan, path, expectations = make_checkpoint(tmp_path)
    source, registry, cleanup = released_evidence(plan)
    events = []
    for expectation in expectations:
        rank = expectation["rank"]
        initial = tmp_path / "initial-state" / f"rank-{rank}.json"
        atomic_json(initial, expectation)
        events.append(
            message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                {"global_rank": rank},
                event="checkpoint_expectation",
                basis="observed",
                data={
                    "path": str(initial),
                    "sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                },
            )
        )
    events.extend(progress_evidence(plan))
    commit = json.loads((path / "commit.json").read_text())
    for rank in plan["training_ranks"]:
        events.append(
            message_envelope(
                plan["run_id"],
                plan["plan_hash"],
                rank,
                event="checkpoint_committed",
                basis="verified",
                data={
                    "path": str(path),
                    "commit_identity": content_hash(commit),
                    "native_cursor": plan["counts"]["native_cursor"],
                    "sample_cursor": plan["counts"]["sample_cursor"],
                },
            )
        )
    if fault == "missing_commit":
        events.pop()
    elif fault == "foreign_commit":
        events[-1]["data"]["commit_identity"] = "different-checkpoint"
    kwargs = dict(
        events=events,
        source=source,
        registries=[registry],
        cleanup=cleanup,
        memory_budget_bytes=2**20,
    )
    if fault:
        with pytest.raises(ValueError, match="commit"):
            verify_execution(plan, path, **kwargs)
    else:
        result = verify_execution(plan, path, **kwargs)
        assert result["verified"]
        assert result["release"] == {"sources_released": 12, "resources_released": 1}
