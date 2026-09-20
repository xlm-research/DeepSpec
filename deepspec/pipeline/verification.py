"""Independent CPU checks of native DCP state and frozen-plan progress."""

import hashlib
import json
import math
from pathlib import Path

from .planning import TopologyPlan
from .runtime import atomic_json_once, message_envelope, validate_message
from .schema import content_hash
from .topology import expected_counts


def _tensor_hash(tensor):
    import torch

    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def checkpoint_expectation(plan, rank, state, *, changed_parameters):
    """Capture the native schema and initial local weights before any update.

    DTensor slices retain global offsets so a CPU verifier can compare them
    against a resharded checkpoint without recreating the original device mesh.
    Call on every rank with the native checkpointer's state dictionaries.
    """
    import torch
    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.checkpoint.metadata import MetadataIndex
    from torch.distributed.checkpoint.planner_helpers import _create_chunk_list
    from torch.distributed.checkpoint.utils import find_tensor_shard

    plan = TopologyPlan.from_dict(plan).to_dict()
    flat = flatten_state_dict(state)[0]
    if (
        flat.get("train_state.step") != 0
        or flat.get("dataloader.next_global_microbatch") != 0
    ):
        raise ValueError("Checkpoint expectations must precede the first update")
    participant = plan["training_ranks"][rank]
    node = next(
        n for n in plan["nodes"].values() if n["node_id"] == participant["node_id"]
    )
    schema, initial = {}, {}
    for key, value in flat.items():
        if torch.is_tensor(value):
            schema[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if key.startswith("model."):
                initial[key] = []
                for index, chunk in enumerate(_create_chunk_list(value)):
                    local = find_tensor_shard(
                        value, MetadataIndex(key, chunk.offsets, index)
                    )
                    initial[key].append(
                        {
                            "offset": list(chunk.offsets),
                            "shape": list(chunk.sizes),
                            "sha256": _tensor_hash(local),
                        }
                    )
        else:
            schema[key] = {"kind": "bytes"}
    changed = sorted(set(changed_parameters))
    if not changed or any(f"model.{name}" not in initial for name in changed):
        raise ValueError("Expected parameter updates must name initial model tensors")
    return message_envelope(
        plan["run_id"],
        plan["plan_hash"],
        {"component": "checkpoint_expectation", **participant},
        rank=rank,
        input_plan_hash=plan["input_plan_hash"],
        model_identity=node["identities"]["model"],
        training_identity=flat["train_state.training_identity"],
        partition_identity=flat["dataloader.feature_identity"],
        schema=schema,
        initial_parameters=initial,
        changed_parameters=changed,
    )


def _expectations(plan, expectations):
    schema, initial, ranks = {}, {}, set()
    identities, changed_sets, partitions = set(), set(), set()
    for value in expectations:
        validate_message(value, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        rank = value["rank"]
        if (
            type(rank) is not int
            or rank in ranks
            or not 0 <= rank < len(plan["training_ranks"])
        ):
            raise ValueError("Checkpoint expectation rank identity differs")
        ranks.add(rank)
        participant = plan["training_ranks"][rank]
        node = next(
            n for n in plan["nodes"].values() if n["node_id"] == participant["node_id"]
        )
        if (
            any(value["sender_identity"].get(k) != v for k, v in participant.items())
            or value["model_identity"] != node["identities"]["model"]
            or value["input_plan_hash"] != plan["input_plan_hash"]
        ):
            raise ValueError("Checkpoint expectation model/input/rank identity differs")
        identities.add(value["training_identity"])
        partitions.add(value["partition_identity"])
        changed_sets.add(tuple(value["changed_parameters"]))
        for key, description in value["schema"].items():
            if key in schema and schema[key] != description:
                raise ValueError(f"Ranks disagree on native state schema: {key}")
            schema[key] = description
        for key, description in value["initial_parameters"].items():
            initial.setdefault(key, []).extend(description)
    if ranks != set(range(len(plan["training_ranks"]))) or any(
        len(values) != 1 for values in (identities, changed_sets, partitions)
    ):
        raise ValueError("Missing ranks or conflicting checkpoint expectation identity")
    for component in (
        "model",
        "optimizer",
        "lr_scheduler",
        "dataloader",
        "train_state",
    ):
        if not any(k.startswith(component + ".") for k in schema):
            raise ValueError(f"Initial native state coverage lacks {component}")
    for rank in ranks:
        for field in ("cpu_rng", "cuda_rng", "python_rng", "numpy_rng"):
            if f"train_state.rank_{rank}.{field}" not in schema:
                raise ValueError(
                    f"Initial native rank state coverage lacks {rank}/{field}"
                )
    changed = next(iter(changed_sets))
    if not changed or any(f"model.{name}" not in initial for name in changed):
        raise ValueError("Initial parameter-update expectations are missing")
    return schema, initial, identities.pop(), changed_sets.pop(), partitions.pop()


def save_checkpoint_expectation(plan, rank, trainer):
    """Write one immutable per-rank schema before its initialization ACK."""
    if trainer.step != 0 or trainer.completed_updates != 0:
        raise ValueError("Initial checkpoint evidence must precede training")
    state = {
        name: value.state_dict() for name, value in trainer.checkpointer.states.items()
    }
    expectation = checkpoint_expectation(
        plan,
        rank,
        state,
        changed_parameters=(
            "fc.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
        ),
    )
    path = Path(plan["config"]["output_dir"]) / "initial-state" / f"rank-{rank}.json"
    if not atomic_json_once(path, expectation):
        raise ValueError("Initial checkpoint expectation already exists")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_checkpoint_expectations(plan, events):
    """Bind immutable initial schemas to each rank's pre-training event."""
    result, seen, started = [], set(), set()
    for event in events:
        validate_message(event, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        rank = event["sender_identity"].get("global_rank")
        if event["event"] == "rank_update_completed":
            started.add(rank)
        if event["event"] != "checkpoint_expectation":
            continue
        if rank in seen or rank in started or event["basis"] != "observed":
            raise ValueError("Initial checkpoint evidence is duplicate or late")
        seen.add(rank)
        path = (
            Path(plan["config"]["output_dir"]) / "initial-state" / f"rank-{rank}.json"
        )
        evidence = event["data"]
        if Path(evidence["path"]).resolve() != path.resolve():
            raise ValueError("Initial checkpoint evidence path differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            raise ValueError("Initial checkpoint evidence identity changed")
        value = json.loads(raw)
        if value["rank"] != rank:
            raise ValueError("Initial checkpoint evidence rank differs")
        result.append(value)
    _expectations(plan, result)
    return result


def _check_chunks(key, tensor):
    shape = tuple(tensor.size)
    chunks = tensor.chunks
    volume = 0
    for index, chunk in enumerate(chunks):
        if (
            len(chunk.offsets) != len(shape)
            or len(chunk.sizes) != len(shape)
            or any(
                offset < 0 or size < 0 or offset + size > bound
                for offset, size, bound in zip(
                    chunk.offsets, chunk.sizes, shape, strict=True
                )
            )
        ):
            raise ValueError(f"DCP tensor range outside shape: {key}")
        volume += math.prod(chunk.sizes)
        for previous in chunks[:index]:
            if all(
                max(a, b) < min(a + sa, b + sb)
                for a, sa, b, sb in zip(
                    chunk.offsets,
                    chunk.sizes,
                    previous.offsets,
                    previous.sizes,
                    strict=True,
                )
            ):
                raise ValueError(f"Overlapping DCP tensor ranges: {key}")
    if volume != math.prod(shape):
        raise ValueError(f"Incomplete DCP tensor range coverage: {key}")


def _load_keys(path, keys):
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    state = {}
    # Pinned native dcp_to_torch_save path; public load copies an empty mapping.
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    return flatten_state_dict(state)[0]


def _finite(value):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return all(_finite(v) for v in value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    return True


def verify_checkpoint(plan, path, expectations, *, memory_budget_bytes):
    """Read every native field on CPU within a caller-approved working budget.

    This result verifies checkpoint state only. Runtime progress, allocation,
    object release and resource cleanup are separate required checks.
    """
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torchtitan.models.dspark_draft.checkpoint import read_commit

    if torch.cuda.is_initialized():
        raise ValueError("Independent checkpoint verification requires a CPU process")
    if type(memory_budget_bytes) is not int or memory_budget_bytes <= 0:
        raise ValueError("CPU verification budget must be a positive byte count")
    plan = TopologyPlan.from_dict(plan).to_dict()
    counts = expected_counts(plan["config"], sample_count=len(plan["samples"]))
    schema, initial, training_id, changed, partition = _expectations(plan, expectations)
    path = Path(path).resolve()
    commit = read_commit(str(path))
    expected_commit = {
        "format_version": 1,
        "run_id": plan["run_id"],
        "training_identity": training_id,
        "input_plan_identity": plan["input_plan_hash"],
        "partition_identity": partition,
        "world_size": len(plan["training_ranks"]),
        "completed_updates": counts["optimizer_steps"],
        "next_global_microbatch": counts["native_cursor"],
    }
    if (
        any(commit.get(k) != v for k, v in expected_commit.items())
        or Path(commit["checkpoint"]).resolve() != path
    ):
        raise ValueError(
            "Native commit identity or progress differs from the frozen plan"
        )
    metadata = FileSystemReader(path).read_metadata()
    if set(metadata.state_dict_metadata) != set(schema):
        raise ValueError("Native checkpoint state coverage differs from initialization")
    if not metadata.storage_data:
        raise ValueError("DCP storage range metadata is missing")
    largest = 0
    for key, description in metadata.state_dict_metadata.items():
        expected = schema[key]
        if isinstance(description, TensorStorageMetadata):
            if {
                "shape": list(description.size),
                "dtype": str(description.properties.dtype),
            } != expected:
                raise ValueError(f"DCP tensor schema differs: {key}")
            _check_chunks(key, description)
            size = (
                math.prod(description.size)
                * torch.empty((), dtype=description.properties.dtype).element_size()
            )
            largest = max(largest, size)
        elif expected != {"kind": "bytes"}:
            raise ValueError(f"DCP field type differs: {key}")
    for key in (k for k in schema if k.startswith("model.")):
        if key not in initial:
            raise ValueError(f"Initial model state coverage is incomplete: {key}")
        pieces = {}
        for piece in initial[key]:
            region = (tuple(piece["offset"]), tuple(piece["shape"]))
            if region in pieces and pieces[region] != piece["sha256"]:
                raise ValueError(f"Replicated initial model identity differs: {key}")
            pieces[region] = piece["sha256"]
        from types import SimpleNamespace

        _check_chunks(
            key,
            SimpleNamespace(
                size=schema[key]["shape"],
                chunks=[SimpleNamespace(offsets=o, sizes=s) for o, s in pieces],
            ),
        )
    for storage in metadata.storage_data.values():
        file = (path / storage.relative_path).resolve()
        if not file.is_relative_to(path) or storage.offset < 0 or storage.length <= 0:
            raise ValueError("Invalid DCP storage range")
        if file.stat().st_size < storage.offset + storage.length:
            raise ValueError(f"Truncated DCP storage range: {file.name}")
        largest = max(largest, storage.length)
    # One CPU tensor, native serialized input and hashing/finite-check scratch.
    required = largest * 4 + (path / ".metadata").stat().st_size * 4
    if required > memory_budget_bytes:
        raise ValueError(
            f"CPU verification budget insufficient: need {required}, approved {memory_budget_bytes}"
        )
    scalars = {
        "train_state.step": counts["optimizer_steps"],
        "train_state.run_id": plan["run_id"],
        "train_state.training_identity": training_id,
        "dataloader.next_global_microbatch": counts["native_cursor"],
        "dataloader.cursor": counts["native_cursor"],
        "dataloader.feature_identity": partition,
    }
    changed_found, loaded = set(), set()
    for key in sorted(schema):
        values = _load_keys(path, {key})
        if set(values) != {key}:
            raise ValueError(
                f"Native DCP did not load exactly the requested field: {key}"
            )
        value = values[key]
        if not _finite(value):
            raise ValueError(f"Native checkpoint contains non-finite state: {key}")
        if key in scalars and value != scalars[key]:
            raise ValueError(f"Native checkpoint identity/step/cursor differs: {key}")
        if key.startswith("optimizer.") and key.endswith(".step"):
            if value != counts["optimizer_steps"]:
                raise ValueError(f"Native optimizer step differs: {key}")
        if key.startswith("model."):
            for piece in initial[key]:
                slices = tuple(
                    slice(o, o + n)
                    for o, n in zip(piece["offset"], piece["shape"], strict=True)
                )
                if _tensor_hash(value[slices]) != piece["sha256"]:
                    changed_found.add(key.removeprefix("model."))
        loaded.add(key)
        del value, values
    if not set(scalars) <= loaded:
        raise ValueError("Native checkpoint progress/identity coverage is incomplete")
    if not set(changed) <= changed_found:
        raise ValueError(
            f"Expected parameters are unchanged: {sorted(set(changed) - changed_found)}"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("Checkpoint verification unexpectedly initialized CUDA")
    return {
        "verified": True,
        "scope": "native_checkpoint",
        "run_id": plan["run_id"],
        "plan_hash": plan["plan_hash"],
        "checkpoint": str(path),
        "counts": counts,
        "fields_loaded": len(loaded),
        "changed_parameters": sorted(changed_found),
        "cpu_working_bound_bytes": required,
        "cuda_initialized": False,
        "commit": commit,
    }


def verify_progress(plan, events):
    """Require every planned rank update and designated full reader ACK."""
    plan = TopologyPlan.from_dict(plan).to_dict()
    counts = expected_counts(plan["config"], sample_count=len(plan["samples"]))
    updates, reads, ids = set(), set(), set()
    samples = {s["position"]: s for s in plan["samples"]}
    for event in events:
        validate_message(event, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        if event["event_id"] in ids:
            raise ValueError("Evidence contains a duplicate event")
        ids.add(event["event_id"])
        kind, value, sender = event["event"], event["data"], event["sender_identity"]
        if kind == "rank_update_completed":
            rank = sender.get("global_rank")
            if type(rank) is not int or not 0 <= rank < len(plan["training_ranks"]):
                raise ValueError("Unplanned training rank")
            if any(
                sender.get(k) != plan["training_ranks"][rank][k]
                for k in ("node_id", "local_rank", "node_rank")
            ):
                raise ValueError("Training rank node identity differs")
            step = value["optimizer_step"]
            if (
                type(step) is not int
                or not 1 <= step <= counts["optimizer_steps"]
                or (rank, step) in updates
                or event["basis"] != "observed"
                or value["native_cursor"] != step * counts["gas"]
                or value["sample_cursor"]
                != step * plan["config"]["training"]["global_batch_size"]
                or not isinstance(value["loss"], (int, float))
                or not math.isfinite(value["loss"])
            ):
                raise ValueError("Invalid, duplicate or non-finite rank updates")
            updates.add((rank, step))
        elif kind == "feature_read":
            position, reader = value["position"], value["reader_rank"]
            sample = samples.get(position)
            if (
                sample is None
                or reader not in sample["reader_ranks"]
                or (position, reader) in reads
                or value["verified"] is not True
                or event["basis"] != "verified"
                or value["nbytes"] != sample["nbytes"]
            ):
                raise ValueError("Invalid, duplicate or incomplete reader evidence")
            reads.add((position, reader))
    if len(updates) != len(plan["training_ranks"]) * counts["optimizer_steps"]:
        raise ValueError("Missing rank optimizer updates")
    if len(reads) != counts["reader_count"]:
        raise ValueError("Missing designated reader evidence")
    return counts


def verify_release(plan, *, source, registries, cleanup):
    """Check source accounting and exact resource release; unknown is failure.

    The caller supplies final source evidence, all ownership registries and
    separately collected cleanup observations. A controller success flag is
    insufficient, even when checkpoint and progress verification passed.
    """
    plan = TopologyPlan.from_dict(plan).to_dict()
    validate_message(source, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    records = source["records"]
    positions = [record["position"] for record in records]
    if len(positions) != len(set(positions)) or set(positions) != set(
        range(len(plan["samples"]))
    ):
        raise ValueError("Source release coverage differs from the plan")
    if source.get("reserved_bytes") != 0 or source.get("resident_bytes") != 0:
        raise ValueError("Source objects still occupy pool bytes")
    for record in records:
        sample = plan["samples"][record["position"]]
        if (
            record["state"] != "released"
            or record.get("delete_confirmed") is not True
            or set(record["acked"]) != set(sample["reader_ranks"])
        ):
            raise ValueError("Source object release was not fully confirmed")
    resources = {}
    for registry in registries:
        if (
            registry.get("schema_version"),
            registry.get("run_id"),
            registry.get("plan_hash"),
        ) != (3, plan["run_id"], plan["plan_hash"]):
            raise ValueError("Resource registry identity differs")
        for resource in registry["resources"]:
            key = resource["allocation_id"]
            if key in resources or resource["release_state"] != "released":
                raise ValueError("Resource release is duplicate, unknown or incomplete")
            if (resource["run_id"], resource["plan_hash"]) != (
                plan["run_id"],
                plan["plan_hash"],
            ):
                raise ValueError("Resource identity differs")
            resources[key] = resource
    if not resources:
        raise ValueError("Resource release evidence is missing")
    observed = {}
    for event in cleanup:
        validate_message(event, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        key = event["resource_id"]
        if (
            key in observed
            or key not in resources
            or event["release_state"] != "released"
        ):
            raise ValueError("Resource cleanup observation is missing or unknown")
        if event["ray_id"] != resources[key]["ray_id"]:
            raise ValueError("Resource cleanup observed a different Ray identity")
        observed[key] = event
    if set(observed) != set(resources):
        raise ValueError("Resource cleanup observations are incomplete")
    return {"sources_released": len(records), "resources_released": len(resources)}


def verify_commits(plan, events, checkpoint, saved):
    counts = plan["counts"]
    committed = set()
    for event in events:
        if event["event"] != "checkpoint_committed":
            continue
        rank = event["sender_identity"].get("global_rank")
        value = event["data"]
        if (
            type(rank) is not int
            or not 0 <= rank < len(plan["training_ranks"])
            or rank in committed
            or event["basis"] != "verified"
            or value["commit_identity"] != content_hash(saved["commit"])
            or Path(value["path"]).resolve() != Path(checkpoint).resolve()
            or value["native_cursor"] != counts["native_cursor"]
            or value["sample_cursor"] != counts["sample_cursor"]
        ):
            raise ValueError("Rank checkpoint commit identity/progress differs")
        committed.add(rank)
    if committed != set(range(len(plan["training_ranks"]))):
        raise ValueError("Not every rank confirmed the native checkpoint commit")


def verify_execution(
    plan, checkpoint, *, events, source, registries, cleanup, memory_budget_bytes
):
    """Combine independent state, rank progress and release evidence."""
    plan = TopologyPlan.from_dict(plan).to_dict()
    events = list(events)
    counts = verify_progress(plan, events)
    saved = verify_checkpoint(
        plan,
        checkpoint,
        load_checkpoint_expectations(plan, events),
        memory_budget_bytes=memory_budget_bytes,
    )
    verify_commits(plan, events, checkpoint, saved)
    release = verify_release(
        plan, source=source, registries=registries, cleanup=cleanup
    )
    return message_envelope(
        plan["run_id"],
        plan["plan_hash"],
        {"component": "cpu_verifier"},
        verified=True,
        counts=counts,
        checkpoint=saved,
        release=release,
    )


def verify_placement(plan, snapshot):
    """Recheck actual all-role GPU/rank reports independently of ready/status."""
    validate_message(snapshot, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
    expected = {
        f"inference/{r['replica_id']}/{tp}": (node, 1)
        for r in plan["replicas"]
        for tp, node in enumerate(r["worker_nodes"])
    }
    expected.update(
        {
            f"training/{r['node_rank']}": (r["node_id"], r["local_world_size"])
            for r in plan["training_ranks"]
            if r["local_rank"] == 0
        }
    )
    allocations, initialized = snapshot["allocations"], snapshot["initialized"]
    if set(allocations) != set(expected):
        raise ValueError("Actual allocation participants differ from plan")
    used = set()
    for participant, (node, count) in expected.items():
        allocation = allocations[participant]
        devices = allocation["gpu_uuids"]
        inventory = {
            g["uuid"]
            for n in plan["nodes"].values()
            if n["node_id"] == node
            for g in n["gpus"]
        }
        if (
            allocation["node_id"] != node
            or len(devices) != count
            or len(set(devices)) != count
            or not set(devices) <= inventory
            or used.intersection((node, d) for d in devices)
        ):
            raise ValueError("Actual allocation GPU/node identity differs or overlaps")
        used.update((node, d) for d in devices)
    infer = {p for p in expected if p.startswith("inference/")}
    if set(initialized) != infer | {
        f"rank/{r['global_rank']}" for r in plan["training_ranks"]
    } | {"store"}:
        raise ValueError("Actual initialized participants differ from plan")
    if set(snapshot["connectors"]) != infer:
        raise ValueError("Not every native worker initialized its connector")
    for participant in infer:
        replica, tp = map(int, participant.split("/")[1:])
        connector = snapshot["connectors"][participant]
        allocation = allocations[participant]
        if (
            any(
                connector[k] != allocation[k]
                for k in ("node_id", "actor_id", "gpu_uuids")
            )
            or connector["tp_rank"] != tp
            or connector["dp_rank"] != replica
            or connector["writer"] is not (tp == 0)
        ):
            raise ValueError("Native writer/worker execution identity differs")
    for rank in plan["training_ranks"]:
        actual = initialized[f"rank/{rank['global_rank']}"]
        if (
            any(
                actual[k] != rank[k]
                for k in (
                    "global_rank",
                    "local_rank",
                    "node_rank",
                    "node_id",
                    "tp_rank",
                    "dp_rank",
                )
            )
            or actual["gas"] != plan["counts"]["gas"]
            or actual["input_plan_hash"] != plan["input_plan_hash"]
            or actual["gpu_uuid"]
            != allocations[f"training/{rank['node_rank']}"]["gpu_uuids"][
                rank["local_rank"]
            ]
        ):
            raise ValueError("Native training rank execution differs from plan")
    return {
        "verified": True,
        "inference_workers": len(infer),
        "training_ranks": len(plan["training_ranks"]),
    }
