"""Real five-layer Qwen DSpark acceptance through the phase entry."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from deepspec.orchestration.io import require_idle
from tests.compare_torchtitan_checkpoints import compare
from torchtitan.models.dspark_draft.planning import make_input_plan


TOPOLOGIES = {
    "hsdp": (8, 1, 1, 1, 2, 0, 0),
    "tp": (2, 4, 1, 1, 1, 0, 0),
    "sp": (2, 4, 1, 1, 1, 1, 0),
    "vocab": (2, 4, 1, 1, 1, 0, 1),
    "sp_vocab": (2, 4, 1, 1, 1, 1, 1),
    "cp": (4, 1, 2, 1, 1, 0, 0),
    "tp_cp": (1, 4, 2, 1, 1, 0, 0),
    "pp": (4, 1, 1, 2, 1, 0, 0),
    "tp_cp_pp": (1, 2, 2, 2, 1, 0, 0),
    "tp_pp": (1, 2, 1, 2, 1, 0, 0),
}


def launch(command, root, environment, devices, *, expect_failure=False):
    shared = environment.get("DEEPSPEC_PARALLEL_SHARED_GPUS") == "1"
    before = resource_state(devices) if shared else require_idle(devices)
    started = time.monotonic()
    with (root / "run.log").open("w") as log:
        process = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
    released = resource_state(devices) if shared else require_idle(devices)
    if (process.returncode != 0) != expect_failure:
        raise RuntimeError(
            f"Unexpected exit {process.returncode}: {root / 'run.log'}\n"
            + (root / "run.log").read_text()[-14000:]
        )
    return {
        "seconds": time.monotonic() - started,
        "release": released,
        "before": before,
        "exclusive_gpu_pool": not shared,
        "returncode": process.returncode,
    }


def resource_state(devices):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(devices),
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "devices": devices,
        "compute_processes": result.stdout.splitlines(),
        "observed_at": time.time(),
    }


def assert_close(actual, expected, path="state", *, exact=False):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(
                f"{path}: different keys: {actual.keys() ^ expected.keys()}"
            )
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}/{key}", exact=exact)
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: different lengths")
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_close(a, b, f"{path}/{index}", exact=exact)
    elif torch.is_tensor(expected) or isinstance(expected, float):
        try:
            torch.testing.assert_close(
                actual, expected, rtol=0 if exact else 1e-4, atol=0 if exact else 1e-6
            )
        except AssertionError as error:
            raise AssertionError(f"{path}: {error}") from error
    elif actual != expected:
        raise AssertionError(f"{path}: {actual} != {expected}")


def check_reference(root, fixtures, topology):
    dp, tp, cp, pp, _, _, vocab = topology
    world = dp * tp * cp * pp
    actual = [
        torch.load(root / f"rank-{rank}.pt", weights_only=True) for rank in range(world)
    ]
    for field in ("parameters", "gradients_after_clip", "adam"):
        owned = set().union(*(state["updates"][0][field] for state in actual))
        expected_names = set(fixtures[0]["result"]["updates"][0][field])
        if owned != expected_names:
            raise AssertionError(
                f"The pipeline does not own the full {field}: {owned ^ expected_names}"
            )
    for rank, state in enumerate(actual):
        dp_rank = (rank % (world // pp)) // (cp * tp)
        reference = fixtures[dp_rank]["result"]
        for index, update in enumerate(state["updates"]):
            expected = reference["updates"][index]
            for field in ("parameters", "gradients_after_clip", "adam"):
                assert_close(
                    update[field],
                    {name: expected[field][name] for name in update[field]},
                    f"rank{rank}/update{index}/{field}",
                )
            assert_close(
                update["scheduler"], expected["scheduler"], f"rank{rank}/scheduler"
            )
        assert_close(
            state["grad_norm"], reference["metrics"]["grad_norm"], f"rank{rank}/norm"
        )
        if state["cursor"] != 4:
            raise AssertionError("An update changed the logical sample grouping")
    for dp_rank in range(dp):
        base_rank = (pp - 1) * (world // pp) + dp_rank * cp * tp
        for index, expected in enumerate(fixtures[dp_rank]["result"]["microbatches"]):
            loss = sum(
                actual[base_rank + c * tp]["microbatches"][index]["loss"]
                for c in range(cp)
            )
            assert_close(
                loss, expected["loss"] / dp / 2, f"dp{dp_rank}/microbatch{index}/loss"
            )
            for key, reference_output in expected["output"].items():
                pieces = []
                for c in range(cp):
                    replicas = [
                        actual[base_rank + c * tp + t]["microbatches"][index]["output"][
                            key
                        ]
                        for t in range(tp)
                    ]
                    if vocab and key in ("draft_logits", "aligned_target_logits"):
                        value = torch.cat(replicas, dim=-1)
                    else:
                        value = replicas[0]
                        for replica in replicas[1:]:
                            assert_close(
                                replica, value, f"{key}/TP replica", exact=True
                            )
                    pieces.append(value)
                combined = torch.cat(pieces, dim=1)
                assert_close(
                    combined, reference_output, f"dp{dp_rank}/microbatch{index}/{key}"
                )
    return {"updates": 2, "logical_gas": 2, "ranks": world, "rtol": 1e-4, "atol": 1e-6}


def run(output, selected, dtypes, phases, *, retry_incomplete=False):
    device_pool = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
    )
    summary_path = output / "summary.json"
    reports = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for name in selected:
        topology = TOPOLOGIES[name]
        dp, tp, cp, pp, replicate, sp, vocab = topology
        world = dp * tp * cp * pp
        if len(device_pool) < world:
            raise ValueError(f"{name} requires {world} physical GPUs")
        devices = device_pool[:world]
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        reference = output / f"reference-dp{dp}"
        if not (reference / "complete.json").is_file():
            if reference.exists():
                raise FileExistsError(
                    f"Incomplete reference needs inspection: {reference}"
                )
            env = dict(
                environment,
                DEEPSPEC_GQA_REFERENCE_WORKERS=str(dp),
                DEEPSPEC_GQA_REFERENCE_LAYERS="5",
                DEEPSPEC_GQA_REFERENCE_OUTPUT=str(reference),
            )
            print(f"Recording independent five-layer DP{dp} reference", flush=True)
            log_root = output / f"reference-dp{dp}-launch"
            log_root.mkdir(exist_ok=True)
            result = launch(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={dp}",
                    "-m",
                    "tests.create_torchtitan_gqa_reference",
                ],
                log_root,
                env,
                devices,
            )
            (reference / "complete.json").write_text(json.dumps(result, indent=2))
        for dtype in dtypes:
            base = output / name / dtype
            base.mkdir(parents=True, exist_ok=True)
            fixtures = [
                torch.load(
                    reference / f"torch.{dtype}_rank{rank}.pt", weights_only=True
                )
                for rank in range(dp)
            ]
            batches = [
                (f"batch-{i}-rank{rank}.pt", fixtures[rank]["features"][i])
                for i in range(4)
                for rank in range(dp)
            ]
            plan = base / "input-plan.json"
            plan.write_text(
                json.dumps(
                    make_input_plan(
                        run_id="dspark-dense-12-19", ordered_batches=batches
                    ),
                    sort_keys=True,
                )
            )
            results = dict(reports.get(f"{name}/{dtype}", {}))
            for phase, start, stop in (
                ("continuous", 0, 2),
                ("first", 0, 1),
                ("resumed", 1, 2),
                ("failure", 1, 2),
            ):
                if phase not in phases:
                    continue
                root = base / phase
                completion = root / "complete.json"
                if completion.is_file():
                    results[phase] = json.loads(completion.read_text())
                    continue
                if root.exists() and retry_incomplete:
                    root.rename(root.with_name(f"{phase}-attempt-{time.time_ns()}"))
                root.mkdir(parents=True, exist_ok=False)
                torch.save(fixtures[0]["initial_weights"], root / "initial-weights.pt")
                entries = []
                for filename, batch in batches[start * 2 * dp : stop * 2 * dp]:
                    torch.save(batch, root / filename)
                    entries.append({"id": filename, "path": filename})
                (root / "features.json").write_text(json.dumps({"batches": entries}))
                env = dict(
                    environment,
                    DEEPSPEC_BASELINE_REFERENCE=str(reference),
                    DEEPSPEC_PARALLEL_ROOT=str(root),
                    DEEPSPEC_PARALLEL_DTYPE=dtype,
                    DEEPSPEC_PARALLEL_TP=str(tp),
                    DEEPSPEC_PARALLEL_CP=str(cp),
                    DEEPSPEC_PARALLEL_PP=str(pp),
                    DEEPSPEC_PARALLEL_REPLICATE=str(replicate),
                    DEEPSPEC_PARALLEL_SP=str(sp),
                    DEEPSPEC_PARALLEL_VOCAB=str(vocab),
                )
                request = {
                    "draft_python": sys.executable,
                    "draft_source": str(repository / "torchtitan"),
                    "workers": world,
                    "devices": devices,
                    "result_path": str(root / "result.json"),
                    "recipe_args": [
                        "--module",
                        "tests.torchtitan_parallel_fixtures",
                        "--config",
                        "parallel_features",
                    ],
                    "phase": {
                        "run_id": "dspark-dense-12-19",
                        "stop_update": stop,
                        "microbatch_start": start * 2,
                        "feature_manifest": str(root / "features.json"),
                        "plan_path": str(plan),
                        "checkpoint_folder": str(root / "checkpoints"),
                        "resume_checkpoint": str(base / "first/checkpoints/step-1")
                        if start
                        else None,
                    },
                }
                request_path = root / "request.json"
                request_path.write_text(json.dumps(request, indent=2))
                if phase == "failure":
                    (root / "checkpoints/step-2/commit.json").mkdir(parents=True)
                print(f"Running {name}/{dtype}/{phase}: {topology}", flush=True)
                result = launch(
                    [
                        sys.executable,
                        "-m",
                        "deepspec.orchestration.draft",
                        str(request_path),
                    ],
                    root,
                    env,
                    devices,
                    expect_failure=phase == "failure",
                )
                if phase == "failure":
                    if (
                        (root / "result.json").exists()
                        or not (root / "checkpoints/step-2/.metadata").is_file()
                        or "IsADirectoryError" not in (root / "run.log").read_text()
                    ):
                        raise AssertionError(
                            "Expected real DCP write followed by failed commit, without handoff"
                        )
                else:
                    returned = json.loads((root / "result.json").read_text())
                    for pid in returned["worker_pids"]:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            continue
                        raise AssertionError(f"Training worker {pid} did not exit")
                    if (
                        returned["completed_updates"] != stop
                        or returned["consumed_range"] != [start * 2, stop * 2]
                        or len(returned["worker_pids"]) != world
                    ):
                        raise AssertionError(
                            "Phase progress or participating ranks differ"
                        )
                    result["phase"] = returned
                    if phase == "continuous":
                        result["numerical"] = check_reference(root, fixtures, topology)
                    elif phase == "resumed":
                        result["checkpoint"] = compare(
                            base / "continuous/checkpoints/step-2",
                            root / "checkpoints/step-2",
                        )
                        if not result["checkpoint"]["bitwise_equal"]:
                            raise AssertionError(result["checkpoint"])
                        for rank in range(world):
                            full = torch.load(
                                base / "continuous" / f"rank-{rank}.pt",
                                weights_only=True,
                            )
                            resumed = torch.load(
                                root / f"rank-{rank}.pt", weights_only=True
                            )
                            for key in ("cpu_rng", "cuda_rng", "cursor"):
                                assert_close(resumed[key], full[key], key, exact=True)
                            assert_close(
                                resumed["updates"],
                                full["updates"][1:],
                                "resumed updates",
                                exact=True,
                            )
                            assert_close(
                                resumed["microbatches"],
                                full["microbatches"][2:],
                                "resumed loss and outputs",
                                exact=True,
                            )
                completion.write_text(json.dumps(result, indent=2))
                results[phase] = result
                print(f"PASS {name}/{dtype}/{phase}", flush=True)
            reports[f"{name}/{dtype}"] = results
            summary_path.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument(
        "--topologies", nargs="+", choices=TOPOLOGIES, default=list(TOPOLOGIES)
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "bfloat16"])
    parser.add_argument(
        "--phases", nargs="+", default=["continuous", "first", "resumed", "failure"]
    )
    args = parser.parse_args()
    run(
        args.output.resolve(),
        args.topologies,
        args.dtypes,
        args.phases,
        retry_incomplete=args.retry_incomplete,
    )
