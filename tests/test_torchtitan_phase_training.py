"""Numerical acceptance driven through DeepSpec and the real Titan process."""

import json
import copy
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

import torch

from tests import test_dspark_training_baseline as baseline


def legacy_reference():
    from deepspec.distributed import ParallelConfig, ParallelContext
    from deepspec.training.loss import configure_loss_reduction_group
    from deepspec.utils.metrics import configure_reduction_group
    from tests.distributed_test_utils import require_torchrun

    runtime = require_torchrun(unittest.TestCase(), world_size=1)
    topology = ParallelContext.build(ParallelConfig(dp_shard=1, reduce_dtype="fp32"))
    configure_loss_reduction_group(topology.loss_mesh.get_group())
    configure_reduction_group(topology.loss_mesh.get_group())
    dtype = getattr(torch, os.environ["DEEPSPEC_PHASE_TEST_DTYPE"])
    fixture = torch.load(
        Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"]) / f"{dtype}_rank0.pt",
        weights_only=True,
    )
    trainer = baseline.FixedFeatureTrainer(runtime, topology, dtype, fixture=fixture)
    torch.set_rng_state(fixture["initial_cpu_rng"])
    torch.cuda.set_rng_state(fixture["initial_cuda_rng"], runtime.device)
    result = trainer.train_and_observe()
    torch.save(result, Path(os.environ["DEEPSPEC_PHASE_TEST_ROOT"]) / "legacy-rank0.pt")


class NativePhaseTrainingTest(unittest.TestCase):
    def test_native_phase_matches_real_qwen_updates(self):
        workers = int(os.environ.get("DEEPSPEC_PHASE_WORKERS", "1"))
        tp = int(os.environ.get("DEEPSPEC_PHASE_TP", "1"))
        self.assertIn(workers, (1, 2, 4, 8))
        reference = os.environ.get("DEEPSPEC_BASELINE_REFERENCE")
        if not reference:
            self.skipTest("requires the immutable DSpark baseline fixture")
        if torch.cuda.device_count() < workers * tp:
            self.skipTest(f"requires {workers * tp} actual CUDA GPUs")
        repository = Path(__file__).resolve().parents[1]
        output = os.environ.get("DEEPSPEC_NATIVE_TEST_OUTPUT")
        with tempfile.TemporaryDirectory(prefix="dspark-native-phase-") as temporary:
            base = Path(output or temporary)
            for dtype in os.environ.get(
                "DEEPSPEC_PHASE_DTYPES", "float32,bfloat16"
            ).split(","):
                with self.subTest(dtype=dtype):
                    root = base / dtype
                    root.mkdir(parents=True, exist_ok=False)
                    fixtures = [
                        torch.load(
                            Path(reference) / f"torch.{dtype}_rank{rank}.pt",
                            weights_only=True,
                        )
                        for rank in range(workers)
                    ]
                    torch.save(
                        fixtures[0]["initial_weights"], root / "initial-weights.pt"
                    )
                    entries = []
                    for index in range(len(fixtures[0]["features"])):
                        for rank, fixture in enumerate(fixtures):
                            name = f"batch-{index}-rank{rank}.pt"
                            torch.save(fixture["features"][index], root / name)
                            entries.append({"id": name, "path": name})
                    (root / "features.json").write_text(
                        json.dumps({"batches": entries})
                    )
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "DEEPSPEC_PHASE_TEST_ROOT": str(root.resolve()),
                            "DEEPSPEC_PHASE_TEST_DTYPE": dtype,
                            "OMP_NUM_THREADS": "1",
                        }
                    )
                    request = {
                        "draft_python": sys.executable,
                        "draft_source": str(repository / "torchtitan"),
                        "workers": workers * tp,
                        "result_path": str((root / "phase-result.json").resolve()),
                        "recipe_args": [
                            "--module",
                            os.environ.get(
                                "DEEPSPEC_PHASE_MODULE",
                                "tests.torchtitan_phase_fixtures",
                            ),
                            "--config",
                            os.environ.get("DEEPSPEC_PHASE_CONFIG", "fixed_features"),
                        ],
                    }
                    (root / "request.json").write_text(json.dumps(request))
                    commands = [
                        (
                            "legacy",
                            [
                                sys.executable,
                                "-m",
                                "torch.distributed.run",
                                "--standalone",
                                "--nproc-per-node=1",
                                "-m",
                                "tests.test_torchtitan_phase_training",
                                "--legacy",
                            ],
                        ),
                        (
                            "native",
                            [
                                sys.executable,
                                "-m",
                                "deepspec.orchestration.draft",
                                str(root / "request.json"),
                            ],
                        ),
                    ]
                    for name, command in commands:
                        if name == "legacy" and workers > 1:
                            continue
                        archived_single = os.environ.get(
                            "DEEPSPEC_SINGLE_GPU_REFERENCE"
                        )
                        if name == "legacy" and archived_single:
                            shutil.copyfile(
                                Path(archived_single) / dtype / "legacy-rank0.pt",
                                root / "legacy-rank0.pt",
                            )
                            continue
                        with (root / f"{name}.log").open("w") as log:
                            result = subprocess.run(
                                command,
                                env=environment,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        self.assertEqual(
                            result.returncode,
                            0,
                            (root / f"{name}.log").read_text()[-12000:],
                        )
                    assert_observed_phase(root, fixtures, workers, tp=tp)
                    self.assertEqual(
                        json.loads((root / "phase-result.json").read_text()),
                        {"completed_updates": 2, "consumed_microbatches": 4},
                    )


def assert_observed_phase(root, fixtures, workers, tp=1):
    for rank in range(workers * tp):
        actual = torch.load(root / f"native-rank{rank}.pt", weights_only=True)
        expected = (
            copy.deepcopy(fixtures[rank // tp]["result"])
            if workers > 1
            else torch.load(root / "legacy-rank0.pt", weights_only=True)
        )
        if workers > 1:
            # Native FSDP sums gradients; the archived loop averages them.
            # Compare un-compensated local objective contributions in both.
            for batch in expected["microbatches"]:
                batch["loss"] /= workers
        expected["metrics"] = {"grad_norm": expected["metrics"]["grad_norm"]}
        baseline.DSparkTrainingBaselineTest().assert_state_close(actual, expected)


if __name__ == "__main__":
    if "--legacy" in sys.argv:
        legacy_reference()
    else:
        unittest.main()
