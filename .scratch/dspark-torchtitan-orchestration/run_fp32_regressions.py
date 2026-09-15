"""Run affected FP32 regressions on independent GPU subsets."""
import os
from pathlib import Path
import subprocess
import sys

repository = Path.cwd()
root = repository / "output/dspark_torchtitan_orchestration_20260914"
original = repository / "output/dspark_torchtitan_baseline_20260914/numerics-final"
cases = [
    ("stable-fp32-single", "0", 1, 1, original),
    ("stable-fp32-tp2", "1,2,3,4", 2, 2, original),
    ("stable-fp32-gqa-tp1", "5,6", 2, 1, root / "gqa-reference"),
]
children = []
for name, devices, dp, tp, reference in cases:
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES=devices,
        DEEPSPEC_PHASE_WORKERS=str(dp),
        DEEPSPEC_PHASE_TP=str(tp),
        DEEPSPEC_PHASE_DTYPES="float32",
        DEEPSPEC_BASELINE_REFERENCE=str(reference),
        DEEPSPEC_SINGLE_GPU_REFERENCE=str(root / "single-gpu"),
        DEEPSPEC_NATIVE_TEST_OUTPUT=str(root / name),
    )
    log = (root / f"{name}-test.log").open("w")
    child = subprocess.Popen(
        [sys.executable, "-m", "unittest", "-v", "tests.test_torchtitan_phase_training"],
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    children.append((name, child, log))
failed = []
for name, child, log in children:
    code = child.wait()
    log.close()
    print(name, code, flush=True)
    if code:
        failed.append(name)
if failed:
    raise SystemExit(f"Failed regressions: {failed}")
