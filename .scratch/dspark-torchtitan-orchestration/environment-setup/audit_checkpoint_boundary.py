"""Read a completed H800 scale phase's small checkpoint state on CPU."""

import argparse
import json
from pathlib import Path

import torch

from deepspec.orchestration.io import digest
from tests.compare_torchtitan_checkpoints import load_checkpoint


def audit(phase_path, observations_dir=None):
    phase = json.loads(phase_path.read_text())
    draft = phase["draft"]
    commit = draft["commit"]
    checkpoint = Path(commit["checkpoint"])
    start, stop = draft["consumed_range"]
    assert digest(checkpoint / ".metadata") == commit["metadata_sha256"]
    state = load_checkpoint(
        checkpoint, keys={"train_state", "dataloader", "lr_scheduler"}
    )
    assert state["train_state.step"] == commit["completed_updates"] == stop // 2
    assert state["dataloader.next_global_microbatch"] == stop
    assert state["train_state.run_id"] == commit["run_id"]
    assert state["lr_scheduler.0.last_epoch"] == commit["completed_updates"]
    observations = observations_dir or Path(
        commit["resolved_recipe"]["dataloader"]["manifest"]
    ).parent
    accepted = json.loads((observations / "draft-result.json").read_text())
    assert accepted["worker_pids"] == draft["worker_pids"]
    assert accepted["commit"] == commit
    ranks = []
    for rank in range(8):
        recorded = torch.load(
            observations / f"supervision-rank{rank}.pt", weights_only=True
        )
        cursors = [value["next_microbatch"] for value in recorded]
        assert cursors == [2 * (index // 2 + 1) for index in range(start, stop)], (
            rank,
            cursors,
        )
        for kind in ("cpu", "cuda"):
            assert torch.equal(
                state[f"train_state.rank_{rank}.{kind}_rng"],
                recorded[-1][f"{kind}_rng"],
            ), (rank, kind)
        assert f"train_state.rank_{rank}.python_rng" in state
        assert f"train_state.rank_{rank}.numpy_rng" in state
        ranks.append(
            {
                "rank": rank,
                "forward_count": len(recorded),
                "prefetched_read_cursors": cursors,
                "checkpoint_cpu_cuda_rng_equal_last_forward": True,
                "python_numpy_rng_present": True,
            }
        )
    assert not torch.cuda.is_initialized()
    return {
        "phase": str(phase_path),
        "checkpoint": str(checkpoint),
        "completed_updates": commit["completed_updates"],
        "global_microbatch_cursor": stop,
        "metadata_sha256_verified": True,
        "scheduler_last_epoch": state["lr_scheduler.0.last_epoch"],
        "ranks": ranks,
        "cuda_initialized": False,
        "note": (
            "Native Trainer prefetches GAS-two inputs before both forwards. "
            "This checks metadata and boundary RNG; it does not replace the "
            "full model/optimizer replay comparison."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--observations-dir", type=Path)
    args = parser.parse_args()
    result = audit(args.phase.resolve(), args.observations_dir)
    args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
