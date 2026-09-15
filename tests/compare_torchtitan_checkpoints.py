"""Compare complete native DCP contents on CPU after real phase acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def load_checkpoint(path, keys=None):
    state = {}
    # This is the pinned implementation used by dcp_to_torch_save. The public
    # load entry copies an initially empty mapping and cannot return its keys.
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    if not state:
        raise ValueError("No checkpoint state was loaded")
    return flatten_state_dict(state)[0]


def equal(actual, expected):
    if torch.is_tensor(expected):
        return (
            torch.is_tensor(actual)
            and actual.dtype == expected.dtype
            and torch.equal(actual, expected)
        )
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, type(expected)) and len(actual) == len(expected) and all(
            equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equal(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def compare(first, second):
    a, b = load_checkpoint(first), load_checkpoint(second)
    if a.keys() != b.keys():
        raise ValueError("Checkpoint state keys differ")
    # A continuous and a partitioned run have different local manifest/cursor
    # representations; the global microbatch position must still be identical.
    partition_fields = {"dataloader.feature_identity", "dataloader.cursor"}
    differences = []
    tensors = 0
    for key, expected in a.items():
        if key in partition_fields:
            continue
        tensors += torch.is_tensor(expected)
        if not equal(b[key], expected):
            differences.append(key)
    if not tensors or "dataloader.next_global_microbatch" not in a:
        raise ValueError("The checkpoints do not contain complete native training state")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checkpoint comparison unexpectedly initialized CUDA")
    return {
        "first": str(first),
        "second": str(second),
        "compared_fields": len(a) - len(partition_fields),
        "compared_tensors": tensors,
        "partition_fields": {
            key: [a[key], b[key]] for key in sorted(partition_fields)
        },
        "differences": differences,
        "bitwise_equal": not differences,
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report["bitwise_equal"]:
        raise SystemExit(1)
