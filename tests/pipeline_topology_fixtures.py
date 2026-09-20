"""Synthetic CPU contract inputs. Never use these facts for real acceptance."""

import copy
import json
import math
from pathlib import Path

GIB = 1024**3
CONTRACTS = (
    Path(__file__).resolve().parents[1] / "specs/001-unify-ray-topology/contracts"
)
LAYOUTS = ("M0", "M1-11", "M1-12", "M1-21", "M1-22", "M2-DP1", "M2-DP2", "M3")


def task_config(layout="M3", *, output_dir="/fixture/run", steps=3):
    config = json.loads((CONTRACTS / "m3.example.json").read_text())
    config.update(
        ray_address="fixture://cluster",
        model_path="/fixture/model",
        output_dir=str(output_dir),
    )
    config["data"]["source_path"] = "/fixture/input.jsonl"
    config["layout"] = layout.split("-")[0]
    inference_dp = int(layout[-1]) if layout.startswith("M2-") else 1
    training_dp = 2 if config["layout"] in ("M2", "M3") else 1
    if layout.startswith("M1-"):
        inference_dp, training_dp = map(int, layout[-2:])
    names = {
        "M0": (["a"], ["a"]),
        "M1": (["a"], ["b"]),
        "M2": (["a", "b"], ["c"]),
        "M3": (["a"], ["b", "c"]),
    }
    inference_nodes, training_nodes = names[config["layout"]]
    config["nodes"] = [
        {"alias": n, "selector": {"node_id": f"node-{n}"}, "cpu_limit": 32}
        for n in sorted(set(inference_nodes + training_nodes))
    ]
    config["inference"].update(
        tp=8 if config["layout"] == "M2" else 4,
        dp=inference_dp,
        nodes=[{"node": n, "gpus": 4 * inference_dp} for n in inference_nodes],
    )
    config["training"].update(
        dp=training_dp,
        steps=steps,
        nodes=[
            {"node": n, "gpus": 4 * training_dp // len(training_nodes)}
            for n in training_nodes
        ],
    )
    config["store"]["node"] = training_nodes[0]
    return config


def node_facts(config, *, now=100.0, gpu_count=8):
    """Explicit invented facts with non-contiguous physical GPU ordinals."""
    return [
        {
            "node_id": n["selector"]["node_id"],
            "ip": f"10.123.0.{i + 1}",
            "hostname": f"fixture-{n['alias']}",
            "alive": True,
            "cpu_available": 64,
            "gpu_available": gpu_count,
            "gpus": [
                {"index": str(g * 2 + 1), "uuid": f"GPU-{n['alias']}-{g}"}
                for g in range(gpu_count)
            ],
            "memory": {
                "physical_bytes": 1024 * GIB,
                "limit_bytes": 1024 * GIB,
                "headroom_bytes": 768 * GIB,
            },
            "identities": {
                "source": "fixture-source",
                "dependencies": "fixture-deps",
                "model": "fixture-model",
                "input": "fixture-input",
            },
            "shared_paths": {"readable": True, "writable": True},
            "capabilities": {
                "borrowed_pg": True,
                "cpu_core": True,
                "allocation_gate": True,
            },
            "boot_id": f"boot-{i}",
            "agent_epoch": f"epoch-{i}",
            "sample_seq": 1,
            "request_sent_at": now,
            "observed_at": now,
            "evidence_level": "synthetic",
        }
        for i, n in enumerate(config["nodes"])
    ]


def input_plan(config):
    """Shape-derived byte counts; no torch tensors or model are constructed."""
    count = config["training"]["steps"] * config["training"]["global_batch_size"]
    batches = []
    for position in range(count):
        length = config["data"]["context_length"] - position % 3
        fields = {
            "input_ids": {"shape": [1, length], "dtype": "int64", "element_size": 8},
            "hidden_states": {
                "shape": [length, 32],
                "dtype": "bfloat16",
                "element_size": 2,
            },
        }
        nbytes = sum(math.prod(f["shape"]) * f["element_size"] for f in fields.values())
        batches.append(
            {
                "position": position,
                "id": f"sample-{position}",
                "length": length,
                "input_identity": f"fixture-input-{position}",
                "fields": fields,
                "nbytes": nbytes,
            }
        )
    return {
        "teacher": "fixture-model",
        "batches": batches,
        "data_parallel_size": config["training"]["dp"],
        "global_batch_size": 4,
        "gradient_accumulation_steps": 4 // config["training"]["dp"],
    }


def negative_fixtures(config):
    """Independent mutations for planner, freshness and allocation tests."""
    duplicate = copy.deepcopy(config)
    duplicate["nodes"][1]["selector"] = copy.deepcopy(duplicate["nodes"][0]["selector"])
    unknown = copy.deepcopy(config)
    unknown["inference"]["unexpected"] = True
    insufficient = node_facts(config)
    insufficient[0]["gpu_available"] = 1
    stale = node_facts(config, now=0)
    return {
        "duplicate_node": duplicate,
        "unknown_field": unknown,
        "insufficient": insufficient,
        "stale": stale,
        "wrong_rank": {"global_rank": 4, "dp_rank": 0, "tp_rank": 0},
    }
