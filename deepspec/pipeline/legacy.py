"""Explicit legacy configuration migration into the versioned run controller."""

import json
import tempfile
from pathlib import Path

from .runtime import PipelineError, atomic_json
from .schema import upgrade_task_config


def run_config(config, *, prepare_only=False, transport_only=False):
    from .cli import preview
    from .execution import run_plan
    from .transport import transport_check

    if config.get("topology_plan_path"):
        return run_plan(config["topology_plan_path"])
    output = Path(config["output_dir"]).resolve()
    if output.exists():
        raise PipelineError(
            "OUTPUT_EXISTS",
            "Use a new output directory; legacy run ledgers are not resumed",
            field_path="output_dir",
        )
    # Validate all supplied options before connecting or starting a local Ray.
    normalized = upgrade_task_config(config).to_dict()
    ray = None
    owned_cluster = False
    try:
        if not config.get("cluster_address") and not config.get("ray_address"):
            import ray

            try:
                ray.init(address="auto", log_to_driver=False)
            except ConnectionError:
                ray.init(
                    num_cpus=24,
                    num_gpus=8,
                    object_store_memory=128 * 1024**2,
                    include_dashboard=False,
                    log_to_driver=False,
                )
                owned_cluster = True
            normalized["ray_address"] = ray.get_runtime_context().gcs_address
        with tempfile.TemporaryDirectory(prefix="deepspec-legacy-") as temporary:
            path = Path(temporary) / "task.json"
            path.write_text(json.dumps(normalized))
            prepared = preview(path)
        if owned_cluster:
            atomic_json(
                output / "cluster-owner.json",
                {
                    "run_id": prepared["run_id"],
                    "plan_hash": prepared["plan_hash"],
                    "owner": "DeepSpec legacy entry",
                    "ray_address": normalized["ray_address"],
                },
            )
        if prepare_only:
            return prepared
        if transport_only:
            return transport_check(prepared["plan_path"])
        return run_plan(prepared["plan_path"])
    finally:
        # Ray only stops a cluster that this driver itself started. Connecting
        # to an external cluster followed by shutdown merely detaches the driver.
        if ray is not None:
            ray.shutdown()
