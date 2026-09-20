"""Process and native-rank handshake at TorchTitan's initialization boundary."""

import hashlib
import json
import os
from pathlib import Path

from .planning import TopologyPlan
from .runtime import Deadline, EventWriter, message_envelope, validate_message
from .schema import content_hash


def native_rank_groups(dims):
    import torch.distributed as dist

    tp = dims.get_mesh("tp")
    # The pinned spmd_types backend preserves the DP storage axis instead of
    # constructing DTensor's flattened fsdp axis. CP is fixed to one here.
    dp = dims.get_mesh("dp_shard" if dims.spmd_backend == "spmd_types" else "fsdp")
    return {
        "tp_rank": tp.get_local_rank(),
        "dp_rank": dp.get_local_rank(),
        "tp_members": dist.get_process_group_ranks(tp.get_group()),
        "dp_members": dist.get_process_group_ranks(dp.get_group()),
    }


class TrainingHandshake:
    def __init__(self, pipeline, plan, gate, node_id, *, environment=None):
        from deepspec.orchestration.process import capture_process

        self.pipeline, self.plan, self.gate = pipeline, plan, gate
        env = os.environ if environment is None else environment
        self.actual = {
            "global_rank": int(env["RANK"]),
            "local_rank": int(env["LOCAL_RANK"]),
            "node_rank": int(env["GROUP_RANK"]),
            "world": int(env["WORLD_SIZE"]),
            "local_world_size": int(env["LOCAL_WORLD_SIZE"]),
            "node_id": node_id,
        }
        index = self.actual["global_rank"]
        if not 0 <= index < len(plan["training_ranks"]):
            raise ValueError("Torchrun global rank is outside the immutable plan")
        self.rank = plan["training_ranks"][index]
        if any(
            self.actual[k] != self.rank[k] for k in self.actual if k != "world"
        ) or self.actual["world"] != len(plan["training_ranks"]):
            raise ValueError(
                "Torchrun process topology differs from the immutable plan"
            )
        self.process = capture_process(os.getpid(), plan["run_id"])
        self.events = EventWriter(
            Path(pipeline["output_dir"])
            / f"events/training-{node_id}-rank{index}.jsonl",
            run_id=plan["run_id"],
            plan_hash=plan["plan_hash"],
            sender_identity={
                "component": "training_rank",
                **self.actual,
                "pid": self.process["pid"],
                "start_ticks": self.process["start_ticks"],
            },
        )

    @classmethod
    def begin(cls, pipeline_path):
        """Run before native Trainer construction, CUDA initialization or NCCL."""
        import ray
        from torchtitan.models.dspark_draft.planning import model_identity

        pipeline = json.loads(Path(pipeline_path).read_text())
        if not pipeline.get("topology_plan_path"):
            return None
        plan = TopologyPlan.from_dict(
            json.loads(Path(pipeline["topology_plan_path"]).read_text())
        ).to_dict()
        if (pipeline["run_id"], pipeline.get("plan_hash")) != (
            plan["run_id"],
            plan["plan_hash"],
        ):
            raise ValueError("Training configuration belongs to another plan")
        if not ray.is_initialized():
            ray.init(
                address=pipeline["ray_address"],
                namespace=pipeline["namespace"],
                log_to_driver=False,
            )
        gate = ray.get_actor(
            pipeline["native_gate_name"], namespace=pipeline["namespace"]
        )
        instance = cls(pipeline, plan, gate, ray.get_runtime_context().get_node_id())
        try:
            instance.model_identity = content_hash(
                model_identity(pipeline["model_path"])
            )
            instance.input_plan_hash = hashlib.sha256(
                Path(pipeline["plan_path"]).read_bytes()
            ).hexdigest()
            node = next(
                n
                for n in plan["nodes"].values()
                if n["node_id"] == instance.actual["node_id"]
            )
            if (
                instance.model_identity != node["identities"]["model"]
                or instance.input_plan_hash != plan["input_plan_hash"]
            ):
                raise ValueError(
                    "Native model or input plan changed before rank initialization"
                )
            instance._call(
                "register_training_process",
                instance.message(**instance.actual, process=instance.process),
                timeout=plan["timeouts_seconds"]["initialization"],
            )
            return instance
        except BaseException as error:
            instance.failed(error)
            instance.close()
            raise

    def failed(self, error):
        try:
            self._call(
                "native_failed",
                self.message(reason=repr(error)),
                timeout=min(5, self.plan["timeouts_seconds"]["cleanup"]),
            )
        except Exception as notification_error:
            error.add_note(
                f"Native gate failure notification failed: {notification_error}"
            )

    def message(self, **payload):
        return message_envelope(
            self.plan["run_id"],
            self.plan["plan_hash"],
            self.events.identity,
            participant=f"rank/{self.rank['global_rank']}",
            **payload,
        )

    def _call(self, method, message, *, timeout):
        import ray

        deadline = Deadline.after(timeout)
        kwargs = (
            {"timeout": deadline.remaining()}
            if method == "register_training_process"
            else {}
        )
        reply = ray.get(
            getattr(self.gate, method).remote(message, **kwargs),
            timeout=deadline.remaining(),
        )
        validate_message(
            reply, run_id=self.plan["run_id"], plan_hash=self.plan["plan_hash"]
        )
        return reply

    def initialized(self, trainer):
        import torch
        import torch.distributed as dist

        from .cluster import gpu_inventory

        dims, expected = trainer.parallel_dims, self.plan["config"]["training"]
        if (dims.tp, dims.dp_shard, dims.dp_replicate, dims.cp, dims.pp) != (
            4,
            expected["dp"],
            1,
            1,
            1,
        ):
            raise ValueError(
                "Native training must use the planned TP/FSDP shard topology"
            )
        device = torch.cuda.current_device()
        if device != self.rank["local_rank"]:
            raise ValueError("Native rank CUDA device differs from its local slot")
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        physical = visible[device]
        inventory = gpu_inventory(
            timeout=self.plan["timeouts_seconds"]["budget_snapshot"]
        )
        devices = {str(g["index"]): g["uuid"] for g in inventory}
        devices.update({g["uuid"]: g["uuid"] for g in inventory})
        groups = native_rank_groups(dims)
        from .verification import save_checkpoint_expectation

        expectation = save_checkpoint_expectation(
            self.plan, self.rank["global_rank"], trainer
        )
        self.events.emit("checkpoint_expectation", expectation, basis="observed")
        report = self.message(
            **{
                **self.actual,
                "global_rank": dist.get_rank(),
                "world": dist.get_world_size(),
            },
            **groups,
            gas=trainer.gradient_accumulation_steps,
            input_plan_hash=trainer.dataloader.plan_identity,
            model_identity=self.model_identity,
            gpu_uuid=devices[physical],
            pid=self.process["pid"],
            start_ticks=self.process["start_ticks"],
            checkpoint_expectation=expectation,
        )
        self._call(
            "report_initialized",
            report,
            timeout=self.plan["timeouts_seconds"]["initialization"],
        )
        self.events.emit(
            "node_environment",
            {
                "identities": {
                    "model": self.model_identity,
                    "input_plan": trainer.dataloader.plan_identity,
                    "training": trainer.training_identity,
                    "process": self.process,
                },
                "placement": report,
            },
            basis="observed",
            causes=(report["event_id"],),
        )

    def update(self, trainer, loss, *, duration_seconds=None):
        if trainer.completed_updates != trainer.step:
            raise ValueError("Native optimizer completion differs from rank progress")
        self.events.emit(
            "rank_update_completed",
            {
                "optimizer_step": trainer.completed_updates,
                "native_cursor": trainer.dataloader.next_global_microbatch,
                "sample_cursor": trainer.dataloader.next_global_microbatch
                * self.plan["config"]["training"]["dp"],
                "loss": loss,
                "loss_basis": "sum_of_native_microbatch_losses_on_this_rank",
                "duration_seconds": duration_seconds,
            },
            basis="observed",
        )

    def checkpoint(self, commit):
        from torchtitan.models.dspark_draft.checkpoint import read_commit

        committed = read_commit(commit["checkpoint"])
        if committed != commit or committed["run_id"] != self.plan["run_id"]:
            raise ValueError("Native checkpoint commit identity differs")
        self.events.emit(
            "checkpoint_committed",
            {
                "path": committed["checkpoint"],
                "commit_identity": content_hash(committed),
                "native_cursor": committed["next_global_microbatch"],
                "sample_cursor": committed["next_global_microbatch"]
                * self.plan["config"]["training"]["dp"],
            },
            basis="verified",
        )

    def close(self):
        self.events.close()
