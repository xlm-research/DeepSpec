"""Ray allocates resources; native vLLM and torchrun own their model workers."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ray

from .runtime import (
    Deadline,
    component_deadline,
    get_with_deadline,
    message_envelope,
    notify_buffer_failure,
)
from .topology import consumer_dp, consumer_nodes, producer_dp, sample_producer


async def run_async_production(config, buffer, generate, *, deadline=None):
    deadline = component_deadline(config) if deadline is None else deadline
    async with asyncio.timeout(deadline.remaining()):
        return await _run_async_production(config, buffer, generate)


async def _run_async_production(config, buffer, generate):
    """Reserve in plan order and preserve the configured native per-replica batch."""
    batch_size = config.get("producer_batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Producer batch size must be a positive integer")
    slots = [asyncio.Semaphore(batch_size) for _ in range(producer_dp(config))]

    async def request(sample, rank):
        try:
            await generate(sample, rank)
        finally:
            slots[rank].release()

    async def watch_failure():
        await buffer.wait_for_failure.remote()

    try:
        async with asyncio.TaskGroup() as tasks:
            watcher = tasks.create_task(watch_failure())
            requests = []
            position = 0
            while position < len(config["samples"]):
                positions = await buffer.reserve_batch.remote(
                    position, batch_size * len(slots)
                )
                if not positions or positions != list(
                    range(position, position + len(positions))
                ):
                    raise ValueError(
                        "Batch reservation changed the frozen sample order"
                    )
                for current in positions:
                    sample = config["samples"][current]
                    rank = sample_producer(config, current)
                    await slots[rank].acquire()
                    requests.append(tasks.create_task(request(sample, rank)))
                position = positions[-1] + 1
            await asyncio.gather(*requests)
            # A completed generation may still have a queued Store write.
            await buffer.finish_production.remote()
            watcher.cancel()
    except BaseException as error:
        try:
            await asyncio.wait_for(
                buffer.fail.remote(f"Async producer failed: {error!r}"),
                config.get("timeouts_seconds", {}).get("cleanup", 35),
            )
        except Exception as secondary:  # noqa: BLE001 -- retain the production failure or cancellation
            error.add_note(f"Failure notification failed: {secondary}")
        raise
    return {"samples": len(config["samples"]), "producer_dp": producer_dp(config)}


def consumer_command(config, node_rank):
    nodes = consumer_nodes(config)
    if not 0 <= node_rank < nodes:
        raise ValueError("Consumer node rank is outside the configured topology")
    rendezvous = ["--standalone"]
    if nodes > 1:
        endpoint = config["consumer_rendezvous"]
        rendezvous = [
            f"--nnodes={nodes}",
            f"--node-rank={node_rank}",
            f"--master-addr={endpoint['host']}",
            f"--master-port={endpoint['port']}",
            f"--rdzv-id={config['run_id']}",
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *rendezvous,
        f"--nproc-per-node={config['consumer_world_size'] // nodes}",
        "--max-restarts=0",
        "-m",
        "torchtitan.models.dspark_draft.train",
        "--module",
        "deepspec.pipeline.recipe",
        "--config",
        "qwen38_streaming",
    ]


class ControlledActor:
    """Keep control RPCs independent of native calls.

    A timed out worker thread remains owned by its actor. Its caller must reclaim
    that actor through the registered NodeAgent; cancelling a future is not proof
    of native process termination.
    """

    def _initialize_control(self):
        self._control_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pipeline-work"
        )
        self._work = self._close_work = None
        self._stop_reason = None
        self._backend_ready = threading.Event()
        self._owned_process = None
        self._cleanup_errors = []
        self._first_error = None
        self._notified_stop = False

    def _reply(self, **payload):
        return message_envelope(
            self.config["run_id"],
            self.config.get("plan_hash", "legacy"),
            {"component": type(self).__name__, "pid": os.getpid()},
            **payload,
        )

    def _check_running(self):
        if self._stop_reason is not None:
            raise RuntimeError(f"Actor stopped: {self._stop_reason}")
        self.run_deadline.remaining()

    def start(self):
        with self._control_lock:
            self._check_running()
            if self._work is None:
                self._work = self._executor.submit(self.run)
        return self._reply(started=True)

    def status(self):
        with self._control_lock:
            result = None
            state = "created" if self._work is None else "running"
            if self._work is not None and self._work.done():
                try:
                    result = self._work.result()
                    state = "finished"
                except BaseException as error:  # noqa: BLE001 -- native worker failures must reach the control plane
                    self._first_error = self._first_error or repr(error)
                    state = "failed"
            if self._stop_reason is not None:
                state = "stopped"
            owned = self._owned_process
            return self._reply(
                state=state,
                ready=self._backend_ready.is_set() and state == "running",
                result=result,
                reason=self._stop_reason,
                error=self._first_error,
                process=None if owned is None else owned.identity,
                supervisor_report_path=None
                if owned is None
                else str(owned.report_path),
                cleanup_errors=list(self._cleanup_errors),
            )

    def ready(self):
        # Backend readiness is local; only AllocationGate may declare group readiness.
        return self.status()

    def identity(self):
        from .runtime import actor_identity

        return actor_identity(self.config)

    def _close_native(self, deadline):
        pass

    def stop(self, reason, *, timeout=None):
        deadline = Deadline.after(
            self.config.get("timeouts_seconds", {}).get("cleanup", 35)
            if timeout is None
            else timeout
        )
        with self._control_lock:
            self._stop_reason = self._stop_reason or reason
            work, owned = self._work, self._owned_process
            notify = work is not None and not work.done() and not self._notified_stop
            self._notified_stop |= notify
        complete = False
        try:
            if notify and getattr(self, "buffer", None) is not None:
                try:
                    get_with_deadline(
                        self.buffer.fail.remote(self._stop_reason),
                        self.config,
                        deadline=deadline,
                    )
                except Exception as error:  # noqa: BLE001 -- continue stopping native processes after notification failure
                    self._cleanup_errors.append(repr(error))
            if owned is not None:
                report = owned.stop(timeout=deadline.remaining())
                if not report["cleanup_complete"]:
                    raise RuntimeError(f"Native process cleanup unconfirmed: {report}")
            if work is not None:
                try:
                    work.result(timeout=deadline.remaining())
                except TimeoutError as error:
                    if not work.done():
                        raise
                    self._first_error = self._first_error or repr(error)
                except BaseException as error:  # noqa: BLE001 -- preserve native failure while completing cleanup
                    self._first_error = self._first_error or repr(error)
            with self._control_lock:
                if self._close_work is None:
                    self._close_work = self._executor.submit(
                        self._close_native, deadline
                    )
                    self._executor.shutdown(wait=False)
                close_work = self._close_work
            close_work.result(timeout=deadline.remaining())
            complete = True
        except Exception as error:  # noqa: BLE001 -- failed cleanup stays unknown for the owner
            self._cleanup_errors.append(repr(error))
        return self._reply(
            cleanup_complete=complete,
            reason=self._stop_reason,
            error=self._first_error,
            errors=list(self._cleanup_errors),
        )


class Producer(ControlledActor):
    def _get(self, ref, *, timeout=None):
        self._check_running()
        return get_with_deadline(
            ref, self.config, deadline=self.run_deadline, timeout=timeout
        )

    def __init__(
        self,
        config_path,
        *,
        native_plan=None,
        placement_groups=None,
        gate=None,
        runtime_env=None,
    ):
        self.config = json.loads(Path(config_path).read_text())
        self.run_deadline = component_deadline(self.config)
        self.buffer = ray.get_actor(
            self.config["buffer_name"], namespace=self.config["namespace"]
        )
        self.llm = None
        self.config_path = config_path
        self.native_adapter = None
        if native_plan is not None:
            from .planning import TopologyPlan
            from .vllm_adapter import NativeInferenceAdapter

            plan = TopologyPlan.from_dict(native_plan).to_dict()
            if (self.config["run_id"], self.config.get("plan_hash")) != (
                plan["run_id"],
                plan["plan_hash"],
            ):
                raise ValueError("Frontend config differs from its frozen native plan")
            self.native_adapter = NativeInferenceAdapter(
                plan, placement_groups, gate, runtime_env=runtime_env
            )
        self.native_initialization_timeout = None
        self._initialize_control()

    def start(self, *, initialization_timeout=None):
        if (
            self.native_initialization_timeout is None
            and initialization_timeout is not None
        ):
            Deadline.after(initialization_timeout)
            self.native_initialization_timeout = initialization_timeout
        return super().start()

    def run(self):
        import torch
        from vllm.config import KVTransferConfig

        from deepspec.trainer.qwen3_8_vllm import teacher_identity
        from vllm import LLM, SamplingParams

        config = self.config
        try:
            if (
                teacher_identity(
                    config["model_path"], config["teacher"]["target_layer_ids"]
                )
                != config["teacher"]
            ):
                raise ValueError(
                    "The target checkpoint changed after input preparation"
                )
            model_args = {
                "model": config["model_path"],
                "dtype": "bfloat16",
                "seed": 0,
                "tensor_parallel_size": (
                    self.native_adapter.plan["config"]["inference"]["tp"]
                    if self.native_adapter is not None
                    else 4
                ),
                "distributed_executor_backend": "ray",
                "max_model_len": config["context_length"] + 1,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": config.get("producer_batch_size", 1),
                "gpu_memory_utilization": 0.7,
                "enforce_eager": True,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "language_model_only": True,
                "speculative_config": {
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": config["teacher"][
                                "aux_layer_ids"
                            ],
                            "extract_final_hidden_state": True,
                        }
                    },
                },
                "kv_transfer_config": KVTransferConfig(
                    kv_connector="MooncakeHiddenStatesConnector",
                    kv_connector_module_path="deepspec.pipeline.connector",
                    kv_role="kv_producer",
                    kv_connector_extra_config={
                        "pipeline_config": self.config_path,
                        "shared_storage_path": str(
                            Path(config["output_dir"]) / "locators"
                        ),
                        "allow_custom_save_path": True,
                        "use_synchronization_lock": False,
                        "num_writer_threads": 1,
                        "separate_hidden_state_pages": True,
                    },
                ),
            }
            if self.native_adapter is not None or producer_dp(config) > 1:
                return asyncio.run(run_native_async(self, model_args))
            self.llm = LLM(**model_args)
            self._check_running()
            self._backend_ready.set()
            self._get(self.buffer.event.remote("producer_initialized", pid=os.getpid()))
            self._get(self.buffer.wait_for_consumer.remote())
            position = 0
            while position < len(config["samples"]):
                positions = self._get(
                    self.buffer.reserve_batch.remote(
                        position, config.get("producer_batch_size", 1)
                    )
                )
                prompts, parameters = [], []
                for current in positions:
                    sample = config["samples"][current]
                    batch = torch.load(
                        sample["input_path"], map_location="cpu", weights_only=True
                    )
                    path = str(Path(config["output_dir"]) / "locators" / str(current))
                    prompts.append({"prompt_token_ids": batch["input_ids"][0].tolist()})
                    parameters.append(
                        SamplingParams(
                            temperature=0,
                            max_tokens=1,
                            extra_args={
                                "kv_transfer_params": {"hidden_states_path": path},
                            },
                        )
                    )
                    self._get(
                        self.buffer.event.remote(
                            "inference_start", position=current, producer_rank=0
                        )
                    )
                self._get(
                    self.buffer.event.remote(
                        "inference_batch", positions=positions, batch_size=len(prompts)
                    )
                )
                self.llm.generate(prompts, parameters, use_tqdm=False)
                for current in positions:
                    self._get(
                        self.buffer.event.remote(
                            "inference_end", position=current, producer_rank=0
                        )
                    )
                position = positions[-1] + 1
            # vLLM request completion only guarantees its D2H lifetime boundary.
            # The connector publishes READY after Store completes all writes.
            self._get(self.buffer.finish_production.remote())
            return {"samples": len(config["samples"])}
        except BaseException as error:
            notify_buffer_failure(
                self.buffer, config, error, message=f"Producer failed: {error!r}"
            )
            raise

    def _close_native(self, deadline):
        if self.native_adapter is not None:
            self.native_adapter.stop(timeout=deadline.remaining())
            self.llm = None
            return
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None

    def close(self):
        report = self.stop("close")
        if not report["cleanup_complete"]:
            raise RuntimeError(f"Producer cleanup unconfirmed: {report}")


async def run_native_async(producer, model_args):
    async with asyncio.timeout(producer.run_deadline.remaining()):
        return await _run_native_async(producer, model_args)


async def _run_native_async(producer, model_args):
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm import SamplingParams

    config = producer.config
    if producer.native_adapter is not None:
        timeout = min(
            producer.run_deadline.remaining(),
            producer.native_initialization_timeout
            or config["timeouts_seconds"]["initialization"],
        )
        producer.llm = producer.native_adapter.start(
            {**model_args, "disable_log_stats": True}, timeout=timeout
        )
    else:
        producer.llm = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                **model_args,
                data_parallel_size=producer_dp(config),
                data_parallel_size_local=producer_dp(config),
                data_parallel_backend="ray",
                data_parallel_address=config["producer_node_ips"][0],
                disable_log_stats=True,
            )
        )
    try:
        producer._check_running()
        producer._backend_ready.set()
        await producer.buffer.event.remote(
            "producer_initialized", pid=os.getpid(), producer_dp=producer_dp(config)
        )
        if producer.native_adapter is not None:
            await asyncio.to_thread(
                producer.native_adapter.ready, timeout=producer.run_deadline.remaining()
            )
        else:
            await producer.buffer.wait_for_consumer.remote()

        async def generate(sample, rank):
            position = sample["position"]
            batch = torch.load(
                sample["input_path"], map_location="cpu", weights_only=True
            )
            path = str(Path(config["output_dir"]) / "locators" / str(position))
            await producer.buffer.event.remote(
                "inference_start", position=position, producer_rank=rank
            )
            result = None
            # Native generate aborts its request when this task is cancelled.
            async for result in producer.llm.generate(
                {"prompt_token_ids": batch["input_ids"][0].tolist()},
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    extra_args={"kv_transfer_params": {"hidden_states_path": path}},
                ),
                request_id=f"{config['run_id']}-{position}",
                data_parallel_rank=rank,
            ):
                pass
            if result is None or not result.finished:
                raise RuntimeError(f"Generation {position} did not finish")
            await producer.buffer.event.remote(
                "inference_end", position=position, producer_rank=rank
            )

        return await run_async_production(
            config, producer.buffer, generate, deadline=producer.run_deadline
        )
    finally:
        # Shutdown while AsyncLLM's owning event loop is still alive.
        if producer.native_adapter is not None:
            producer.native_adapter.stop(timeout=config["timeouts_seconds"]["cleanup"])
        else:
            producer.llm.shutdown(timeout=10)
        producer.llm = None


class OwnedMasterActor(ControlledActor):
    """CPU service host whose constructor never launches a native service."""

    owns_supervised_processes = True

    def __init__(self, config, host, endpoint=None):
        from .runtime import MooncakeMaster
        from .store import free_port

        self.config = dict(config)
        self.run_deadline = component_deadline(config)
        self._initialize_control()
        self.endpoint = endpoint or f"{host}:{free_port()}"
        self.master = MooncakeMaster(
            self.endpoint,
            Path(config["output_dir"]) / "mooncake-master.log",
            metrics_port=free_port(),
            env=dict(os.environ, DEEPSPEC_PIPELINE_RUN_ID=config["run_id"]),
            run_timeout=self.run_deadline.remaining(),
            cleanup_timeout=config["timeouts_seconds"]["cleanup"],
        )

    def configuration(self):
        return self._reply(endpoint=self.endpoint)

    def run(self):
        self._check_running()
        self.master.start(
            timeout=min(
                self.run_deadline.remaining(),
                self.config["timeouts_seconds"]["initialization"],
            )
        )
        with self._control_lock:
            self._owned_process = self.master.handle
        self._check_running()
        self._backend_ready.set()
        return {
            "endpoint": self.endpoint,
            "supervisor_report_path": str(self.master.handle.report_path),
        }

    def ready(self):
        report = self.status()
        report["ready"] = report["state"] == "finished" and self.master.poll() is None
        return report

    def _close_native(self, deadline):
        return self.master.stop(timeout=deadline.remaining())


class Consumer(ControlledActor):
    owns_supervised_processes = True

    def _get(self, ref, *, timeout=None):
        self._check_running()
        return get_with_deadline(
            ref, self.config, deadline=self.run_deadline, timeout=timeout
        )

    def __init__(self, config_path, node_rank=0, *, native_plan=None, gate=None):
        self.path = config_path
        self.config = json.loads(Path(config_path).read_text())
        self.run_deadline = component_deadline(self.config)
        self.node_rank = node_rank
        self.native_plan, self.gate = native_plan, gate
        self._rendezvous_reservation = None
        self.initialization_deadline = None
        if native_plan is not None:
            from .planning import TopologyPlan

            self.native_plan = TopologyPlan.from_dict(native_plan).to_dict()
            if (self.config["run_id"], self.config.get("plan_hash")) != (
                native_plan["run_id"],
                native_plan["plan_hash"],
            ):
                raise ValueError(
                    "Training launcher configuration differs from its native plan"
                )
        self.buffer = None
        self._initialize_control()

    def reserve_rendezvous(self):
        import socket

        if (
            self.node_rank != 0
            or self.native_plan is None
            or consumer_nodes(self.config) != 2
        ):
            raise ValueError("Only the first planned launcher may reserve rendezvous")
        node = self.native_plan["training_ranks"][0]["node_id"]
        host = next(
            n["ip"] for n in self.native_plan["nodes"].values() if n["node_id"] == node
        )
        if self._rendezvous_reservation is None:
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                reservation.bind((host, 0))
            except BaseException:
                reservation.close()
                raise
            self._rendezvous_reservation = reservation
        return {
            **self.identity(),
            "host": host,
            "port": self._rendezvous_reservation.getsockname()[1],
        }

    def configure_rendezvous(self, endpoint):
        if self._work is not None or self.native_plan is None:
            raise ValueError("Rendezvous must be frozen before native startup")
        first = self.native_plan["training_ranks"][0]["node_id"]
        host = next(
            n["ip"] for n in self.native_plan["nodes"].values() if n["node_id"] == first
        )
        if (
            set(endpoint) != {"host", "port"}
            or endpoint["host"] != host
            or type(endpoint["port"]) is not int
            or not 0 < endpoint["port"] < 65536
        ):
            raise ValueError("Rendezvous endpoint differs from the first training node")
        old = self.config.get("consumer_rendezvous")
        if old is not None and old != endpoint:
            raise ValueError("Cannot change a frozen rendezvous endpoint")
        if self.node_rank == 0 and (
            self._rendezvous_reservation is None
            or self._rendezvous_reservation.getsockname()[1] != endpoint["port"]
        ):
            raise ValueError("Rendezvous port was not reserved by this launcher")
        self.config["consumer_rendezvous"] = dict(endpoint)
        return self._reply(configured=True)

    def _close_native(self, deadline):
        if self._rendezvous_reservation is not None:
            self._rendezvous_reservation.close()
            self._rendezvous_reservation = None

    def start(self, *, initialization_timeout=None):
        if self.initialization_deadline is None:
            self.initialization_deadline = Deadline.after(
                min(
                    self.run_deadline.remaining(),
                    initialization_timeout
                    or self.config.get("timeouts_seconds", {}).get(
                        "initialization", self.config["timeout_seconds"]
                    ),
                )
            )
        return super().start()

    def allocation(self, *, timeout):
        from ray._private.worker import get_resource_ids

        from .cluster import gpu_inventory
        from .runtime import actor_identity
        from .vllm_adapter import observed_bundle

        deadline = Deadline.after(timeout)
        report = actor_identity(self.config)
        group = ray.util.get_current_placement_group()
        if group is None or self.native_plan is None:
            raise ValueError(
                "Native training launcher requires an owned placement group"
            )
        devices = ray.get_runtime_context().get_accelerator_ids()["GPU"]
        inventory = gpu_inventory(timeout=deadline.remaining())
        by_id = {str(g["index"]): g["uuid"] for g in inventory}
        by_id.update({g["uuid"]: g["uuid"] for g in inventory})
        report.update(
            participant=f"training/{self.node_rank}",
            pg_id=group.id.hex(),
            bundle_index=observed_bundle(get_resource_ids(), group.id.hex()),
            gpu_uuids=[by_id[str(d)] for d in devices],
            pid=report["process"]["pid"],
            start_ticks=report["process"]["start_ticks"],
        )
        return report

    def run(self):
        from deepspec.orchestration.process import start_owned

        config = self.config
        buffer = ray.get_actor(config["buffer_name"], namespace=config["namespace"])
        self.buffer = buffer
        context = ray.get_runtime_context()
        devices = context.get_accelerator_ids()["GPU"]
        local_world = config["consumer_world_size"] // consumer_nodes(config)
        if len(devices) != local_world:
            raise ValueError(
                f"TorchTitan requires exactly {local_world} allocated GPUs"
            )
        if self.gate is not None:
            startup = self.initialization_deadline or Deadline.after(
                min(
                    self.run_deadline.remaining(),
                    config["timeouts_seconds"]["initialization"],
                )
            )
            self._get(
                self.gate.wait_for_allocation.remote(timeout=startup.remaining()),
                timeout=startup.remaining(),
            )
        self._get(
            buffer.event.remote(
                "consumer_launcher",
                pid=os.getpid(),
                node_id=context.get_node_id(),
                ray_gpu_ids=devices,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                node_rank=self.node_rank,
                consumer_nodes=consumer_nodes(config),
                collective_environment={
                    name: os.environ.get(name)
                    for name in (
                        "NCCL_IB_DISABLE",
                        "NCCL_NET",
                        "NCCL_SOCKET_IFNAME",
                        "GLOO_SOCKET_IFNAME",
                    )
                },
            )
        )
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(devices),
                "DEEPSPEC_PIPELINE_CONFIG": self.path,
                "DEEPSPEC_PHASE_RESULT": str(
                    Path(config["output_dir"]) / "training-result.json"
                ),
                "RAY_ADDRESS": config["ray_address"],
                "DEEPSPEC_PIPELINE_RUN_ID": config["run_id"],
                "DEEPSPEC_RUN_REMAINING_SECONDS": str(self.run_deadline.remaining()),
            }
        )
        command = consumer_command(config, self.node_rank)
        # Rank zero hands its reserved port to static torchrun once. A competing
        # bind fails the shared job; no rank is reassigned and no new port chosen.
        self._close_native(self.run_deadline)
        log_name = (
            "consumer.log"
            if self.node_rank == 0
            else f"consumer-node{self.node_rank}.log"
        )
        try:
            with (Path(config["output_dir"]) / log_name).open("w", buffering=1) as log:
                self._check_running()
                handle = start_owned(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.run_deadline.remaining(),
                    cleanup_timeout=config.get("timeouts_seconds", {}).get(
                        "cleanup", 35
                    ),
                    report_path=Path(config["output_dir"]) / f"{log_name}.cleanup.json",
                )
                with self._control_lock:
                    self._owned_process = handle
                    stopped = self._stop_reason is not None
                if stopped:
                    handle.stop()
                    self._check_running()
                if self.gate is not None:
                    registration = message_envelope(
                        config["run_id"],
                        config["plan_hash"],
                        {"component": "training_launcher"},
                        participant=f"training/{self.node_rank}",
                        node_id=context.get_node_id(),
                        actor_id=context.get_actor_id(),
                        process=handle.identity,
                        supervisor_report_path=str(handle.report_path),
                    )
                    try:
                        self._get(
                            self.gate.register_training_process.remote(
                                registration, timeout=startup.remaining()
                            ),
                            timeout=startup.remaining(),
                        )
                    except BaseException:
                        handle.stop(timeout=config["timeouts_seconds"]["cleanup"])
                        raise
                handle.result()
            if consumer_dp(config) > 1:
                self._get(
                    buffer.event.remote(
                        "consumer_node_finished", node_rank=self.node_rank
                    )
                )
            if self.node_rank != 0:
                return {
                    "node_rank": self.node_rank,
                    "launcher_exit_code": 0,
                    "log": log_name,
                }
            result = json.loads(Path(environment["DEEPSPEC_PHASE_RESULT"]).read_text())
            if consumer_dp(config) == 1:
                self._get(
                    buffer.event.remote(
                        "consumer_finished",
                        completed_updates=result["completed_updates"],
                    )
                )
            return result
        except BaseException as error:
            notify_buffer_failure(
                buffer, config, error, message=f"Consumer failed: {error!r}"
            )
            raise
