"""Ray allocates resources; native vLLM and torchrun own their model workers."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import ray

from .topology import consumer_dp, consumer_nodes, producer_dp, sample_producer


async def run_async_production(config, buffer, generate):
    """Admit in plan order, with one generation per native DP group at a time."""
    slots = [asyncio.Semaphore(1) for _ in range(producer_dp(config))]

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
            for sample in config["samples"]:
                rank = sample_producer(config, sample["position"])
                await slots[rank].acquire()
                try:
                    await buffer.reserve.remote(sample["position"])
                except BaseException:
                    slots[rank].release()
                    raise
                requests.append(tasks.create_task(request(sample, rank)))
            await asyncio.gather(*requests)
            # A completed generation may still have a queued Store write.
            await buffer.finish_production.remote()
            watcher.cancel()
    except BaseException as error:
        await buffer.fail.remote(f"Async producer failed: {error!r}")
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


class Producer:
    def __init__(self, config_path):
        self.config = json.loads(Path(config_path).read_text())
        self.buffer = ray.get_actor(self.config["buffer_name"])
        self.llm = None
        self.config_path = config_path

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
                "tensor_parallel_size": 4,
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
            if producer_dp(config) > 1:
                return asyncio.run(run_native_async(self, model_args))
            self.llm = LLM(**model_args)
            ray.get(self.buffer.event.remote("producer_initialized", pid=os.getpid()))
            ray.get(self.buffer.wait_for_consumer.remote())
            position = 0
            while position < len(config["samples"]):
                positions = ray.get(
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
                    ray.get(
                        self.buffer.event.remote(
                            "inference_start", position=current, producer_rank=0
                        )
                    )
                ray.get(
                    self.buffer.event.remote(
                        "inference_batch", positions=positions, batch_size=len(prompts)
                    )
                )
                self.llm.generate(prompts, parameters, use_tqdm=False)
                for current in positions:
                    ray.get(
                        self.buffer.event.remote(
                            "inference_end", position=current, producer_rank=0
                        )
                    )
                position = positions[-1] + 1
            # vLLM request completion only guarantees its D2H lifetime boundary.
            # The connector publishes READY after Store completes all writes.
            ray.get(self.buffer.finish_production.remote())
            return {"samples": len(config["samples"])}
        except BaseException as error:
            ray.get(self.buffer.fail.remote(f"Producer failed: {error!r}"))
            raise

    def close(self):
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None


async def run_native_async(producer, model_args):
    import torch
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm import SamplingParams

    config = producer.config
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
        await producer.buffer.event.remote(
            "producer_initialized", pid=os.getpid(), producer_dp=producer_dp(config)
        )
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

        return await run_async_production(config, producer.buffer, generate)
    finally:
        # Shutdown while AsyncLLM's owning event loop is still alive.
        producer.llm.shutdown(timeout=10)
        producer.llm = None


class Consumer:
    def __init__(self, config_path, node_rank=0):
        self.path = config_path
        self.config = json.loads(Path(config_path).read_text())
        self.node_rank = node_rank

    def run(self):
        from deepspec.orchestration.process import run_owned

        config = self.config
        buffer = ray.get_actor(config["buffer_name"])
        context = ray.get_runtime_context()
        devices = context.get_accelerator_ids()["GPU"]
        local_world = config["consumer_world_size"] // consumer_nodes(config)
        if len(devices) != local_world:
            raise ValueError(
                f"TorchTitan requires exactly {local_world} allocated GPUs"
            )
        ray.get(
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
            }
        )
        command = consumer_command(config, self.node_rank)
        log_name = (
            "consumer.log"
            if self.node_rank == 0
            else f"consumer-node{self.node_rank}.log"
        )
        try:
            with (Path(config["output_dir"]) / log_name).open("w", buffering=1) as log:
                run_owned(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            if consumer_dp(config) > 1:
                ray.get(
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
                ray.get(
                    buffer.event.remote(
                        "consumer_finished",
                        completed_updates=result["completed_updates"],
                    )
                )
            return result
        except BaseException as error:
            ray.get(buffer.fail.remote(f"Consumer failed: {error!r}"))
            raise
