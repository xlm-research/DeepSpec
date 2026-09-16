"""Ray allocates resources; native vLLM and torchrun own their model workers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import ray


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
            self.llm = LLM(
                model=config["model_path"],
                dtype="bfloat16",
                seed=0,
                tensor_parallel_size=4,
                distributed_executor_backend="ray",
                max_model_len=config["context_length"] + 1,
                max_num_batched_tokens=8192,
                max_num_seqs=1,
                gpu_memory_utilization=0.7,
                enforce_eager=True,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                language_model_only=True,
                speculative_config={
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
                kv_transfer_config=KVTransferConfig(
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
            )
            ray.get(self.buffer.event.remote("producer_initialized", pid=os.getpid()))
            ray.get(self.buffer.wait_for_consumer.remote())
            for sample in config["samples"]:
                position = sample["position"]
                ray.get(self.buffer.reserve.remote(position))
                batch = torch.load(
                    sample["input_path"], map_location="cpu", weights_only=True
                )
                path = str(Path(config["output_dir"]) / "locators" / str(position))
                ray.get(self.buffer.event.remote("inference_start", position=position))
                self.llm.generate(
                    [{"prompt_token_ids": batch["input_ids"][0].tolist()}],
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        extra_args={
                            "kv_transfer_params": {"hidden_states_path": path},
                        },
                    ),
                    use_tqdm=False,
                )
                ray.get(self.buffer.event.remote("inference_end", position=position))
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


class Consumer:
    def __init__(self, config_path):
        self.path = config_path
        self.config = json.loads(Path(config_path).read_text())

    def run(self):
        from deepspec.orchestration.process import run_owned

        config = self.config
        buffer = ray.get_actor(config["buffer_name"])
        context = ray.get_runtime_context()
        devices = context.get_accelerator_ids()["GPU"]
        if len(devices) != 4:
            raise ValueError("TorchTitan requires exactly four allocated GPUs")
        ray.get(
            buffer.event.remote(
                "consumer_launcher",
                pid=os.getpid(),
                node_id=context.get_node_id(),
                ray_gpu_ids=devices,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--max-restarts=0",
            "-m",
            "torchtitan.models.dspark_draft.train",
            "--module",
            "deepspec.pipeline.recipe",
            "--config",
            "qwen38_streaming",
        ]
        try:
            with (Path(config["output_dir"]) / "consumer.log").open(
                "w", buffering=1
            ) as log:
                run_owned(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            result = json.loads(Path(environment["DEEPSPEC_PHASE_RESULT"]).read_text())
            ray.get(
                buffer.event.remote(
                    "consumer_finished", completed_updates=result["completed_updates"]
                )
            )
            return result
        except BaseException as error:
            ray.get(buffer.fail.remote(f"Consumer failed: {error!r}"))
            raise
