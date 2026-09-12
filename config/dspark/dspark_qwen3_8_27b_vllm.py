"""Qwen3.8 DSpark with a full 64-layer vLLM teacher and bounded feature caching."""

import copy
import os
import sys

from config.dspark import dspark_qwen3_8_27b as native
from deepspec.trainer.qwen3_8_vllm_trainer import Qwen3_8VllmDSparkTrainer

project_name = native.project_name
exp_name = "dspark_block7_qwen3_8_27b_vllm"
seed = native.seed
model = copy.deepcopy(native.model)
train = copy.deepcopy(native.train)
logging = copy.deepcopy(native.logging)
data = copy.deepcopy(native.data)
finalize_cfg = native.finalize_cfg

train["trainer_cls"] = Qwen3_8VllmDSparkTrainer
# Match the launcher's draft topology when loading this config directly.
train["context_parallel_size"] = 1
train["parallel"]["cp"] = 1
train["offline_target_parallel"]["cp"] = 1
train["qwen_vllm"] = dict(
    python_executable=os.environ.get("VLLM_PYTHON_BIN", sys.executable),
    source_dir=os.environ.get("VLLM_SOURCE_DIR"),
    tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4")),
    max_num_batched_tokens=int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "8192")),
    gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.45")),
    load_format=os.environ.get("VLLM_LOAD_FORMAT", "auto"),
    timeout_seconds=int(os.environ.get("VLLM_TIMEOUT_SECONDS", "86400")),
    verify_logits=os.environ.get("VLLM_VERIFY_LOGITS", "false").lower() == "true",
    logprob_atol=0.1,
)
