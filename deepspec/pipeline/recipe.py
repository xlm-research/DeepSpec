"""Real Qwen3.8 DSpark structure with a four-GPU streaming data recipe."""

import json
import os
from dataclasses import fields
from pathlib import Path

from torchtitan.models.dspark_draft.config_registry import qwen38_27b_tp4

from .data import MooncakeFeatureLoader
from .trainer import StreamingDSparkTrainer


def qwen38_preparation():
    pipeline_path = os.environ["DEEPSPEC_PIPELINE_CONFIG"]
    pipeline = json.loads(Path(pipeline_path).read_text())
    config = qwen38_27b_tp4()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 4
    config.training.max_context_length = pipeline["context_length"]
    config.training.num_tokens_per_microbatch_per_dp_rank = pipeline["context_length"]
    config.training.num_tokens_per_train_step = (
        pipeline["context_length"] * pipeline["samples_per_update"]
    )
    config.training.steps = pipeline["steps"]
    config.phase_stop_update = pipeline["steps"]
    config.run_id = pipeline["run_id"]
    config.dump_folder = str(Path(pipeline["output_dir"]) / "training")
    config.preparation.source_paths = [pipeline["source_path"]]
    config.preparation.epochs = 1
    config.metrics.log_freq = 1
    config.checkpoint.folder = str(Path(pipeline["output_dir"]) / "checkpoints")
    config.measure_phase = True
    return config


def qwen38_streaming():
    pipeline_path = os.environ["DEEPSPEC_PIPELINE_CONFIG"]
    pipeline = json.loads(Path(pipeline_path).read_text())
    base = qwen38_preparation()
    config = StreamingDSparkTrainer.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    config.dataloader = MooncakeFeatureLoader.Config(
        manifest=pipeline["manifest_path"],
        plan_path=pipeline["plan_path"],
        target_layer_ids=base.dataloader.target_layer_ids,
        hidden_size=base.dataloader.hidden_size,
        require_producer_manifest=True,
        pipeline_config=pipeline_path,
    )
    return config
