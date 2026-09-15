"""Native component recipe for the initial Qwen DSpark integration."""

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig

from . import model_spec
from .data import FeatureLoader, PreparedTokens
from .loss import DSparkLoss
from .optimizer import DraftOptimizers
from .scheduler import DraftSchedulers
from .trainer import DSparkTrainer


def qwen38_debug() -> DSparkTrainer.Config:
    model = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        layer_types=["full_attention"] * 2,
    )

    model.target_layer_ids = [1, 3]
    model.num_target_layers = 4
    model.block_size = 3
    model.mask_token_id = 127
    model.num_anchors = 2
    model.enable_confidence_head = True
    model.markov_rank = 8
    model.markov_head_type = "vanilla"
    model.confidence_head_with_markov = True
    return DSparkTrainer.Config(
        model_spec=model_spec(model.to_dict()),
        tokenizer=PreparedTokens.Config(vocab_size=128),
        dataloader=FeatureLoader.Config(manifest=""),
        loss=DSparkLoss.Config(),
        optimizer=DraftOptimizers.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(
                    pattern=".*",
                    optimizer_name="MasterWeightAdamW",
                    optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
                )
            ],
        ),
        lr_scheduler=DraftSchedulers.Config(
            warmup_steps=1,
            total_steps=4,
            decay_type="cosine",
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=16,
            num_tokens_per_train_step=32,
            max_context_length=16,
            steps=2,
            max_norm=0.5,
            dtype="float32",
            mixed_precision_param="float32",
            fp32_matmul_precision="ieee",
            disable_cuda_graphs=True,
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=None,
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        debug=DebugConfig(seed=1000),
    )


def qwen38_27b() -> DSparkTrainer.Config:
    """Released Qwen teacher geometry with the retained five-layer DSpark recipe."""
    import json
    from pathlib import Path

    from .checkpoint import PhaseCheckpointer
    from .preparation import PreparationConfig

    assets = "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"
    text = json.loads((Path(assets) / "config.json").read_text())["text_config"]
    expected = {
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
    }
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("The Qwen3.8 recipe requires the released 27B teacher")
    model = Qwen3_5TextConfig(**text)
    model.num_target_layers = 64
    model.num_hidden_layers = 5
    model.layer_types = ["full_attention"] * 5
    model.target_layer_ids = [1, 16, 31, 46, 61]
    model.block_size = 7
    model.mask_token_id = 248077
    model.num_anchors = 512
    model.enable_confidence_head = True
    model.markov_rank = 256
    model.markov_head_type = "vanilla"
    model.confidence_head_with_markov = True
    model.tie_word_embeddings = False
    model.partial_rotary_factor = 1.0
    model.rope_parameters = {"rope_type": "default", "rope_theta": 10000000.0}
    model.target_context_layout = "native_head_tail"
    config = qwen38_debug()
    config.model_spec = model_spec(model.to_dict())
    config.hf_assets_path = assets
    config.initial_target_path = assets
    config.tokenizer = PreparedTokens.Config(vocab_size=model.vocab_size)
    config.dataloader = FeatureLoader.Config(
        manifest="",
        target_layer_ids=model.target_layer_ids,
        hidden_size=model.hidden_size,
        require_producer_manifest=True,
    )
    config.training = TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=131072,
        num_tokens_per_train_step=131072 * 512,
        max_context_length=131072,
        steps=1000,
        max_norm=1.0,
        dtype="bfloat16",
        mixed_precision_param="bfloat16",
        fp32_matmul_precision="ieee",
        disable_cuda_graphs=True,
    )
    config.parallelism = ParallelismConfig(data_parallel_shard_degree=-1)
    config.optimizer.param_groups[0].optimizer_kwargs["lr"] = 6e-4
    config.lr_scheduler = DraftSchedulers.Config(
        warmup_steps=40, total_steps=1000, decay_type="cosine"
    )
    config.checkpoint = PhaseCheckpointer.Config(
        enable=True, interval=1000, keep_latest_k=2
    )
    config.debug.seed = 42
    config.preparation = PreparationConfig(
        source_paths=["train_data/spec_o3_coldstartsft.repeat60.deepspec.jsonl"],
        epochs=10,
    )
    return config


def qwen38_27b_tp4() -> DSparkTrainer.Config:
    """Eight-GPU DSpark topology with full context and native SelectiveAC."""
    from torchtitan.distributed.activation_checkpoint import SelectiveAC

    config = qwen38_27b()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 4
    config.parallelism.enable_sequence_parallel = False
    config.activation_checkpoint = SelectiveAC.Config()
    return config
