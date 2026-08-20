import os
import time

from deepspec.data import CacheCollator
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.gemma4.config import (
    build_draft_config as build_gemma4_draft_config,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)
from deepspec.modeling.dspark.qwen3_6 import Qwen3_6DSparkModel
from deepspec.modeling.dspark.qwen3_6.config import (
    build_draft_config as build_qwen3_6_draft_config,
)
from deepspec.trainer.base_trainer import BaseTrainer


def _debug_progress(message):
    if os.environ.get("DEEPSPEC_DEBUG_PROGRESS", "false").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        f"[rank={rank} local_rank={local_rank}] {message}",
        flush=True,
    )


class Qwen3DSparkTrainer(BaseTrainer):
    data_collator_cls = CacheCollator

    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3DSparkModel(draft_config)

    # Training step.
    def run_batch(self, batch):
        model_inputs = dict(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        if self.context_parallel_size > 1:
            context_layout = getattr(
                self.draft_model.config,
                "target_context_layout",
                "contiguous",
            )
            if context_layout == "native_head_tail":
                model_inputs.update(
                    context_chunk_len=batch["context_len"],
                    seq_len=batch["seq_len"],
                )
            else:
                model_inputs.update(
                    context_start=batch["context_start"],
                    context_len=batch["context_len"],
                    seq_len=batch["seq_len"],
                )
        outputs = self.model(**model_inputs)
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            l1_loss_alpha=float(self.args.model.l1_loss_alpha),
            confidence_head_alpha=float(self.args.model.confidence_head_alpha),
        )
        return loss


class Gemma4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_gemma4_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Gemma4DSparkModel(draft_config)


class DeepseekV4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        # Keep the optional V4 dependency lazy so existing Qwen/Gemma configs
        # remain importable with older Transformers builds.
        from deepspec.modeling.dspark.deepseek_v4 import (
            DeepseekV4DSparkModel,
            build_draft_config,
        )

        draft_config = build_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return DeepseekV4DSparkModel(draft_config)

    def build_online_target(self):
        from deepspec.modeling.target import DeepseekV4OnlineTarget

        return DeepseekV4OnlineTarget(
            model_name_or_path=self.args.model.target_model_name_or_path,
            target_layer_ids=self.args.model.target_layer_ids,
            topology=self.parallel,
            device=self.device,
            rank_local_cache_dir=os.path.join(
                self.checkpoint_dir_root,
                "target_rank_local",
            ),
        )

    def run_batch(self, batch):
        if self.online_target_enabled:
            if self.online_target is None:
                raise RuntimeError("Online DeepSeek-V4 target is not initialized.")
            _debug_progress("dspark: online target forward start")
            batch = self.online_target.forward_training_batch(batch)
            _debug_progress(
                "dspark: online target forward done "
                + ", ".join(
                    f"{key}=shape{tuple(value.shape)} dtype={value.dtype}"
                    for key, value in batch.items()
                    if hasattr(value, "shape")
                )
            )
        _debug_progress("dspark: draft loss forward start")
        loss = super().run_batch(batch)
        _debug_progress(f"dspark: draft loss forward done loss={loss.detach().float().item()}")
        return loss


class Qwen3_6DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_6_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3_6DSparkModel(draft_config)
