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
import os
from torch.profiler import record_function

from deepspec.trainer.base_trainer import BaseTrainer


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
        needs_target_logits = (
            float(self.args.model.l1_loss_alpha) > 0.0
            or float(self.args.model.confidence_head_alpha) > 0.0
        )
        if needs_target_logits:
            target_last_hidden_states = batch["target_last_hidden_states"]
        else:
            # DFlash is CE-only, so the target model's final-layer feature is
            # unused.  Release it before the draft forward starts.
            batch.pop("target_last_hidden_states", None)
            target_last_hidden_states = None
        with record_function("deepspec::draft_forward"):
            outputs = self.forward_model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=target_last_hidden_states,
                context_chunk_len=batch["context_chunk_len"],
                seq_len=batch["seq_len"],
            )
        with record_function("deepspec::loss"):
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


class Qwen3_6DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_6_draft_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3_6DSparkModel(draft_config)


class DeepseekV4DSparkTrainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        from deepspec.modeling.dspark.deepseek_v4 import (
            DeepseekV4DSparkModel,
            build_draft_config,
        )

        return DeepseekV4DSparkModel(
            build_draft_config(target_config=target_config, model_args=model_args)
        )

    def build_online_target(self):
        from deepspec.modeling.target import DeepseekV4OnlineTarget

        return DeepseekV4OnlineTarget(
            model_name_or_path=self.args.model.target_model_name_or_path,
            target_layer_ids=self.args.model.target_layer_ids,
            topology=self.target_parallel,
            device=self.device,
            rank_local_cache_dir=os.path.join(
                self.checkpoint_dir_root, "target_rank_local"
            ),
        )

    def run_batch(self, batch):
        if self.online_target_enabled:
            with record_function("deepspec::target_forward"):
                target_batch = self.online_target.forward_training_batch(batch)
            # Keep the generated supervision on the outer training-loop batch.
            # This lets BaseTrainer explicitly drop its final references as soon
            # as this micro-batch's backward has consumed the tensors.
            batch.clear()
            batch.update(target_batch)
        needs_target_logits = (
            float(self.args.model.l1_loss_alpha) > 0.0
            or float(self.args.model.confidence_head_alpha) > 0.0
        )
        if not needs_target_logits:
            batch.pop("target_last_hidden_states", None)
        with record_function("deepspec::draft_forward"):
            outputs = self.forward_model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch.get("target_last_hidden_states"),
                context_start=batch["context_start"],
                context_len=batch["context_len"],
                seq_len=batch["seq_len"],
            )
        with record_function("deepspec::loss"):
            return compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=self.args.model.loss_decay_gamma,
                ce_loss_alpha=float(self.args.model.ce_loss_alpha),
                l1_loss_alpha=float(self.args.model.l1_loss_alpha),
                confidence_head_alpha=float(self.args.model.confidence_head_alpha),
            )
