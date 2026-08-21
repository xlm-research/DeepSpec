from deepspec.modeling.dflash2 import (
    Qwen3_8DFlash2Model,
    compute_dflash2_loss,
)
from deepspec.modeling.dflash2.qwen3_8.config import (
    build_draft_config as build_qwen3_8_dflash2_config,
)
from deepspec.trainer.dspark_trainer import Qwen3DSparkTrainer


class Qwen3_8DFlash2Trainer(Qwen3DSparkTrainer):
    def _build_draft_model(self, *, target_config, model_args):
        draft_config = build_qwen3_8_dflash2_config(
            target_config=target_config,
            model_args=model_args,
        )
        return Qwen3_8DFlash2Model(draft_config)

    def _checkpoint_kwargs(self):
        kwargs = super()._checkpoint_kwargs()
        # DFlash2 checkpoints are public HF weights, not rank-local resume
        # shards; existing trainers keep the Dev/Base checkpoint path.
        kwargs["parallel"] = None
        return kwargs

    def run_batch(self, batch):
        # DFlash2 never consumes target final logits. Drop both cache
        # references as soon as the draft forward owns what autograd needs.
        batch.pop("target_last_hidden_states", None)
        target_hidden_states = batch.pop("target_hidden_states")
        model_inputs = dict(
            input_ids=batch["input_ids"],
            target_hidden_states=target_hidden_states,
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=None,
        )
        if self.context_parallel_size > 1:
            model_inputs.update(
                context_chunk_len=batch["context_len"],
                seq_len=batch["seq_len"],
            )
        outputs = self.model(**model_inputs)
        return compute_dflash2_loss(
            outputs=outputs,
            loss_decay_gamma=self.args.model.loss_decay_gamma,
            ce_loss_alpha=float(self.args.model.ce_loss_alpha),
            selector_loss_alpha=float(self.args.model.selector_loss_alpha),
            selector_loss_decay_gamma=self.args.model.get(
                "selector_loss_decay_gamma"
            ),
        )


__all__ = ["Qwen3_8DFlash2Trainer"]
