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

    def run_batch(self, batch):
        # DFlash2 uses CE plus selector path supervision; neither term needs
        # target final-layer logits.
        batch.pop("target_last_hidden_states", None)
        outputs = self.model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=None,
            context_chunk_len=batch["context_chunk_len"],
            seq_len=batch["seq_len"],
        )
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
