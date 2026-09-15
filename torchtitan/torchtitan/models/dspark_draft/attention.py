"""Stable FP32 inner attention for head-sharded DSpark training."""

import torch
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, flex_attention


# Device-local autotuning can choose different reduction trees for the
# attention backward delta. Batch invariance fixes that tree across head
# partitions; a single tile axis avoids shape-dependent multi-axis layouts.
# This applies only to the inner attention compiler, not the enclosing model.
_compiled_attention = torch.compile(
    flex_attention,
    options={"batch_invariant": True, "triton.max_tiles": 1},
)


def fp32_attention_forward(
    module,
    query_BHTD,
    key_BHSD,
    value_BHSD,
    attention_mask,
    scaling=None,
    **kwargs,
):
    if kwargs.get("dropout", 0.0):
        raise ValueError("DSpark FlexAttention requires zero attention dropout")
    if not isinstance(attention_mask, BlockMask):
        raise TypeError("DSpark FlexAttention requires its native block mask")

    def score_mod(score, batch, head, query, key):
        return score

    output_BHTD, aux = _compiled_attention(
        query_BHTD,
        key_BHSD,
        value_BHSD,
        score_mod=score_mod,
        block_mask=attention_mask,
        scale=scaling,
        enable_gqa=False,
        return_aux=AuxRequest(lse=True),
        kernel_options=kwargs.get("kernel_options"),
    )
    return output_BHTD.transpose(1, 2).contiguous(), aux.lse.to(value_BHSD.dtype)
