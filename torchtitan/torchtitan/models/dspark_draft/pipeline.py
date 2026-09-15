"""Two-stage native 1F1B for DSpark's query and differentiable context."""

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B

from torchtitan.config import TORCH_DTYPE_MAP

from .common import DSparkForwardOutput


class DSparkSchedule1F1B(Schedule1F1B):
    """Feed native 1F1B the microbatches already grouped by TorchTitan.

    The pinned torch build's public ``step`` still splits a whole batch, while
    this TorchTitan revision supplies ``arg_mbs``/``kwarg_mbs`` directly.
    """

    def step(
        self,
        *,
        arg_mbs,
        kwarg_mbs,
        target_mbs=None,
        losses=None,
        loss_kwargs=None,
        return_outputs=False,
    ):
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError("Pipeline training requires gradients to be enabled")
        if return_outputs:
            raise ValueError("DSpark Trainer consumes pipeline losses only")
        self._stage.has_backward = self._has_backward
        self._stage.clear_runtime_states()
        self._step_microbatches(
            arg_mbs,
            kwarg_mbs,
            target_mbs,
            losses,
            return_outputs=False,
            loss_kwargs=loss_kwargs,
        )


def pipeline_draft(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
    device,
    model_config,
    parallelize_fn,
    loss_fn,
):
    if parallel_dims.pp != 2 or parallelism.pipeline_parallel_schedule != "1F1B":
        raise ValueError("DSpark pipeline training requires PP2 and the 1F1B schedule")
    if (
        parallelism.pipeline_parallel_schedule_csv
        or parallelism.module_fqns_per_model_part
    ):
        raise ValueError("DSpark owns its dual-stream two-stage split")
    if parallelism.num_pp_microbatches < 2 or len(model.layers) < 2:
        raise ValueError("DSpark 1F1B requires at least two microbatches and layers")
    if model.config.attention_dropout:
        raise ValueError("DSpark 1F1B currently requires zero attention dropout")
    mesh = parallel_dims.get_mesh("pp")
    rank = mesh.get_local_rank()
    model.pipeline_stage = rank
    model.config.pipeline_sequence_length = training.max_context_length
    split = (len(model.layers) + 1) // 2
    for index in range(len(model.layers)):
        if (index < split) != (rank == 0):
            model.layers[index] = None
    unused = (
        ("norm", "lm_head", "markov_head", "confidence_head")
        if rank == 0
        else ("embed_tokens", "fc", "hidden_norm")
    )
    for name in unused:
        setattr(model, name, None)
    model = parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )

    # Explicit metadata avoids an extra model/loss pass during schedule startup.
    # In particular, anchor sampling and loss counters see only actual updates.
    dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    sequence = training.max_context_length
    batch, remainder = divmod(training.num_tokens_per_microbatch_per_dp_rank, sequence)
    if remainder or batch < 1 or (parallel_dims.cp > 1 and batch != 1):
        raise ValueError(
            "DSpark PP token budget must describe complete local sequences"
        )
    anchors = model.num_anchors // parallel_dims.cp
    query = anchors * model.verification_block_size
    context = (
        sequence
        if parallel_dims.cp == 1
        else 2 * ((sequence + 2 * parallel_dims.cp - 1) // (2 * parallel_dims.cp))
    )
    if model.sequence_parallel_group is not None:
        tp_rank = parallel_dims.get_mesh("tp").get_local_rank()
        query = query // parallel_dims.tp + (tp_rank < query % parallel_dims.tp)
        context = context // parallel_dims.tp + (tp_rank < context % parallel_dims.tp)

    def example(shape, tensor_dtype=dtype, requires_grad=False):
        return torch.empty(
            shape, dtype=tensor_dtype, device="meta", requires_grad=requires_grad
        )

    crossing = (
        example((batch, query, model.config.hidden_size), requires_grad=True),
        example((batch, context, model.config.hidden_size), requires_grad=True),
        example((batch, anchors), torch.int64),
        example((batch, anchors), torch.bool),
    )
    tokens = example((batch, sequence), torch.int64)
    prediction_shape = (batch, anchors, model.block_size)
    vocab = model.config.vocab_size // (
        parallel_dims.tp if loss_fn.config.enable_vocab_parallel else 1
    )
    outputs = (
        example((*prediction_shape, vocab), requires_grad=True),
        example(prediction_shape, torch.int64),
        example(prediction_shape, torch.bool),
        example((batch, anchors), torch.bool),
        example(prediction_shape, dtype, True)
        if model.enable_confidence_head
        else example((0,)),
        example((*prediction_shape, vocab)),
    )
    stage = PipelineStage(
        model,
        rank,
        2,
        device,
        input_args=(tokens,) if rank == 0 else crossing,
        output_args=crossing if rank == 0 else outputs,
        input_grads=(None,) if rank == 0 else (crossing[0], crossing[1], None, None),
        output_grads=(crossing[0], crossing[1], None, None)
        if rank == 0
        else (
            outputs[0],
            None,
            None,
            None,
            outputs[4] if model.enable_confidence_head else None,
            None,
        ),
        group=mesh.get_group(),
    )

    def scalar_loss(output, labels, **kwargs):
        logits, ids, mask, keep, confidence, teacher = output
        prediction = DSparkForwardOutput(
            draft_logits=logits,
            target_ids=ids,
            eval_mask=mask,
            block_keep_mask=keep,
            confidence_pred=confidence if confidence.numel() else None,
            aligned_target_logits=teacher if teacher.numel() else None,
        )
        return loss_fn(prediction, labels, **kwargs)[0]

    schedule = DSparkSchedule1F1B(
        stage,
        n_microbatches=parallelism.num_pp_microbatches,
        loss_fn=scalar_loss,
        scale_grads=False,
    )
    return schedule, [model], rank == 0, rank == 1
