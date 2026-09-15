"""Opt-in GLM draft training check, using synthetic cached teacher features.

Run in the isolated DeepEP environment on two otherwise idle GPUs:
    DEEPSPEC_TEST_DEEPEP=1 EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=4,5 python -m \
        torch.distributed.run --standalone --nproc-per-node=2 -m unittest \
        tests.test_glm5_deepep_integration -v

No target model or checkpoint is loaded. EP_DISABLE_GIN is for this node-local
check only. DEEPSPEC_TEST_DEEPEP_HIDDEN can select another DeepEP-supported
hidden dimension (default 512).
Set DEEPSPEC_TEST_DEEPEP_DP_REPLICATE=2 and --nproc-per-node=4 to include
HSDP replica reduction across two EP groups.
"""

import copy
import gc
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

from deepspec.distributed import ParallelConfig, ParallelContext, apply_parallelism
from deepspec.distributed.deepep_dispatch import require_deepep
from deepspec.distributed.draft_expert_dispatch import close_draft_expert_dispatchers
from deepspec.modeling.dspark.glm5_next.modeling import Glm5NextDSparkModel
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.pure_ep import (
    get_pure_expert_modules,
    synchronize_pure_expert_gradients,
)
from deepspec.training.loss import configure_loss_reduction_group
from tests.distributed_test_utils import require_torchrun


def _tiny_config():
    hidden = int(os.environ.get("DEEPSPEC_TEST_DEEPEP_HIDDEN", "512"))
    config = Glm5NextTextConfig(
        hidden_size=hidden,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=0,
        v_head_dim=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        vocab_size=128,
        pad_token_id=127,
        layer_types=["deepseek_sparse_attention"] * 2,
        mlp_layer_types=["sparse"] * 2,
        attention_dropout=0.0,
    )
    config._experts_implementation = "grouped_mm"
    config._attn_implementation = "eager"
    config.target_layer_ids = [0, 1]
    config.sliding_window = 8
    config.block_size = 7
    config.mask_token_id = 127
    config.num_anchors = 1
    config.enable_confidence_head = True
    config.confidence_head_with_markov = True
    config.markov_rank = 16
    config.markov_head_type = "gated"
    return config


def _local(tensor):
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().float()


def _batch(config, runtime):
    # The two ranks dispatch 7 and 21 tokens per layer, with a capacity of 8:
    # the shorter rank must join every forward and backward chunk collective.
    batch_size = 1 + 2 * (runtime.global_rank % 2)
    generator = torch.Generator(device=runtime.device).manual_seed(418 + runtime.global_rank)
    sequence = 32
    return {
        "input_ids": torch.randint(
            0, 127, (batch_size, sequence), generator=generator, device=runtime.device
        ),
        "loss_mask": torch.ones(batch_size, sequence, dtype=torch.bool, device=runtime.device),
        "target_hidden_states": torch.randn(
            batch_size, sequence, len(config.target_layer_ids) * config.hidden_size,
            generator=generator, device=runtime.device, dtype=torch.bfloat16,
        ),
        "target_last_hidden_states": torch.randn(
            batch_size, sequence, config.hidden_size,
            generator=generator, device=runtime.device, dtype=torch.bfloat16,
        ),
    }


@unittest.skipUnless(os.environ.get("DEEPSPEC_TEST_DEEPEP") == "1", "opt-in DeepEP GPU test")
class Glm5DeepEPIntegrationTest(unittest.TestCase):
    def _assert_parameters_close(self, native, deepep, *, gradients):
        native_parameters = dict(native.named_parameters())
        deepep_parameters = dict(deepep.named_parameters())
        self.assertEqual(native_parameters.keys(), deepep_parameters.keys())
        statistics = torch.zeros(3, 2, device="cuda", dtype=torch.float64)
        for name, actual_parameter in deepep_parameters.items():
            reference_parameter = native_parameters[name]
            actual = actual_parameter.grad if gradients else actual_parameter
            reference = reference_parameter.grad if gradients else reference_parameter
            self.assertEqual(actual is None, reference is None, name)
            if actual is None:
                continue
            actual, reference = _local(actual), _local(reference)
            self.assertTrue(bool(torch.isfinite(actual).all()), name)
            maximum = float(reference.abs().max()) if reference.numel() else 0.0
            torch.testing.assert_close(
                actual, reference,
                rtol=0.08 if gradients else 0.03,
                atol=max(1e-7, maximum * (0.02 if gradients else 0.005)),
                msg=lambda message, name=name: f"{name}: {message}",
            )
            category = 0 if ".mlp.experts." in name else (1 if ".mlp.gate." in name else 2)
            statistics[category, 0] += (actual - reference).double().square().sum()
            statistics[category, 1] += reference.double().square().sum()
        dist.all_reduce(statistics)
        # Per-tensor absolute tolerances alone can hide a gradient-scale error.
        # Check aggregate relative error independently for experts, routers,
        # and the remaining dense parameters.
        errors = {}
        for category, (squared_error, squared_reference) in zip(
            ("experts", "routers", "dense"), statistics.tolist()
        ):
            self.assertGreater(squared_reference, 0.0, category)
            relative_error = (squared_error / squared_reference) ** 0.5
            self.assertLess(relative_error, 0.06 if gradients else 0.02, category)
            errors[category] = relative_error
        return errors

    def _forward(self, model, batch, seed):
        torch.manual_seed(seed)
        with patch("deepspec.modeling.dspark.common.add_metric"):
            return model(**batch)

    def _loss(self, outputs):
        with patch("deepspec.modeling.dspark.loss.add_metric"):
            return compute_dspark_loss(
                outputs=outputs,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.2,
                l1_loss_alpha=0.8,
                confidence_head_alpha=0.1,
            )

    def _assert_outputs_close(self, native, deepep):
        for field in ("target_ids", "eval_mask", "block_keep_mask"):
            torch.testing.assert_close(getattr(deepep, field), getattr(native, field))
        for field in ("draft_logits", "aligned_target_logits", "confidence_pred"):
            torch.testing.assert_close(
                getattr(deepep, field), getattr(native, field), rtol=0.03, atol=0.02
            )

    def test_two_layer_fsdp_training_matches_native_and_recreates_buffer(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        dp_replicate = int(os.environ.get("DEEPSPEC_TEST_DEEPEP_DP_REPLICATE", "1"))
        if dp_replicate < 1:
            raise ValueError("DEEPSPEC_TEST_DEEPEP_DP_REPLICATE must be positive.")
        runtime = require_torchrun(self, world_size=2 * dp_replicate)
        require_deepep()
        config = _tiny_config()
        parallel = ParallelConfig(
            dp_replicate=dp_replicate, dp_shard=2, ep=2, expert_dispatch_backend="native",
            expert_dispatch_max_tokens_per_rank=8, reduce_dtype="fp32",
        )
        context = ParallelContext.build(parallel, device_type="cuda")
        configure_loss_reduction_group(context.loss_mesh.get_group())
        batch = _batch(config, runtime)
        previous_dispatcher = None
        try:
            for cycle in range(2):
                torch.manual_seed(171 + cycle)
                native = Glm5NextDSparkModel(copy.deepcopy(config))
                native.set_embedding_head_trainable(False)
                deepep = copy.deepcopy(native)
                native.to(runtime.device, dtype=torch.bfloat16)
                deepep.to(runtime.device, dtype=torch.bfloat16)
                with patch.dict(os.environ, {"DEEPSPEC_V4_EP_TOKEN_CHUNK": "8"}):
                    native = apply_parallelism(
                        native, context, parallel, param_dtype=torch.bfloat16
                    )
                deepep = apply_parallelism(
                    deepep, context, replace(parallel, expert_dispatch_backend="deepep"),
                    param_dtype=torch.bfloat16,
                )
                try:
                    self.assertIsInstance(native, FSDPModule)
                    self.assertIsInstance(deepep, FSDPModule)
                    for model in (native, deepep):
                        self.assertTrue(all(isinstance(layer, FSDPModule) for layer in model.layers))
                        self.assertFalse(model.embed_tokens.weight.requires_grad)
                        self.assertFalse(model.lm_head.weight.requires_grad)
                    expert_modules = get_pure_expert_modules(deepep)
                    self.assertEqual(len(expert_modules), 2)
                    dispatcher = expert_modules[0]._deepspec_deepep_dispatcher
                    self.assertTrue(all(
                        module._deepspec_deepep_dispatcher is dispatcher for module in expert_modules
                    ))
                    self.assertIsNot(dispatcher, previous_dispatcher)
                    self.assertIsNone(dispatcher._buffer)
                    self.assertFalse(dispatcher._closed)
                    optimizers = [
                        torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
                        for model in (native, deepep)
                    ]
                    native_outputs = self._forward(native, batch, 817 + cycle)
                    native_loss = self._loss(native_outputs)
                    native_loss.backward()
                    deepep_outputs = self._forward(deepep, batch, 817 + cycle)
                    deepep_loss = self._loss(deepep_outputs)
                    deepep_loss.backward()
                    self._assert_outputs_close(native_outputs, deepep_outputs)
                    torch.testing.assert_close(deepep_loss, native_loss, rtol=0.02, atol=0.005)
                    self.assertIsNotNone(dispatcher._buffer)
                    for model in (native, deepep):
                        synchronize_pure_expert_gradients(
                            get_pure_expert_modules(model), sparse_mesh=context.sparse_mesh
                        )
                    gradient_errors = self._assert_parameters_close(native, deepep, gradients=True)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    parameter_errors = self._assert_parameters_close(native, deepep, gradients=False)
                    if runtime.global_rank == 0:
                        for phase, errors in (("gradients", gradient_errors), ("parameters", parameter_errors)):
                            details = " ".join(
                                f"{category}_relative_l2={error:.8g}"
                                for category, error in errors.items()
                            )
                            print(
                                f"[glm5-deepep-accuracy] cycle={cycle} "
                                f"dp_replicate={dp_replicate} phase={phase} {details}",
                                flush=True,
                            )
                    with torch.no_grad():
                        native_updated = self._forward(native, batch, 817 + cycle)
                        deepep_updated = self._forward(deepep, batch, 817 + cycle)
                    self._assert_outputs_close(native_updated, deepep_updated)
                    del native_outputs, deepep_outputs, native_updated, deepep_updated
                    del native_loss, deepep_loss, optimizers
                finally:
                    close_draft_expert_dispatchers(deepep)
                close_draft_expert_dispatchers(deepep)
                self.assertIsNone(dispatcher._buffer)
                self.assertTrue(dispatcher._closed)
                previous_dispatcher = dispatcher
                del native, deepep, expert_modules
                gc.collect()
                torch.cuda.synchronize(runtime.device)
                dist.barrier()
        finally:
            configure_loss_reduction_group(None)


if __name__ == "__main__":
    unittest.main()
