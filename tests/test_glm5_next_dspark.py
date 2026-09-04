import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextTextConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextLinearAttention,
    chunk_kimi_delta_attention as transformers_chunk_kimi_delta_attention,
)

from deepspec.data.parser import preprocess_record
from deepspec.data.target_cache_dataset import validate_train_cache
from deepspec.modeling.dspark.glm5_next import (
    Glm5NextDSparkModel,
    build_draft_config,
)
from deepspec.modeling.dspark.glm5_next.config import (
    validate_glm5_next_target_config,
    validate_glm5_next_tokenizer,
)
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.glm5_next_parallel import parallelize_glm5_next_model
from deepspec.modeling.target import Glm5NextOnlineTarget
from deepspec.modeling.target.glm5_next import (
    _causal_conv1d_prefill,
    _flashinfer_supports_glm_nope,
    _native_nope_sparse_attention,
    _pack_kpool_topk_for_native_kernel,
    install_glm5_next_bounded_target_prefill,
)
from deepspec.modeling.target.online import _build_rank_local_ep_target_model
from deepspec.trainer import Glm5NextDSparkTrainer
from deepspec.utils import load_config
from scripts.data.prepare_deepseek_v4_target_cache import (
    _resolve_target_runtime,
    _retained_target_has_routed_experts,
)

TARGET = "/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash"


def tiny_target_config():
    config = AutoConfig.from_pretrained(TARGET)
    text = config.text_config
    text.hidden_size = 64
    text.num_attention_heads = 4
    text.num_key_value_heads = 4
    text.q_lora_rank = 32
    text.kv_lora_rank = 16
    text.qk_nope_head_dim = 16
    text.qk_rope_head_dim = 0
    text.qk_head_dim = 16
    text.v_head_dim = 16
    text.intermediate_size = 128
    text.moe_intermediate_size = 32
    text.n_routed_experts = 8
    text.n_shared_experts = 1
    text.num_experts_per_tok = 2
    text.hc_mult = 2
    text.vocab_size = 128
    text.pad_token_id = 127
    text.num_hidden_layers = 4
    text.layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
    ]
    text.mlp_layer_types = ["dense", "dense", "dense", "sparse"]
    text.index_n_heads = 4
    text.index_head_dim = 8
    text.index_topk = 8
    text.index_kpool = 2
    return config


def model_args():
    return copy.deepcopy(
        load_config("config/dspark/dspark_glm5_3_flash.py").model
    )


def tiny_draft_config():
    config = build_draft_config(
        AutoConfig.from_pretrained(TARGET),
        model_args(),
    )
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.q_lora_rank = 32
    config.kv_lora_rank = 16
    config.qk_nope_head_dim = 16
    config.qk_rope_head_dim = 0
    config.qk_head_dim = 16
    config.v_head_dim = 16
    config.intermediate_size = 128
    config.moe_intermediate_size = 32
    config.n_routed_experts = 8
    config.n_shared_experts = 1
    config.num_experts_per_tok = 2
    config.hc_mult = 2
    config.vocab_size = 128
    config.pad_token_id = 127
    config.mask_token_id = 127
    config.num_anchors = 2
    config.sliding_window = 16
    config.markov_rank = 16
    return config


def tiny_sparse_attention_config():
    config = Glm5NextTextConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        layer_types=["deepseek_sparse_attention"] * 2,
        mlp_layer_types=["dense"] * 2,
        indexer_types=["full", "shared"],
        index_topk=4,
        index_kpool=2,
        index_n_heads=2,
        index_head_dim=4,
        index_kpool_always_select_tail=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return config


class Glm5NextDSparkTest(unittest.TestCase):
    def test_config_selects_glm_trainer_and_parallel_layout(self):
        config = load_config("config/dspark/dspark_glm5_3_flash.py")
        self.assertIs(config.train.trainer_cls, Glm5NextDSparkTrainer)
        self.assertEqual(config.model.target_model_name_or_path, TARGET)
        self.assertEqual(config.model.target_layer_ids, [2, 22, 42])
        self.assertEqual(config.train.parallel.ep, 8)
        self.assertEqual(config.train.target_parallel.ep, 8)
        self.assertEqual(config.train.offline_target_parallel.dp_shard, 2)
        self.assertEqual(config.train.offline_target_parallel.tp, 4)
        self.assertEqual(config.train.offline_target_parallel.ep, 8)
        self.assertEqual(config.train.data_batch_size, 256)
        self.assertIsNone(config.train.max_train_steps)
        self.assertFalse(config.data.online_target)
        self.assertTrue(config.data.offline_target_data_batches)
        self.assertTrue(config.data.train_data_path)
        self.assertTrue(config.data.source_jsonl_path)
        self.assertTrue(config.data.store_target_last_hidden_states)
        self.assertIsNone(config.data.target_cache_path)
        self.assertEqual(config.data.chat_template, "glm5_next")

    def test_config_derives_multi_node_topologies_from_torchrun(self):
        cases = (
            # Node-local target TP4/FSDP2 groups on two 8-GPU nodes.
            (16, 8, (2, 8, 8), (2, 2, 8)),
            # Six GPUs per node cannot contain whole TP4 groups. Fall back to
            # one global target mesh without restricting the node shape.
            (12, 6, (2, 6, 6), (1, 3, 12)),
        )
        for world_size, local_size, draft, target in cases:
            with self.subTest(world_size=world_size, local_size=local_size):
                with patch.dict(
                    os.environ,
                    {
                        "WORLD_SIZE": str(world_size),
                        "LOCAL_WORLD_SIZE": str(local_size),
                    },
                ):
                    config = load_config(
                        "config/dspark/dspark_glm5_3_flash.py"
                    )
                self.assertEqual(config.train.global_batch_size, world_size)
                self.assertEqual(
                    (
                        config.train.parallel.dp_replicate,
                        config.train.parallel.dp_shard,
                        config.train.parallel.ep,
                    ),
                    draft,
                )
                self.assertEqual(
                    (
                        config.train.offline_target_parallel.dp_replicate,
                        config.train.offline_target_parallel.dp_shard,
                        config.train.offline_target_parallel.ep,
                    ),
                    target,
                )
                self.assertEqual(config.train.offline_target_parallel.tp, 4)

    def test_offline_target_runner_selects_glm_wrapper(self):
        target_name, target_cls = _resolve_target_runtime(
            SimpleNamespace(model_type="glm5_next")
        )
        self.assertEqual(target_name, "GLM-5.3")
        self.assertIs(target_cls, Glm5NextOnlineTarget)

    def test_target_ep_matches_retained_glm_layers(self):
        target_config = AutoConfig.from_pretrained(TARGET)
        self.assertTrue(
            _retained_target_has_routed_experts(target_config, [0, 1, 2])
        )
        self.assertTrue(
            _retained_target_has_routed_experts(target_config, [0, 1, 2, 3])
        )

    def test_released_checkpoint_contract_and_draft_metadata(self):
        target_config = AutoConfig.from_pretrained(TARGET)
        text_config = validate_glm5_next_target_config(target_config)
        self.assertEqual(text_config.num_hidden_layers, 45)
        self.assertEqual(
            [
                index
                for index, layer_type in enumerate(text_config.layer_types)
                if layer_type == "deepseek_sparse_attention"
            ],
            list(range(3, 44, 4)),
        )
        self.assertEqual(text_config.mlp_layer_types[:3], ["dense"] * 3)
        self.assertEqual(text_config.mlp_layer_types[3:], ["sparse"] * 42)

        draft_config = build_draft_config(target_config, model_args())
        self.assertEqual(draft_config.target_layer_ids, [2, 22, 42])
        self.assertEqual(draft_config.num_target_layers, 45)
        self.assertEqual(draft_config.deepspec_target_execution, "full_model")
        self.assertEqual(draft_config.deepspec_target_final_hidden_layer, 44)
        self.assertEqual(draft_config.target_context_layout, "contiguous")
        self.assertEqual(draft_config.block_size, 7)

        incompatible = copy.deepcopy(target_config)
        incompatible.text_config.index_topk = 1024
        with self.assertRaisesRegex(ValueError, "index_topk"):
            build_draft_config(incompatible, model_args())

    def test_draft_constructor_allocates_only_rank_local_experts(self):
        torch.manual_seed(42)
        rank_zero = Glm5NextDSparkModel(
            tiny_draft_config(),
            expert_parallel_size=2,
            expert_parallel_rank=0,
        )
        torch.manual_seed(42)
        rank_one = Glm5NextDSparkModel(
            tiny_draft_config(),
            expert_parallel_size=2,
            expert_parallel_rank=1,
        )

        zero_experts = rank_zero.layers[0].mlp.experts
        one_experts = rank_one.layers[0].mlp.experts
        self.assertEqual(tuple(zero_experts.gate_up_proj.shape), (4, 64, 64))
        self.assertEqual(tuple(one_experts.down_proj.shape), (4, 64, 32))
        self.assertEqual(tuple(rank_one.layers[0].mlp.gate.weight.shape), (8, 64))
        self.assertTrue(zero_experts._deepspec_expert_parameters_distributed)
        self.assertFalse(
            torch.equal(zero_experts.gate_up_proj, one_experts.gate_up_proj)
        )
        torch.testing.assert_close(
            rank_zero.layers[0].self_attn.q_a_proj.weight,
            rank_one.layers[0].self_attn.q_a_proj.weight,
        )

        topology = SimpleNamespace(
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            tensor_parallel_group=None,
            expert_parallel_size=2,
            expert_parallel_rank=1,
            expert_parallel_group=None,
            pure_expert_parallel=True,
        )
        parallelize_glm5_next_model(rank_one, topology=topology, draft=True)
        self.assertEqual(tuple(one_experts.gate_up_proj.shape), (4, 64, 64))
        self.assertEqual(one_experts.num_experts, 4)
        self.assertTrue(one_experts._deepspec_pure_expert_parallel)

    def test_reserved_mask_row_matches_tokenizer_and_checkpoint_shapes(self):
        tokenizer = AutoTokenizer.from_pretrained(TARGET)
        validate_glm5_next_tokenizer(tokenizer, mask_token_id=154879)
        self.assertIsNone(tokenizer.mask_token_id)
        self.assertNotIn(154879, set(tokenizer.get_vocab().values()))

        with open(
            Path(TARGET) / "model.safetensors.index.json",
            "r",
            encoding="utf-8",
        ) as handle:
            weight_map = json.load(handle)["weight_map"]
        for tensor_name in (
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ):
            with safe_open(
                Path(TARGET) / weight_map[tensor_name],
                framework="pt",
                device="cpu",
            ) as checkpoint:
                self.assertEqual(
                    checkpoint.get_slice(tensor_name).get_shape(),
                    [154880, 4096],
                )

    def test_target_linear_attention_and_dense_mlp_support_tp4(self):
        config = tiny_target_config()
        config.vision_config.depth = 0
        text = config.text_config
        text.linear_num_heads = 4
        text.linear_head_dim = 16
        with torch.device("meta"):
            target = AutoModel.from_config(config).language_model
        topology = SimpleNamespace(
            tensor_parallel_size=4,
            tensor_parallel_rank=1,
            tensor_parallel_group=None,
            expert_parallel_size=1,
            expert_parallel_rank=0,
            expert_parallel_group=None,
            pure_expert_parallel=False,
        )

        parallelize_glm5_next_model(target, topology=topology, draft=False)

        attention = target.layers[0].self_attn
        self.assertEqual(tuple(attention.q_proj.weight.shape), (16, 64))
        self.assertEqual(tuple(attention.conv1d.weight.shape), (48, 1, 4))
        self.assertEqual(tuple(attention.forget_gate.A_log.shape), (1,))
        self.assertEqual(tuple(attention.o_proj.weight.shape), (64, 16))
        mlp = target.layers[0].mlp
        self.assertEqual(tuple(mlp.gate_proj.weight.shape), (32, 64))
        self.assertEqual(tuple(mlp.down_proj.weight.shape), (64, 32))
        sparse_attention = target.layers[3].self_attn
        self.assertEqual(tuple(sparse_attention.q_b_proj.weight.shape), (16, 32))
        self.assertEqual(tuple(sparse_attention.kv_b_proj.weight.shape), (32, 16))
        self.assertEqual(tuple(sparse_attention.o_proj.weight.shape), (64, 16))
        self.assertEqual(sparse_attention.num_heads, 1)
        self.assertEqual(tuple(target.embed_tokens.weight.shape), (32, 64))

    def test_draft_forward_and_backward(self):
        config = tiny_draft_config()
        self.assertEqual(config.architectures, ["Glm5NextDSparkModel"])
        self.assertEqual(config.model_type, "glm5_next_text")
        self.assertEqual(config.mlp_layer_types, ["sparse"] * 3)
        self.assertEqual(config._experts_implementation, "grouped_mm")

        model = Glm5NextDSparkModel(config).float()
        model.initialize_embedding_and_head_weights(
            embed_weight=torch.randn(128, 64),
            lm_head_weight=torch.randn(128, 64),
            freeze=True,
        )
        seq_len = 32
        output = model(
            input_ids=torch.randint(0, 127, (1, seq_len)),
            target_hidden_states=torch.randn(1, seq_len, 3 * 64),
            loss_mask=torch.ones(1, seq_len, dtype=torch.bool),
            target_last_hidden_states=torch.randn(1, seq_len, 64),
            context_start=torch.tensor([0]),
            context_len=torch.tensor([seq_len]),
            seq_len=torch.tensor([seq_len]),
        )
        self.assertEqual(tuple(output.draft_logits.shape), (1, 2, 7, 128))
        self.assertEqual(tuple(output.target_ids.shape), (1, 2, 7))
        self.assertEqual(tuple(output.aligned_target_logits.shape), (1, 2, 7, 128))
        self.assertEqual(tuple(output.confidence_pred.shape), (1, 2, 7))

        with patch("deepspec.modeling.dspark.loss.add_metric"):
            loss = compute_dspark_loss(
                outputs=output,
                loss_decay_gamma=4.0,
                ce_loss_alpha=0.1,
                l1_loss_alpha=0.9,
                confidence_head_alpha=1.0,
            )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertIsNone(model.embed_tokens.weight.grad)
        self.assertIsNone(model.lm_head.weight.grad)
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
        )

    def test_checkpoint_roundtrip_preserves_glm_heads_and_contract(self):
        model = Glm5NextDSparkModel(tiny_draft_config()).float()
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Glm5NextDSparkModel.from_pretrained(tmpdir)

        self.assertIsNotNone(loaded.markov_head)
        self.assertIsNotNone(loaded.confidence_head)
        self.assertTrue(loaded.confidence_head_with_markov)
        self.assertEqual(loaded.config.deepspec_target_execution, "full_model")
        self.assertEqual(loaded.config.deepspec_target_final_hidden_layer, 44)
        self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    @patch.dict(
        os.environ,
        {
            "DEEPSPEC_GLM_INDEX_QUERY_CHUNK": "3",
            "DEEPSPEC_GLM_ATTN_QUERY_CHUNK": "2",
        },
    )
    def test_bounded_target_sparse_attention_matches_transformers(self):
        torch.manual_seed(7)
        config = tiny_sparse_attention_config()
        original = Glm5NextTextAttention(config, 0).eval()
        bounded = copy.deepcopy(original)
        backbone = SimpleNamespace(
            config=config,
            layers=[SimpleNamespace(self_attn=bounded)],
        )
        install_glm5_next_bounded_target_prefill(backbone)

        hidden_states = torch.randn(1, 9, config.hidden_size)
        attention_mask = torch.ones(1, 9, dtype=torch.bool)
        with torch.no_grad():
            expected, _, expected_topk = original(
                hidden_states,
                attention_mask,
            )
        with patch.object(
            bounded,
            "build_attention_mask_from_topk",
            side_effect=AssertionError("dense sparse mask must not be built"),
        ), torch.no_grad():
            actual, _, actual_topk = bounded(
                hidden_states,
                attention_mask,
            )
        self.assertTrue(torch.equal(actual_topk, expected_topk))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

        padded_mask = attention_mask.clone()
        padded_mask[:, -1] = False
        with self.assertRaisesRegex(ValueError, "unpadded 2D mask"):
            bounded(hidden_states, padded_mask)

    def test_native_sparse_index_packing_preserves_causal_pool_tail(self):
        index_topk = 4
        index_kpool = 2
        indices = torch.full((1, 5, 5), -1, dtype=torch.int32)
        for position in range(5):
            full_pool_tokens = min(
                ((position + 1) // index_kpool) * index_kpool,
                index_topk,
            )
            tail_tokens = (position + 1) % index_kpool
            indices[0, position, :full_pool_tokens] = torch.arange(
                full_pool_tokens,
                dtype=torch.int32,
            )
            indices[
                0,
                position,
                index_topk : index_topk + tail_tokens,
            ] = torch.arange(
                full_pool_tokens,
                full_pool_tokens + tail_tokens,
                dtype=torch.int32,
            )

        packed, active_lengths = _pack_kpool_topk_for_native_kernel(
            indices,
            query_start=0,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )

        self.assertEqual(tuple(packed.shape), (1, 5, 8))
        self.assertTrue(
            torch.equal(
                active_lengths,
                torch.arange(1, 6, dtype=torch.int32),
            )
        )
        for position in range(5):
            active_length = int(active_lengths[position])
            self.assertTrue(
                torch.equal(
                    packed[0, position, :active_length],
                    torch.arange(active_length, dtype=torch.int32),
                )
            )
            self.assertTrue(packed[0, position, active_length:].eq(-1).all())

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and _flashinfer_supports_glm_nope(),
        "Native NoPE sparse MLA requires SM100/SM103 and FlashInfer >=0.6.18",
    )
    @patch.dict(
        os.environ,
        {"DEEPSPEC_GLM_NATIVE_ATTN_QUERY_CHUNK": "16"},
    )
    def test_native_nope_sparse_attention_matches_reference(self):
        torch.manual_seed(1234)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        sequence_length = 64
        num_heads = 16
        q_lora_rank = 32
        kv_lora_rank = 512
        head_dim = 256
        index_topk = 8
        index_kpool = 4
        attention = SimpleNamespace(
            num_heads=num_heads,
            qk_nope_head_dim=head_dim,
            qk_rope_head_dim=0,
            v_head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            scaling=head_dim**-0.5,
            config=SimpleNamespace(
                index_topk=index_topk,
                index_kpool=index_kpool,
            ),
            q_b_proj=torch.nn.Linear(
                q_lora_rank,
                num_heads * head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            kv_b_proj=torch.nn.Linear(
                kv_lora_rank,
                num_heads * 2 * head_dim,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            _deepspec_native_workspace_state={},
        )
        q_resid = torch.randn(
            1,
            sequence_length,
            q_lora_rank,
            device=device,
            dtype=dtype,
        )
        kv_pass = torch.randn(
            1,
            1,
            sequence_length,
            kv_lora_rank,
            device=device,
            dtype=dtype,
        )
        indices = torch.full(
            (1, sequence_length, index_topk + index_kpool - 1),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for position in range(sequence_length):
            full_pool_end = ((position + 1) // index_kpool) * index_kpool
            full_pool_tokens = min(full_pool_end, index_topk)
            tail_tokens = (position + 1) % index_kpool
            indices[0, position, :full_pool_tokens] = torch.arange(
                full_pool_end - full_pool_tokens,
                full_pool_end,
                device=device,
                dtype=torch.int32,
            )
            indices[
                0,
                position,
                index_topk : index_topk + tail_tokens,
            ] = torch.arange(
                full_pool_end,
                full_pool_end + tail_tokens,
                device=device,
                dtype=torch.int32,
            )

        with torch.no_grad():
            actual = _native_nope_sparse_attention(
                attention,
                q_resid=q_resid,
                kv_pass=kv_pass,
                topk_indices=indices,
            )
            query = attention.q_b_proj(q_resid).view(
                1,
                sequence_length,
                num_heads,
                head_dim,
            )
            expanded_kv = attention.kv_b_proj(
                kv_pass.transpose(1, 2)
            ).view(1, sequence_length, num_heads, 2 * head_dim)
            key, value = expanded_kv.split(head_dim, dim=-1)
            expected = torch.empty_like(actual)
            for position in range(sequence_length):
                selected = indices[0, position]
                selected = selected[selected.ge(0)].long()
                scores = torch.einsum(
                    "bhd,bthd->bht",
                    query[:, position],
                    key[:, selected],
                )
                scores = scores * attention.scaling
                probability = torch.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(dtype)
                expected[:, position] = torch.einsum(
                    "bht,bthd->bhd",
                    probability,
                    value[:, selected],
                )

        self.assertTrue(attention._deepspec_used_native_sparse_attention)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)

    @patch.dict(
        os.environ,
        {"DEEPSPEC_GLM_KDA_CHUNKS_PER_BATCH": "1"},
    )
    def test_bounded_target_linear_attention_matches_transformers(self):
        torch.manual_seed(11)
        config = tiny_target_config().text_config
        config.linear_num_heads = 4
        config.linear_head_dim = 8
        config.linear_conv_kernel_dim = 4
        config.linear_lower_bound = -5.0
        device = torch.device("cpu")
        original = Glm5NextTextLinearAttention(config, 0).to(device).eval()
        bounded = copy.deepcopy(original)
        backbone = SimpleNamespace(
            config=config,
            layers=[SimpleNamespace(self_attn=bounded)],
        )
        install_glm5_next_bounded_target_prefill(backbone)

        hidden_states = torch.randn(1, 73, config.hidden_size, device=device)
        attention_mask = torch.ones(1, 73, dtype=torch.bool, device=device)
        with patch(
            "transformers.models.glm5_next.modeling_glm5_next.causal_conv1d_fn",
            _causal_conv1d_prefill,
        ), patch(
            "transformers.models.glm5_next.modeling_glm5_next."
            "chunk_kimi_delta_attention",
            transformers_chunk_kimi_delta_attention.__wrapped__,
        ), torch.no_grad():
            expected = original(
                hidden_states,
                attention_mask=attention_mask,
            )
        with patch(
            "transformers.models.glm5_next.modeling_glm5_next."
            "chunk_kimi_delta_attention",
            side_effect=AssertionError("sequence-sized KDA path must not run"),
        ), torch.no_grad():
            actual = bounded(
                hidden_states,
                attention_mask=attention_mask,
            )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

        padded_mask = attention_mask.clone()
        padded_mask[:, -1] = False
        with self.assertRaisesRegex(ValueError, "unpadded 2D mask"):
            bounded(hidden_states, attention_mask=padded_mask)

    def test_glm_chat_template_marks_only_assistant_content(self):
        tokenizer = AutoTokenizer.from_pretrained(TARGET)
        processed = preprocess_record(
            record={
                "conversations": [
                    {"role": "user", "content": "first-user"},
                    {"role": "assistant", "content": "first-answer"},
                    {"role": "user", "content": "second-user"},
                    {"role": "assistant", "content": "second-answer"},
                ]
            },
            tokenizer=tokenizer,
            chat_template="glm5_next",
            max_length=256,
        )
        selected = tokenizer.decode(
            processed["input_ids"][processed["loss_mask"].bool()]
        )
        self.assertIn("first-answer", selected)
        self.assertIn("second-answer", selected)
        self.assertNotIn("first-user", selected)
        self.assertNotIn("second-user", selected)
        self.assertNotIn("<|user|>", selected)

    def test_glm_chat_template_excludes_tool_observations(self):
        tokenizer = AutoTokenizer.from_pretrained(TARGET)
        processed = preprocess_record(
            record={
                "conversations": [
                    {"role": "user", "content": "look this up"},
                    {
                        "role": "assistant",
                        "content": "calling-tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": {"query": "weather"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "SECRET_TOOL_OBSERVATION",
                    },
                    {"role": "assistant", "content": "final-answer"},
                ]
            },
            tokenizer=tokenizer,
            chat_template="glm5_next",
            max_length=512,
        )
        selected = tokenizer.decode(
            processed["input_ids"][processed["loss_mask"].bool()]
        )
        self.assertIn("calling-tool", selected)
        self.assertIn("<tool_call>", selected)
        self.assertIn("final-answer", selected)
        self.assertNotIn("SECRET_TOOL_OBSERVATION", selected)
        self.assertNotIn("<|observation|>", selected)

    def test_full_target_cache_contract_rejects_truncated_cache(self):
        draft = SimpleNamespace(
            target_layer_ids=[2, 22, 42],
            config=SimpleNamespace(
                hidden_size=64,
                deepspec_target_execution="full_model",
                deepspec_target_final_hidden_layer=44,
            ),
        )
        manifest = {
            "target_layer_ids": [2, 22, 42],
            "hidden_size": 64,
            "target_model_name_or_path": TARGET,
            "stores_target_last_hidden_states": True,
            "target_execution": "truncated_model",
        }
        dataset = SimpleNamespace(manifest=manifest)
        with self.assertRaisesRegex(AssertionError, "truncated teacher"):
            validate_train_cache(
                train_dataset=dataset,
                draft_model=draft,
                target_model_name_or_path=TARGET,
            )

        manifest.update(
            target_execution="full_model",
            target_execution_num_hidden_layers=45,
            target_final_hidden_layer=44,
            target_final_hidden_source="full_model_final_norm",
        )
        validate_train_cache(
            train_dataset=dataset,
            draft_model=draft,
            target_model_name_or_path=TARGET,
        )

    def test_online_target_keeps_all_layers_and_installs_bounded_prefill(self):
        class FakeBackbone(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(45)]
                )

        class FakeTarget(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.language_model = FakeBackbone(config.text_config)
                self.sentinel = torch.nn.Parameter(torch.ones(()))
                self.visual = torch.nn.Module()
                self.visual.rotary_pos_emb = torch.nn.Module()
                self.visual.rotary_pos_emb.theta = 10000.0
                self.visual.rotary_pos_emb.dim = 4
                self.visual.rotary_pos_emb.register_buffer(
                    "inv_freq",
                    torch.empty(2, device="meta"),
                    persistent=False,
                )

        target_config = AutoConfig.from_pretrained(TARGET)
        fake_target = FakeTarget(target_config)
        topology = SimpleNamespace(
            expert_parallel_size=8,
            tensor_parallel_size=4,
            context_parallel_size=1,
        )
        with (
            patch(
                "deepspec.modeling.target.online.AutoConfig.from_pretrained",
                return_value=target_config,
            ),
            patch(
                "deepspec.modeling.target.online.AutoModel.from_config",
                return_value=fake_target,
            ) as from_config,
            patch(
                "deepspec.modeling.target.online.AutoModel.from_pretrained"
            ) as from_pretrained,
            patch(
                "deepspec.modeling.target.online.parallelize_glm5_next_model"
            ) as parallelize,
            patch(
                "deepspec.modeling.target.online.install_glm5_next_bounded_target_prefill"
            ) as install_bounded,
            patch(
                "deepspec.modeling.target.online._wrap_target_model_with_fsdp",
                side_effect=lambda target_model, **_kwargs: target_model,
            ) as wrap_fsdp,
            patch(
                "deepspec.modeling.target.online.load_glm5_huggingface_checkpoint"
            ) as dcp_load,
        ):
            target = Glm5NextOnlineTarget(
                model_name_or_path=TARGET,
                target_layer_ids=[2, 22, 42],
                topology=topology,
                device=torch.device("cpu"),
                rank_local_cache_dir="unused",
            )

        loaded_config = from_config.call_args.args[0]
        self.assertEqual(loaded_config.text_config.num_hidden_layers, 45)
        self.assertEqual(loaded_config.vision_config.depth, 0)
        self.assertEqual(target.target_num_hidden_layers, 45)
        self.assertEqual(target.feature_output_device, torch.device("cpu"))
        self.assertFalse(fake_target.sentinel.requires_grad)
        parallelize.assert_called_once_with(
            fake_target.language_model,
            topology=topology,
            draft=False,
        )
        install_bounded.assert_called_once_with(fake_target)
        from_pretrained.assert_not_called()
        self.assertTrue(wrap_fsdp.call_args.kwargs["deferred_init"])
        dcp_load.assert_called_once_with(
            model=fake_target,
            checkpoint_dir=TARGET,
            config=loaded_config,
            topology=topology,
        )
        self.assertEqual(
            fake_target.visual.rotary_pos_emb.inv_freq.device,
            torch.device("cpu"),
        )

    def test_rank_local_loader_uses_pinned_transformers_distributed_api(self):
        class FakeExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = torch.nn.Parameter(torch.randn(4, 2, 2))
                self.down_proj = torch.nn.Parameter(torch.randn(4, 2, 2))
                self.num_experts = 4

        class FakeLayer(torch.nn.Module):
            def __init__(self, *, sparse):
                super().__init__()
                self.mlp = torch.nn.Module()
                if sparse:
                    self.mlp.experts = FakeExperts()

        class FakeTarget(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.layers = torch.nn.ModuleList(
                    [FakeLayer(sparse=False), FakeLayer(sparse=True)]
                )

        config = tiny_target_config()
        config.text_config.n_routed_experts = 8
        topology = SimpleNamespace(
            expert_parallel_size=2,
            expert_parallel_group=object(),
        )
        fake_target = FakeTarget()
        ep_mesh = object()
        with (
            patch(
                "deepspec.modeling.target.online.DeviceMesh.from_group",
                return_value=ep_mesh,
            ),
            patch(
                "deepspec.modeling.target.online._dequantizing_config",
                return_value=None,
            ),
            patch(
                "deepspec.modeling.target.online.AutoModel.from_pretrained",
                return_value=fake_target,
            ) as from_pretrained,
        ):
            loaded = _build_rank_local_ep_target_model(
                model_name_or_path="target",
                target_config=config,
                topology=topology,
            )

        self.assertIs(loaded, fake_target)
        load_kwargs = from_pretrained.call_args.kwargs
        self.assertNotIn("tp_plan", load_kwargs)
        self.assertIs(load_kwargs["device_mesh"], ep_mesh)
        distributed_config = load_kwargs["distributed_config"]
        self.assertEqual(distributed_config.tp_size, 2)
        self.assertEqual(distributed_config.tp_plan, "auto")
        self.assertTrue(distributed_config.enable_expert_parallel)
        self.assertEqual(
            config.text_config.base_model_ep_plan,
            {
                "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
                "layers.*.mlp.experts.down_proj": "grouped_gemm",
            },
        )
        experts = fake_target.language_model.layers[1].mlp.experts
        self.assertEqual(experts.num_experts, 4)
        self.assertTrue(experts._deepspec_expert_parameters_distributed)

    def test_online_target_rejects_unimplemented_parallel_axes_before_loading(self):
        base = {
            "expert_parallel_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        }
        for field, value, error_type, message in (
            (
                "expert_parallel_size",
                5,
                ValueError,
                "must divide n_routed_experts",
            ),
            (
                "context_parallel_size",
                2,
                NotImplementedError,
                "parallel.cp=1",
            ),
        ):
            values = dict(base)
            values[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                error_type, message
            ):
                Glm5NextOnlineTarget(
                    model_name_or_path=TARGET,
                    target_layer_ids=[2, 22, 42],
                    topology=SimpleNamespace(**values),
                    device=torch.device("cpu"),
                    rank_local_cache_dir="unused",
                )


if __name__ == "__main__":
    unittest.main()
