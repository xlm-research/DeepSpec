import unittest
from types import SimpleNamespace

from deepspec.distributed.config import ParallelConfig


class ParallelConfigTest(unittest.TestCase):
    def test_degrees_and_sparse_view(self):
        config = ParallelConfig(dp_shard=2, cp=2, tp=2, ep=4)
        config.validate_world_size(8)
        self.assertEqual(config.expert_fsdp, 2)
        self.assertEqual(config.fsdp_shard_size, 4)
        self.assertEqual(config.data_parallel_size, 2)

    def test_world_size_mismatch_fails_early(self):
        with self.assertRaisesRegex(ValueError, "world_size does not match"):
            ParallelConfig(dp_shard=2, tp=2).validate_world_size(8)

    def test_tp_model_constraints(self):
        config = ParallelConfig(tp=4, use_fsdp=False)
        config.validate_world_size(4)
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            config.validate_model(
                SimpleNamespace(
                    hidden_size=32,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                )
            )

    def test_dense_model_rejects_ep(self):
        config = ParallelConfig(dp_shard=2, ep=2)
        config.validate_world_size(2)
        with self.assertRaisesRegex(ValueError, "no MoE"):
            config.validate_model(
                SimpleNamespace(
                    hidden_size=32,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                )
            )

    def test_dynamic_cp_is_not_a_fake_toggle(self):
        with self.assertRaisesRegex(NotImplementedError, "micro-batch scheduler"):
            ParallelConfig(dynamic_context_parallel=True).validate_world_size(1)

    def test_deepep_requires_supported_training_layout(self):
        ParallelConfig(dp_shard=2, ep=2, expert_dispatch_backend="deepep").validate_world_size(2)
        for options in (
            {"ep": 1},
            {"tp": 2, "dp_shard": 1},
            {"cp": 2, "dp_shard": 1},
            {"use_compile": True},
            {"use_activation_checkpoint": True},
            {"expert_dispatch_max_tokens_per_rank": 0},
        ):
            with self.subTest(options=options):
                values = dict(dp_shard=2, ep=2, expert_dispatch_backend="deepep")
                values.update(options)
                with self.assertRaises((ValueError, NotImplementedError)):
                    ParallelConfig(**values).validate_world_size(2)

    def test_deepep_rejects_models_without_the_training_adapter(self):
        config = ParallelConfig(dp_shard=2, ep=2, expert_dispatch_backend="deepep")
        config.validate_model(SimpleNamespace(model_type="glm5_next_text", n_routed_experts=8))
        with self.assertRaisesRegex(NotImplementedError, "GLM-5.3"):
            config.validate_model(SimpleNamespace(model_type="deepseek_v4", n_routed_experts=8))

    def test_legacy_config_maps_without_changing_world_layout(self):
        config = ParallelConfig.from_mapping(
            {
                "context_parallel_size": 2,
                "fsdp_size": 4,
                "torch_compile": False,
                "sharding_strategy": "full_shard",
            },
            world_size=8,
        )
        self.assertEqual((config.dp_shard, config.cp), (4, 2))
        self.assertEqual(config.context_parallel_backend, "model_native")


if __name__ == "__main__":
    unittest.main()
