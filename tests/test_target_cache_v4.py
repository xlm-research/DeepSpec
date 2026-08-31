import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from deepspec.data.dataset_quality import (
    analyze_conversation_jsonl,
    dataset_quality_failures,
)
from scripts.data.prepare_target_cache import (
    _materialize_meta_module,
    _run_target_forward_context_parallel,
)

from deepspec.data.target_cache_dataset import (
    CacheCollator,
    CacheDataset,
    LocalTargetCacheWriter,
    build_source_jsonl_fingerprints,
    build_target_cache_manifest,
    expected_target_cache_tensor_nbytes,
    pack_index_record,
    unpack_index_record,
    validate_target_cache_identity,
    write_target_cache_manifest,
)
from deepspec.modeling.target.common import TargetForwardResult
from deepspec.utils import load_config
from scripts.data.prepare_target_cache import (
    _recv_linear_attention_cache_layer,
    _send_linear_attention_cache_layer,
)


class TargetCacheStateTransportTest(unittest.TestCase):
    def test_qwen35_dict_linear_cache_round_trip(self):
        messages = []
        cpu_group = object()
        gpu_group = object()

        def send(tensor, *, dst, group):
            messages.append((tensor.detach().clone(), dst, group))

        def recv(tensor, *, src, group):
            payload, dst, send_group = messages.pop(0)
            self.assertEqual(dst, src)
            self.assertIs(send_group, group)
            self.assertEqual(payload.dtype, tensor.dtype)
            self.assertEqual(tuple(payload.shape), tuple(tensor.shape))
            tensor.copy_(payload)

        class LinearAttentionLayer:
            def __init__(self, *, conv_states, recurrent_states):
                self.number_of_states = 2
                self.conv_states = conv_states
                self.recurrent_states = recurrent_states
                self.is_conv_states_initialized = {
                    index: state is not None
                    for index, state in conv_states.items()
                }
                self.is_recurrent_states_initialized = {
                    index: state is not None
                    for index, state in recurrent_states.items()
                }
                self.has_previous_state = {
                    index: (
                        conv_states[index] is not None
                        and recurrent_states[index] is not None
                    )
                    for index in conv_states
                }

            def lazy_initialization(
                self,
                *,
                conv_states=None,
                recurrent_states=None,
                state_idx=0,
            ):
                if conv_states is not None:
                    self.conv_states[state_idx] = torch.empty_like(conv_states)
                    self.is_conv_states_initialized[state_idx] = True
                if recurrent_states is not None:
                    self.recurrent_states[state_idx] = torch.empty_like(
                        recurrent_states
                    )
                    self.is_recurrent_states_initialized[state_idx] = True

        expected_conv = {
            0: torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            1: torch.arange(4, dtype=torch.float32).reshape(1, 4),
        }
        expected_recurrent = {
            0: torch.arange(8, dtype=torch.float16).reshape(2, 4),
            1: torch.arange(3, dtype=torch.bfloat16).reshape(1, 3),
        }
        send_cache = SimpleNamespace(
            layers=[
                LinearAttentionLayer(
                    conv_states=expected_conv,
                    recurrent_states=expected_recurrent,
                )
            ]
        )
        recv_layer = LinearAttentionLayer(
            conv_states={0: None, 1: None},
            recurrent_states={0: None, 1: None},
        )
        recv_cache = SimpleNamespace(layers=[recv_layer])
        with (
            mock.patch(
                "scripts.data.prepare_target_cache.dist.send",
                side_effect=send,
            ),
            mock.patch(
                "scripts.data.prepare_target_cache.dist.recv",
                side_effect=recv,
            ),
        ):
            _send_linear_attention_cache_layer(
                cache=send_cache,
                layer_idx=0,
                dst=7,
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )
            _recv_linear_attention_cache_layer(
                cache=recv_cache,
                layer_idx=0,
                src=7,
                device=torch.device("cpu"),
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )

        for state_idx in range(2):
            torch.testing.assert_close(
                recv_layer.conv_states[state_idx],
                expected_conv[state_idx],
            )
            torch.testing.assert_close(
                recv_layer.recurrent_states[state_idx],
                expected_recurrent[state_idx],
            )
        self.assertEqual(recv_layer.is_conv_states_initialized, {0: True, 1: True})
        self.assertEqual(
            recv_layer.is_recurrent_states_initialized,
            {0: True, 1: True},
        )
        self.assertEqual(recv_layer.has_previous_state, {0: True, 1: True})
        self.assertFalse(messages)

        missing_cache = SimpleNamespace(
            layers=[
                LinearAttentionLayer(
                    conv_states={0: expected_conv[0], 1: None},
                    recurrent_states=expected_recurrent,
                )
            ]
        )
        with (
            mock.patch(
                "scripts.data.prepare_target_cache.dist.send",
                side_effect=send,
            ),
            mock.patch(
                "scripts.data.prepare_target_cache.dist.recv",
                side_effect=recv,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Missing .*linear-attention state",
            ):
                _send_linear_attention_cache_layer(
                    cache=missing_cache,
                    layer_idx=0,
                    dst=7,
                    gpu_group=gpu_group,
                    cpu_group=cpu_group,
                )
        self.assertFalse(messages)


class TargetCacheV4Test(unittest.TestCase):
    def _build_cache_without_final_hidden(self, root: Path):
        source_path = root / "source.jsonl"
        source_path.write_text(
            json.dumps({"conversations": [{"role": "user", "content": "x"}]})
            + "\n",
            encoding="utf-8",
        )
        writer = LocalTargetCacheWriter(
            rank_dir=str(root),
            max_shard_bytes=1024 * 1024,
        )
        writer.write_sample(
            sample_id=0,
            context_start=0,
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.ones(3),
            loss_mask=torch.tensor([0, 1, 1]),
            target_hidden_states=torch.arange(12).reshape(3, 4),
            target_last_hidden_states=None,
        )
        writer.close()
        manifest = build_target_cache_manifest(
            num_samples=1,
            shards=[{"shard_id": 0, "file_name": "shard-local-00000.bin"}],
            target_layer_ids=[1, 3],
            hidden_size=2,
            extra_fields={
                "target_model_name_or_path": "/models/target",
                "source_jsonl_paths": [str(source_path)],
                "source_jsonl_fingerprints": build_source_jsonl_fingerprints(
                    [source_path]
                ),
                "chat_template": "qwen",
                "max_length": 3,
                "cache_context_parallel_size": 1,
                "context_layout": "contiguous",
                "index_files": ["samples.local.idx"],
                "stores_target_last_hidden_states": False,
            },
        )
        write_target_cache_manifest(output_dir=str(root), manifest=manifest)
        return source_path

    def test_round_trip_without_final_hidden_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._build_cache_without_final_hidden(root)
            dataset = CacheDataset(str(root))
            sample = dataset[0]
            self.assertNotIn("target_last_hidden_states", sample)
            self.assertEqual(tuple(sample["target_hidden_states"].shape), (3, 4))
            batch = CacheCollator()([sample])
            self.assertNotIn("target_last_hidden_states", batch)
            dataset.close()

            nbytes = expected_target_cache_tensor_nbytes(
                seq_len=3,
                context_len=3,
                hidden_size=2,
                num_target_layers=2,
                stores_target_last_hidden_states=False,
            )
            self.assertEqual(nbytes["target_last_hidden_states"], 0)

    def test_round_trip_with_final_hidden_state_for_dspark(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            writer = LocalTargetCacheWriter(
                rank_dir=str(root),
                max_shard_bytes=1024 * 1024,
            )
            expected_last_hidden = torch.arange(6).reshape(3, 2)
            writer.write_sample(
                sample_id=0,
                context_start=0,
                input_ids=torch.tensor([1, 2, 3]),
                attention_mask=torch.ones(3),
                loss_mask=torch.tensor([0, 1, 1]),
                target_hidden_states=torch.arange(12).reshape(3, 4),
                target_last_hidden_states=expected_last_hidden,
            )
            writer.close()
            manifest = build_target_cache_manifest(
                num_samples=1,
                shards=[
                    {"shard_id": 0, "file_name": "shard-local-00000.bin"}
                ],
                target_layer_ids=[1, 3],
                hidden_size=2,
                extra_fields={
                    "target_model_name_or_path": "/models/target",
                    "source_jsonl_paths": ["source.jsonl"],
                    "source_jsonl_fingerprints": [
                        {
                            "path": "source.jsonl",
                            "size": 0,
                            "sha256": "0" * 64,
                        }
                    ],
                    "chat_template": "deepseek_v4",
                    "max_length": 3,
                    "cache_context_parallel_size": 1,
                    "context_layout": "contiguous",
                    "index_files": ["samples.local.idx"],
                    "stores_target_last_hidden_states": True,
                },
            )
            write_target_cache_manifest(output_dir=str(root), manifest=manifest)

            dataset = CacheDataset(str(root))
            sample = dataset[0]
            torch.testing.assert_close(
                sample["target_last_hidden_states"],
                expected_last_hidden.to(torch.bfloat16),
            )
            batch = CacheCollator()([sample])
            self.assertEqual(
                tuple(batch["target_last_hidden_states"].shape),
                (1, 3, 2),
            )
            dataset.close()

            nbytes = expected_target_cache_tensor_nbytes(
                seq_len=3,
                context_len=3,
                hidden_size=2,
                num_target_layers=2,
                stores_target_last_hidden_states=True,
            )
            self.assertEqual(nbytes["target_last_hidden_states"], 12)

    def test_source_content_fingerprint_rejects_stale_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = self._build_cache_without_final_hidden(root)
            kwargs = dict(
                cache_dir=str(root),
                source_jsonl_paths=[source_path],
                target_model_name_or_path="/models/target",
                target_layer_ids=[1, 3],
                chat_template="qwen",
                max_length=3,
                context_parallel_size=1,
                stores_target_last_hidden_states=False,
            )
            validate_target_cache_identity(**kwargs)
            with self.assertRaisesRegex(
                AssertionError,
                "Final-hidden-state storage mode does not match",
            ):
                validate_target_cache_identity(
                    **{**kwargs, "stores_target_last_hidden_states": True}
                )
            source_path.write_text(
                json.dumps(
                    {"conversations": [{"role": "user", "content": "changed"}]}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "does not match"):
                validate_target_cache_identity(**kwargs)

    def test_contiguous_cp_cache_exposes_deepseek_partition_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard_metadata = []
            index_files = []
            for cp_rank, length in enumerate((3, 2)):
                rank_dir = root / f"rank-{cp_rank}"
                rank_dir.mkdir()
                writer = LocalTargetCacheWriter(
                    rank_dir=str(rank_dir),
                    max_shard_bytes=1024 * 1024,
                )
                writer.write_sample(
                    sample_id=0,
                    context_start=0 if cp_rank == 0 else 3,
                    input_ids=torch.arange(5),
                    attention_mask=torch.ones(5),
                    loss_mask=torch.ones(5),
                    target_hidden_states=torch.full((length, 2), cp_rank),
                    target_last_hidden_states=None,
                )
                writer.close()
                shard_name = f"shard-{cp_rank:05d}.bin"
                (rank_dir / "shard-local-00000.bin").replace(root / shard_name)
                index_name = f"samples.cp{cp_rank:03d}.idx"
                index_bytes = (rank_dir / "samples.local.idx").read_bytes()
                record = unpack_index_record(index_bytes)
                record["shard_id"] = cp_rank
                (root / index_name).write_bytes(pack_index_record(**record))
                shard_metadata.append(
                    {"shard_id": cp_rank, "file_name": shard_name}
                )
                index_files.append(index_name)

            manifest = build_target_cache_manifest(
                num_samples=1,
                shards=shard_metadata,
                target_layer_ids=[0],
                hidden_size=2,
                extra_fields={
                    "target_model_name_or_path": "/models/deepseek-v4",
                    "source_jsonl_paths": ["source.jsonl"],
                    "source_jsonl_fingerprints": [
                        {"path": "source.jsonl", "size": 0, "sha256": "0" * 64}
                    ],
                    "chat_template": "deepseek_v4",
                    "max_length": 5,
                    "cache_context_parallel_size": 2,
                    "context_layout": "contiguous",
                    "index_files": index_files,
                    "stores_target_last_hidden_states": False,
                },
            )
            write_target_cache_manifest(output_dir=str(root), manifest=manifest)

            dataset = CacheDataset(
                str(root),
                context_parallel_size=2,
                context_parallel_rank=1,
                expected_context_layout="contiguous",
            )
            sample = dataset[0]
            self.assertEqual(sample["context_start"], 3)
            self.assertEqual(sample["context_len"], 2)
            batch = CacheCollator()([sample])
            self.assertEqual(batch["context_start"].tolist(), [3])
            self.assertEqual(batch["context_len"].tolist(), [2])
            dataset.close()

    def test_deepseek_training_configs_select_expected_target_mode(self):
        dspark = load_config("config/dspark/dspark_deepseek_v4.py")
        self.assertTrue(dspark.data.online_target)
        self.assertTrue(dspark.data.train_data_path)
        self.assertIsNone(dspark.data.target_cache_path)

        for path, stores_last_hidden in (
            ("config/dflash/dflash_deepseek_v4.py", False),
            ("config/dflash2/dflash2_deepseek_v4.py", False),
        ):
            config = load_config(path)
            self.assertFalse(config.data.online_target, path)
            self.assertEqual(
                config.data.store_target_last_hidden_states,
                stores_last_hidden,
                path,
            )
            self.assertTrue(config.data.source_jsonl_path, path)


class TargetCacheMaterializationTest(unittest.TestCase):
    def test_fsdp_style_meta_parameters_are_materialized(self):
        module = nn.Sequential(nn.Linear(2, 2, device="meta"))
        for submodule in module.modules():
            _materialize_meta_module(submodule, device=torch.device("cpu"))
        self.assertFalse(any(param.is_meta for param in module.parameters()))

    def test_cp_validation_keeps_pre_forward_global_length(self):
        def forward_context_parallel(*, model_inputs, **_kwargs):
            attention_mask = model_inputs["attention_mask"]
            attention_mask.set_(attention_mask[:, :2].clone())
            hidden = torch.zeros(1, 2, 4)
            return TargetForwardResult(
                target_hidden_states=hidden,
                target_last_hidden_states=hidden,
            )

        target_model = SimpleNamespace(
            _deepspec_context_layout="native_head_tail",
            forward_context_parallel=forward_context_parallel,
        )
        topology = SimpleNamespace(
            context_parallel_group=None,
            context_parallel_rank=0,
            context_parallel_size=2,
            tensor_parallel_group=None,
            tensor_parallel_rank=0,
            tensor_parallel_size=1,
        )
        model_inputs = {"attention_mask": torch.ones(1, 4)}
        result = _run_target_forward_context_parallel(
            target_model=target_model,
            model_inputs=model_inputs,
            target_layer_ids=[0],
            topology=topology,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(result.target_hidden_states.shape), (1, 2, 4))
        self.assertEqual(tuple(model_inputs["attention_mask"].shape), (1, 4))


class DatasetQualityTest(unittest.TestCase):
    def test_mechanical_repeats_fail_production_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data.jsonl"
            rows = [
                {"conversations": [{"role": "user", "content": "a"}]},
                {"conversations": [{"role": "user", "content": "b"}]},
            ]
            with path.open("w", encoding="utf-8") as handle:
                for _ in range(3):
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
            summary = analyze_conversation_jsonl(path)
            self.assertEqual(summary["total_records"], 6)
            self.assertEqual(summary["unique_records"], 2)
            self.assertEqual(summary["max_repeat_count"], 3)
            self.assertTrue(
                dataset_quality_failures(
                    summary,
                    min_unique_records=4,
                    min_unique_ratio=0.5,
                )
            )


if __name__ == "__main__":
    unittest.main()
