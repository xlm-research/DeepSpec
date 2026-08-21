import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

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
    validate_target_cache_identity,
    write_target_cache_manifest,
)
from deepspec.modeling.target.common import TargetForwardResult


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
            source_path.write_text(
                json.dumps(
                    {"conversations": [{"role": "user", "content": "changed"}]}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "does not match"):
                validate_target_cache_identity(**kwargs)


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
