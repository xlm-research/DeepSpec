"""Validate native preparation and producer shard reconstruction with real inputs."""

import json
import copy
import os
from pathlib import Path
import tempfile
import unittest
import pickle

import torch
from transformers import AutoTokenizer

from deepspec.data.parser import preprocess_record as retained_preprocess
from deepspec.trainer.qwen3_8_vllm import convert_hidden_states
from torchtitan.models.dspark_draft.features import ProducerFeatures, file_digest
from torchtitan.models.dspark_draft.planning import input_identity


class NativeInputsTest(unittest.TestCase):
    def test_invalid_producer_shapes_dtypes_shards_and_supervision(self):
        fixture = torch.load(
            Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"]) / "torch.bfloat16_rank0.pt",
            weights_only=True,
        )
        batch = fixture["features"][0]
        actual = {**batch, "context_chunk_len": torch.tensor([16])}
        identity = input_identity(batch)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_path = root / "features.pt"
            manifest_path = root / "producer.json"
            for change in (
                "layers",
                "declared-dtype",
                "hidden-dtype",
                "hidden-shape",
                "tokens",
                "mask",
                "length",
                "missing-shard",
                "partial-file",
            ):
                with self.subTest(change=change):
                    features = copy.deepcopy(actual)
                    if change == "hidden-dtype":
                        features["target_hidden_states"] = features[
                            "target_hidden_states"
                        ].float()
                    elif change == "hidden-shape":
                        features["target_last_hidden_states"] = features[
                            "target_last_hidden_states"
                        ][..., :-1]
                    elif change == "tokens":
                        features["input_ids"][0, 0] += 1
                    elif change == "mask":
                        features["loss_mask"][0, 0] = False
                    elif change == "length":
                        features["seq_len"] = torch.tensor([15])
                    torch.save(features, feature_path)
                    if change == "partial-file":
                        feature_path.write_bytes(b"partial feature write")
                    manifest = {
                        "version": 1,
                        "teacher": {
                            "target_layer_ids": [3, 1]
                            if change == "layers"
                            else [1, 3],
                            "hidden_size": 64,
                            "activation_dtype": "float32"
                            if change == "declared-dtype"
                            else "bfloat16",
                            "target_final_hidden_source": "full_model_final_norm_output",
                        },
                        "samples": [
                            {
                                "sample_id": "archived-0",
                                "position": 0,
                                "input_identity": identity,
                                "length": 16,
                                "shards": [
                                    {
                                        "cp_rank": 0,
                                        "path": str(feature_path),
                                        "sha256": file_digest(feature_path),
                                    }
                                ],
                            }
                        ],
                    }
                    manifest_path.write_text(json.dumps(manifest))
                    if change == "missing-shard":
                        feature_path.unlink()
                    with self.assertRaises(
                        (
                            ValueError,
                            FileNotFoundError,
                            pickle.UnpicklingError,
                            EOFError,
                        )
                    ):
                        reader = ProducerFeatures(
                            manifest_path,
                            file_digest(manifest_path),
                            layer_ids=[1, 3],
                            hidden_size=64,
                            vocab_size=128,
                        )
                        reader.read("archived-0", identity)

    def test_preparation_preserves_retained_tokens_and_epoch_order(self):
        root = Path(os.environ["DEEPSPEC_PREPARED_INPUTS"])
        plan = json.loads((root / "input-plan.json").read_text())
        records = [
            json.loads(line)
            for line in Path(os.environ["DEEPSPEC_LIVE_DATA"]).read_text().splitlines()
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B", local_files_only=True
        )
        order = torch.randperm(
            len(records), generator=torch.Generator().manual_seed(42)
        )[:8].tolist()
        self.assertEqual(plan["samples_per_epoch"], 8)
        self.assertEqual(plan["global_batch_size"], 4)
        self.assertEqual(plan["gradient_accumulation_steps"], 2)
        self.assertEqual(len(plan["batches"]), 8)
        for position, entry in enumerate(plan["batches"]):
            self.assertEqual(entry["sample_id"], f"epoch-0/sample-{order[position]}")
            actual = torch.load(entry["input_path"], weights_only=True)
            expected = retained_preprocess(
                records[order[position]], tokenizer, "qwen", 256
            )
            for key in ("input_ids", "loss_mask"):
                self.assertTrue(torch.equal(actual[key], expected[key].unsqueeze(0)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_and_head_tail_features_preserve_archived_inputs(self):
        reference = Path(os.environ["DEEPSPEC_BASELINE_REFERENCE"])
        fixture = torch.load(reference / "torch.bfloat16_rank0.pt", weights_only=True)
        batch = fixture["features"][0]
        ids = batch["input_ids"]
        hidden = torch.cat(
            (
                batch["target_hidden_states"].reshape(1, ids.shape[1], 2, 64),
                batch["target_last_hidden_states"].unsqueeze(2),
            ),
            dim=2,
        )[0].bfloat16()
        raw = {"hidden_states": hidden, "token_ids": ids[0]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = None
            for size in (1, 2, 3):
                shards = []
                for rank in range(size):
                    path = root / f"size{size}-rank{rank}.pt"
                    converted = convert_hidden_states(
                        raw,
                        batch,
                        hidden_size=64,
                        num_layers=2,
                        cp_size=size,
                        cp_rank=rank,
                    )
                    torch.save(converted, path)
                    shards.append(
                        {
                            "cp_rank": rank,
                            "path": str(path),
                            "sha256": file_digest(path),
                        }
                    )
                manifest = {
                    "version": 1,
                    "teacher": {
                        "target_layer_ids": [1, 3],
                        "hidden_size": 64,
                        "activation_dtype": "bfloat16",
                        "target_final_hidden_source": "full_model_final_norm_output",
                    },
                    "samples": [
                        {
                            "sample_id": "archived-0",
                            "position": 0,
                            "input_identity": input_identity(batch),
                            "length": ids.shape[1],
                            "shards": shards,
                        }
                    ],
                }
                path = root / "producer.json"
                path.write_text(json.dumps(manifest))
                reader = ProducerFeatures(
                    path,
                    file_digest(path),
                    layer_ids=[1, 3],
                    hidden_size=64,
                    vocab_size=128,
                )
                actual = reader.read("archived-0", input_identity(batch))
                if expected is None:
                    expected = actual
                for key in expected:
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                with self.assertRaisesRegex(ValueError, "native input plan"):
                    reader.read("archived-0", "wrong-input")
                manifest["teacher"]["target_final_hidden_source"] = "decoder_output"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "target_final_hidden_source"):
                    ProducerFeatures(
                        path,
                        file_digest(path),
                        layer_ids=[1, 3],
                        hidden_size=64,
                        vocab_size=128,
                    )
                with open(shards[0]["path"], "ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "bytes have changed"):
                    reader.read("archived-0", input_identity(batch))


if __name__ == "__main__":
    unittest.main()
