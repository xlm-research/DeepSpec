"""Read immutable producer features through the draft input index."""

from pathlib import Path
import tempfile
import unittest

import torch

from deepspec.data.draft_feature_reader import DraftFeatureIndex, feature_input_identity


class DraftFeatureReaderTest(unittest.TestCase):
    def test_producer_shards_preserve_tokens_features_and_resume_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = torch.arange(7).unsqueeze(0)
            hidden = torch.arange(42).reshape(1, 7, 6).float()
            final = torch.arange(21).reshape(1, 7, 3).float()
            common = {
                "input_ids": ids,
                "loss_mask": ids.remainder(2).bool(),
                "seq_len": torch.tensor([7]),
            }
            shards = []
            # Existing producer CP2 uses head/tail shards: [0,1,6,pad], [2,3,4,5].
            for rank, positions in enumerate(([0, 1, 6, 7], [2, 3, 4, 5])):
                positions = torch.tensor(positions)
                batch = dict(common, context_chunk_len=torch.tensor([4]))
                for name, full in (
                    ("target_hidden_states", hidden),
                    ("target_last_hidden_states", final),
                ):
                    batch[name] = full[:, positions.clamp_max(6)].clone()
                    batch[name][:, positions == 7] = 0
                path = root / f"producer{rank}.pt"
                torch.save(batch, path)
                shards.append({"path": str(path), "owner": 6 + rank, "cp_rank": rank})
            identity = {"teacher": "fixed-teacher", "layout": {"cp": 2}}
            index = DraftFeatureIndex.create(
                samples=[
                    {
                        "position": 0,
                        "sample_id": "epoch0:sample5",
                        "epoch": 0,
                        "shards": shards,
                    }
                ],
                producer_identity=identity,
                partition_id=1,
                start_micro_step=0,
                data_parallel_size=1,
                gradient_accumulation_steps=2,
                samples_per_epoch=8,
            )
            manifest = root / "draft-index.json"
            index.save(manifest)
            restored = DraftFeatureIndex.load(
                manifest,
                producer_identity=identity,
                next_micro_step=0,
                data_parallel_size=1,
                gradient_accumulation_steps=2,
            )
            batch = restored.read(micro_step=0, data_parallel_rank=0)
            for name, expected in dict(
                common, target_hidden_states=hidden, target_last_hidden_states=final
            ).items():
                torch.testing.assert_close(batch[name], expected, rtol=0, atol=0)
            self.assertEqual(batch["context_chunk_len"].item(), 7)
            for rank in range(2):
                batch = restored.read(
                    micro_step=0,
                    data_parallel_rank=0,
                    context_parallel_size=2,
                    context_parallel_rank=rank,
                )
                expected = torch.load(shards[rank]["path"], weights_only=True)
                for name, value in expected.items():
                    torch.testing.assert_close(batch[name], value, rtol=0, atol=0)

            self.assertEqual(restored.owned_paths(0), ())
            self.assertEqual(restored.owned_paths(7), (shards[1]["path"],))
            for override in (
                {
                    "producer_identity": {
                        "teacher": "another-teacher",
                        "layout": {"cp": 2},
                    }
                },
                {"next_micro_step": 2},
                {"data_parallel_size": 2},
                {"gradient_accumulation_steps": 1},
            ):
                options = dict(
                    producer_identity=identity,
                    next_micro_step=0,
                    data_parallel_size=1,
                    gradient_accumulation_steps=2,
                )
                options.update(override)
                with self.subTest(override=override), self.assertRaises(ValueError):
                    DraftFeatureIndex.load(manifest, **options)
            (root / "producer1.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                restored.read(micro_step=0, data_parallel_rank=0)

    def test_wrong_sample_is_rejected_even_before_the_file_was_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "producer.pt"
            expected_input = {
                "input_ids": torch.tensor([[1, 2]]),
                "loss_mask": torch.tensor([[False, True]]),
            }
            actual = dict(
                expected_input,
                input_ids=torch.tensor([[3, 4]]),
                target_hidden_states=torch.zeros(1, 2, 6),
                target_last_hidden_states=torch.zeros(1, 2, 3),
                seq_len=torch.tensor([2]),
                context_chunk_len=torch.tensor([2]),
            )
            torch.save(actual, path)
            index = DraftFeatureIndex.create(
                samples=[
                    {
                        "position": 0,
                        "sample_id": "0:5",
                        "epoch": 0,
                        "input_identity": feature_input_identity(expected_input),
                        "shards": [{"path": str(path), "owner": 0, "cp_rank": 0}],
                    }
                ],
                producer_identity={"teacher": "fixed", "layout": {"cp": 1}},
                partition_id=1,
                start_micro_step=0,
                data_parallel_size=1,
                gradient_accumulation_steps=2,
                samples_per_epoch=4,
            )
            with self.assertRaisesRegex(ValueError, "input identity"):
                index.read(micro_step=0, data_parallel_rank=0)
            # Replacing a file after indexing must also fail its checksum check.
            torch.save(dict(actual, **expected_input), path)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                index.read(micro_step=0, data_parallel_rank=0)

    def test_missing_or_reused_producer_shards_cannot_change_token_order(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = []
            for rank, value in enumerate((10.0, 11.0)):
                path = Path(directory) / f"cp{rank}.pt"
                torch.save(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "loss_mask": torch.tensor([[False, True, True]]),
                        "target_hidden_states": torch.tensor([[[value], [0.0]]]),
                        "target_last_hidden_states": torch.tensor([[[value], [0.0]]]),
                        "seq_len": torch.tensor([3]),
                        "context_chunk_len": torch.tensor([2]),
                    },
                    path,
                )
                shards.append({"path": str(path), "owner": rank, "cp_rank": rank})
            # CP4 and CP2 both have two slots for this length. Shape checks alone
            # cannot see missing shards; the fixed producer CP degree must rule.
            for cp_size, records in (
                (4, shards),
                (2, [shards[0], dict(shards[0], cp_rank=1)]),
            ):
                with self.subTest(cp_size=cp_size), self.assertRaises(ValueError):
                    DraftFeatureIndex.create(
                        samples=[
                            {
                                "position": 0,
                                "sample_id": "0:0",
                                "epoch": 0,
                                "shards": records,
                            }
                        ],
                        producer_identity={
                            "teacher": "fixed",
                            "layout": {"cp": cp_size},
                        },
                        partition_id=1,
                        start_micro_step=0,
                        data_parallel_size=1,
                        gradient_accumulation_steps=2,
                        samples_per_epoch=4,
                    )


if __name__ == "__main__":
    unittest.main()
