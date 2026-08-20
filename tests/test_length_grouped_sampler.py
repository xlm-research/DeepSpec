import json

import torch

from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils.distributed import StatelessResumableDistributedSampler


class _LengthDataset:
    def __init__(self, lengths):
        self.lengths = list(lengths)

    def __len__(self):
        return len(self.lengths)

    def get_length_hint(self, index):
        return self.lengths[index]


def _global_groups(samplers):
    return list(zip(*(list(sampler) for sampler in samplers)))


def _straggler_cost(groups, lengths):
    return sum(max(lengths[index] for index in group) for group in groups)


def test_length_grouping_preserves_samples_and_reduces_straggler_cost():
    lengths = [1 + ((index * 37) % 1000) for index in range(96)]
    dataset = _LengthDataset(lengths)
    common = dict(
        dataset=dataset,
        num_replicas=4,
        total_size=96,
        seed=17,
    )
    plain = [
        StatelessResumableDistributedSampler(rank=rank, **common)
        for rank in range(4)
    ]
    grouped = [
        StatelessResumableDistributedSampler(
            rank=rank,
            length_fn=dataset.get_length_hint,
            length_bucket_size=24,
            **common,
        )
        for rank in range(4)
    ]

    plain_groups = _global_groups(plain)
    grouped_groups = _global_groups(grouped)
    assert sorted(index for group in grouped_groups for index in group) == list(range(96))
    assert _straggler_cost(grouped_groups, lengths) <= _straggler_cost(
        plain_groups, lengths
    )
    assert grouped_groups == _global_groups(
        [
            StatelessResumableDistributedSampler(
                rank=rank,
                length_fn=dataset.get_length_hint,
                length_bucket_size=24,
                **common,
            )
            for rank in range(4)
        ]
    )


def test_length_grouped_sampler_resume_matches_stream_suffix():
    dataset = _LengthDataset(range(40))
    common = dict(
        dataset=dataset,
        num_replicas=4,
        total_size=40,
        seed=3,
        length_fn=dataset.get_length_hint,
        length_bucket_size=20,
    )
    full = StatelessResumableDistributedSampler(rank=2, **common)
    resumed = StatelessResumableDistributedSampler(
        rank=2,
        start_global_offset_samples=6,
        num_samples=4,
        **common,
    )
    assert list(resumed) == list(full)[6:10]


def test_jsonl_dataset_length_hint_uses_record_bytes(tmp_path):
    path = tmp_path / "samples.jsonl"
    records = [
        {"conversations": [{"role": "user", "content": "a"}]},
        {"conversations": [{"role": "user", "content": "longer text"}]},
    ]
    encoded_lines = [
        (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    ]
    path.write_bytes(b"".join(encoded_lines))

    dataset = JsonLineDataset([path])

    assert dataset.get_length_hint(0) == len(encoded_lines[0])
    assert dataset.get_length_hint(1) == len(encoded_lines[1])
