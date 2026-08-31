# DSpark evaluation and confidence scheduling

The fixed-block setting used for paper-style offline draft-quality evaluation is:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --target_name_or_path TARGET \
  --draft_name_or_path DRAFT \
  --temperature 1 \
  --top-k 0 \
  --top-p 1 \
  --confidence-threshold 0
```

`acceptance_length` is the mean number of newly committed tokens per target
verification, including the target bonus token. Position metrics are deliberately
separate:

- `conditional_accept_rates_by_position[k]` is the paper's step-conditional rate.
- `prefix_accept_rates_by_position[k]` is prefix survival through position `k`.
- `full_block_acceptance_rate` is computed only over proposals scheduled at the
  configured maximum length.

## Fit Sequential Temperature Scaling

Collect raw confidence observations on a held-out set with scheduling disabled:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --target_name_or_path TARGET \
  --draft_name_or_path DRAFT \
  --task alpaca:500 \
  --confidence-threshold 0 \
  --confidence-observations-jsonl output/sts-observations.jsonl
```

Distributed evaluation writes one rank-suffixed JSONL file per worker. Pass every
file to the fitter with a repeated `--observations-jsonl` argument:

```bash
python scripts/eval/fit_dspark_sts.py \
  --observations-jsonl output/sts-observations.jsonl \
  --output output/sts-calibration.json
```

Calibration artifacts are tied to the target, draft, block size, temperature,
top-k and top-p. Evaluation rejects a mismatched artifact instead of silently
using uncalibrated confidence values.

## Hardware-aware prefix scheduling

Provide an engine profile covering every integer token batch size required by the
active request batch. The current per-sample evaluator has one active request, so
a block size of 7 requires entries 1 through 8:

```json
{
  "schema_version": 1,
  "steps_per_second": {
    "1": 100.0,
    "2": 99.0,
    "3": 97.0,
    "4": 94.0,
    "5": 90.0,
    "6": 85.0,
    "7": 79.0,
    "8": 72.0
  }
}
```

Then enable the paper's Algorithm 1 scheduler:

```bash
python eval.py \
  --target_name_or_path TARGET \
  --draft_name_or_path DRAFT \
  --scheduler-mode hardware-aware \
  --confidence-calibration-json output/sts-calibration.json \
  --sps-profile-json output/sps-profile.json
```

The reusable scheduler accepts a batch-shaped confidence tensor `[requests,
block_size]`; a serving engine can call `HardwareAwarePrefixScheduler.schedule`
once for all active requests. The repository evaluator remains batch size 1 and
therefore exercises the same algorithm for a single active request.
