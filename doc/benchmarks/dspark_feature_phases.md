# Qwen producer bindings and draft feature consumption

The Qwen vLLM trainer now plans target requests from the existing deterministic
sample stream using the fixed target mesh. Draft DP/CP/TP degrees select a view
of indexed producer files. The vLLM worker, inference kernels, feature tensors,
DSpark objective, and optimizer implementation remain the existing versions.

Source `env.sh` before launching. It binds `VLLM_PYTHON_BIN` to the existing
`deepspec_vllm_torchtitan_envs` interpreter and `VLLM_SOURCE_DIR` to the local
source-built checkout. Explicit producer bindings take precedence. The vLLM
launcher preserves the established eight-GPU-node producer defaults:
`TARGET_FSDP_SIZE=2`, `TARGET_CONTEXT_PARALLEL_SIZE=1`, and
`TARGET_TENSOR_PARALLEL_SIZE=4`; producer DP replication covers the remaining
world size. Draft options use the existing separate names. A different producer
layout must be configured explicitly with the `TARGET_*` variables.

`qwen38_vllm_producer.json` records the target identity, inference configuration,
resolved interpreter and source directory, owner/device mapping, and dataset
sample plan. A changed binding is rejected when reusing the checkpoint root.
The source-built vLLM and its PyTorch/CUDA stack were not installed or rebuilt.

Each draft phase writes `draft_feature_indexes/micro_<cursor>.json`. This records
sample identities, global positions, epoch and partition, original CP shards
and file owners, input and file checksums, and logical microbatch/update
membership. Resume requires the expected producer identity, DP degree, GAS and
cursor. CP shards are reconstructed in head/tail token order before forming
the consumer view. A missing or changed file stops all consumers before that
microbatch enters forward/backward. Cache deletion uses producer ownership and
waits for every consumer after the last update. Profiler regions
`deepspec::draft_feature_index` and `deepspec::draft_feature_read` attribute the
consumer overhead to draft work.

## Verification

The immutable baseline is described in [dspark_torchtitan_baseline.md](dspark_torchtitan_baseline.md).
Full state comparisons retain its `rtol=1e-4`, `atol=1e-6` bounds.

- `tests/test_draft_feature_reader.py`: exact seven-token CP2 reconstruction,
  padding, original masks, consumer CP views, wrong producer/DP/GAS/cursor,
  actual file ownership, missing files, changed file contents and mismatched
  requested inputs. CPU tests passed.
- `tests/test_qwen_draft_feature_training.py`: actual retained Qwen/FSDP training
  from CP2 producer shards matched the captured FP32 and BF16 loss, gradients,
  two Adam updates, scheduler, metrics, frozen weights and RNG. A shard missing
  on one consumer stopped both ranks at microstep zero without forward or update.
- `tests/test_qwen_producer_isolation.py`: full Qwen constructor and training
  entry on two GPUs; the external producer alone supplies immutable fixture
  features. Fixed producer TP2/CP1 serves draft FSDP2 and replicated DP2, with
  GAS2 and two updates. Final model, Adam, scheduler and RNG matched; a delayed
  consumer verified files remained until its optimizer finished. Both ranks
  passed in 75.99 seconds in the first full integration check. Enhanced checks
  also compare every loss, model output and clipped gradient. This is correctness evidence for
  the tiny real Qwen draft, not a full-size 128K performance result.
- `tests/test_qwen_producer_launcher.py` and the existing native launcher tests:
  nine tests passed, including draft CP/TP changes with fixed producer settings.

Logs are under `output/dspark_torchtitan_implementation/`. The first integration
run failed at the old matching-layout restriction. An intermediate fixture run
needed the baseline's explicit `flex_attention` selection restored after
loading its serialized model configuration; production model math was unchanged.

## Complete-update phase boundaries

Only the Qwen vLLM trainer enables optimizer-aligned per-epoch partitions.
`train.data_partitions` is capped at the number of complete updates in an epoch;
counts are balanced in update units. The sampler still shuffles each epoch with
seed `42 + epoch`, truncates to complete global batches, and stops at the same
complete-update cursor. GAS and per-microbatch weighting are unchanged.

The real two-GPU entry test uses nine records, global batch four, GAS two, two
epochs and a three-update stop. The first eight records of each shuffled epoch
remain eligible; the stop consumes four from the second epoch. Requested counts
one and three yield producer phase sizes `[8, 4]` and `[4, 4, 4]`, respectively.
Every phase begins without accumulated gradients. All sample identities,
losses, outputs, clipped gradients, final model, Adam, scheduler and RNG match.
The pre-change red test reproduced the third producer phase starting at
microstep three with a half-completed accumulation window.

`feature-phases-green.log` records all three enhanced tests passing on both
actual ranks in 66.32 seconds. The existing CPU reference for gradients spanning
old half-update partitions remains in `tests/test_qwen38_vllm.py` and passed
along with that file's other eleven tests. The native partition-schedule
regression also passed. This is not evidence of DCP persistence, unloading or
128K acceptance; those belong to subsequent tickets.

Type checking and Ruff pass for the changed reader, Qwen adapter and tests.
The shared `BaseTrainer` has the same 21 mypy diagnostics before and after this
change, with no new diagnostic messages; logs preserve that comparison.

Reproduce the distributed checks from the repository root:

```bash
source env.sh
export PYTHONPATH="$PWD/torchtitan:$PWD/vllm:$PWD"
export DEEPSPEC_BASELINE_REFERENCE="$PWD/output/dspark_torchtitan_baseline_20260914/numerics-final"
OMP_NUM_THREADS=1 "$CONDA_PREFIX/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m unittest \
  tests.test_qwen_producer_isolation.QwenProducerIsolationTest \
  tests.test_qwen_draft_feature_training.QwenDraftFeatureTrainingTest
```

Code review found two missing checks, now addressed: fixed producer CP degree
must govern shard completeness even when padding hides missing data, and the
persisted sample plan must include `chat_template` and `min_loss_tokens`.
The reader also rejects reuse of a producer file for another shard coordinate.
CPU regressions reproduced missing CP4 shards at length three and reused CP2
files before the fix; both now fail validation. The real constructor regression
reproduced acceptance of changed preprocessing before the identity fix. Standards
review reported no actionable findings; Spec re-review found both issues resolved.

After the review fixes, `feature-phases-reviewed.log` records all three enhanced
GPU tests passing on both ranks in 74.16 seconds, including rejection of changed
preprocessing in the real constructor. A subsequent CPU run passed all consumer
checks but exposed an existing one-second startup timeout in the worker cleanup
test on the shared filesystem. The test now allows ordinary success/error
children 30 seconds to start, and gives the deliberate timeout case ten seconds;
production worker timeouts are unchanged.

The final worker cleanup recheck passed all three success/failure/timeout cases
(`worker-cleanup-final.log`). The prior combined CPU run had passed the other
fifteen tests, including all revised reader and launcher checks.
