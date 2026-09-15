# Native TorchTitan DSpark draft acceptance

Tickets 02 and 03 are verified on 2026-09-14. DeepSpec starts an independent
`torch.distributed.run` process; TorchTitan builds its native Trainer, model,
feature loader, loss, optimizer and scheduler and executes its own update loop.
The retained DeepSpec trainer is used only to obtain the numerical reference.

## Numerical evidence

| Topology | Precision | Updates / GAS | Result |
| --- | --- | --- | --- |
| Single GPU, FSDP degree 1 | FP32 | 2 / 2 | pass |
| Single GPU, FSDP degree 1 | BF16 | 2 / 2 | pass |
| Two GPUs, FSDP2 | FP32 | 2 / 2 | pass |
| Two GPUs, FSDP2 | BF16 | 2 / 2 | pass |

Both integration runs completed without skips:

- `output/dspark_torchtitan_orchestration_20260914/single-gpu-v2-test.log`
- `output/dspark_torchtitan_orchestration_20260914/two-gpu-v2-test.log`

`tests/test_torchtitan_phase_training.py` compares every forward output, weighted
loss term, per-microbatch denominator, clipped trainable gradient, parameter,
FP32 master weight, Adam state, scheduler state, gradient norm, RNG state and
consumption cursor after both updates. The real Qwen DSpark model uses the
immutable short-sequence fixture described in
[`dspark_orchestration_baseline.md`](dspark_orchestration_baseline.md).
Frozen embedding/head parameters are included in the comparison. Unequal
supervision counts and a zero-denominator microbatch are included.

The two-rank reference is the original immutable baseline. The single-rank
reference was computed separately through the retained production loop with
rank 0's immutable inputs and saved under `single-gpu/{float32,bfloat16}`.
The final single-rank test reuses that reference through
`DEEPSPEC_SINGLE_GPU_REFERENCE`; it reruns the real native training process.
Tensor/float tolerance remains `rtol=1e-4, atol=1e-6`; it was not loosened to
accept the BF16 integration.

Native FSDP gradient division is disabled, matching TorchTitan's SUM convention.
Each logical microbatch divides by its global weighted denominator and GAS once.
For comparing the *local* loss with the averaging-based legacy loop, the test
removes the reference's rank-count compensation. All gradients and optimizer
states are compared directly. Gradient reduction occurs at the end of the
accumulation window, preserving the retained FP32 reduction behavior.

## Pinned-environment integration

Training uses
`/mnt/afs-agentpro/share/env/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python`,
PyTorch `2.13.0+cu130`, Transformers `5.16.1`, and NVIDIA B300 GPUs. TorchTitan is
based on `f6b9152e9bedcc18f5dc339b9f88265e5a07e988` plus the recorded working changes.

- DSpark explicitly selects IEEE FP32 matmuls. The pinned build lacks the newer
  BFX9 interface assumed by TorchTitan; existing native recipes retain their
  BFX9 default. Seven distributed-utility tests passed.
- The process-group timeout helper supports the pinned build's private API and
  the newer public spelling. Real multi-update tests exercise that transition.
- DSpark casts nonpersistent RoPE buffers with model parameters, as the retained
  implementation does. Both BF16 buffers match bitwise after meta materialization.
  Keeping these buffers in FP32 caused a detectable first-forward mismatch.
- Pure FSDP uses TorchTitan's `spmd_types` backend and its one-axis `dp_shard`
  storage mesh. The pinned PyTorch n-D `dp_mesh_dims` interface requires DTensors;
  the pure-FSDP adapter supplies ordinary parameters directly to the one-axis API.
- Only Grain and four additive dependencies were installed. Existing installed
  versions and all twelve source-built vLLM binary hashes remain unchanged.
  Hugging Face data-source imports are lazy, so prepared-feature training does
  not require the incompatible optional `datasets` dependency.

## Reproduction

Use the training interpreter above, `OMP_NUM_THREADS=1`, and
`PYTHONPATH="$PWD:$PWD/torchtitan:$PWD/vllm:/tmp/deepspec-validation-tools"`.
Set `DEEPSPEC_BASELINE_REFERENCE` to the absolute immutable `numerics-final`
directory and `DEEPSPEC_NATIVE_TEST_OUTPUT` to a fresh output directory, then run:

```bash
python -m tests.test_torchtitan_phase_training
```

Use one visible GPU for the default single-rank run, or two visible GPUs with
`DEEPSPEC_PHASE_WORKERS=2`. `python` in this example means the pinned interpreter.

Larger parallel configurations have separate acceptance tests and are not
implied by this four-case result.

## Full phase recovery (ticket 04)

Real two-rank DCP tests passed for FP32 (563.909 seconds) and BF16 (551.738
seconds), without skips. Each compares a continuous two-update process with
one update, a committed full checkpoint, complete worker exit, and a second
independent process completing update two. The restored process deliberately
consumes extra CPU/CUDA random numbers during construction. Every saved output,
loss, gradient, parameter, FP32 master/Adam state, scheduler value, RNG and input
cursor matches the continuous trajectory bitwise.

- `checkpoint-v2-float32-test.log`
- `checkpoint-v2-bfloat16-test.log`
- `native-optimizer-state-test.log`

These files are under `output/dspark_torchtitan_orchestration_20260914`. The
integration test is `tests/test_torchtitan_phase_checkpoint.py`; the separate
optimizer test verifies that BF16 loading preserves incoming FP32 master and
moment tensors, including an identical next update.

DeepSpec supplies a phase envelope with the immutable whole-run input plan,
feature partition, update stop, run identity and optional resume checkpoint.
The recipe retains its full scheduler horizon. The native state extension
includes rank-specific RNG and buffers, full resolved configuration and plan
identity. A synchronous native DCP save, GPU synchronization and rank barrier
precede the durable commit marker; the marker authenticates DCP metadata.
DCP-only runs write no Hugging Face checkpoint. The launcher returns only after
all native workers exit; tests also check their absence from GPU process lists.

The real save-failure test passed in 496.249 seconds. It makes the final commit
path unwritable as a file while allowing the actual model/optimizer DCP write.
The process fails, emits no phase success result, leaves no GPU workers, and
does not modify the previous committed checkpoint. An incomplete directory is
not a valid resume checkpoint. Evidence: `checkpoint-save-failure-test.log` and
`checkpoint-save-failure/native.log`.

## SelectiveAC (ticket 05)

Native `SelectiveAC.Config(preserve_rng_state=True)` wraps each real DSpark
block once, before native FSDP2. Outer model compilation remains disabled. The
FP32/BF16, two-rank, GAS-two trajectories match the no-AC native results bitwise
for both updates, including recomputation RNG and full optimizer state.
`selective-ac-state-comparison-final.log` records all four comparisons. The
initial observation harness exposed checkpoint-wrapper prefixes in parameter
names; canonicalizing those names fixes observation only, without changing
training or numeric tolerance.

`checkpoint-selective-ac-test.log` passed without skips in 1149.779 seconds. It
executes continuous, first-phase and independent resumed processes for both
dtypes through the DeepSpec entry, verifying complete DCP state and bitwise
next-update equivalence. The native recipe reconstructs the same AC policy on
resume. Reproduce the checkpoint test with `DEEPSPEC_PHASE_SELECTIVE_AC=1`.

The tiny acceptance workload has insufficient activation volume or warmup to
measure useful performance savings. Its native logs record actual peak memory
and per-update timing; this is a correctness test, not an AC performance claim.
Performance tuning remains ticket 25 and requires at least ten updates.

For the BF16 two-update observation, peak allocated memory was 0.04 GiB in
both modes. Update two took about 0.19 seconds without AC and 0.52 seconds with
SelectiveAC (170 versus 62 reported tokens/second). These single observations
include runtime overhead and do not isolate recomputation; they are recorded
for acceptance transparency, not used to choose the production AC policy.

## Retention and HF export (ticket 08)

`phase-retention-v2-test.log` passed without skips in 661.690 seconds. Real BF16
FSDP2 training generates successive complete DCP points, preserves milestone
one, and retains the latest two ordinary points. Separate native processes
resume each of the two retained points and match the continuous trajectory
bitwise, including every parameter, FP32 optimizer state, scheduler, RNG and
cursor. Resumed phases deliberately omit the external initialization file.
After update four, the retained directories are `step-1`, `step-3`, `step-4`.

HF export is optional (`checkpoint.export_hf_steps` / `export_hf_final`) and
writes to `checkpoints/hf/step-N`, separately from complete DCP. In the small
acceptance runs, export at update three took 0.128-0.152 seconds; updates
one, two and four did not request HF export. The existing
DeepSpec Qwen model consumer loads the real exported safetensors and matches
all parameters at the corresponding native update bitwise. Ordinary DCP
folders contain no HF configuration or weight files.

`retention-save-failure-test.log` passed in 127.784 seconds. It restores real
update-three state, computes update four and successfully writes native DCP
payload/metadata, then fails the actual final marker rename. The milestone and
both previous recovery points remain unchanged, no success result is published,
and all GPU workers exit. Retention runs only after a new durable commit.

Post-training export is also available without a GPU model or optimizer update:

```bash
python -m torchtitan.models.dspark_draft.export CHECKPOINT OUTPUT_DIRECTORY
```

The pinned interpreter and source `PYTHONPATH` are the same as above.
`cpu-hf-export-comparison.log` verifies the CPU-exported artifact with the
existing Qwen consumer and compares all weights bitwise with native update
three; CUDA remains uninitialized. Conversion and serialization took 0.207
seconds after imports for this small model; interpreter startup is separate.
The orchestration journal can finish a pending HF export from committed DCP
without repeating the already committed update.
