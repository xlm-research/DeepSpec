# Native preparation and real Qwen feature handoff

Ticket 06 passed on 2026-09-14. The live two-phase run uses the released Qwen3.8-27B
teacher and the full five-layer draft geometry, with short inputs for lifecycle
acceptance. This document does not claim 128K acceptance or steady-state speed.

## Ownership and entry points

- `deepspec.orchestration.run` schedules complete-update partitions, launches
  CPU preparation, alternates producer/draft processes and observes the GPU pool.
- `torchtitan.models.dspark_draft.preparation` resolves the native recipe and
  owns tokenizer, chat formatting, truncation, mask, epoch order and update plan.
  It prepares CPU input tensors without constructing a draft model or CUDA context.
- `deepspec.orchestration.target` calls the existing Qwen vLLM worker unchanged.
  Producer manifests record teacher identity, sample position, input identity,
  length and original context shards. They do not define draft microbatch/GAS.
- The consumption manifest refers separately to the immutable whole-run plan
  and producer manifest. Native `ProducerFeatures` authenticates files, checks
  tokens/masks, layer order, dtype, shape and final-normalized hidden semantics,
  and reconstructs original head/tail shard order before training consumption.
- `deepspec.orchestration.draft` launches the native Trainer and returns after
  workers exit. Complete DCP state and a durable commit precede phase success.

The request supplies `run_id`, `output_dir`, `draft_python`, `draft_source`,
`workers`, `draft_devices`, `recipe_args`, `updates_per_partition` and a separate
`producer` configuration. `retain_features` preserves evidence when true;
otherwise completed feature caches are reclaimed only after committed updates
and worker release. The native recipe defines batch, GAS, model and optimizer.

## Input acceptance

`native-inputs-v3-test.log` passed all three tests without skips. A real local Qwen
tokenizer produces exactly the retained parser's tokens and masks. Eleven source
records truncate to eight samples per epoch at global batch four. Seed 42 fixes
the sample permutation; the plan contains two updates, GAS two and two DP ranks.
CPU preparation reports `cuda_initialized: false`.

The immutable baseline inputs also pass the native reader after the existing
producer converter writes full features or two/three head-tail shards. Every
reconstructed tensor is bitwise identical. Changed feature bytes, wrong input
identity and incorrect final-hidden semantics fail explicitly. Test source:
`tests/test_torchtitan_inputs.py`.

## Live acceptance configuration

Evidence root: `output/dspark_torchtitan_orchestration_20260914`.

| Component | Actual configuration |
| --- | --- |
| Teacher | `/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B`, full 64 layers |
| Producer | Two independent TP4 replicas on GPUs 0-7, original source-built vLLM |
| Inference | BF16, memory fraction 0.45, max batched tokens 8192, max sequence 131073, eager |
| Draft | Five full-attention layers, hidden 5120, 24 query / 4 KV heads, head width 256 |
| Draft phase | Two GPUs with FSDP2, BF16, local batch one, GAS two, global batch four |
| Short workload | Eight prepared samples, 106/110 tokens, eight anchors, two one-update phases |
| Supervision | Target layers 1/16/31/46/61 plus actual final normalized hidden |

The live recipe is `tests.torchtitan_phase_fixtures.qwen38_live`; it specializes
the native `qwen38_27b` recipe for this bounded acceptance workload. The target's
model, environment, source, TP, memory and context settings remain fixed.

`live-request.json` records the exact orchestration request and
`live-two-phase.log` captures native training. Per-phase target job logs and
producer manifests preserve production and resource observations. The first
phase completed update one, wrote about 31 GiB of complete DCP state in 62.2
seconds and exited all native workers. Peak allocated draft memory was 41.53
GiB per rank. Both the pre-draft and post-draft GPU process lists were empty.

## Accepted interruption and artifact recovery

`phase-interrupt-v2-test.log` passed the real two-GPU interruption suite in
1102.611 seconds, without skips. Corrupt input before the first commit and
after update one, parent SIGKILL and GPU-worker SIGKILL all fail the phase,
release owned GPU processes and resume in a new native process. Every resumed
parameter, gradient, FP32 master/Adam state, scheduler, RNG and data cursor is
bitwise identical to the uninterrupted two-update reference. The killed runs
have no phase-success journal, so recovery reconstructs progress from the
authenticated native commit.

The Linux subreaper also terminates and reaps descendants that start separate
sessions (`process-supervisor-v2/result.txt`). Rank exceptions exit promptly;
they do not wait in process-group teardown while peers remain in collectives.
The run lock and request identity prevent concurrent or incompatible owners.
Incomplete partitions retain their features and last committed recovery point.

`phase-retention-v2-test.log` passed in 661.690 seconds: both retained recovery
points restore bitwise-equal trajectories, milestone one survives rolling
retention, and the existing DeepSpec HF consumer loads the optional step-three
export with all parameters bitwise equal. `retention-save-failure-test.log`
passed in 127.784 seconds with a real commit-rename failure after DCP payload
write: prior commits survive, no phase succeeds and all GPU workers exit.

CPU-only export also reconstructs HF weights from full DCP without a training
update or CUDA initialization. The run journal repairs a pending optional
export after a committed phase (`pending-hf-export-recovery.json`), preserving
completed update three; that recovery cost is 69.180 seconds including startup.

## Completed live result

`live-handoff-acceptance.json` verifies both real target/draft phases, all eight
sample identities, feature file hashes, complete DCP metadata, consumption
ranges `[0, 2]` and `[2, 4]`, and absence of every native worker after its phase.
Both vLLM replicas also release their GPU processes before each draft launch.
Teacher identity and inference configuration are identical in the two phases.

| Phase | Native update | Draft launch through exit | Target production | DCP save |
| --- | --- | --- | --- | --- |
| First | 1 | 303.21 s | 434.95 s | 62.22 s |
| Restored | 2 | 377.03 s | 463.76 s | 60.81 s |

The second native process restores complete state in 145.21 seconds and starts
at update two. Peak allocated memory is 45.64 GiB per rank in that phase. These
measurements include cold imports/compilation and network filesystem I/O;
optimization comparisons require their own fixed-workload runs. Target time
is reported separately and excluded from draft elapsed time.

Feature artifacts are retained for this acceptance run (`retain_features=true`).
The production default reclaims only fully committed consumption ranges after
worker release; interrupted ranges retain their inputs and prior recovery point.
