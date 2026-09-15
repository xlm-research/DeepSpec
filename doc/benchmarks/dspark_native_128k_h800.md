# Qwen 128K acceptance on H800

> 本文保留截至 2026-09-15 15:24 的运行与排障记录。后续两轮均已完成，
> 连续训练与分阶段恢复的全量一致性核验通过；最终结果见
> [H800 连续训练与恢复对照报告](dspark_native_128k_h800_comparison.md)。

Status at 2026-09-15 15:24 Asia/Shanghai: **in progress, resuming after an NCCL
memory fix**. The first five full-scale updates and their committed DCP completed.
All eight new second-phase workers loaded step 5 at 14:46:44 (approximately
47 seconds), but update 6 failed during gradient-norm reduction at 14:50:26 with
an NCCL CUDA out-of-memory error. The supervisor released all workers; progress
remains at the valid step-5 checkpoint. No update-6 checkpoint or run completion
was published. `h800-phase2-failure.json` records the failure.

The first full-128K probe reproduced the failure. A second probe with eager
parameter-mesh communication initialization completed update 6, saved its DCP
and released all workers. The fix is now in the native trainer; the original
orchestrator restarted from step 5 at 15:07:18 (PID 21016). New workers
25533–25540 restored the checkpoint at 15:19:20 and completed update 6 at
15:22:48, again with loss 3.72804. Updates 7–10 are in progress; the latest
durable checkpoint remains step 5 until the phase boundary. Full restart
equivalence and performance comparisons remain pending.

## Reproduction

From `/mnt/afs_share/lezewei/DeepSpec_torchtitan_vllm`:

```bash
bash .scratch/dspark-torchtitan-orchestration/resume-128k-after-gpu-release.sh
```

Check the existing orchestrator and its command line before launching. The
original process was PID 4130253, started at 13:06 and exited after the resume
failure; the active retry is PID 21016. Do not launch a duplicate. GPU idleness during
feature validation does not mean a run has ended. The run root's `complete.json` is
the terminal success record; individual phase completions are intermediate.

Artifacts below are relative to
`outputs/dspark_torchtitan_orchestration_20260914/` (**outputs** with an s):

- Request: `128k-h800-request.json`.
- Run: `qwen38-128k-h800/`; log: `qwen38-128k-h800.log`.
- Input: `128k-source.jsonl`, 80 packed records; the plan selects 40.
- Initialization: `scale-initialization-h800/initial-weights.pt` and eight
  `rng-rank*.pt` files. Replays read this location from the original plan's
  `resolved_recipe.capture_initialization`; they must not overwrite it.
- Model: `/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B`.
- Run ID: `native-qwen38-128k-h800-20260915`.
- Input-plan SHA256:
  `efd6377dd1b18e4f2dadf32161d4ce151ee59933068609434642cd7275ae1987`.

Environment setup, pinned supplements and smoke-test logs are under
`.scratch/dspark-torchtitan-orchestration/environment-setup/`. `env.sh` activates
the local system-site-packages venv over `/tmp/deepspec_vllm_torchtitan_envs`.
Only pip was used. Hardware is eight H800 80GB GPUs, driver 580.95.05;
Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.16.1 and
the repository's source-built vLLM 0.26.1rc1.dev719 are in use.

## Workload and measurement contract

The geometry, optimizer and measurement rules in
[the 128K acceptance contract](dspark_native_128k.md) apply. This H800 run uses
five DSpark layers, hidden 5120, FFN 17408, vocabulary 248320, 24 Q / 4 KV
heads with head width 256, 512 anchors, block seven and Markov rank 256.
All 40 planned samples have 131072 tokens. Native topology is TP4 × DP shard2,
CP1 / PP1, BF16 parameters, FP32 reduction/master/Adam state and SelectiveAC.
Local batch is one; global batch four gives GAS two. Ten updates use the
unchanged 1000-step scheduler horizon, 40-step warmup and LR 6e-4.
Full-scale compiler threads are 32; HF export is disabled for this run.

## First-phase observations

`phase-0-5/complete.json` and `checkpoints/step-5/commit.json` exist. All eight
draft workers exited after synchronous checkpoint completion. The recorded
resource check found no GPU compute processes before the next target phase.

| Update | Logged loss | Longest-rank update seconds |
| --- | --- | --- |
| 1 | 3.73530 | 215.051 |
| 2 | 3.73460 | 198.938 |
| 3 | 3.73396 | 200.093 |
| 4 | 3.73274 | 202.190 |
| 5 | 3.73085 | 201.918 |

The loss decreases across these five updates; this short warmup interval does
not establish convergence. Timing source: `h800-first-phase-timing.json`,
produced from the real phase using `phase_measurements()`. The corrected
`h800-first-phase-timing-v2.json` explicitly scopes the end-of-phase memory
counters; timing values are unchanged.

| First-phase component | Seconds |
| --- | --- |
| Orchestration feature validation and preparation | 684.794 |
| Worker lifecycle, launch through exit | 1105.434 |
| GPU release check | 0.155 |
| Complete draft cost | 1790.383 |
| ↳ Launch (within worker lifecycle) | 18.593 |
| ↳ Native all-rank wall span | 1078.673 |
| ↳ Exit | 8.168 |

Within the longest native rank's timeline: initialization 29.102 seconds,
one-time validation capture 14.740, training including feature reads 1018.189,
checkpoint save 14.601, close 0.028, and other native work 2.013.
These nested intervals are not added to the lifecycle total again. Target
production is excluded. First-phase GPU release was verified independently.

The phase-end CUDA counters contain allocated 25,043,702,784 bytes and reserved
78,674,657,280 bytes, but native metrics reset peak counters after each logged
update. These values **do not establish whole-phase memory peaks**.
`h800-first-phase-memory.json` reads the five pre-reset TensorBoard intervals:
rank 0 reached 61.627 GiB active and 73.271 GiB reserved, with zero allocation
retries and OOMs. Only rank 0's detailed metrics were recorded in this run.
Active bytes and allocated bytes are distinct allocator measures. The console's
73.27 GiB value is reserved memory.

The uninterrupted replay enables native `metrics.save_for_all_ranks` so all
eight ranks' per-update peaks can be reported. Its CPU configuration preflight
passed (`h800-replay-preflight.log`); this changes metrics recording only and
is outside the training-state identity. The original trainer is unchanged.
The summary requires every expected update and the actual worker's event file;
it reports which ranks are covered instead of treating rank 0 as all eight.

`h800-step5-state-audit.json` verifies the committed metadata digest, update 5,
global microbatch cursor 10, scheduler epoch 5, presence of all eight ranks'
Python/NumPy RNG, and bitwise equality between checkpoint CPU/CUDA RNG and
each rank's last forward observation. This CPU audit does not compare complete
model or optimizer trajectories; that still requires the uninterrupted replay.
The reproducible audit entry is
`.scratch/dspark-torchtitan-orchestration/environment-setup/audit_checkpoint_boundary.py`.

### Forward observation cursor correction

The original CPU audit rejected `[2,2,4,4,6,6,8,8,10,10]` as repeated samples.
Source inspection and all eight real rank artifacts show that native Trainer
prefetches both GAS microbatches before running either forward. The scale hook
records that **read cursor**, whereas the retained resident hook records the
completed forward position. There are ten native forward observations, with
the expected cursor at every observation.

`tests/summarize_torchtitan_scale.py` now validates the exact phase boundaries,
forward counts and expected prefetched cursors, then assigns ordered forward
positions for comparison. Original artifacts and all supervision/RNG tensors
remain unchanged. The running trainer and its mathematics were not modified.
`python -m unittest tests.test_torchtitan_scale_summary -v` passes six tests,
including malformed cursor, missing/repeated forward, phase gap and wrong worker
evidence rejection. Two additional tests cover complete eight-rank memory
intervals and rejection of missing updates; the combined eight-test log is
`h800-summary-memory-regression.log`. Red and green cursor logs are archived as
`h800-summary-regression-{red,green}.log`. This is validation-tool evidence;
the completed full-run summary remains pending.

## Resume OOM evidence and correction

`h800-resume-oom-diagnosis.json` records the isolated reproduction. Immediately
before gradient clipping, rank 0 had 25,043,701,248 allocated bytes but
79,968,600,064 reserved bytes, leaving only 157,286,400 bytes available to the
CUDA driver. NCCL's first parameter-mesh norm reduction then failed while
allocating peer-connection resources outside PyTorch's allocator. All eight
ranks had the same allocated bytes before and after DCP restore, ruling out
extra live tensor storage from loading as the observed cause.

The second probe changed only the initialization of existing parameter-mesh
communication: one zero-scalar all-reduce per unique non-singleton axis before
training. It retained the full geometry, original step-5 checkpoint and exact
128K feature inputs. Update 6 completed with loss 3.72804; its checkpoint metadata,
unchanged training identity, worker exit and GPU release are recorded in
`h800-resume-warmup-acceptance.json`. The separate boundary audit verifies all
eight ranks' CPU/CUDA RNG against their final forward, global cursor 12 and
scheduler step 6. These artifacts are diagnostic evidence, not a replacement
for completing and comparing all ten updates.

The passing probe recorded all eight ranks: peak active memory was 61.628 GiB
and peak reserved memory 74.477 GiB. Rank 0 had one allocator retry that reclaimed
cached memory; no rank recorded an OOM. See `h800-resume-probe-2-memory.json`.

`DSparkTrainer._initialize_parameter_collectives` now performs this warmup
inside the measured native initialization interval. It uses independent zero
scalars and leaves model, gradients and RNG unchanged. Diagnostic hooks and
NCCL debug logging are absent from the production run. Both probe versions,
the pre-fix trainer and the final patch are archived alongside their results.

## Remaining acceptance

1. Finish updates 6–10 from the committed step-5 DCP and verify all workers exit.
2. Run uninterrupted full-scale native replay with the same initialization and
   real features. Compare full DCP and per-forward supervision/RNG.
3. Measure the retained resident reference on the same workload. Report its
   two-GPU FSDP topology separately from native eight-GPU TP4. H800 memory fit
   is unverified; do not replace the workload with a smaller context.
4. Reproduce missing prerequisite evidence on this environment as necessary,
   including save failure protection, retention and CPU HF export. Previous
   B300 evidence is historical and is not a new H800 pass.
5. Archive full timing, variability and numerical results before closing
   ticket 10 or delivering ticket 11's DP replicate8 / shard8 support.
