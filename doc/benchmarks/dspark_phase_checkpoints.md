# Qwen draft phase checkpoints

Ticket 05 implements complete durable Qwen phase checkpoints. GPU state unloading
and rolling retention are handled by tickets 06 and 07.

The draft phase checkpoint entry uses the retained Qwen training loop and the
existing PyTorch distributed checkpoint adapter. It saves the model (including
frozen weights), FP32 master Adam state, scheduler and rank-specific training
RNG. Ordinary phase saves omit HF export. Model metadata, the resolved training
configuration, the original configuration source and the complete draft input
index accompany the distributed state.

A save writes into a new hidden `.step_<n>.incomplete-*` directory. After DCP
returns, the writer verifies the referenced storage extents and required state
namespaces, hashes and fsyncs every saved file, writes `draft_phase_commit.json`,
then fsyncs the directories. The directory rename precedes publication of the
relative `step_latest` symlink; symlink replacement and its directory fsync are
synchronous. Rank-zero file operations broadcast their result to all workers.
The next producer phase is reached only after this entry returns successfully.

Discovery checks committed directories in descending update order, validates
file hashes, DCP storage extents and rank RNG namespaces, and cross-checks the
input index, topology, accumulation and progress metadata. Hidden incomplete
attempts are ignored. A damaged newer checkpoint falls back to the previous
valid commit; the rejected directory is preserved under a hidden name before
the replayed phase is saved. If every committed checkpoint is damaged, the job
fails explicitly. An initial attempt that never committed starts from scratch.

Resume checks the resolved model/training configuration, complete draft topology,
model geometry, fixed producer identity and sample plan before training. DCP's
load template explicitly requests phase fields, and restored per-rank progress
is checked collectively against the committed metadata. Each checkpoint carries
its producer and teacher identities in its copied input index, so moving the
checkpoint to a new storage root does not require root-level sidecar files.
Suspend requests finish the current aligned phase and use the same DCP entry.

The first real two-GPU Qwen test ran two optimizer updates with GAS two. Its
pre-change run failed because `model.safetensors` was still emitted alongside
DCP. The new path passed on both ranks in 65.32 seconds, producing committed
DCPs, copied feature indexes and no HF files. Logs:
`output/dspark_torchtitan_implementation/phase-checkpoint-red.log` and
`phase-checkpoint-first.log`.

The new-process comparison launches three independent two-rank jobs: continuous
training, one update followed by process exit, and restoration for the next
update. The restore fixture initializes all model weights, including frozen
weights, to zero and reads model geometry from checkpoint metadata. The passing
comparisons include inputs, every loss and model output, clipped gradients,
final weights, optimizer/scheduler state and each rank's RNG. The three-process check passed in 501.22 seconds including process startup and
shutdown. This is functional test wall time, not a draft performance measurement.
Artifacts and per-process logs are under
`output/dspark_torchtitan_implementation/checkpoint-restart-green/`.
The final rerun with automatic integrity, configuration and loaded-progress
validation passed all three fresh processes and all comparisons in 483.82 seconds:
`checkpoint-restart-validated.log` and `checkpoint-restart-validated/`.

The restart test first exposed that the Qwen batch generator ignored the retained
loop's `_active_train_end_step`. Its red run performed two updates when asked to
stop after one (`checkpoint-restart-red/save.log`). The generator now limits its
ready-feature phase to that complete-update cursor without changing the overall
scheduler or GAS. Phase identities derive from the starting update cursor and
therefore remain stable across a fresh process. The existing half-update CPU
reference still passes after supplying its explicit no-checkpoint configuration.

Additional acceptance on the same two-GPU real Qwen fixture:

- Standalone directory restore: copied only `step_1` into a fresh checkpoint
  root, restored from zero weights, and completed update two. Both ranks passed
  in 31.06 seconds; every observation, gradient, final model/Adam/scheduler tensor,
  progress and Python/NumPy/CPU/CUDA RNG matches continuous training.
  Evidence: `checkpoint-standalone-green.log`, `checkpoint-standalone-comparison.log`.
- Configuration rejection: changed loss weight, learning rate, global batch,
  draft topology or preprocessing settings. Both ranks rejected each mismatch
  before training (23.39 seconds). Evidence: `checkpoint-identity-red.log` and
  `checkpoint-identity-green.log`.
- Actual storage faults: first-save fsync failure, fsync failure after a valid
  commit, truncated DCP storage, and commit rename failure. All four cases passed
  on both ranks (44.68 seconds): no next target phase, no failed published step,
  and the previous valid checkpoint remains discoverable.
  Evidence: `checkpoint-failures-red.log`, `checkpoint-failures-green.log`.
- Fresh replay after failure: copied the previous valid commit to a new root
  alongside damaged `step_2`, recovered `step_1`, replayed update two and committed
  a replacement while preserving the damaged directory (32.08 seconds).
  Evidence: `checkpoint-failure-recovered.log`.
  The replayed inputs, losses, gradients, complete optimizer/model state and all
  four RNG states also match the original failed update on both ranks:
  `checkpoint-failure-recovery-comparison.log`.
- Suspend requested after the first update of a two-update phase: both ranks
  finish the phase, then suspend through a valid full DCP with no HF files
  (32.95 seconds). Evidence: `checkpoint-suspend-green.log`.
- Six file-validation and discovery regressions passed against real saved DCP
  files, including corruption, missing storage hidden by an incomplete manifest,
  inconsistent progress, initial uncommitted attempts and fallback discovery.
  Evidence: `checkpoint-validation-red.log`, `checkpoint-discovery-green.log`.

Evidence paths above are relative to
`output/dspark_torchtitan_implementation/`. Timings are functional test wall time,
not training throughput claims. These tests retain the actual Qwen model, DSpark
loss, optimizer and DCP; only external feature production and the external
suspend service use fixtures. Storage failures occur at real filesystem calls.

To reproduce the focused distributed cases, run
`tests.test_qwen_phase_checkpoint.QwenDraftPhaseCheckpointTest` with two torchrun
ranks and `DEEPSPEC_BASELINE_REFERENCE` set to the immutable `numerics-final`
baseline. `DEEPSPEC_PHASE_TEST_MODE` selects `full`, `save`, `resume`, `reject`,
`failures`, `resume_failure` or `suspend`; `DEEPSPEC_PHASE_TEST_ROOT` preserves logs
and states, and `DEEPSPEC_PHASE_INPUT_ROOT` retains the original dataset identity
when testing a new storage root. The orchestration test
`tests.test_qwen_checkpoint_restart` launches fresh continuous/save/resume jobs
and compares their complete trajectories.

Both Standards and Spec review findings were resolved: phase fields are requested
in DCP's load template, rejected newer destinations no longer block replayed
saves, and suspension cannot bypass the DCP protocol. Scoped Ruff/mypy checks
pass. BaseTrainer retains exactly its 21 pre-existing mypy diagnostics, with no
new diagnostics. The full suite is reserved for completion of the implementation
series; the baseline run's unrelated failures are recorded separately.
