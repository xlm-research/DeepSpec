# Implementation baseline

Date: 2026-09-20. Feature: `001-unify-ray-topology`.

## Workflow and source state

Loaded `/mnt/afs_share/lezewei/VideoLingo-main/.agents/skills/speckit-implement/SKILL.md` and the equivalent installed `specify-cli/core_pack/commands/implement.md`. Project-local scripts are absent; the installed `core_pack/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`, run from this repository, successfully resolved this feature. Requirements checklist: 16/16 checked, unchanged. No extension hooks or constitution file are present.

All source-manifest digests match the implementation starting point. The original manifest is preserved. `implementation-baseline.json` records the full pre-implementation Git status, including existing deletions and untracked files; these are not implementation changes.

| Repository | Branch | HEAD |
|---|---|---|
| DeepSpec | dev/vllm_torchtitan | 589c8b8ebef766a6f979cc7007e3b8f19f707c4c |
| vLLM | lzw/support_dspark_v.0.1.1 | 1ee54c40df7ffe2c8934f5bd1c79917f34cb954e |
| TorchTitan | dev/vllm_torchtitan | ae56a6d6e585875a206733a308b2eecd18b45258 |

Existing edits include `cluster.py`, its tests, training scripts/documentation and the vLLM checkout. Preserve these and the existing deleted files. Git uses command-local `safe.directory` because the shared checkout has another owner; global settings are unchanged. No branch switch, dependency installation or model launch was performed for this baseline.

## Interpreter and dependencies

Activation: `source ./h800conda.sh`. Interpreter: `/mnt/afs_share/miniconda3/envs/deepspec_vllm_torchtitan_envs/bin/python`, Python 3.12.14. Metadata inspection:

| Package | Version |
|---|---|
| torch | 2.13.0 |
| ray | 2.58.0 |
| vllm | 0.26.1rc1.dev719+g1ee54c40d.d20260912 |
| mooncake-transfer-engine | 0.3.13.post1 |
| transformers | 5.16.1 |
| pytest | 9.1.1 |
| jsonschema | 4.26.0 |

Local metadata does not establish remote environment agreement or native backend capability.

## Existing integration boundaries

- `run.prepare`: native CPU preparation, teacher/input identity, actual tensor byte accounting, legacy compatibility snapshot.
- `topology.py`: producer routing, DP-major/TP-minor readers, native microstep cursor.
- `cluster.NodeMonitor`: node/environment/memory observations and owned-process tracking; preserve existing PID/start-time fixes.
- `actors.consumer_command` and `recipe.py`: native torchrun/TorchTitan DSpark, shard DP and TP4; retain optimizer/loss math.
- vLLM `ParallelConfig` → `CoreEngineActorManager` → `RayExecutorV2`: planned borrowed-PG, CPU-core and preload-gate seams, not yet implemented or verified.
- `memory.py`, `BufferLedger`, Store/prefetch: byte reservations, designated-reader ACK and confirmed deletion.
- `runtime.MooncakeMaster` / `orchestration.process`: owned service and subprocess supervision.
- Native `dspark_draft/checkpoint.py`: authoritative DCP/commit; fixed shard-count assumptions are invalid.

## Validation log

With the activated interpreter and `CUDA_VISIBLE_DEVICES=''`, before implementation:

```text
python -m pytest -q tests/test_pipeline_buffer.py tests/test_pipeline_cluster.py -k 'not real and not smoke'
28 passed in 12.56s
```

This is local unit/subprocess evidence, not Ray placement, transport or model-training acceptance. New implementation results are appended below.

T002/T003: `tests/test_pipeline_acceptance.py`: 28 passed in 0.42s. Synthetic M0–M3 configurations cover 1/3/5 updates; acceptance storage refuses unexecuted/missing evidence and preserves prior records. No real case executed.

T004–T018: 107 tests passed in 10.80s (`test_pipeline_acceptance`, `foundations`, `plan`, `cli`, `inspection`, existing `buffer` and `cluster`). Expected red stage: missing controller, planner and CLI imports before implementation. Synthetic 13-case preview contracts recorded separately in `outputs/ray-topology-acceptance/results.json` using `python -m tests.run_pipeline_acceptance --cpu-contract --case all`. They are CPU contracts only, not training acceptance. Real native capability currently reports missing until T045–T048. No live Ray/model probe was executed.


T019–T027: implemented per-node startup/remaining bounds, request-specific fresh admission at every update boundary, reader-copy lifetimes independent of source deletion, strict writer shape/dtype and completed-transfer publication, bounded prefetch credits retained on timeout, and confirmed-absence deletion before source credit returns. Red tests exposed missing admission fencing and same-byte shape/deletion/prefetch gaps; those were corrected. Conservative retained memory credit remains zero unless explicitly proven by a node report.

CPU evidence: broad regression reached 126 passing tests before the last admission/writer cases; focused acceptance/memory/buffer tests subsequently passed 59 tests. The final `python -m tests.run_pipeline_acceptance --capacity-contracts` passed 43 CPU tests (8 real-service tests explicitly deselected). The final suite evidence is `outputs/ray-topology-acceptance/cpu-contracts/capacity/e80d73b7376344ccbbadb68dfae5346a/`; the append-only index distinguishes its test-suite identity from a topology plan. These are explicit backend doubles and CPU operations, not model/128K acceptance.

Real local transport regression: `python -m pytest -q tests/test_pipeline_store.py -k 'not cpu_contract'` under a task-owned subprocess supervisor passed 8 tests in 377.12 seconds. Evidence: `outputs/ray-topology-acceptance/local-store-probe/39de3527fe3a4361923f90845f19eb6c/`. This used owned local Ray, 64 MiB Mooncake pools and CPU/Gloo ranks, with no GPU/model or three-node run. The launch request was recorded, but no source snapshot was captured at probe launch; the index explicitly preserves this limitation. Existing external Ray head/GCS/raylet PIDs remained alive after the probe.

Real M0–M3 training and native borrowed-PG integration remain unexecuted. Three-node address/selectors and acceptance input are requested from the user; no response at this checkpoint. Lifecycle work must pass its own gates before model launches.

Phase 5 is in progress, not complete. Added two CPU gate contracts, exact-PG allocation/rollback primitives, PID/start-time and run-marker process checks, finite supervised process handles, lease fencing and a node watchdog. Controller/group/CLI integration is still required before native backends are enabled. The user reaffirmed that execution must use `h800conda.sh`; the active interpreter remains its `PIPELINE_PYTHON`.

- Process/lifecycle/cluster regression reached 34 passing tests. It includes driver SIGKILL before/after child launch, new-session grandchildren, finite stop, PID reuse, unreadable ownership, lease expiry while control is blocked and bystander survival.
- A real four-test CPU Ray/Mooncake regression passed its test assertions but failed the supervisor cleanup gate (`local-store-probe/1b5695b7d8ca4cd8ba92cece3df99301`). Three previously tracked processes had transient `/proc/environ` failures during exit, which were retained after the processes exited. The fix retains observation history and resolves uncertainty only after exit is observed; live/unreadable identities remain unknown. The independent run `local-store-probe/6de8fb33b80f4b269093c13144c14796` passed all four tests and confirmed cleanup. Both results remain in the index.
- Owned-master regression `local-store-probe/9f8cc8477f0b40edb5516b3e91f57841` passed real local TCP/CPU writes, full reads, confirmed deletion, retained reader-copy validity and bounded master cleanup. It used a 64 MiB pool and no GPU/model.
- Later lifecycle/process/store CPU tests: 28 passed, 8 real-service tests deselected. RPC timeout integration plus buffer/writer/cluster tests: 50 passed. Latest buffer/lifecycle/process selection: 43 passed. Native-close errors/timeouts retain buffer ownership; stale supervisor cleanup files cannot establish success.

Remaining Phase 5 work includes nonblocking Producer/Consumer control, concrete StoreService and transport-check, RunController/cancel, full phase-fault coverage and real CPU lifecycle probes. Native vLLM seams, training integration, verifier, full event sources and GPU acceptance remain pending. Standalone helper or transport success does not complete these tasks.

Phase 5 continuation (2026-09-20): Producer/Consumer now expose background start and independent ready/status/stop controls; late completion cannot revive a stopped actor. Consumer retains its native supervisor handle. FeatureBuffer supports deferred native pool construction so ownership is registered before allocation begins. ActorAllocator persists exact actor IDs before readiness waits, confirms DEAD on release, disables actor restarts and limits detached lifetime to leased NodeAgents. NodeAgents runs independent heartbeats and registers native process identities before service startup. StoreService connects one FeatureBuffer pool to an owned or borrowed master; external master cleanup never stops the external service.

Added an operations-based RunController with a one-use execution claim, shared run/cleanup deadlines, independent verification as a success condition and first-cause retention. Native inference/training operations are not wired yet, so this does **not** enable CLI `run`. CLI `cancel --run-dir` validates run/plan/config/execution identity, atomically publishes a persistent request, waits boundedly for cleanup and preserves existing terminal state. CLI `transport-check --plan` now runs a separately supervised CPU driver with a unique probe namespace and 64 MiB pool across the plan's actual writer/reader node set. It checks every tensor, confirms deletion and records owned-resource cleanup; it cannot establish full-feature capacity or training correctness.

Validation: controller/CLI/groups/acceptance selection passed 62 tests; transport matrix/CLI/group selection passed 13. The broad CPU regression passed **169 tests** (14 upstream torch.jit deprecation warnings). Two real local CPU lifecycle probes passed with independent supervisor cleanup and bystander survival: `lifecycle-probe/17c60e0c6e6b4c2692d2d48ad87b8fb0` covered owned service startup/stop, and `lifecycle-probe/c80916229ede462fb6a5ca121ee4921e` additionally exercised the actual transport-check entrypoint through a separate driver. The latter's transport evidence is under `probes/c4b59d4df57348d48122791860eb6f30/`. These probes used synthetic topology metadata and an owned zero-GPU local Ray cluster; they are not real M0/M1 placement or M2/M3 evidence. Source files were still being edited during these development probes, so their launch snapshots do not establish a frozen final-source gate; a stable-source rerun is required.

A new real process regression exposed a cleanup race: NodeAgent could kill a supervisor at the same one-second escalation point at which that supervisor was about to reap a SIGTERM-resistant child. Supervisors now retain their grace period until near the common cleanup deadline; ordinary registered processes still escalate promptly. The focused regression is being checked before further native integration.

The SIGTERM-resistant-child fix passed all 11 process tests. A subsequent fixed-source CPU Ray probe (`lifecycle-probe/d8893eec5f55431ea6d216cc746504f0`) passed its 12,619,792-byte full transport check but **failed** the driver-SIGKILL/orphan gate. Raylet's per-worker process-group cleanup also killed the supervisor, which had inherited its actor parent's process group; the native master in its own session survived without cleanup proof. The installed Ray `dashboard/modules/job/job_supervisor.py` documents this same watcher requirement. `start_owned` now requires a separate session for the supervisor itself, in addition to its native child. A new CPU regression first reproduced the shared-process-group defect before the fix. The failed result is retained. Its one remaining master (PID 2509865) was separately checked against the fault run marker and start time, terminated through `signal_process`, and confirmed released; this repair is recorded in `manual-cleanup.json` and does not change the failed acceptance result. Existing external Ray processes remained alive.

The acceptance runner now verifies that source hashes are unchanged throughout a real lifecycle probe and preserves readable partial results when the child exits unsuccessfully. A new complete fixed-source probe must pass after the supervisor-session fix before closing the lifecycle gate.

Still pending in Phase 5: concrete full-run operations wiring, complete blocked-phase/driver-loss CPU Ray fault probes, final stable-source evidence and closure of all T028–T040 acceptance requirements. No Phase 5 task is marked complete solely from the partial helpers above. Original external Ray head/GCS/raylet PIDs 2367074/2367370/2367663 remained alive after the new probes.

Supervisor diagnosis continuation: the separate-session change alone did **not**
fix the original fault. Stable-source run
`lifecycle-probe/9f3ba174023243bf9082b5b28a0175b2` failed with the same missing
supervisor report. Its leaked native master PID 2513941 was independently
verified, stopped and confirmed released in `manual-cleanup.json`; the failed
result remains unchanged. Thus the earlier process-group explanation was
incomplete, not a demonstrated complete diagnosis.

Using the `diagnosing-bugs` workflow, `tests/pipeline_supervisor_probe.py` reduced
the problem to one real zero-GPU Ray actor, `start_owned`, and a sleeping Python
child. Two baseline attempts and one explicit enabled-setting attempt reproduced
the missing report and surviving child. The exact actor worker logs name the
supervisor PID in `core_worker_shutdown_executor.cc`'s direct-child kill list.
Ray's CoreWorker kills direct children regardless of the new session. Disabling
`RAY_kill_child_processes_on_worker_exit` **only for that actor** produced confirmed
cleanup in 0.126 seconds, without changing Raylet or cluster settings.

The retained regression uses the production `ActorAllocator`/`RayActorBackend`
entrypoint. Before the fix, `supervisor-regression-red/d1849282fca04873bcc1c6f848bc4b38`
failed. After the fix,
`supervisor-regression-green/43a0cc37bae04ab6adf0068da034b2c7` passed and recorded
parent-loss cleanup in 0.141 seconds. Only `OwnedMasterActor` and `Consumer`, which
launch native children through the independent bounded subreaper, opt into the
actor-local policy. The legacy Consumer creation path applies the same policy.
Other actors and shared Ray settings retain their original cleanup behavior.
The diagnostic harness remains explicitly labeled as a CPU process regression;
its direct-actor controls are diagnostic variants, not topology acceptance.

Latest focused lifecycle/process/CLI/group/transport regression: 48 passed in
21.17 seconds. The corresponding files pass Ruff. Earlier separately recorded
stable suites are `lifecycle-tests/lifecycle-contracts/55fab1bcefbe4731b605a2ebca1c38d5`
(36 CPU contracts) and
`lifecycle-tests/process-contracts/616426dbc8d344ec835c17200698cc4d`
(12 CPU/process tests). A new frozen-source full lifecycle probe includes the
original driver SIGKILL case plus actual blocked-RPC cancellation in all six
controller phases. Its result must pass before Phase 5 closure.

Phase 5 closed after stable-source run
`lifecycle-probe/0429fb9898b84a179eb03203ce5b757c` passed all seven acceptance gates
in 212.64 seconds (`record_id=59a86f258f5040e6991237cdbdc0f022`). It confirmed the
real CPU service and full transport path, bystander survival, outer-supervisor
cleanup, driver-SIGKILL cleanup, six actual blocked-RPC cancellations, and
unchanged source hashes. Driver-loss cleanup took 4.196 seconds, with every
registered process released, all actor IDs DEAD, and the native master
supervisor's report present. Cancellation used the real CLI entrypoint in
allocate/initialize/ready/run/drain/verify, returned 130, preserved CANCELLED as
the first cause, and retained identical results on repeat. No GPU/model was
started. T028–T040 are now checked on this bounded CPU lifecycle scope; the
concrete native operations adapter and `run`/independent DCP CLI wiring remain
T055, rather than prerequisites being claimed complete here. Actual multi-node
and native-training faults still require T075/T077.

Phase 6 implementation has started; T041 onward remain unchecked until their
complete requirements are exercised. Added explicit v1/v2-to-v3 migration and
connected it to `preview`. The first 17 migration tests failed at the missing
upgrade entrypoint, then passed after implementation. Subsequent cluster/CLI/plan
selection passed 75 tests; expanded cluster/CLI selection passed 46. Migration
preserves independent pDP/cDP, batch, inflight/window/prefetch limits and exact
RDMA-device strings (including TCP configurations). Conflicting old/new fields
name both paths; ambiguous historical split-DP node declarations require explicit
selectors. New grouped fields remain strict. Explicit unsupported legacy tuning
options are rejected instead of silently dropped. Prepared-run replay/legacy
launcher unification is still pending T057.

Native vLLM changes are limited to the planned config/manager/core/V2 seams.
`ParallelConfig.bind_ray_placement` owns copied declarations, excludes handles and
callbacks from JSON, excludes placement from graph hash, and validates group IDs,
one-GPU worker bundles, a separate CPU core bundle, local core ranks and finite
startup policy. The borrowed CPU core branch preserves local DP identity without
computing a local TP GPU range. CoreEngineActorManager reads the bound groups,
uses a common startup deadline, registers cores through an optional runtime
callback, cleans partially created cores on error, and never removes borrowed
PGs. Native V2 reports deferred worker placement, awaits the external all-role
gate, then performs its existing worker initialization; callbacks also report
worker initialization and failures. Native execution/math is unchanged.

`tests/test_pipeline_vllm_placement.py`: the initial eight missing-API tests failed,
then passed; two manager tests reproduced unwanted automatic placement before
the manager fix; two V2 tests failed at the missing gated initialization seam
before the V2 implementation. The latest selection passed all 12 CPU seam tests
in 52.80 seconds, including blocked-gate zero-initialization and non-contiguous
physical GPU mapping. Ruff checks pass for the four native files and seam tests.
These tests import the actual native modules with explicit CPU doubles. They do
not establish real Ray GPU placement, complete adapter integration, model output
equivalence, or native training acceptance. Native capability markers remain
disabled until the DeepSpec adapter is wired and validated.

Phase 6 continuation: preflight now takes a fresh, fenced memory sample after
slow identity hashing and resource inspection. Source identity covers nested
pipeline modules, the process supervisor, native trainer and all four vLLM seams.
Inspection/CLI/plan regression: 47 passed in 9.24 seconds. The ownership registry
can bind an observed process and GPU UUIDs to an existing allocation without
changing its owner/borrower; conflicting/reused/late observations are rejected.
Foundation/lifecycle regression for that addition: 35 passed in 10.94 seconds.

`NativeCoordination` now records native actor creation and actual process/device
observations in one ledger, acknowledges NodeAgent process ownership before the
allocation gate, tolerates creation/self-report arrival order, and retains
unconfirmed cleanup as unknown. Native callbacks derive the PG bundle from Ray's
actual indexed resources and GPUs from physical indices plus UUID inventory.
Runtime handles stay outside JSON. `NativeInferenceAdapter` binds normalized
native config and uses `AsyncLLM.from_vllm_config` for both DP1 and DP2. Tests
exposed the native singleton fallback's local-rank-zero/offline behavior; the
adapter restores online rank/address defaults only after rejecting explicit
environment overrides. Capability markers now cover all four native seams;
markers enable hook detection, not training acceptance.

`InferenceGroup` creates one separately budgeted CPU frontend and borrows the
controller's exact PG handles. Producer native mode uses the adapter and all-role
gate. Async production reserves in frozen input order and permits the configured
per-replica batch, with existing window and writer credits still enforced.
The group/buffer/lifecycle selection passed 49 tests in 12.31 seconds. Expanded
adapter/native-seam/group/inspection selection passed 42 tests in 58.80 seconds.
These are CPU doubles and native module tests, not actual GPU allocation or
training. T041 onward remain unchecked pending their complete requirements.

All commands continue to source `h800conda.sh` and invoke `$PIPELINE_PYTHON` with
`CUDA_VISIBLE_DEVICES=''` for CPU validation. Upstream test collection initially
failed because `tblib` was absent. Test-only helpers and mypy 1.20.2 (the version
in vLLM's pre-commit configuration) were installed with `--no-deps --target` under
`outputs/ray-topology-acceptance/test-tools`; the conda environment was not changed.
With this directory on PYTHONPATH, the upstream DP placement allowlist, Ray output
copying, and background worker shutdown tests passed: 8 tests in 2.28 seconds.
The four-file Python 3.12 mypy check reported 28 errors, including pre-existing
native annotations and new optional/type narrowing issues. This check is not
passed; introduced issues and baseline attribution are being resolved.

## Phase 6 continuation: independent checkpoint verification

Reloaded the local `speckit-implement` skill and ran the installed core-pack
prerequisite script from this repository. It resolved this feature and all
required documents. The requirements checklist remains 16/16, unchanged; no
extension hooks or constitution were found. Existing worktree changes, deleted
files, training environments and vendor edits were preserved.

Added `deepspec/pipeline/verification.py` and the T043 CPU tests. The verifier
uses native `read_commit` and the pinned DCP loader, checks complete schema,
tensor range coverage, storage extents, shape/dtype and finite state, and reads
every saved field on CPU. A caller-approved working-memory budget is checked
before loading state tensors. Counts come from the frozen plan, including
DP1/DP2 with one, three and five updates; neither rank count nor shard-file count
is used as a checkpoint completeness shortcut.

`TrainingHandshake.initialized` now captures each rank's native state schema and
initial model chunk hashes before its initialization ACK. Initial records are
write-once, and their file digest is retained in the rank's event stream and
initialization report. This allows final CPU parameters to be checked against
pre-update state, rather than accepting a checkpoint's own success claim. The
context projections (`fc`, first-layer K and V) must change. The capture follows
DCP's chunk protocol, including multiple local chunks. A real
`LocalShardsWrapper` regression first exposed a failed reshape in the naive
single-tensor implementation; native chunk enumeration/shard lookup fixes it.

Additional checks require all planned rank updates, finite losses, exact sample
and reader coverage, each rank's matching native commit, confirmed source
deletion, and matching released resources/cleanup observations. Unknown cleanup,
missing readers, forged success flags, late/modified initial evidence, absent
state and truncated storage are rejected. `verify_execution` combines these
checks without modifying controller status. Runtime allocation/execution
placement and metric completeness still require their later acceptance gates.

Validation commands use `source ./h800conda.sh` and `CUDA_VISIBLE_DEVICES=''`:

```bash
"$PIPELINE_PYTHON" -m pytest -q tests/test_pipeline_training.py \
  tests/test_pipeline_vllm_adapter.py tests/test_pipeline_vllm_placement.py \
  tests/test_pipeline_groups.py
"$PIPELINE_PYTHON" -m tests.run_pipeline_acceptance --verification-contracts
ruff check deepspec/pipeline/verification.py deepspec/pipeline/training.py \
  deepspec/pipeline/runtime.py tests/test_pipeline_verification.py \
  tests/run_pipeline_acceptance.py
```

The existing native adapter/group/handshake selection passed 55 tests in 57.02s.
The initial verifier test collection failed at the missing module, followed by
26 passing CPU DCP tests. Expanded stable-source runs passed 72 tests
(`record_id=35042501f9d8486aa0887b5edb2b75be`) and 102 tests
(`record_id=f78b9047a5584a85b0aa3981dee83b41`). The latter includes immutable
initial-record and existing acceptance-recorder regressions. Native optimizer
and scheduler coverage uses actual `DraftOptimizers`/`DraftSchedulers` on a tiny
CPU model; topology and runtime evidence are explicitly synthetic.

The acceptance runner now supports `--verification-contracts`, retains source
hashes, JUnit and full logs under separate run IDs, and rejects a passing record
if the source changes during testing. The final multi-chunk run passed **103
tests** (`record_id=06c317408b5a460d83f891b10160b2e7`,
`run_id=438c6cb9b5334e8a8c2d8728e834dbfb`), with unchanged sources and passing
Ruff checks. Evidence is under
`outputs/ray-topology-acceptance/cpu-contracts/verification-contracts/438c6cb9b5334e8a8c2d8728e834dbfb/`;
the result is indexed in `outputs/ray-topology-acceptance/results.json`.

T043's CPU verification contract is implemented. T054 has its verifier and
initial-state capture implementation, but remains unchecked pending the full
native-run integration. T055 must still wire bounded CPU execution and its
preflight budget into `run`/`verify`; T080 must supply actual source/cleanup and
metric evidence. These changes do not enable or establish a real M0–M3 training
run, multi-node FSDP acceptance, GPU/model output equivalence, or throughput
improvement. The previously recorded vLLM mypy issues remain unresolved by this
continuation.


## Continuation: unified execution and native integration (2026-09-20)

The six v3 CLI commands now share frozen artifacts, bounded supervised drivers,
actual per-node resource observers and independent CPU checkpoint verification.
`operations.py` owns allocation, initialization, training, draining, model/PG
release, CPU verification and exact-resource cleanup. The compatibility launchers
now call the same preview/controller; the old debug wrapper selects v3 verification
for new runs. TCP/RDMA device strings remain distinct settings. The legacy
`retain_for_peak` pressure option remains explicitly unsupported by the v3 schema.

M3 has one four-GPU launcher per training node, one static endpoint reserved on the
first training node, frozen before launch, `max_restarts=0`, and a common native
TorchTitan world. CPU tests exercise the same eight-rank identity/handshake gates
for M1-12 and M3, plus actual local socket exclusivity and release. Native inference
placement tests also cover M2 DP1 and DP2. These are implementation/contract
results, not multi-node training evidence.

Validation in the existing h800conda environment: execution-contracts passed
210 tests (one deselected), record `8e9df14fe3784a72b5e9bcad23583a29`; subsequent
legacy/M3/Store regressions passed 74 tests, the affected native/CLI/plan selection
passed 127 tests, and the expanded training/placement selection passed 56 tests.
The legacy debug and M3 socket selection passed four tests. DSpark baseline,
TP numerics and native phase-checkpoint selection produced one pass and three
skips on CPU, not a GPU numerical acceptance result.

Real M0 allocation probe `345cccd710354d3e925d692bd3fc72b7` and partial-allocation
rollback `8ee754ce9c1946b9bbf733f203cc077e` passed, including independent supervisor
cleanup. First native attempt `47cac586b60540cca8c972725012e6ff` passed transport but
failed before models started because FeatureBuffer still indexed the old
`feature_memory_budget` field. A second regression showed that a failed initializer
prevented closing its partially allocated Store. Both errors were reproduced and
fixed; this failed run remains in the append-only index as record
`41d68b3af4f14ddc958e2c47289c6083`. Its owned actor/PG/process deaths were independently
confirmed, but its overall cleanup flag correctly remained false due to the close
error. Retry `eb6d86aafdf04fc187565c649b368502` has a new frozen plan, passed a fresh
allocation probe (`0bdb5244c240454abe94852914a2c76d`) and reached native initialization.
Its terminal result is recorded separately in the acceptance report/index.

The formal vLLM mypy hook was run with Python 3.12 from an isolated uv venv using
the existing training environment's system packages, without changing training
dependencies. A same-command `--shadow-file` baseline comparison found 17 existing
errors versus 11 current errors, with zero introduced error messages. This is not
a passing type check. Logs and the comparison are under
`outputs/ray-topology-acceptance/type-check/{ci.log,ci-baseline.log,comparison.json}`.
The reduced follow-imports profile also retained the same 32 preexisting errors.

The live external Ray cluster currently has one node and eight GPUs. Eleven M1,
M2 and M3 normal cases have explicit blocked inventory records; they have not been
substituted with synthetic passes. The remaining real training, fault and capacity
tasks stay unchecked until their required physical-node evidence exists.

## Continuation: native initialization and single-node evidence (2026-09-20)

The full execution-contract selection passed **366 tests**, with one deselected
and 58 warnings in 405.60 seconds (record `1556031ca4eb4c45b27284cc8722d6e7`).
After the fixes below, the affected native initialization, training handshake,
execution/status, cluster address/inspection, CLI, group, collective and acceptance
selection passed **124 tests**, with 14 warnings in 53.98 seconds (record
`ea02cd3639e54482b7071454b87fdbfb`). These overlapping selections are not added
together. The updated real eight-CPU-rank Store data probe also passed its DP2
regression (one test, 144.17 seconds; `plan-data-probe-regression-final.log`).
Ruff over the pipeline, acceptance/probe tools and affected tests, and
`git diff --check`, passed after these source changes.

The standalone eight-rank CPU Gloo probe now records all rank identities, TP/DP
collectives and independent supervisor cleanup. Its normal case passed (record
`e292f5715268491f83f0e0059767ea79`), as did an injected rank-7 failure with the
expected nonzero outcome and cleanup (record `6ba57f023c114dbc92671f178105cd8e`).
Both records explicitly say one physical node, no model/GPU, and do not establish
M3 cross-node behavior. The plan-driven rank/collective probe code checks frozen
node/rank identities and derives samples, GAS and both cursors from the plan.

Native M0 run `eb6d86aafdf04fc187565c649b368502` failed when external CI began using
the allocated GPUs. Its cleanup was confirmed; the foreign processes were not
terminated (training record `78c0bb1ddda8472e856dc9d1057dc9e4`). Structured
`PipelineError` now survives native Ray exception serialization, preserving code,
message, details and exit code. A later preview exposed `ray_address=auto`
re-resolving to a temporary CPU-test cluster. Preview now freezes the first
resolved GCS endpoint before native input preparation and all later inspections.

Native M0 run `f114b5b60ad6491ba25b1aaa7b4995e1` used the explicit external GCS
endpoint, passed real allocation and transport, and loaded all four vLLM workers
plus the native four-rank TorchTitan model. The handshake then failed because the
pinned `spmd_types` backend exposes its shard mesh as `dp_shard`, whereas `dtensor`
uses `fsdp`. `native_rank_groups` now selects the backend's actual mesh dimension.
The controller now polls role failure during both native gates and trainer
construction reports the original exception to the gate, instead of waiting out
the initialization deadline. Both backend names, DP1/DP2 and both gate/role
failure combinations have passing regressions. The old failed run remains
immutable (record `6cd54a78c40941edb36efa728823e506`, cleanup confirmed); a fresh
real-model retry is required to validate the fixes in training.

The user confirmed that only one physical node is available. At 20:40 local time
all eight GPUs were occupied by external CI (inventory
`resource-inventory/e8b9779370c34afcbc58c6701bee86c2.json`); M0-4K and M0-128K were
recorded blocked at that observation. They became idle later and a fresh M0-4K
attempt was started under `native-m0/98186a02a69541f389dd6c7039a1ca42`. Its terminal
outcome belongs in the acceptance report and append-only index; starting it is
not evidence of passing. Multi-node matrix cases remain blocked.

T055, T078, T081 and T082 now have implementation plus execution/status/verifier
contract coverage: sealed-run checks, post-GPU-release budgeted CPU verification,
read-only status with read/commit separation and lease/orphan handling, and strict
source/checkpoint/cleanup success requirements. The status tests live in
`tests/test_pipeline_execution.py` alongside their fixtures. Actual training and
fault acceptance tasks remain separate from these implementation completions.

The fresh M0 attempt `7e2f569ba6414086b5d227c315660385` has now terminated.
Real allocation passed (`a31d4d5f644643758416263a4dee28b9`), as did TCP transport
and probe cleanup. External CI started another eight-GPU job after the initially
idle observation. The all-role allocation gate rejected foreign processes on the
four training GPUs before native model initialization. The actual failed training
attempt is record `0ccb0c559ed74f1da7689635217d08e8`; it did not reach the repaired
training handshake. All six cleanup phases passed, all eight independently
observed resources were released, and the driver supervisor confirmed cleanup
with no unknown or unverifiable processes. A read-only local status observer
recorded healthy allocation, failure, cleanup and the final terminal state under
`native-m0/98186a02a69541f389dd6c7039a1ca42/status-observation-6f2f2fd8b95c477ab4ebc725e04f5a30.jsonl`.
This is failure-state evidence, not a normal training or complete FR-018 pass.
No external process or Ray service was stopped. M0 needs a sustained eight-GPU
idle window; single-node availability alone cannot unblock the multi-node cases.

The acceptance report now selects normal-case outcomes only from training and
inventory evidence, so historical same-name CPU preview passes cannot appear as
successful M0-128K training. Example JSON schemas and report/quickstart links were
checked. The task checklist contains 89 unique tasks, of which 66 are complete.
