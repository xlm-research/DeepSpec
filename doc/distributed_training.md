# Modular distributed training

This document records the repository audit, the new training architecture, and
the commands that were actually exercised on 2026-08-21. The implementation is
fixed-degree Context Parallel; `dynamic_context_parallel` is an explicit future
scheduler/backend boundary and is not a variable-only pseudo implementation.

## Repository audit

The selected production config is
`config/dflash2/dflash2_qwen3_8_27b.py`. Its target checkpoint is
Qwen3.8-27B, but the trainable model is a dense five-layer DFlash2 draft model,
not the 64-layer target model.

Exact trainable model FQNs:

- root: `Qwen3_8DFlash2Model`
- Transformer blocks: `layers.{0..4}`
  (`Qwen3_8DFlash2DecoderLayer`)
- Attention: `layers.*.self_attn` (`Qwen3DSparkAttention`)
- Q/K/V/O projections: `layers.*.self_attn.{q_proj,k_proj,v_proj,o_proj}`
- MLP: `layers.*.mlp` (`Qwen3MLP`)
- MLP projections: `layers.*.mlp.{gate_proj,up_proj,down_proj}`
- norms: `layers.*.{input_layernorm,post_attention_layernorm}`
- DFlash2 convolutions: `layers.*.{attention_conv,mlp_conv}`
- target-feature projection: `fc`; hidden norm: `hidden_norm`
- final norm/head: `norm`, `lm_head`; token embedding: `embed_tokens`
- path selector: `candidate_selector`, with
  `{predecessor_codebook,successor_codebook,hidden_projection}`

There is no trainable MoE router or expert container in this draft model.
`deepspec/modeling/pure_ep.py` is an untracked external-model adapter in the
audited workspace, not a DFlash2 MoE layer. Therefore `ep=1` works and `ep>1`
is rejected before model transformation. The native top-k token dispatcher is
implemented and tested for a future model adapter, but the current dense model
does not claim FSDP2+TP+EP support.

The old training path used a parent-side `torch.multiprocessing.spawn`, FSDP1,
a detached FP32-master AdamW wrapper, and per-rank
`training_state.rank<N>.pt` files. Gradient accumulation was derived from
`global_batch_size / (data_parallel_size * local_batch_size)`. DSpark/DFlash2
losses used valid-token denominators; Eagle3 intentionally uses its historical
per-sequence local mean. Models are explicitly stored/executed in BF16 rather
than using autocast, and no GradScaler was active. The scheduler is cosine with
linear warmup and advances once per optimizer update.

No DDP, DeepSpeed, Megatron Core, Accelerate, or `nn.DataParallel` training path
was found. FSDP1 remains only in the offline target-cache preparation script;
the new trainable draft-model engine uses FSDP2 `fully_shard` exclusively.

The previous checkpoint contained:

- a Hugging Face-compatible full draft model (`config.json` plus safetensors),
- copied `train_config.py`,
- optimizer/scheduler/RNG state in one `torch.save` file per rank.

The new checkpoint retains the HF export for evaluation compatibility and adds
`distributed_checkpoint/`, containing model, optimizer, scheduler, progress,
data position, RNG state, parallel config, and model config through
`torch.distributed.checkpoint`.

## Architecture and ordering

`ParallelConfig` is loaded from `train.parallel`. Configs without that nested
mapping are translated from the old `fsdp_size`, `context_parallel_size`, and
`torch_compile` keys. `ParallelContext` constructs every mesh/process group
once before model construction is transformed. No forward or training step
creates a process group.

The dense rank layout is:

```text
(dp_replicate, dp_shard, cp, tp)
```

The sample/data-loader rank is `(dp_replicate, dp_shard)`; CP and TP ranks see
the same logical sample. Dense FSDP uses `(dp_replicate, dp_shard * cp)` so that
different CP token/anchor contributions update one parameter state. Loss and
metric normalization covers `(dp_replicate, dp_shard, cp)` and excludes TP,
which prevents TP-duplicated token counts.

For future MoE adapters, the same ranks have an alternative sparse view:

```text
(dp_replicate, expert_fsdp, ep)
dp_shard * cp * tp == expert_fsdp * ep
```

The transformation order is TP, model-provided EP, activation checkpoint,
per-block `Module.compile`, bottom-up block FSDP2, then root FSDP2. The optimizer
is created only after these transformations. The trainer always calls the
module (`model(...)`) so compile/FSDP hooks are not bypassed.

The current DFlash2/DSpark attention is not standard self-attention: its draft
queries attend to a CP-sharded cached target context plus replicated draft K/V
using FlexAttention masks. PyTorch's prototype `context_parallel` intercepts
SDPA and cannot express this mixed-input operation. Consequently:

- generic SDPA modules use `context_parallel_backend="pytorch"`;
- DFlash2/DSpark use the existing exact fixed-degree ring FlexAttention path
  with `context_parallel_backend="model_native"`;
- TP Sequence Parallel is not treated as CP;
- Sequence Parallel and Loss Parallel are rejected for DFlash2 because its
  multi-input block and full-vocabulary candidate selector do not preserve the
  required layouts.

## Runnable profiles

The profiles use the existing DFlash2 model/data config and differ only in
`train.parallel`:

| Profile | Degrees | Status for current model |
| --- | --- | --- |
| `single_gpu.py` | all 1, FSDP off | supported |
| `fsdp2_8gpu.py` | DP-shard 8 | supported |
| `tp_8gpu.py` | TP 8 | supported |
| `cp_8gpu.py` | CP 8, model-native fixed CP | supported |
| `ep_8gpu.py` | EP 8 | validation template; rejected because model is dense |
| `fsdp2_tp_8gpu.py` | DP-shard 2, TP 4 | supported |
| `fsdp2_tp_cp_8gpu.py` | DP-shard 2, TP 2, CP 2 | supported |
| `fsdp2_tp_ep_8gpu.py` | DP-shard 2, TP 4, EP 4 | MoE template; rejected for current model |

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 train.py \
  --config config/distributed/single_gpu.py \
  --opts "data.target_cache_path=/path/to/cp1/cache"
```

Eight-GPU examples (replace the profile as needed):

```bash
TARGET_CACHE_DIR=/path/to/cp1/cache \
CONFIG_PATH=config/distributed/fsdp2_8gpu.py \
bash scripts/train/torchrun_train.sh

TARGET_CACHE_DIR=/path/to/cp2/cache \
CONFIG_PATH=config/distributed/fsdp2_tp_cp_8gpu.py \
bash scripts/train/torchrun_train.sh
```

The target cache CP degree must equal the selected training CP degree. A CP1
cache cannot be reused by CP2/CP8.

Multi-node follows ordinary torchrun semantics:

```bash
torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" --master-port="${MASTER_PORT}" \
  train.py --config config/distributed/fsdp2_tp_8gpu.py \
  --opts "data.target_cache_path=/shared/cache"
```

## Checkpoint save, load, and migration

No extra save flag is required. Each configured checkpoint step writes:

```text
step_<N>/
  config.json + model safetensors       # existing evaluator/HF format
  train_config.py
  distributed_checkpoint/              # reshardable training state
  distributed_checkpoint_metadata.json
```

`step_latest` is updated only after all distributed writers complete. Restart
the same command to auto-resume model, FP32 master optimizer, scheduler,
progress/data position, and RNG state. A different DP/FSDP world size may load
the distributed checkpoint; the 2-rank to 1-rank test command below exercises
that path.

Old HF model checkpoints remain directly loadable and evaluable. Old per-rank
optimizer files are accepted only when their local parameter shard layout
matches the new optimizer template. In particular, an old CP+FSDP1 optimizer
whose FSDP domain excluded CP cannot be safely reinterpreted as the new
`dp_shard*cp` domain. Keep the old checkpoint, use its HF weights as a warm
start, and generate the first new distributed checkpoint before changing world
size. There is no destructive in-place conversion.

## Support matrix

“Tested” below means the listed local command actually completed on the audited
host; it does not mean a full 27B-data training run was performed.

| Capability | Implementation | Validation |
| --- | --- | --- |
| single GPU | implemented | optimizer/checkpoint/unit paths tested; full DFlash2 run not run |
| torchrun + named DeviceMesh | implemented | 2 GPU tested |
| FSDP2 bottom-up | implemented | 2 GPU forward/backward/clip/update tested |
| TP QKV/O and MLP | implemented | exact Qwen FQNs checked; 2 GPU numerics tested |
| Sequence Parallel | not enabled for current multi-input blocks | early error |
| Loss Parallel | not enabled for current full-logit selector | early error |
| PyTorch fixed CP | implemented for SDPA | 2 GPU loss/backward tested |
| DFlash2 fixed CP | existing ring backend retained | static/inherited tests; full 27B run not rerun |
| native EP dispatcher | implemented | 2 GPU top-k dispatch/combine/backward tested |
| EP on current DFlash2 | not applicable (dense model) | early error tested |
| FSDP2 + TP | implemented | 4 GPU update tested |
| FSDP2 + TP + CP | implemented | 4 GPU native-CP update tested |
| FSDP2 + TP + EP | adapter boundary implemented, no current MoE model | not claimed |
| distributed checkpoint | implemented | CPU and 2 GPU same-world round-trip tested |
| checkpoint reshard | implemented by PyTorch state-dict APIs | 2 GPU save → 1 GPU load tested |
| activation checkpoint | implemented | CPU forward/backward tested |
| `torch.compile` | optional per block | 1 GPU block forward/backward tested; full model not run |
| DeepEP | auto fallback to native | unavailable in audited environment |
| Dynamic Context Parallel | scheduler/backend interface only | intentionally not implemented |

## Environment and actual test commands

Audited runtime:

- 8 × NVIDIA B300 SXM6 AC, compute capability 10.3, 275040 MiB each
- NVIDIA driver 580.95.05
- Python 3.12.13
- PyTorch 2.11.0+cu130, CUDA build 13.0, NCCL 2.28.9
- Transformers 5.12.1, Triton 3.6.0, flash-attn 2.8.3
- safetensors 0.8.0, NumPy 1.26.4, PyYAML 6.0.3
- DeepEP and pytest absent; tests use the standard-library `unittest`

`requirements.txt` currently pins PyTorch 2.9.1, Transformers 5.10.2,
Triton 3.5.1, and NumPy 2.4.4. The composed TP+FSDP mesh uses PyTorch's
flattened-root mesh view so both DTensor dimensions retain the same root-mesh
identity; PyTorch 2.11 emits a deprecation warning for that slice even though
its FSDP+TP concatenation currently requires the shared identity. Production
validation in this refactor was on the versions above.
Re-run the distributed tests before deploying the pinned 2.9.1 environment.

Commands exercised:

```bash
python -m unittest discover -s tests -p 'test_*.py'

torchrun --standalone --nproc-per-node=2 -m unittest tests.test_mesh
torchrun --standalone --nproc-per-node=2 -m unittest tests.test_tp_numerics
torchrun --standalone --nproc-per-node=2 -m unittest tests.test_fsdp_numerics
torchrun --standalone --nproc-per-node=2 -m unittest tests.test_cp_numerics
torchrun --standalone --nproc-per-node=2 -m unittest tests.test_ep_numerics
torchrun --standalone --nproc-per-node=2 -m unittest tests.test_checkpoint_roundtrip
torchrun --standalone --nproc-per-node=4 -m unittest tests.test_composed_numerics
torchrun --standalone --nproc-per-node=4 -m unittest tests.test_fsdp_tp_cp_numerics
torchrun --standalone --nproc-per-node=1 -m unittest tests.test_compile

torchrun --standalone --nproc-per-node=2 \
  -m tests.distributed_checkpoint_reshard_worker \
  --phase save --checkpoint-dir /tmp/deepspec-reshard
torchrun --standalone --nproc-per-node=1 \
  -m tests.distributed_checkpoint_reshard_worker \
  --phase load --checkpoint-dir /tmp/deepspec-reshard
```

The first native-CP attempt used an unrealistically small four-token local
sequence and exposed a PyTorch 2.11 efficient-attention LSE tile shape error.
The production-representative test uses sequence 128/head dimension 64 and
passes. Very short native-CP sequences remain a prototype-API risk.

Observed final results were: local discovery `41 tests, OK, 10 skipped`
(distributed-only cases skip without torchrun); every listed 1/2/4-GPU command
completed with `OK`; and the separate 2-rank save plus 1-rank load process exited
successfully. PyTorch 2.11 emits the flattened-mesh deprecation warning described
above, and the local environment emits unrelated SWIG/JIT deprecation warnings.

## Remaining risks

- Run at least several optimizer steps of the real Qwen3.8 DFlash2 model with a
  completed CP-matched target cache before production deployment; no complete
  cache was available in the audited workspace.
- Benchmark peak memory and throughput for each 8-GPU profile. Toy numerical
  tests do not provide meaningful MFU or 27B-model memory data.
- Certify PyTorch 2.9.1 separately or update the dependency pin to the tested
  2.11 build after cluster rollout policy is decided.
- Add a concrete MoE model adapter before claiming FSDP2+TP+EP. It must expose
  router semantics and expert modules to the provided dispatcher; DeepEP must
  be version-certified rather than selected only because it imports.
- If Dynamic Context Parallel becomes a hard requirement, choose a compatible
  main engine and implement micro-batch packing, GPU-domain assignment,
  communication-group selection, and token-normalized loss together.
