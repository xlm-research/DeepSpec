# Qwen3.8-27B DSpark with a vLLM teacher

The production entry point is `scripts/fsdp/qwen3.8-27b_dspark.sh`. It follows
`scripts/fsdp/glm5.3-flash_dspark.sh`: copy the environment tar to a unique
directory under `/tmp`, unpack it locally, set `VLLM_WORKER_MULTIPROC_METHOD=fork`,
and use the unpacked Python for both training and the teacher. It invokes
`scripts/train/train_qwen3_8_27b_dspark_vllm.sh` with
`config/dspark/dspark_qwen3_8_27b_vllm.py`. The original Qwen entry point and
the GLM-5.3-Flash trainer retain their existing behavior.

## Teacher features

`model.target_layer_ids=[1,31,61]` selects zero-based decoder outputs.
The vLLM auxiliary slots are `[2,32,62,64]`. All 64 teacher layers run.
The three intermediate states are concatenated into `[1,T,15360]`. With
`extract_final_hidden_state=true`, the extractor replaces the last slot with
the actual normalized state sent to the LM head, producing `[1,T,5120]` final
supervision. Qwen's offset RMSNorm, `(1 + weight)`, executes inside the teacher.
The exported final state is not normalized again: reconstructing it from the
rounded BF16 residual sum loses precision from the fused final normalization.
The full conversation is evaluated
as the prompt; the token sampled to finish extraction is excluded from the
cache. Input IDs and assistant loss masks come from the existing Qwen collator.

This differs from the original bounded Qwen native teacher, which truncates
the backbone at layer 61. Its normalized last state is not the full teacher's
final state. Use a separate checkpoint directory for this new training run.
The new directory records `qwen38_vllm_teacher.json`; resumes require the same
teacher identity. Draft checkpoints also record the vLLM backend and the
full-model final-state semantics.

## Partition and parallel behavior

Draft training defaults to CP=1 (no context parallelism) and TP=4 in both the
Python config and launcher. The teacher's TP is configured separately with
`VLLM_TENSOR_PARALLEL_SIZE` (default 4). The CP2 validation below is an optional
compatibility check, not a requirement for draft training.

The existing Qwen `train.data_partitions` schedule and optimizer loop are reused.
Partitions may end inside gradient accumulation. The draft, optimizer, and
accumulated gradients remain resident while an isolated vLLM subprocess generates
the next partition. This is not GLM's model-swap lifecycle.

Each node-local draft CP x TP group owns one vLLM replica. Teacher TP must divide
the group's size. Within that group, each input is inferred once; each CP shard
is written once and read by all draft TP ranks in that CP group. For CP > 1,
features are padded and rearranged into the native head/tail layout expected by
the Qwen draft. Input IDs and loss masks retain their full sequence order.

CPU/Gloo collectives synchronize extraction stages. Draft training resumes only
after all extraction workers have exited. Successful partitions are deleted
after their draft updates; failed partition inputs and worker logs are retained.
Checkpoint/resume and gradient accumulation use the original Qwen training loop.

At 131072 tokens, four BF16 states of width 5120 occupy about 5 GiB per sample
before storage overhead. The worker handles one sample at a time. Its default
GPU budget is 0.45 because the draft is still resident; adjust it against actual
free memory, teacher TP and sequence length.

This entry point enables `separate_hidden_state_pages` in the bundled vLLM
checkout. The hidden-state cache keeps its own page size; attention and Mamba
retain their original contiguous storage. Without this option, Qwen TP4's hybrid
allocator can shrink the hidden page to one token, making 128K extraction
require over 1 TiB of cache per GPU. The option is disabled by default, and GLM's
specialized cache allocation takes precedence. Both this option and
`extract_final_hidden_state` require the vLLM changes included with this entry.

## Launch

Run once on each node with the usual shared rendezvous configuration:

```bash
bash scripts/fsdp/qwen3.8-27b_dspark.sh
```

This wrapper locates the repository relative to its own path and reads
`envs/deepspec_vllm_torchtitan_env.tar`. Set `DEEPSPEC_ENV_TAR` to use another
archive, or `DEEPSPEC_ENV_DIR` to reuse an existing environment without copying
the archive. It defaults to draft CP=1, draft TP=4 and teacher TP=4.
Its default output directory is `output/dspark_qwen3_8_27b_vllm`; set
`OUTPUT_ROOT` to override it. It uses the unpacked environment directly without
creating a venv. Each node prepares its own local environment.

For a single node with four free visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 \
OUTPUT_ROOT="$PWD/output/qwen38_vllm_run" \
bash scripts/fsdp/qwen3.8-27b_dspark.sh
```

For diagnostics using an already prepared environment, the lower-level
`scripts/train/train_qwen3_8_27b_dspark_vllm.sh` still accepts `PYTHON_BIN` and
`VLLM_PYTHON_BIN` directly.

The launcher retains the original Qwen production defaults: 128K context,
1024 partitions per epoch, and the configured full training dataset. It also
retains existing controls for topology, learning rate, global batch size,
checkpoint frequency and resume. `BOUNDED_OFFLINE=false` is rejected by this
entry point. Set `PRODUCTION_RUN=false` when using diagnostic overrides such
as `MAX_TRAIN_STEPS` and a small `SOURCE_JSONL_PATH`.

Additional teacher controls:

| Environment variable | Default |
| --- | --- |
| `VLLM_PYTHON_BIN` | Training interpreter |
| `VLLM_SOURCE_DIR` | This repository's `vllm/` checkout |
| `VLLM_TENSOR_PARALLEL_SIZE` | 4 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 8192 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.45 |
| `VLLM_LOAD_FORMAT` | auto |
| `VLLM_TIMEOUT_SECONDS` | 86400 |
| `VLLM_VERIFY_LOGITS` | false |

`VLLM_VERIFY_LOGITS=true` checks the final prompt token's top-20 log probabilities
using the exported final state and the original LM head (absolute tolerance 0.1).
This adds an LM-head copy to the driver's GPU. Per-partition verification results
are stored as `vllm_rank*_partition*.json` in the checkpoint directory.

## Cache publication and failure handling

The teacher saves each feature file to `.pt.tmp`, flushes and synchronizes the
file, then renames it to `.pt`. Publication checks that the final file is readable,
has the saved size, and that the temporary path is gone. Transient filesystem
failures receive at most three attempts; persistent failures retain the temporary
file and stop extraction.

After extraction, every rank checks the metadata of every expected cache file
through a CPU memory mapping. This catches missing files, incomplete archives,
and invalid tensor shapes without loading all activations into RAM twice. It does
not perform a full payload CRC scan. All ranks agree on the validation result via
Gloo before any training batch is yielded. Each subsequent sample load also uses
that error exchange, so a consumer that loses access after validation causes all
ranks to fail before entering that sample's model collectives.

Job inputs and worker logs remain available until the partition finishes training.
They are retained on failure. A recovered feature file alone cannot restore lost
model/optimizer state; resume requires a matching training checkpoint.

Qwen launchers generate a cache directory containing the model, topology and
timestamp when `DATA_BATCH_CACHE_DIR` is unset. For a multi-node run, set the same
`DATA_BATCH_CACHE_TIMESTAMP=YYYYMMDD_HHMMSS` on every node. An explicitly supplied
`DATA_BATCH_CACHE_DIR` should belong to only one job. Use separate output roots
for diagnostic and production runs.

### Local debugging environment

The validated local interpreter is `.envs/qwen38-debug/bin/python`. It uses
Python 3.12.14 and PyTorch 2.13.0+cu130 from the extracted environment at
`/tmp/deepspec_vllm_torchtitan_envs`, with pytest 8.4.2 installed in the local venv.
The base environment must remain available. To recreate the venv:

```bash
/tmp/deepspec_vllm_torchtitan_envs/bin/python -m venv --system-site-packages .envs/qwen38-debug
.envs/qwen38-debug/bin/python -m pip install pytest==8.4.2
```

To use it with the current checkout's production wrapper:

```bash
DEEPSPEC_ENV_DIR="$PWD/.envs/qwen38-debug" \
  bash scripts/fsdp/qwen3.8-27b_dspark.sh
```

Model/data paths and distributed topology still come from the normal launcher
settings. Supply W&B credentials through the calling environment. The vLLM
launcher enables unbuffered Python output.

## Validation

Use the selected environment with both this repository and its `vllm/` checkout
on `PYTHONPATH`:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/vllm:$PWD" \
  /path/to/python -m pytest \
  tests/test_qwen38_vllm.py \
  tests/test_glm5_partitioned_model_swap.py \
  tests/test_qwen38_dspark.py \
  tests/test_qwen38_multinode_launcher.py -q

HF_HUB_OFFLINE=1 PYTHONPATH="$PWD/vllm:$PWD" \
  /path/to/python -m pytest --confcutdir=vllm/tests/v1 \
  vllm/tests/v1/core/test_kv_cache_utils.py \
  vllm/tests/v1/worker/test_gpu_model_runner.py -q -k hidden_state
```

The CPU tests preserve final features exactly, verify head/tail CP ordering and
padding, reject corrupt outputs, release child processes on success/failure/timeout,
and run two partitions through the actual training iterator with one optimizer
window. Bundled vLLM tests verify 128K cache admission and the opt-in final-state
extraction while checking the unchanged default behavior.
On 2026-09-08, the copied environment (Torch 2.13.0, Transformers 5.16.1,
vLLM 0.26.1rc1.dev716 with the bundled source) passed 70 DeepSpec tests with one
skip. The six focused vLLM cache/extraction checks passed. The broader cache
suite had six existing failures and one missing fixture when run with
`--confcutdir`; the same failures were reproduced against the original source.

On B300 hardware, a TP1 teacher exported a complete 131072-token example:
`target_hidden_states=[1,131072,25600]` and
`target_last_hidden_states=[1,131072,5120]`, both BF16. The last-token top-20
log-probability error was zero. A separate TP4 teacher supplied a four-GPU
CP2 x TP2 draft run: each CP shard had 65536 feature positions, the optimizer
completed an update with loss 3.1728, and a distributed checkpoint was saved.
That synthetic training sample contained 131008 supervised assistant tokens.
For mid-schedule resume, a two-step run stopped after step 1, retaining the
original scheduler. The normal launcher restored `next_micro_step=1` and learning
rate `3e-4`, completed step 2 with loss 2.7562, and saved the next checkpoint.
Comparing HF exports found 5039 changed confidence-head weights, with maximum
absolute change 0.0003052. Both 128K teacher extractions had zero last-token
log-probability error. This additional test used `VLLM_LOAD_FORMAT=instanttensor`
from the copied environment; its teacher weight loading took about 40 seconds.
The transient feature files were removed after training. Resuming preserves the
saved learning-rate schedule: increasing `MAX_TRAIN_STEPS` after that schedule
has ended does not restart its learning rate. Multi-node 128-GPU execution and
training quality have not been measured in these smoke tests.

### Numerical scope

These are vLLM teacher features. They are not bitwise interchangeable with the
Transformers teacher or with different prefill chunk sizes. On two short test
prompts, the full native teacher versus vLLM TP1 had mean final-state cosine
similarities of 0.9806 and 0.9952. The exported vLLM final states reproduced the
same engine's last-token top-20 log probabilities exactly on both prompts and on
a 131072-token prompt. This verifies extraction against the serving teacher;
it does not establish equivalent training quality between teacher backends.
Keep teacher TP and prefill settings fixed when comparing training runs.

### Cache failure regression checks

```bash
.envs/qwen38-debug/bin/python -m pytest -q \
  tests/test_qwen38_vllm.py tests/test_qwen38_multinode_launcher.py \
  tests/test_qwen38_dspark.py

# Two free GPUs: real CUDA backward/NCCL followed by a rank-local read failure.
.envs/qwen38-debug/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 --module tests.qwen38_cache_distributed_worker \
  /tmp/qwen38-cache-failure-check --device cuda
```

The regression cases include a success record with a missing `.pt` file, an
unpublished `.pt.tmp`, a corrupt archive, transient and persistent publication
failures, and a consumer-local read error after a successful first batch. They
also retain the cross-partition gradient-accumulation check.
