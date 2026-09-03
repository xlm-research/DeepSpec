# Data Preparation

This directory contains an example data preparation pipeline using `Qwen/Qwen3-4B` as the target model.

DeepSpec trains draft models against a target model. The data pipeline does three things:

1. download and split prompt data,
2. regenerate assistant answers with the target model,
3. precompute the target cache used by training.

The example below targets `Qwen/Qwen3-4B`, but the same pipeline applies to other models (e.g. Gemma). To switch targets, change the model name (`--model` / `model_path`) and adjust the sampling parameters (`--temperature`, `--top-p`, `--top-k` and `--min-p`) to match the recommended generation settings for that model. Output paths in the examples reference `qwen3_4b`; rename them as needed.

The wrapper script [prepare_data.sh](./prepare_data.sh) records the default settings. The individual Python scripts are also documented below for users who want to run each stage manually.

## Outputs

Default outputs:

```text
train_datasets/perfectblend_train.jsonl
train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
~/.cache/deepspec/qwen3_4b_target_cache
```

The example scripts assume a single machine with eight visible GPUs by default. For fewer GPUs, edit `num_workers` and `CUDA_VISIBLE_DEVICES` in the shell scripts.

## Step 1: Download And Split Data

The source dataset is `mlabonne/open-perfectblend`. The train split is written as JSONL, and the held-out user turns are written under `eval_datasets/`.

```bash
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

This produces:

```text
train_datasets/perfectblend_train.jsonl
eval_datasets/perfectblend.jsonl
```

## Step 2: Regenerate Answers With Qwen3-4B

This step serves the target model and regenerates assistant answers against it. Any OpenAI-compatible inference engine works (SGLang, vLLM, TGI, etc.) — the example below uses [SGLang](https://github.com/sgl-project/sglang), but you can swap in whatever engine you prefer as long as it exposes an OpenAI-compatible `/v1` endpoint. SGLang is not in `requirements.txt`; install it separately, e.g. `pip install "sglang[all]"`.

Start local sglang servers in one terminal:

```bash
bash scripts/data/launch_sglang_server.sh
```

By default this starts eight `Qwen/Qwen3-4B` workers on ports `30000` to `30007` and writes logs to:

```text
logs/sglang_qwen3_4b/
```

In another terminal, regenerate the assistant answers:

```bash
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3-4B \
    --server-address \
        127.0.0.1:30000 \
        127.0.0.1:30001 \
        127.0.0.1:30002 \
        127.0.0.1:30003 \
        127.0.0.1:30004 \
        127.0.0.1:30005 \
        127.0.0.1:30006 \
        127.0.0.1:30007 \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

This produces:

```text
train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

If any samples fail, the script writes them to:

```text
train_datasets/qwen3_4b/perfectblend_train_regen_error.jsonl
```

Stop the sglang servers before the next step if they are using the same GPUs.

## Step 3: Prepare Target Cache

The training loop reads a precomputed target cache instead of repeatedly running the target model. Prepare it with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b.py \
    --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
    --output-dir ${HOME}/.cache/deepspec/qwen3_4b_target_cache \
    --local-batch-size 16
```

> **Storage warning:** The target cache stores per-token hidden states for the
> full training set and can be very large. With the default `Qwen/Qwen3-4B`
> setting it takes roughly **38 TB** of disk. Make sure the `--output-dir`
> filesystem has enough free space (scaling with dataset size, sequence length,
> and target hidden dimension) before running this step. If storage is limited,
> use a smaller training set and/or reduce `model.target_layer_ids` in the config
> (fewer captured layers means proportionally less cache).

This produces the cache consumed by [scripts/train/train.sh](../train/train.sh):

```text
~/.cache/deepspec/qwen3_4b_target_cache
```

### DeepSeek-V4 target feature modes

DeepSeek-V4 DSpark processes a bounded target-first data batch, stages that
batch's selected and final hidden states in a transient disk cache, trains the
draft on the block, and deletes the consumed cache before advancing. Global
CUDA synchronization and rank barriers isolate target inference from draft
training; draft training cannot fall back to inline target inference. DFlash and
DFlash2 use the offline EP/FSDP2 and ring-CP target runner because they only need
selected hidden states. For DSpark, `DATA_BATCH_SIZE` is the number of
partitions, not a sample count.
For example, 15,000 planned samples with `DATA_BATCH_SIZE=3` produce three
5,000-sample blocks when the optimizer-step boundaries divide evenly.

To prepare an eight-GPU DFlash2 cache:

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/data/prepare_deepseek_v4_target_cache.py \
  --config config/dflash2/dflash2_deepseek_v4.py \
  --train-data-path /path/to/train.jsonl \
  --output-dir /shared/cache/deepseek_v4_dflash2_cp2

torchrun --standalone --nproc-per-node=8 train.py \
  --config config/dflash2/dflash2_deepseek_v4.py \
  --opts data.target_cache_path=/shared/cache/deepseek_v4_dflash2_cp2 \
  --opts data.source_jsonl_path=/path/to/train.jsonl
```

The preparation command validates and reuses a completed cache. It refuses to
overwrite a partial, stale, or incompatible directory. Keep the cache on a
filesystem visible to every training node and set `TARGET_CACHE_DIR` when using
the DeepSeek-V4 DFlash/DFlash2 scripts under `scripts/fsdp/`. Their caches omit
the unused final hidden state. DSpark uses `DATA_BATCH_CACHE_DIR` only for its
bounded transient blocks; it does not require a completed offline manifest.

GLM-5.3 DSpark uses the same bounded target-first lifecycle, but retains the
final hidden state required by its L1 and confidence losses. It keeps
`data.online_target=false`: each target partition is completed on disk before
the draft consumes it, and the partition is deleted before the next one is
generated. Configure the partition count and transient location with:

```bash
MAX_TRAIN_STEPS=100 DATA_BATCH_SIZE=100 \
DATA_BATCH_CACHE_DIR=/local-nvme/glm5-target-blocks \
  bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

`DATA_BATCH_SIZE` is the number of optimizer-aligned partitions. Its default is
256. If fewer than 256 optimizer steps remain, the effective count is capped at
the remaining step count so every partition stays non-empty. A partition cannot
split below an optimizer boundary. With the current three selected
states plus final-state supervision, eight exact-128K BF16 samples occupy about
32 GiB, which is therefore the minimum transient-cache peak for
`GLOBAL_BATCH_SIZE=8`; shorter records scale with their actual token counts.
When `MAX_TRAIN_STEPS` is unset, the launcher trains the full usable dataset for
`NUM_TRAIN_EPOCHS` (default 1). Increase `DATA_BATCH_SIZE` for a smaller cache
footprint, or decrease it for fewer target/draft phase switches.
`MAX_TRAIN_STEPS` is an optional diagnostic override and is not imposed by the
multi-node launcher.
On an eight-GPU node, the target phase uses `FSDP2 x TP4`, providing two
effective sample replicas with CP disabled. Four serial target passes cover the
eight draft sample owners per micro-step. EP stays at 1 because
`target_layer_ids=[0,1,2]` truncates GLM before its first MoE layer (index 3).
The draft uses `FSDP8 x EP8` on that node. With multiple eight-GPU nodes, both
meshes become HSDP automatically: they shard within a node and replicate across
nodes, while keeping target TP at 4.

For a scheduler-managed SenseCore or Slurm multi-node job, configure this exact
command once; the scheduler runs it on every node and the wrapper consumes the
injected topology automatically:

```bash
bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

No DP/FSDP/TP/EP variables are required. On multi-node jobs, the bundled full
training JSONL is selected automatically; `TRAIN_DATA_PATH` remains an optional
dataset override. A scheduler must provide a reachable `MASTER_ADDR` (Slurm
node lists are resolved automatically), because independent machines cannot
discover a rendezvous endpoint themselves. The transient target cache defaults
to a node-local directory under `TMPDIR` (or `/tmp`) and materializes only one
of the effective, at-most-256 partitions at a time; checkpoints and the JSONL
index remain under the shared `OUTPUT_ROOT`.

For a non-scheduler manual torchrun deployment, launch the wrapper once on every
node with the same rendezvous address. For two eight-GPU nodes, use
`NODE_RANK=0` on the first node and `NODE_RANK=1` on the second:

```bash
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.10 MASTER_PORT=29501 \
TRAIN_DATA_PATH=/shared/data/glm5-train.jsonl \
OUTPUT_ROOT=/shared/jobs/glm5-dspark \
JSONL_INDEX_CACHE_DIR=/shared/jobs/glm5-dspark/jsonl-index \
DATA_BATCH_CACHE_DIR=/local-nvme/glm5-target-blocks \
  bash scripts/fsdp/train_glm5_3_flash_dspark_fsdp2.sh
```

The launcher also accepts scheduler-provided `WORLD_SIZE`/`RANK` or
`SENSECORE_PYTORCH_NNODES`/`SENSECORE_PYTORCH_NODE_RANK`. It does not fix the
node count or GPUs per node. The total GPU count must be divisible by 4 because
the requested target TP degree is 4. Draft EP is selected as the greatest
common divisor of the node-local FSDP width and GLM's 288 experts; explicit
`DP_REPLICATE`, `DP_SHARD`, `DRAFT_EP`, `TARGET_DP_REPLICATE`, and
`TARGET_DP_SHARD` overrides remain available.

The model, training JSONL, `OUTPUT_ROOT`, and `JSONL_INDEX_CACHE_DIR` must be
visible at the same path on every node (or identically provisioned where
applicable). `DATA_BATCH_CACHE_DIR` may be node-local: every draft owner writes,
reads, and deletes only its own rank directory. This avoids requiring shared
disk capacity for transient target activations.

For workflows that have ample disk and want a reusable full cache, the separate
`prepare_glm5_next_target_cache.py` runner remains available.

Qwen3.8-27B DSpark uses exact per-epoch micro-batch partitions, so the requested
count is not capped by the number of optimizer steps. The 128K launcher splits
each dataset epoch into 512 partitions by default and deletes each partition
only after its draft backward passes have finished on every rank:

```bash
bash scripts/train/train_qwen3_8_27b_dspark_128gpu.sh
```

`DATA_PARTITIONS` defaults to 512 and may be set to any positive integer. The
default transient cache directory is created under `OUTPUT_ROOT` with the
target model name, DP/FSDP/CP/TP configuration, and launch time, for example
`Qwen3.8-27B_dp16_fsdp1_cp2_tp4_20260902_153045`. Set the same
`DATA_BATCH_CACHE_TIMESTAMP=YYYYMMDD_HHMMSS` on every node when a multi-node
launch must use one shared timestamp, or set `DATA_BATCH_CACHE_DIR` explicitly
to override the generated path.

If the partition count exceeds the per-rank micro-batches in one epoch, the
effective count is capped so no partition is empty. Ten epochs therefore run
ten repetitions of the 512-part lifecycle, but only one 1/512-epoch feature
block is resident on disk at a time. The draft retains its configured CP/TP
topology. During target inference, Qwen's native causal CP shards the 128K
sequence and the draft TP ranks act as additional target FSDP shards; the
generated hidden states remain replicated across the TP consumers. Only one of
those identical copies is written per TP group, so `DATA_BATCH_CACHE_DIR` must
be visible at the same path to all ranks in that node-local TP group. Set
`BOUNDED_OFFLINE=false` only when a complete reusable target cache is
intentionally desired.

### Multimodal targets

`prepare_target_cache.py` detects image-text target configurations such as
Qwen3.5/Qwen3.6 and loads their `AutoProcessor`. The processor expands visual
placeholders, runs the target ViT/merger and decoder, and stores decoder hidden
states in the same cache format used by text-only training.

Both OpenAI-style content blocks and ms-swift-style top-level media lists are
accepted. For example:

```json
{"messages":[{"role":"user","content":"<image>Describe it."},{"role":"assistant","content":"..."}],"images":["sample.jpg"]}
```

or:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"sample.jpg"},{"type":"text","text":"Describe it."}]},{"role":"assistant","content":"..."}]}
```

Use `--media-root /path/to/media` when media paths in JSONL are relative. The
multimodal collator preserves `pixel_values`, grid metadata and multimodal token
types required by Qwen3.6. If `max_length` would cut through a visual token
block, that sample is skipped instead of producing a mismatched cache entry.

## Wrapper Script

The wrapper script combines the default public commands:

```bash
bash scripts/data/prepare_data.sh
```

Use the manual commands above if you want to stop and restart services between stages, change sampling parameters, use fewer GPUs, or inspect intermediate outputs.
