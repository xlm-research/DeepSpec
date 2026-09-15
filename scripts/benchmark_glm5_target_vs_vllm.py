#!/usr/bin/env python3
"""Compare the existing GLM target prefill with vLLM on identical token IDs.

Run prepare once, then target under torchrun (four ranks), then vllm with
four visible GPUs. Each backend runs one cold full-length request followed
by --repeats measured requests. See the accompanying benchmark report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
PROCESS_START = time.perf_counter()


def save(args, name, value):
    path = args.output_dir / f"{name}.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(args):
    from transformers import AutoTokenizer
    from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY

    source_bytes = args.source.read_bytes()
    records = [json.loads(line) for line in source_bytes.splitlines() if line]
    assert len(records) == 1, "Use the existing single long packed record."
    record = records[0]
    messages = record.get("messages", record.get("conversations"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    parser = GeneralParser(tokenizer, TEMPLATE_REGISTRY.get("glm5_next"))
    # Use the training renderer, without computing the unused training loss mask.
    messages = parser._prepare_render_messages(messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(all_ids) >= args.length, (len(all_ids), args.length)
    ids = all_ids[: args.length]
    save(args, "input_ids", ids)
    metadata = {
        "model": args.model,
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_row": 0,
        "rendered_tokens_before_truncation": len(all_ids),
        "input_tokens": len(ids),
        "token_ids_sha256_int32_le": hashlib.sha256(
            struct.pack(f"<{len(ids)}i", *ids)
        ).hexdigest(),
        "chat_template": "glm5_next, training non-thinking assistant prefix",
        "batch_size": 1,
    }
    save(args, "input_metadata", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def target(args):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from deepspec.distributed import ParallelConfig, ParallelContext
    from deepspec.modeling.target import Glm5NextOnlineTarget

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    rank = dist.get_rank()
    assert dist.get_world_size() == 4, "This comparison uses TP4 / EP4, one sample."
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    topology = ParallelContext.build(ParallelConfig(tp=4, ep=4, use_fsdp=True))
    started = time.perf_counter()
    model = Glm5NextOnlineTarget(
        model_name_or_path=args.model,
        target_layer_ids=[2, 22, 42],
        topology=topology,
        device=device,
        rank_local_cache_dir=str(args.output_dir / f"rank_{rank}"),
    )
    torch.cuda.synchronize()
    dist.barrier()
    load_time = torch.tensor(time.perf_counter() - started, device=device)
    dist.all_reduce(load_time, op=dist.ReduceOp.MAX)
    load_seconds = load_time.item()
    ids = torch.tensor(
        json.loads((args.output_dir / "input_ids.json").read_text()),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    assert ids.shape == (1, args.length)
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "loss_mask": torch.ones_like(ids),
    }
    report = {
        "backend": "Glm5NextOnlineTarget.forward_training_batch",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "parallel": topology.config.to_dict(),
        "parameter_dtype": "bfloat16 (dequantized FP8 checkpoint)",
        "input_tokens": args.length,
        "model_load_seconds_max_rank": load_seconds,
        "process_to_ready_seconds": time.perf_counter() - PROCESS_START,
        "includes": "full 45-layer forward, selected and final features to CPU",
        "excludes": "tokenization, checkpoint load, disk cache writing, LM head",
        "runs": [],
    }
    if rank == 0:
        print(
            f"[benchmark] target ready: {report['process_to_ready_seconds']:.3f}s",
            flush=True,
        )
        save(args, "target", report)

    # CUDA events observe each layer without synchronizing between layers.
    events = []
    handles = []
    if rank == 0:
        for index, layer in enumerate(model.model.language_model.layers):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            events.append((start, end))
            handles.append(
                layer.register_forward_pre_hook(
                    lambda module, inputs, event=start: event.record()
                )
            )

            def finished(module, inputs, output, event=end, layer_index=index):
                event.record()
                if layer_index % 5 == 0 or layer_index == 44:
                    print(
                        f"[benchmark] layer {layer_index + 1}/45 enqueued", flush=True
                    )

            handles.append(layer.register_forward_hook(finished))

    for run in range(args.repeats + 1):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(
                f"[benchmark] target run={run} tokens={args.length} start", flush=True
            )
        started = time.perf_counter()
        output = model.forward_training_batch(batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        measurement = torch.tensor(
            [
                seconds,
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(measurement, op=dist.ReduceOp.MAX)
        shapes = {
            key: list(output[key].shape)
            for key in ("target_hidden_states", "target_last_hidden_states")
        }
        for key in shapes:
            assert shapes[key][0:2] == [1, args.length]
            assert torch.isfinite(output[key][0, ::1024, ::64]).all().item()
        row = {
            "run": run,
            "kind": "cold_full_length" if run == 0 else "warm",
            "seconds_max_rank": measurement[0].item(),
            "peak_allocated_gib_max_rank": measurement[1].item(),
            "peak_reserved_gib_max_rank": measurement[2].item(),
            "output_shapes": shapes,
        }
        if rank == 0:
            row["layer_cuda_seconds"] = [a.elapsed_time(b) / 1000 for a, b in events]
            report["runs"].append(row)
            if run:
                report["warm_median_seconds"] = statistics.median(
                    item["seconds_max_rank"] for item in report["runs"][1:]
                )
            save(args, "target", report)
            print(
                f"[benchmark] target run={run} seconds={row['seconds_max_rank']:.3f}",
                flush=True,
            )
        del output
    for handle in handles:
        handle.remove()
    dist.barrier()
    dist.destroy_process_group()


def vllm(args):
    # Keep heavy imports inside main, so spawn does not re-execute them.
    # Match vllm/vllm_demo.py: the outer checkout is not the Python package.
    sys.path.insert(0, str(REPO_ROOT / "vllm"))
    import vllm as vllm_package
    from vllm import LLM, SamplingParams, __version__

    ids = json.loads((args.output_dir / "input_ids.json").read_text())
    assert len(ids) == args.length
    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        load_format="instanttensor",
        tensor_parallel_size=4,
        max_model_len=args.length + 1,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        language_model_only=True,
        disable_log_stats=False,
        seed=42,
    )
    report = {
        "backend": "vLLM LLM.generate",
        "vllm": __version__,
        "vllm_source": vllm_package.__file__,
        "tensor_parallel_size": 4,
        "quantization": "checkpoint default FP8",
        "input_tokens": len(ids),
        "output_tokens": 1,
        "max_num_batched_tokens": 8192,
        "prefix_caching": False,
        "enforce_eager": True,
        "engine_init_seconds": time.perf_counter() - started,
        "process_to_ready_seconds": time.perf_counter() - PROCESS_START,
        "includes": "chunked prefill, KV cache, last-token LM head and sampling",
        "excludes": "tokenization, engine initialization, intermediate features",
        "runs": [],
    }
    save(args, "vllm", report)
    print(
        f"[benchmark] vllm ready: {report['process_to_ready_seconds']:.3f}s", flush=True
    )
    params = SamplingParams(
        temperature=0, max_tokens=1, ignore_eos=True, detokenize=False
    )
    for run in range(args.repeats + 1):
        print(f"[benchmark] vllm run={run} tokens={len(ids)} start", flush=True)
        started = time.perf_counter()
        output = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        seconds = time.perf_counter() - started
        assert len(output.prompt_token_ids) == args.length
        assert len(output.outputs[0].token_ids) == 1
        assert output.num_cached_tokens == 0, output.num_cached_tokens
        row = {
            "run": run,
            "kind": "cold_full_length" if run == 0 else "warm",
            "seconds": seconds,
            "num_cached_tokens": output.num_cached_tokens,
            "output_token_ids": list(output.outputs[0].token_ids),
            "metrics": vars(output.metrics) if output.metrics is not None else None,
        }
        report["runs"].append(row)
        if run:
            report["warm_median_seconds"] = statistics.median(
                item["seconds"] for item in report["runs"][1:]
            )
        save(args, "vllm", report)
        print(f"[benchmark] vllm run={run} seconds={seconds:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["prepare", "target", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT
        / "train_data"
        / "spec_o3_coldstartsft.repeat60.deepspec.automodel_context128k_1.jsonl",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.backend](args)


if __name__ == "__main__":
    main()
