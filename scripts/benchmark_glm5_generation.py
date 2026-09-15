#!/usr/bin/env python3
"""Time full greedy generation on the same four GPUs and identical token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def save(args, name, data):
    (args.output_dir / f"{name}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )


def target(args, ids):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoTokenizer, DynamicCache
    from deepspec.distributed import ParallelConfig, ParallelContext
    from deepspec.modeling.target import Glm5NextOnlineTarget

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    assert dist.get_world_size() == 4
    rank = dist.get_rank()
    torch.set_float32_matmul_precision("high")
    topology = ParallelContext.build(ParallelConfig(tp=4, ep=4))
    teacher = Glm5NextOnlineTarget(
        model_name_or_path=args.model,
        target_layer_ids=[2, 22, 42],
        topology=topology,
        device=device,
        rank_local_cache_dir=str(args.output_dir / f"rank_{rank}"),
    )
    text_config = teacher.model.language_model.config
    index = json.loads((Path(args.model) / "model.safetensors.index.json").read_text())
    head_file = Path(args.model) / index["weight_map"]["lm_head.weight"]
    with safe_open(head_file, framework="pt", device="cpu") as checkpoint:
        weight = checkpoint.get_slice("lm_head.weight")
        vocab_size, hidden_size = weight.get_shape()
        assert (
            vocab_size == text_config.vocab_size
            and hidden_size == text_config.hidden_size
        )
        assert vocab_size % 4 == 0
        local_vocab = vocab_size // 4
        head = weight[rank * local_vocab : (rank + 1) * local_vocab].to(
            device, torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    full_logits = torch.empty(vocab_size, dtype=head.dtype, device=device)
    inputs = torch.tensor([ids], dtype=torch.long, device=device)

    def next_token(hidden):
        logits = F.linear(hidden[:, -1], head).flatten().contiguous()
        dist.all_gather_into_tensor(
            full_logits, logits, group=topology.tensor_parallel_group
        )
        return full_logits.argmax().reshape(1, 1)

    def generate(prompt, count, label):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        cache = DynamicCache(config=text_config)
        output = teacher.model(input_ids=prompt, use_cache=True, past_key_values=cache)
        token = next_token(output.last_hidden_state)
        generated = [token]
        del output
        torch.cuda.synchronize()
        first_token_seconds = time.perf_counter() - started
        if rank == 0:
            print(
                f"[generation] {label} first_token={first_token_seconds:.3f}s",
                flush=True,
            )
        decode_started = time.perf_counter()
        for step in range(1, count):
            output = teacher.model(
                input_ids=token, use_cache=True, past_key_values=cache
            )
            token = next_token(output.last_hidden_state)
            generated.append(token)
            del output
            if rank == 0 and (step + 1) % 32 == 0:
                print(f"[generation] {label} generated={step + 1}/{count}", flush=True)
        token_ids = torch.cat(generated, dim=1)[0].cpu().tolist()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        values = torch.tensor(
            [
                first_token_seconds,
                finished - decode_started,
                finished - started,
                torch.cuda.max_memory_allocated() / 1024**3,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        expected_cache_len = prompt.shape[1] + count - 1
        assert cache.get_seq_length() == expected_cache_len
        for layer_index, kind in enumerate(text_config.layer_types):
            layer_cache = cache.layers[layer_index]
            if kind == "linear_attention":
                assert cache.has_previous_state(layer_index)
                assert torch.isfinite(layer_cache.recurrent_states[0]).all()
            else:
                assert layer_cache.keys.shape[-2] == expected_cache_len
                if layer_cache.is_indexer_initialized:
                    assert layer_cache.indexer_keys.shape[1] == expected_cache_len
        assert torch.isfinite(full_logits).all()
        assert len(token_ids) == count
        result = {
            "input_tokens": prompt.shape[1],
            "output_tokens": count,
            "ttft_seconds": values[0].item(),
            "decode_seconds": values[1].item(),
            "total_seconds": values[2].item(),
            "decode_steps": count - 1,
            "decode_tokens_per_second": (count - 1) / values[1].item(),
            "decode_ms_per_token": values[1].item() * 1000 / (count - 1),
            "peak_allocated_gib_max_rank": values[3].item(),
            "final_cache_length": expected_cache_len,
            "generated_token_ids": token_ids,
            "generated_text": tokenizer.decode(token_ids, skip_special_tokens=False),
        }
        del cache
        return result

    with torch.no_grad():
        if rank == 0:
            print("[generation] target ready; short warmup start", flush=True)
        generate(inputs[:, :128], 4, "warmup")
        if rank == 0:
            print(
                f"[generation] target measured input={len(ids)} output={args.new_tokens}",
                flush=True,
            )
        result = generate(inputs, args.new_tokens, "measured")
    if rank == 0:
        result.update(
            {
                "backend": "custom target + native cached decode + TP4 LM head",
                "parameter_dtype": "bfloat16",
                "tp": 4,
                "ep": 4,
                "torch": torch.__version__,
                "includes_initialization": False,
                "input_ids_sha256": args.input_sha256,
            }
        )
        save(args, "target", result)
        print(
            f"[generation] target DONE total={result['total_seconds']:.3f}s decode={result['decode_seconds']:.3f}s",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


def vllm(args, ids):
    sys.path.insert(0, str(REPO_ROOT / "vllm"))
    from vllm import LLM, SamplingParams, __version__

    llm = LLM(
        model=args.model,
        load_format="instanttensor",
        tensor_parallel_size=4,
        max_model_len=len(ids) + args.new_tokens,
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
    print("[generation] vllm ready; short warmup start", flush=True)
    llm.generate(
        [{"prompt_token_ids": ids[:128]}],
        SamplingParams(
            temperature=0,
            max_tokens=4,
            min_tokens=4,
            ignore_eos=True,
            detokenize=False,
        ),
        use_tqdm=False,
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=args.new_tokens,
        min_tokens=args.new_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    print(
        f"[generation] vllm measured input={len(ids)} output={args.new_tokens}",
        flush=True,
    )
    started = time.perf_counter()
    output = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
    total = time.perf_counter() - started
    token_ids = list(output.outputs[0].token_ids)
    stats = output.metrics
    assert output.finished and output.outputs[0].finish_reason == "length"
    assert len(output.prompt_token_ids) == len(ids)
    assert len(token_ids) == args.new_tokens and output.num_cached_tokens == 0
    assert stats is not None and not stats.is_corrupted
    decode = stats.last_token_ts - stats.first_token_ts
    result = {
        "backend": "vLLM",
        "vllm": __version__,
        "quantization": "fp8",
        "tp": 4,
        "input_tokens": len(ids),
        "output_tokens": len(token_ids),
        "ttft_seconds": stats.first_token_latency,
        "decode_seconds": decode,
        "total_seconds": total,
        "decode_steps": len(token_ids) - 1,
        "decode_tokens_per_second": (len(token_ids) - 1) / decode,
        "decode_ms_per_token": decode * 1000 / (len(token_ids) - 1),
        "num_cached_tokens": output.num_cached_tokens,
        "finish_reason": output.outputs[0].finish_reason,
        "generated_token_ids": token_ids,
        "generated_text": llm.get_tokenizer().decode(
            token_ids, skip_special_tokens=False
        ),
        "metrics": vars(stats),
        "includes_initialization": False,
        "input_ids_sha256": args.input_sha256,
    }
    save(args, "vllm", result)
    print(f"[generation] vllm DONE total={total:.3f}s decode={decode:.3f}s", flush=True)
    llm.llm_engine.engine_core.shutdown(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["target", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-tokens", type=int, default=256)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3", (
        "Both benchmarks must use GPUs 0,1,2,3."
    )
    assert args.new_tokens > 1
    input_bytes = args.input_ids.read_bytes()
    ids = json.loads(input_bytes)
    assert len(ids) == 131072
    args.input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.backend](args, ids)


if __name__ == "__main__":
    main()
