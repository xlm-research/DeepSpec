#!/usr/bin/env python3
"""Export identical GLM target features on eight GPUs and time through fsync."""

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
LAYERS = (2, 22, 42)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_features(path, tensors):
    from safetensors.torch import save_file

    started = time.perf_counter()
    save_file(
        tensors,
        str(path),
        metadata={
            "target_layer_ids_zero_based": json.dumps(LAYERS),
            "hidden_layout": "token, concatenated layer features",
            "selected_layer_semantics": "post decoder layer, HC stream mean",
            "last_layer_semantics": "final HC mean and output RMSNorm",
        },
    )
    with Path(path).open("rb+") as handle:
        os.fsync(handle.fileno())
    return time.perf_counter() - started


def feature_probe(tensors):
    import torch

    probes = {}
    for key in ("target_hidden_states", "target_last_hidden_states"):
        rows = torch.linspace(0, tensors[key].shape[0] - 1, 16).long()
        sample = tensors[key][rows][:, ::128].contiguous()
        assert torch.isfinite(sample).all()
        raw = sample.view(torch.uint8).numpy().tobytes()
        probes[key] = {"sha256": hashlib.sha256(raw).hexdigest(), "bf16_hex": raw.hex()}
    return probes


def compare_rank_probes(probes):
    import torch

    result = {}
    for key in probes[0]:
        reference = torch.frombuffer(
            bytearray.fromhex(probes[0][key]["bf16_hex"]), dtype=torch.bfloat16
        ).float()
        rows = []
        for rank, probe in enumerate(probes):
            candidate = torch.frombuffer(
                bytearray.fromhex(probe[key]["bf16_hex"]), dtype=torch.bfloat16
            ).float()
            delta = candidate - reference
            cosine = torch.dot(reference, candidate) / (
                reference.norm() * candidate.norm()
            )
            rows.append(
                {
                    "rank": rank,
                    "bitwise_equal": torch.equal(reference, candidate),
                    "max_abs_error": delta.abs().max().item(),
                    "relative_l2": (delta.norm() / reference.norm()).item(),
                    "cosine": cosine.item(),
                }
            )
        result[key] = rows
    return result


def target(args, ids):
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from deepspec.distributed import ParallelConfig, ParallelContext
    from deepspec.modeling.target import Glm5NextOnlineTarget

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(hours=2), device_id=device)
    assert dist.get_world_size() == 8
    rank = dist.get_rank()
    torch.set_float32_matmul_precision("high")
    topology = ParallelContext.build(ParallelConfig(tp=8, ep=8))
    started = time.perf_counter()
    teacher = Glm5NextOnlineTarget(
        model_name_or_path=args.model,
        target_layer_ids=LAYERS,
        topology=topology,
        device=device,
        rank_local_cache_dir=str(args.output_dir / f"rank_{rank}"),
    )
    torch.cuda.synchronize()
    report = {
        "backend": "custom target forward_training_batch",
        "parallel": topology.config.to_dict(),
        "parameter_dtype": "bfloat16",
        "torch": torch.__version__,
        "model_load_seconds_rank0": time.perf_counter() - started,
        "layer_ids_zero_based": LAYERS,
        "input_ids_sha256": args.input_sha256,
        "cpu_features_on_all_tp_ranks": True,
        "file_writer_rank": 0,
        "runs": [],
    }
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    progress = [""]
    if rank == 0:
        for index, layer in enumerate(teacher.model.language_model.layers):
            if (index + 1) % 10 == 0 or index == 44:
                layer.register_forward_hook(
                    lambda module, inputs, output, index=index: print(
                        f"[hidden-bench] target {progress[0]} layer={index + 1}/45",
                        flush=True,
                    )
                )
        print("[hidden-bench] target ready", flush=True)

    for label, length in [("warmup", 8192)] + [
        (f"run_{index}", len(ids)) for index in range(args.repeats)
    ]:
        progress[0] = label
        prompt = input_tensor[:, :length]
        batch = {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "loss_mask": torch.ones_like(prompt),
        }
        path = args.output_dir / f"target_{label}.safetensors"
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(f"[hidden-bench] target {label} start tokens={length}", flush=True)
        started = time.perf_counter()
        output = teacher.forward_training_batch(batch)
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        tensors = {
            "input_ids": prompt[0].to(device="cpu", dtype=torch.int32),
            "target_hidden_states": output["target_hidden_states"][0],
            "target_last_hidden_states": output["target_last_hidden_states"][0],
        }
        save_seconds = write_features(path, tensors) if rank == 0 else 0.0
        dist.barrier()
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        values = torch.tensor(
            [
                forward_seconds,
                save_seconds,
                total_seconds,
                torch.cuda.max_memory_allocated() / 1024**3,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        probes = [None] * 8
        dist.all_gather_object(probes, feature_probe(tensors))
        rank_comparison = compare_rank_probes(probes)
        if rank == 0:
            save_json(
                args.output_dir / f"target_{label}_rank_comparison.json",
                rank_comparison,
            )
            print(
                f"[hidden-bench] target {label} rank comparison: {rank_comparison}",
                flush=True,
            )
        assert all(
            row["cosine"] > 0.99 for rows in rank_comparison.values() for row in rows
        )
        assert tensors["target_hidden_states"].shape == (length, 3 * 4096)
        assert tensors["target_last_hidden_states"].shape == (length, 4096)
        if rank == 0:
            row = {
                "label": label,
                "input_tokens": length,
                "forward_and_cpu_copy_seconds": values[0].item(),
                "save_fsync_seconds": values[1].item(),
                "end_to_end_seconds": values[2].item(),
                "peak_allocated_gib_max_rank": values[3].item(),
                "file": str(path.resolve()),
                "file_bytes": path.stat().st_size,
                "shapes": {k: list(v.shape) for k, v in tensors.items()},
                "tp_rank_feature_probes_identical": all(
                    probe == probes[0] for probe in probes
                ),
                "tp_rank_feature_comparison": rank_comparison,
                "feature_probe": probes[0],
            }
            report["runs"].append(row)
            save_json(args.output_dir / "target.json", report)
            print(
                f"[hidden-bench] target {label} DONE forward_copy={values[0]:.3f}s "
                f"save_fsync={values[1]:.3f}s total={values[2]:.3f}s",
                flush=True,
            )
        del tensors, output
    dist.barrier()
    dist.destroy_process_group()


def install_capture(model):
    import torch
    import torch.distributed as dist

    language_model = model.language_model
    backbone = language_model.model
    language_model.set_aux_hidden_state_layers(tuple(index + 1 for index in LAYERS))
    state = {"active": False, "rank": dist.get_rank()}
    model._deepspec_hidden_benchmark = state

    def capture(module, inputs, kwargs, output):
        final, auxiliary = output
        assert len(auxiliary) == len(LAYERS)
        if state["active"]:
            positions = kwargs.get("positions")
            if positions is None:
                positions = inputs[1]
            positions = positions.detach().to(device="cpu", dtype=torch.long)
            start = state["cursor"]
            end = start + final.shape[0]
            assert end <= state["length"]
            assert torch.equal(positions, torch.arange(start, end))
            supplied_ids = kwargs.get("input_ids")
            if supplied_ids is None and inputs:
                supplied_ids = inputs[0]
            if supplied_ids is not None:
                assert torch.equal(
                    supplied_ids.detach().to(device="cpu", dtype=torch.int32),
                    state["tensors"]["input_ids"][start:end],
                )
                state["input_ids_verified_tokens"] += end - start
            copy_started = time.perf_counter()
            state["tensors"]["target_hidden_states"][start:end].copy_(
                torch.cat(auxiliary, dim=-1).detach().to("cpu")
            )
            state["tensors"]["target_last_hidden_states"][start:end].copy_(
                final.detach().to("cpu")
            )
            state["copy_seconds"] += time.perf_counter() - copy_started
            state["chunks"].append([start, end])
            state["cursor"] = end
            if state["rank"] == 0:
                print(
                    f"[hidden-bench] vllm {state['label']} captured={end}/{state['length']}",
                    flush=True,
                )
        # Retain the ordinary engine output; auxiliary states go to the exporter.
        return final

    backbone.register_forward_hook(capture, with_kwargs=True)
    return {
        "rank": dist.get_rank(),
        "backbone": type(backbone).__name__,
        "aux_layer_ids": list(backbone.aux_hidden_state_layers),
        "sequence_parallel": backbone.is_sequence_parallel,
    }


def begin_capture(model, *, input_path, length, label):
    import torch

    state = model._deepspec_hidden_benchmark
    ids = json.loads(Path(input_path).read_text())[:length]
    state.update(
        active=True,
        length=length,
        label=label,
        cursor=0,
        chunks=[],
        copy_seconds=0.0,
        input_ids_verified_tokens=0,
        tensors={
            "input_ids": torch.tensor(ids, dtype=torch.int32),
            "target_hidden_states": torch.empty(length, 3 * 4096, dtype=torch.bfloat16),
            "target_last_hidden_states": torch.empty(
                length, 4096, dtype=torch.bfloat16
            ),
        },
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def finish_capture(model, *, path):
    import torch

    state = model._deepspec_hidden_benchmark
    torch.cuda.synchronize()
    assert state["cursor"] == state["length"]
    tensors = state["tensors"]
    save_seconds = write_features(path, tensors) if state["rank"] == 0 else 0.0
    row = {
        "rank": state["rank"],
        "captured_tokens": state["cursor"],
        "position_ranges": state["chunks"],
        "input_ids_verified_tokens": state["input_ids_verified_tokens"],
        "copy_seconds": state["copy_seconds"],
        "save_fsync_seconds": save_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "feature_probe": feature_probe(tensors),
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
    }
    state["active"] = False
    del state["tensors"]
    return row


class HiddenCaptureWorkerExtension:
    def deepspec_install_capture(self):
        return install_capture(self.get_model())

    def deepspec_begin_capture(self, input_path, length, label):
        return begin_capture(
            self.get_model(), input_path=input_path, length=length, label=label
        )

    def deepspec_finish_capture(self, path):
        return finish_capture(self.get_model(), path=path)


def vllm(args, ids):
    sys.path.insert(0, str(REPO_ROOT / "vllm"))
    from vllm import LLM, SamplingParams, __version__

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        load_format="instanttensor",
        tensor_parallel_size=8,
        max_model_len=len(ids) + 1,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        language_model_only=True,
        disable_log_stats=False,
        worker_extension_cls="scripts.benchmark_glm5_hidden_states.HiddenCaptureWorkerExtension",
        seed=42,
    )
    report = {
        "backend": "vLLM native auxiliary hidden states + chunked CPU export",
        "vllm": __version__,
        "tp": 8,
        "quantization": "fp8",
        "engine_init_seconds": time.perf_counter() - started,
        "layer_ids_zero_based": LAYERS,
        "input_ids_sha256": args.input_sha256,
        "cpu_features_on_all_tp_ranks": True,
        "file_writer_rank": 0,
        "capture_installation": llm.collective_rpc("deepspec_install_capture"),
        "runs": [],
    }
    params = SamplingParams(
        temperature=0, max_tokens=1, ignore_eos=True, detokenize=False
    )
    for label, length in [("warmup", 8192)] + [
        (f"run_{index}", len(ids)) for index in range(args.repeats)
    ]:
        path = args.output_dir / f"vllm_{label}.safetensors"
        llm.collective_rpc(
            "deepspec_begin_capture",
            args=(str(args.input_ids.resolve()), length, label),
        )
        print(f"[hidden-bench] vllm {label} start tokens={length}", flush=True)
        started = time.perf_counter()
        output = llm.generate(
            [{"prompt_token_ids": ids[:length]}], params, use_tqdm=False
        )[0]
        forward_seconds = time.perf_counter() - started
        workers = llm.collective_rpc(
            "deepspec_finish_capture", args=(str(path.resolve()),)
        )
        total_seconds = time.perf_counter() - started
        assert output.finished and len(output.outputs[0].token_ids) == 1
        assert output.num_cached_tokens == 0
        assert output.metrics is not None and not output.metrics.is_corrupted
        assert len(workers) == 8
        rank_comparison = compare_rank_probes(
            [worker["feature_probe"] for worker in workers]
        )
        assert all(
            row["cosine"] > 0.99 for rows in rank_comparison.values() for row in rows
        )
        row = {
            "label": label,
            "input_tokens": length,
            "forward_and_cpu_copy_seconds": forward_seconds,
            "save_fsync_seconds": max(
                worker["save_fsync_seconds"] for worker in workers
            ),
            "end_to_end_seconds": total_seconds,
            "peak_allocated_gib_max_rank": max(
                worker["peak_allocated_gib"] for worker in workers
            ),
            "file": str(path.resolve()),
            "file_bytes": path.stat().st_size,
            "workers": workers,
            "num_cached_tokens": output.num_cached_tokens,
            "generated_token_ids": list(output.outputs[0].token_ids),
            "metrics": vars(output.metrics),
            "tp_rank_feature_probes_identical": all(
                worker["feature_probe"] == workers[0]["feature_probe"]
                for worker in workers
            ),
            "tp_rank_feature_comparison": rank_comparison,
        }
        report["runs"].append(row)
        save_json(args.output_dir / "vllm.json", report)
        print(
            f"[hidden-bench] vllm {label} DONE forward_copy={forward_seconds:.3f}s "
            f"save_fsync={row['save_fsync_seconds']:.3f}s total={total_seconds:.3f}s",
            flush=True,
        )
    llm.llm_engine.engine_core.shutdown(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["target", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3,4,5,6,7"
    raw = args.input_ids.read_bytes()
    ids = json.loads(raw)
    assert len(ids) == 131072 and args.repeats > 0
    args.input_sha256 = hashlib.sha256(raw).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    globals()[args.backend](args, ids)


if __name__ == "__main__":
    main()
