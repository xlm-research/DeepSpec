"""Run a fresh, complete image-text DSpark pass with local acceptance artifacts."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

from omegaconf import OmegaConf

from torchspec.config.train_config import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]


def write_pass_config(
    manifest_path: Path,
    data_path: Path,
    output_dir: Path,
    cache_dir: Path,
    cli_args: list[str] | None = None,
) -> Path:
    manifest = json.loads(manifest_path.read_text())
    config = load_config(str(SCRIPT_DIR / "config.yaml"))
    endpoint = math.ceil(manifest["window_count"] / 16)
    required = {
        "model.target_model_path": manifest["model"],
        "model.initial_draft_model_path": None,
        "dataset.train_data_path": str(data_path.resolve()),
        "dataset.supervision_manifest_path": str(manifest_path.resolve()),
        "dataset.renderer": "qwen3.5-full-data",
        "dataset.last_turn_loss_only": False,
        "dataset.min_loss_tokens": manifest["min_loss_tokens"],
        "training.complete_data_pass": True,
        "training.load_path": None,
        "training.ckpt_step": None,
        "training.continual_training": False,
        "training.num_epochs": 1,
        "training.num_train_steps": endpoint,
        "training.lr_total_steps": endpoint,
        "training.max_seq_length": manifest["max_length"],
        "training.micro_batch_size": 1,
        "training.draft_accumulation_steps": 4,
        "training.training_num_nodes": 1,
        "training.training_num_gpus_per_node": 4,
        "training.train_with_decode": False,
        "inference.inference_engine_type": "vllm",
        "inference.inference_num_gpus": 4,
        "inference.inference_batch_size": 1,
        "inference.vllm.tp_size": 4,
        "inference.vllm.pp_size": 1,
        "inference.vllm.mem_fraction_static": 0.7,
        "inference.vllm.extra_args.max_num_seqs": 1,
        "inference.vllm.extra_args.max_model_len": manifest["max_length"],
        "inference.vllm.extra_args.language_model_only": False,
        "inference.vllm.extra_args.limit_mm_per_prompt.image": max(1, manifest["max_images"]),
        "decode.max_new_tokens": 1,
        "decode.min_new_tokens": 1,
        "decode.weight_sync_enabled": False,
        "decode.speculative_algorithm": None,
        "logging.report_to": "tensorboard",
    }
    for key, value in required.items():
        OmegaConf.update(config, key, value)
    config.output_dir = str(output_dir.resolve())
    config.cache_dir = str(cache_dir.resolve())
    config.training.save_interval = 0
    config.training.save_per_epoch = True
    config.inference.max_sample_pool_size = 4
    config.mooncake.global_segment_size = "128GB"
    config.mooncake.local_buffer_size = "16GB"
    if cli_args:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(cli_args))
    for key, expected in required.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f"Full-data contract requires {key}={expected!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    OmegaConf.save(config, path)
    load_config(str(path))
    return path


def prepare_preflight_inputs(
    data_path: Path, manifest_path: Path, output: Path
) -> tuple[Path, Path]:
    """Select distinct real windows covering changed behavior and a 17-window tail."""
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    selected: dict[str, dict[str, Any]] = {}

    def choose(row: dict[str, Any]) -> None:
        selected.setdefault(row["data_id"], row)

    choose(max(rows, key=lambda row: row["metadata"]["supervision_window"]["expanded_tokens"]))
    choose(max(rows, key=lambda row: row["metadata"]["supervision_window"]["image_count"]))
    for source_index in (111, 232, 233, 236, 296):
        matches = [
            row
            for row in rows
            if row["metadata"]["supervision_window"]["source_index"] == source_index
        ]
        if matches:
            choose(matches[0])
    for predicate in (
        lambda row: any(m["role"] == "system" for m in row["messages"][1:]),
        lambda row: any(
            m["role"] == "tool"
            and isinstance(m.get("content"), list)
            and any(block.get("type") == "image_url" for block in m["content"])
            for m in row["messages"]
        ),
        lambda row: row["metadata"]["supervision_window"]["assistant_count"]
        > len(row["metadata"]["supervision_window"]["supervised_assistant_ordinals"]),
        lambda row: ".gif" in json.dumps(row["messages"]).lower(),
        lambda row: "/full-data/media/" in json.dumps(row["messages"]),
    ):
        matches = [row for row in rows if predicate(row)]
        if matches:
            choose(matches[0])
    for row in sorted(
        rows, key=lambda row: row["metadata"]["supervision_window"]["expanded_tokens"]
    ):
        if len(selected) >= 17:
            break
        choose(row)
    if len(selected) != 17:
        raise ValueError("GPU tail preflight requires 17 distinct real supervision windows")
    chosen = list(selected.values())
    windows = [row["metadata"]["supervision_window"] for row in chosen]
    sources: dict[str, dict[str, Any]] = {}
    for window in windows:
        source = sources.setdefault(
            window["source_id"],
            {
                "source_id": window["source_id"],
                "source_index": window["source_index"],
                "assistant_message_indices": [],
                "window_ids": [],
            },
        )
        source["assistant_message_indices"].extend(window["supervised_message_indices"])
        source["window_ids"].append(window["window_id"])
    subset = {
        **manifest,
        "purpose": "GPU preflight subset; not full-source acceptance",
        "parent_source_count": manifest["source_count"],
        "source_count": len(sources),
        "window_count": len(windows),
        "sources": list(sources.values()),
        "windows": windows,
    }
    output.mkdir(parents=True, exist_ok=True)
    data = output / "train.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen))
    report = output / "manifest.json"
    report.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n")
    return data, report


def run_stage(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    environment["TORCHSPEC_LOG_DIR"] = str(log_path.parent / "actor_logs")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def prepared_cache_is_valid(directory: Path, source: Path, model: str) -> bool:
    from torchspec.controller.data_pass import CompleteDataPass
    from torchspec.data.supervision_windows import WINDOW_CACHE_VERSION

    try:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["cache_version"] != WINDOW_CACHE_VERSION
            or manifest["source_count"] != 327
            or manifest["max_length"] != 131072
            or manifest["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
            or manifest["model"] != str(Path(model).resolve())
        ):
            return False
        for media in manifest["media"].values():
            path = Path(unquote(urlsplit(media["local_uri"]).path))
            if hashlib.sha256(path.read_bytes()).hexdigest() != media["sha256"]:
                return False
        rows = [json.loads(line) for line in (directory / "train.jsonl").read_text().splitlines()]
        CompleteDataPass(str(manifest_path), 4, 4).validate_dataset(rows)
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def main() -> None:
    from torchspec.data.supervision_windows import prepare_full_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    cli, overrides = parser.parse_known_args()
    output = cli.output_dir.resolve()
    cache = cli.cache_dir.resolve() / "full-data"
    source = ROOT / "mydata/image-text-327.jsonl"
    base = load_config(str(SCRIPT_DIR / "config.yaml"))
    model = base.model.target_model_path
    if not prepared_cache_is_valid(cache, source, model):
        prepare_full_data(
            SimpleNamespace(
                source=str(source),
                output=str(cache),
                media_root="/mnt/afs_toolcall/juxiaolong1/Data/draftData",
                model=model,
                max_length=131072,
                min_loss_tokens=14,
                expected_sources=327,
            )
        )
    config = write_pass_config(
        cache / "manifest.json",
        cache / "train.jsonl",
        output,
        cache / "tokenization",
        overrides,
    )
    if cli.prepare_only:
        return
    preflight = output / "preflight"
    data, manifest = prepare_preflight_inputs(
        cache / "train.jsonl",
        cache / "manifest.json",
        preflight / "data",
    )
    preflight_config = write_pass_config(
        manifest,
        data,
        preflight / "training",
        cache / "preflight-tokenization",
    )
    run_stage(
        [
            sys.executable,
            str(SCRIPT_DIR / "prefill.py"),
            "--config",
            str(preflight_config),
            "--data",
            str(data),
            "--output",
            str(preflight / "prefill"),
        ],
        preflight / "prefill.log",
    )
    run_stage(
        [
            sys.executable,
            str(SCRIPT_DIR / "prefill.py"),
            "--legacy-text-only",
            "--output",
            str(preflight / "legacy-text"),
        ],
        preflight / "legacy-text.log",
    )
    run_stage(
        [sys.executable, "-m", "torchspec.train_entry", "--config", str(preflight_config)],
        preflight / "training.log",
    )
    if cli.preflight_only:
        return
    run_stage(
        [sys.executable, "-m", "torchspec.train_entry", "--config", str(config)],
        output / "training.log",
    )


if __name__ == "__main__":
    main()
