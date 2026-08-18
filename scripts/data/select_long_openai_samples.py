#!/usr/bin/env python3
"""Select the longest DeepSpec-compatible samples below a token limit.

The input used by this project is a JSON catalog whose ``annotation`` fields
point at OpenAI-style JSONL files.  This script scans the catalog without
loading the full corpus into memory, keeps a bounded pool of the longest
text candidates, tokenizes that pool with the target tokenizer, and writes
DeepSpec-style ``conversations`` JSONL.

By default only plain user/assistant data is accepted.  Tool definitions,
tool messages, and tool calls are deliberately rejected because the current
DeepSeek-V4 DeepSpec chat template does not serialize them.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


DEFAULT_CATALOG = (
    "/mnt/afs_toolcall/zhengnairong/data/aaa_data_clean_swift_train/"
    "data_openai_format/all/coding_related.json"
)
DEFAULT_MODEL = "/mnt/afs_agents/share_models/deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_OUTPUT = (
    "/mnt/afs_share/wzj/deepspec1/train_data/"
    "deepseek_v4_long_under_128k.jsonl"
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    source_type: str
    domain: str


@dataclass(frozen=True)
class CandidateRef:
    char_count: int
    serial: int
    source: Source
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the longest plain user/assistant samples whose exact "
            "DeepSeek-V4 rendered token length is below 128K."
        )
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=18,
        help="Number of samples to write (18 fills the default 144-GPU layout once).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=32768,
        help="Reject shorter samples. Defaults to 32K tokens.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=131071,
        help="Inclusive maximum. 131071 is strictly below 128 Ki tokens.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=4096,
        help=(
            "Number of longest character-count candidates to tokenize exactly. "
            "Raise this if too few candidates fall below the token limit."
        ),
    )
    parser.add_argument(
        "--source-type",
        choices=("basic", "all"),
        default="basic",
        help="Use only catalog entries marked basic, or scan every entry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output and its sidecar manifest.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        parser.error("require 0 < --min-tokens <= --max-tokens")
    if args.candidate_pool_size < args.num_samples:
        parser.error("--candidate-pool-size must be at least --num-samples")
    return args


def load_sources(catalog_path: str, source_type: str) -> list[Source]:
    with open(catalog_path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict):
        raise TypeError("The catalog root must be a JSON object.")

    sources: list[Source] = []
    for name, metadata in catalog.items():
        if not isinstance(metadata, dict):
            continue
        annotation = metadata.get("annotation")
        entry_type = str(metadata.get("from", ""))
        if source_type == "basic" and entry_type != "basic":
            continue
        if not isinstance(annotation, str) or not annotation:
            continue
        if not os.path.isfile(annotation):
            raise FileNotFoundError(f"Annotation file does not exist: {annotation}")
        sources.append(
            Source(
                name=str(name),
                path=annotation,
                source_type=entry_type,
                domain=str(metadata.get("domain", "")),
            )
        )
    if not sources:
        raise RuntimeError("No usable annotation files were found in the catalog.")
    return sources


def normalize_plain_conversation(record: Any) -> list[dict[str, str]] | None:
    if not isinstance(record, dict):
        return None
    if record.get("tools"):
        return None
    messages = record.get("messages", record.get("conversations"))
    if not isinstance(messages, list) or not messages:
        return None

    normalized: list[dict[str, str]] = []
    has_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            return None
        if not isinstance(content, str):
            return None
        if message.get("tool_calls") or message.get("tool_call_id"):
            return None
        if role == "system" and normalized:
            return None
        if role == "assistant":
            has_assistant = True
        normalized.append({"role": role, "content": content})

    first_role = normalized[0]["role"]
    if first_role == "system":
        if len(normalized) < 2 or normalized[1]["role"] != "user":
            return None
    elif first_role != "user":
        return None
    return normalized if has_assistant else None


def rendered_char_count(conversation: Iterable[dict[str, str]]) -> int:
    # Headers are short and fixed, so content characters are a sufficiently
    # accurate first-pass proxy while avoiding tokenizing the full corpus.
    return sum(len(message["content"]) for message in conversation)


def scan_candidate_refs(
    sources: list[Source], candidate_pool_size: int
) -> tuple[list[CandidateRef], dict[str, int]]:
    heap: list[tuple[int, int, CandidateRef]] = []
    stats = {"lines": 0, "valid_plain": 0, "invalid_json": 0}
    serial = 0
    for source_index, source in enumerate(sources, start=1):
        with open(source.path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                stats["lines"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue
                conversation = normalize_plain_conversation(record)
                if conversation is None:
                    continue
                stats["valid_plain"] += 1
                serial += 1
                ref = CandidateRef(
                    char_count=rendered_char_count(conversation),
                    serial=serial,
                    source=source,
                    line_number=line_number,
                )
                item = (ref.char_count, ref.serial, ref)
                if len(heap) < candidate_pool_size:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        print(
            f"[scan {source_index}/{len(sources)}] {source.name}: "
            f"plain={stats['valid_plain']:,}, pool={len(heap):,}",
            flush=True,
        )
    return [item[2] for item in heap], stats


def reload_candidates(
    refs: list[CandidateRef],
) -> list[tuple[CandidateRef, list[dict[str, str]]]]:
    wanted: dict[str, dict[int, CandidateRef]] = defaultdict(dict)
    for ref in refs:
        wanted[ref.source.path][ref.line_number] = ref

    loaded: list[tuple[CandidateRef, list[dict[str, str]]]] = []
    for path, line_map in wanted.items():
        remaining = set(line_map)
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number not in remaining:
                    continue
                record = json.loads(line)
                conversation = normalize_plain_conversation(record)
                if conversation is None:
                    raise RuntimeError(
                        f"Candidate changed between scans: {path}:{line_number}"
                    )
                loaded.append((line_map[line_number], conversation))
                remaining.remove(line_number)
                if not remaining:
                    break
        if remaining:
            raise RuntimeError(f"Could not reload {len(remaining)} candidates from {path}")
    return loaded


def render_deepseek_v4(
    conversation: list[dict[str, str]], bos_token: str, eos_token: str
) -> str:
    messages = list(conversation)
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    pieces = [bos_token]
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            pieces.append(content)
        elif role == "user":
            pieces.extend(("<｜User｜>", content))
        elif role == "assistant":
            pieces.extend(("<｜Assistant｜></think>", content, eos_token))
    return "".join(pieces)


def exact_select(
    candidates: list[tuple[CandidateRef, list[dict[str, str]]]],
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    num_samples: int,
) -> tuple[list[tuple[int, CandidateRef, list[dict[str, str]]]], dict[str, int]]:
    selected: list[
        tuple[int, int, CandidateRef, list[dict[str, str]]]
    ] = []
    stats = {"below_min": 0, "above_max": 0, "within_range": 0}
    bos_token = tokenizer.bos_token or ""
    eos_token = tokenizer.eos_token or ""

    ordered = sorted(candidates, key=lambda item: item[0].char_count, reverse=True)
    for index, (ref, conversation) in enumerate(ordered, start=1):
        rendered = render_deepseek_v4(conversation, bos_token, eos_token)
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if token_count < min_tokens:
            stats["below_min"] += 1
            continue
        if token_count > max_tokens:
            stats["above_max"] += 1
            continue
        stats["within_range"] += 1
        item = (token_count, ref.serial, ref, conversation)
        if len(selected) < num_samples:
            heapq.heappush(selected, item)
        elif item[:2] > selected[0][:2]:
            heapq.heapreplace(selected, item)
        if index % 100 == 0 or index == len(ordered):
            print(
                f"[tokenize {index}/{len(ordered)}] "
                f"in_range={stats['within_range']:,}, above={stats['above_max']:,}",
                flush=True,
            )

    result = [(tokens, ref, conv) for tokens, _, ref, conv in selected]
    result.sort(key=lambda item: item[0], reverse=True)
    return result, stats


def write_outputs(
    output_path: str,
    selected: list[tuple[int, CandidateRef, list[dict[str, str]]]],
    args: argparse.Namespace,
    scan_stats: dict[str, int],
    token_stats: dict[str, int],
) -> None:
    output = Path(output_path)
    manifest = Path(f"{output_path}.manifest.json")
    if not args.overwrite:
        for path in (output, manifest):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for token_count, ref, conversation in selected:
            record = {
                "conversations": conversation,
                "source_dataset": ref.source.name,
                "source_path": ref.source.path,
                "source_line": ref.line_number,
                "num_tokens": token_count,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    token_lengths = [item[0] for item in selected]
    manifest_data = {
        "catalog": os.path.abspath(args.catalog),
        "model_path": os.path.abspath(args.model_path),
        "output": os.path.abspath(args.output),
        "source_type": args.source_type,
        "requested_samples": args.num_samples,
        "num_samples": len(selected),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "candidate_pool_size": args.candidate_pool_size,
        "selected_token_min": min(token_lengths),
        "selected_token_max": max(token_lengths),
        "selected_token_mean": sum(token_lengths) / len(token_lengths),
        "scan_stats": scan_stats,
        "tokenization_stats": token_stats,
        "samples": [
            {
                "num_tokens": token_count,
                "source_dataset": ref.source.name,
                "source_path": ref.source.path,
                "source_line": ref.line_number,
            }
            for token_count, ref, _ in selected
        ],
    }
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    sources = load_sources(args.catalog, args.source_type)
    print(f"Scanning {len(sources)} annotation files...", flush=True)
    refs, scan_stats = scan_candidate_refs(sources, args.candidate_pool_size)
    print(f"Reloading {len(refs):,} bounded candidates...", flush=True)
    candidates = reload_candidates(refs)

    print(f"Loading tokenizer from {args.model_path}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    selected, token_stats = exact_select(
        candidates,
        tokenizer,
        args.min_tokens,
        args.max_tokens,
        args.num_samples,
    )
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Only found {len(selected)} matching samples, requested {args.num_samples}. "
            "Lower --min-tokens or raise --candidate-pool-size."
        )
    write_outputs(args.output, selected, args, scan_stats, token_stats)
    print(
        f"Wrote {len(selected)} samples to {args.output}; "
        f"tokens={selected[-1][0]:,}..{selected[0][0]:,}",
        flush=True,
    )
    print(f"Manifest: {args.output}.manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
