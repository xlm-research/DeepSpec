#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SPEC_RE = re.compile(r"['\"]([^'\"]+\.jsonl)(?:#(\d+))?['\"]")
MEDIA_FIELDS = ("images", "image", "videos", "video", "audios", "audio", "media", "files")
MEDIA_MARKERS = ("s3://", "oss://", "aoss://", "<image>", "image_url", "video_url", "audio_url", "data:image")


def parse_specs(input_path: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    section = None
    for line_no, line in enumerate(input_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("# ==="):
            section = stripped.strip("# ").strip()
            continue
        m = SPEC_RE.search(line)
        if not m:
            continue
        comment = ""
        # Split after closing quote conservatively so '#N' inside spec is not treated as the comment.
        tail = line[m.end():]
        if "#" in tail:
            comment = tail.split("#", 1)[1].strip()
        specs.append({
            "index": len(specs),
            "line_no": line_no,
            "path": m.group(1),
            "declared_n": int(m.group(2) or 0),
            "name": comment,
            "section": section,
        })
    return specs


def non_empty(value: Any) -> bool:
    if value is None:
        return False
    if value is False:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value)
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def has_multimodal(obj: dict[str, Any]) -> tuple[bool, str]:
    for key in MEDIA_FIELDS:
        if key in obj and non_empty(obj.get(key)):
            return True, f"field:{key}"

    messages = obj.get("messages") or obj.get("conversations") or []
    if isinstance(messages, list):
        for mi, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                return True, f"list_content:msg{mi}"
            if isinstance(content, str):
                low = content.lower()
                for marker in MEDIA_MARKERS:
                    if marker in low:
                        return True, f"marker:{marker}"
    # Some rows store media markers outside content; avoid full-row scan unless no message list exists.
    if not isinstance(messages, list) or not messages:
        raw = json.dumps(obj, ensure_ascii=False).lower()
        for marker in MEDIA_MARKERS:
            if marker in raw:
                return True, f"marker:{marker}"
    return False, ""


def normalize_role(role: Any) -> str:
    if role is None:
        return "user"
    role = str(role)
    mapping = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    return mapping.get(role, role)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Text output should not receive list content because it is classified multimodal.
    # Keep a conservative JSON representation for odd scalar/dict content.
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def to_conversations(obj: dict[str, Any]) -> list[dict[str, str]] | None:
    src = obj.get("conversations")
    if src is None:
        src = obj.get("messages")
    if not isinstance(src, list) or not src:
        return None
    convs: list[dict[str, str]] = []
    for msg in src:
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg.get("role", msg.get("from")))
        content = msg.get("content", msg.get("value", ""))
        convs.append({"role": role, "content": content_to_text(content)})
    return convs or None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert v42 MS-Swift pack sources to DeepSpec conversations JSONL, full-volume, text/mm split.")
    ap.add_argument("--input", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/input_v42.txt")
    ap.add_argument("--text-output", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-all.jsonl")
    ap.add_argument("--multimodal-output", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-multimodal-all.jsonl")
    ap.add_argument("--summary", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-all.summary.json")
    ap.add_argument("--source-specs", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/sensenova-flash-lite-v42-all.source_specs.tsv")
    ap.add_argument("--train-data-path", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/train_data_path_v42_all.txt")
    ap.add_argument("--train-data-args", default="/mnt/afs_agents/hongjiawei/code/DeepSpec_basemain/train_dataset/train_data_args_v42_all.sh")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50000)
    args = ap.parse_args()

    input_path = Path(args.input)
    text_output = Path(args.text_output)
    multimodal_output = Path(args.multimodal_output)
    summary_path = Path(args.summary)
    specs_path = Path(args.source_specs)
    train_data_path = Path(args.train_data_path)
    train_data_args = Path(args.train_data_args)

    for out in [text_output, multimodal_output, summary_path, specs_path, train_data_path, train_data_args]:
        if out.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {out} (pass --overwrite)")
    text_output.parent.mkdir(parents=True, exist_ok=True)

    specs = parse_specs(input_path)
    if not specs:
        raise SystemExit(f"No jsonl specs parsed from {input_path}")

    missing = [s for s in specs if not Path(s["path"]).exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} source files; first: {missing[0]['path']}")

    text_tmp = Path(str(text_output) + ".tmp")
    mm_tmp = Path(str(multimodal_output) + ".tmp")
    summary_tmp = Path(str(summary_path) + ".tmp")
    specs_tmp = Path(str(specs_path) + ".tmp")
    for tmp in [text_tmp, mm_tmp, summary_tmp, specs_tmp]:
        if tmp.exists():
            tmp.unlink()

    totals = collections.Counter()
    section_stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    reason_counts = collections.Counter()
    source_rows: list[dict[str, Any]] = []
    t0 = time.time()

    with text_tmp.open("w", encoding="utf-8") as text_f, mm_tmp.open("w", encoding="utf-8") as mm_f:
        for si, spec in enumerate(specs, 1):
            spath = Path(spec["path"])
            sc = collections.Counter()
            with spath.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    totals["raw_lines"] += 1
                    sc["raw_lines"] += 1
                    line = line.rstrip("\n")
                    if not line:
                        totals["empty_lines"] += 1
                        sc["empty_lines"] += 1
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        totals["json_errors"] += 1
                        sc["json_errors"] += 1
                        continue
                    if not isinstance(obj, dict):
                        totals["non_object"] += 1
                        sc["non_object"] += 1
                        continue

                    convs = to_conversations(obj)
                    if convs is None:
                        totals["no_conversations"] += 1
                        sc["no_conversations"] += 1
                        continue

                    is_mm, reason = has_multimodal(obj)
                    if is_mm:
                        totals["multimodal_rows"] += 1
                        sc["multimodal_rows"] += 1
                        reason_counts[reason] += 1
                        # Keep original row for multimodal recovery, plus source metadata.
                        mm_obj = dict(obj)
                        mm_obj.setdefault("_source_path", str(spath))
                        mm_obj.setdefault("_source_line", line_no)
                        mm_obj.setdefault("_multimodal_reason", reason)
                        mm_f.write(json.dumps(mm_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                    else:
                        totals["text_rows"] += 1
                        sc["text_rows"] += 1
                        text_f.write(json.dumps({"conversations": convs}, ensure_ascii=False, separators=(",", ":")) + "\n")

                    done = totals["raw_lines"]
                    if args.progress_every and done % args.progress_every == 0:
                        elapsed = time.time() - t0
                        print(
                            f"[progress] raw={done} text={totals['text_rows']} mm={totals['multimodal_rows']} "
                            f"json_err={totals['json_errors']} elapsed={elapsed:.1f}s",
                            file=sys.stderr,
                            flush=True,
                        )

            section = spec.get("section") or ""
            for k, v in sc.items():
                section_stats[section][k] += v
            source_rows.append({
                **spec,
                "raw_lines": sc["raw_lines"],
                "text_rows": sc["text_rows"],
                "multimodal_rows": sc["multimodal_rows"],
                "json_errors": sc["json_errors"],
                "empty_lines": sc["empty_lines"],
                "no_conversations": sc["no_conversations"],
            })
            print(
                f"[source {si}/{len(specs)}] raw={sc['raw_lines']} text={sc['text_rows']} mm={sc['multimodal_rows']} path={spath}",
                file=sys.stderr,
                flush=True,
            )

    with specs_tmp.open("w", encoding="utf-8") as f:
        f.write("index\tline_no\tsection\tname\tdeclared_n\traw_lines\ttext_rows\tmultimodal_rows\tjson_errors\tempty_lines\tno_conversations\tpath\n")
        for r in source_rows:
            f.write("\t".join(str(r.get(k, "")) for k in ["index", "line_no", "section", "name", "declared_n", "raw_lines", "text_rows", "multimodal_rows", "json_errors", "empty_lines", "no_conversations", "path"]) + "\n")

    summary = {
        "input": str(input_path),
        "text_output": str(text_output),
        "multimodal_output": str(multimodal_output),
        "source_specs": str(specs_path),
        "num_specs": len(specs),
        "declared_sum_ignored_for_full_conversion": sum(s["declared_n"] for s in specs),
        "totals": dict(totals),
        "multimodal_reason_counts": dict(reason_counts),
        "section_stats": {k: dict(v) for k, v in section_stats.items()},
        "elapsed_seconds": round(time.time() - t0, 3),
        "text_output_bytes": text_tmp.stat().st_size if text_tmp.exists() else 0,
        "multimodal_output_bytes": mm_tmp.stat().st_size if mm_tmp.exists() else 0,
    }
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_data_path.write_text(str(text_output) + "\n", encoding="utf-8")
    train_data_args.write_text(f"export DATA_PATH={text_output}\nexport DATA_MIX_NAME=sensenova_flash_lite_v42_all_text\n", encoding="utf-8")

    os.replace(text_tmp, text_output)
    os.replace(mm_tmp, multimodal_output)
    os.replace(summary_tmp, summary_path)
    os.replace(specs_tmp, specs_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
