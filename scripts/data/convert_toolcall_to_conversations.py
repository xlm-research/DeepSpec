#!/usr/bin/env python3
"""Convert native function-calling JSONL data to DeepSpec conversations JSONL.

Input records are OpenAI/Hermes-style:
  {"messages": [...], "tools": [...]}

Output records are DeepSpec-style:
  {"conversations": [{"role": "system|user|assistant", "content": "..."}, ...]}

The DeepSeek-V4-Flash Jinja template is used to render tool schemas, tool
results, and assistant tool_calls. Top-level `tools` are folded into the first
system message so DeepSpec's existing `conversations` loader can consume the
result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, StrictUndefined

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_CLOSE = "</think>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert OpenAI/Hermes native tool-call JSONL into DeepSpec "
            "conversations JSONL."
        )
    )
    parser.add_argument(
        "--template",
        default="/mnt/afs-agentpro/hongjiawei/code/DeepSpec/scripts/train/flash.jinja",
        help="DeepSeek-V4-Flash Jinja template path.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Input JSONL files. Entries may be either /path/file.jsonl or /path/file.jsonl#N.",
    )
    parser.add_argument(
        "--input-list",
        help=(
            "Text file containing input JSONL paths, one per line. Lines may use "
            "/path/file.jsonl#N; the #N suffix is ignored by default unless "
            "--respect-entry-counts is set."
        ),
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--respect-entry-counts",
        action="store_true",
        help="If an input entry has #N, keep at most N records from that file.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap across all input files, useful for smoke tests.",
    )
    parser.add_argument(
        "--thinking-mode",
        default="non-thinking",
        choices=["thinking", "non-thinking"],
        help=(
            "Template thinking mode used for rendering assistant bodies. "
            "Default non-thinking matches DeepSpec's current deepseek_v4 "
            "parser, which already emits <Assistant></think> before content."
        ),
    )
    parser.add_argument(
        "--keep-reasoning",
        action="store_true",
        help=(
            "Keep reasoning_content in rendered assistant text. By default it is "
            "dropped to avoid double thinking markers with DeepSpec's current "
            "deepseek_v4 chat template."
        ),
    )
    parser.add_argument(
        "--assistant-content-mode",
        default="deepspec_current",
        choices=["deepspec_current", "flash_body"],
        help=(
            "How to store rendered assistant text inside conversations.content. "
            "deepspec_current strips a leading </think> because the current "
            "DeepSpec deepseek_v4 template injects it. flash_body stores the "
            "Flash body literally, e.g. <think>...</think>answer/tool_calls; use "
            "with a DeepSpec chat template whose assistant header is <｜Assistant｜>."
        ),
    )
    parser.add_argument(
        "--skip-bad-records",
        action="store_true",
        help="Skip malformed records instead of failing fast.",
    )
    return parser.parse_args()


def load_template(path: str):
    text = Path(path).read_text(encoding="utf-8")
    env = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)
    # Jinja2's built-in tojson is HTML-safe and escapes '<', '>' and '&'.
    # Transformers chat templates emit raw text, so use a plain JSON filter.
    env.filters["tojson"] = json_dumps_compact
    return env.from_string(text)


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Make OpenAI wire-format tool_calls acceptable to flash.jinja.

    flash.jinja expects `function.arguments` to be a mapping because Jinja has a
    tojson filter but no fromjson filter. Most native SFT files store arguments
    as JSON strings, so parse them here.
    """
    if not isinstance(message, dict) or not message.get("tool_calls"):
        return message
    out = dict(message)
    normalized = []
    for tool_call in message.get("tool_calls") or []:
        tc = dict(tool_call)
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                parsed = {"arguments": args}
            if not isinstance(parsed, dict):
                parsed = {"arguments": parsed}
            args = parsed
        elif args is None:
            args = {}
        elif not isinstance(args, dict):
            args = {"arguments": args}
        fn["arguments"] = args
        tc["function"] = fn
        normalized.append(tc)
    out["tool_calls"] = normalized
    return out


def render(template, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
           thinking_mode: str = "non-thinking", drop_thinking: bool = True) -> str:
    kwargs = dict(
        messages=messages,
        add_generation_prompt=False,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
    )
    if tools:
        kwargs["tools"] = tools
    return template.render(**kwargs)


def strip_bos(text: str) -> str:
    return text[len(BOS):] if text.startswith(BOS) else text


def strip_eos(text: str) -> str:
    return text[:-len(EOS)] if text.endswith(EOS) else text


def extract_after_marker(rendered: str, marker: str) -> str:
    text = strip_bos(rendered)
    idx = text.find(marker)
    if idx < 0:
        raise ValueError(f"Rendered template did not contain marker {marker!r}: {text[:200]!r}")
    return strip_eos(text[idx + len(marker):])


def render_system_with_tools(template, system_content: str, tools: list[dict[str, Any]] | None) -> str:
    rendered = render(
        template,
        messages=[{"role": "system", "content": system_content or ""}],
        tools=tools,
        thinking_mode="non-thinking",
        drop_thinking=True,
    )
    return strip_bos(rendered)


def render_tool_as_user_content(template, message: dict[str, Any]) -> str:
    rendered = render(
        template,
        messages=[{"role": "tool", "content": message.get("content") or ""}],
        thinking_mode="non-thinking",
        drop_thinking=True,
    )
    return extract_after_marker(rendered, USER)


def render_assistant_content(
    template,
    message: dict[str, Any],
    *,
    thinking_mode: str,
    keep_reasoning: bool,
    assistant_content_mode: str,
) -> str:
    normalized = normalize_tool_calls(message)
    rendered = render(
        template,
        messages=[normalized],
        thinking_mode=thinking_mode,
        drop_thinking=not keep_reasoning,
    )
    body = extract_after_marker(rendered, ASSISTANT)
    if assistant_content_mode == "flash_body":
        return body
    # DeepSpec's current `deepseek_v4` parser emits '<Assistant></think>' before
    # every assistant content. For non-thinking renders, flash.jinja also emits
    # the same leading '</think>'; remove it here to avoid duplication.
    if body.startswith(THINK_CLOSE):
        body = body[len(THINK_CLOSE):]
    return body


def get_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if messages is None:
        messages = record.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record must contain a non-empty messages/conversations list")
    return messages


def convert_record(
    record: dict[str, Any],
    template,
    *,
    thinking_mode: str,
    keep_reasoning: bool,
    assistant_content_mode: str,
) -> dict[str, Any]:
    messages = get_messages(record)
    tools = record.get("tools") or None
    if tools is not None and not isinstance(tools, list):
        raise ValueError("record.tools must be a list when present")

    conversations: list[dict[str, str]] = []
    pending_system_parts: list[str] = []
    start_idx = 0

    if messages and messages[0].get("role") == "system":
        first_system = messages[0].get("content") or ""
        start_idx = 1
    else:
        first_system = ""

    system_text = render_system_with_tools(template, first_system, tools)
    if system_text:
        conversations.append({"role": "system", "content": system_text})

    for message in messages[start_idx:]:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        role = message.get("role")
        if role == "system" or role == "developer":
            # DeepSpec's parser only supports a leading system turn. Fold any
            # later system/developer instructions into the leading system text.
            pending_system_parts.append(str(message.get("content") or ""))
            continue
        if pending_system_parts:
            extra = "\n\n".join(part for part in pending_system_parts if part)
            if extra:
                if conversations and conversations[0]["role"] == "system":
                    conversations[0]["content"] += "\n\n" + extra
                else:
                    conversations.insert(0, {"role": "system", "content": extra})
            pending_system_parts.clear()

        if role == "user":
            conversations.append({"role": "user", "content": message.get("content") or ""})
        elif role == "tool":
            conversations.append({"role": "user", "content": render_tool_as_user_content(template, message)})
        elif role == "assistant":
            conversations.append(
                {
                    "role": "assistant",
                    "content": render_assistant_content(
                        template,
                        message,
                        thinking_mode=thinking_mode,
                        keep_reasoning=keep_reasoning,
                        assistant_content_mode=assistant_content_mode,
                    ),
                }
            )
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    if pending_system_parts:
        extra = "\n\n".join(part for part in pending_system_parts if part)
        if extra:
            if conversations and conversations[0]["role"] == "system":
                conversations[0]["content"] += "\n\n" + extra
            else:
                conversations.insert(0, {"role": "system", "content": extra})

    if not any(item["role"] == "assistant" for item in conversations):
        raise ValueError("converted record has no assistant turn")
    # DeepSpec parser requires first non-system turn to be user.
    first_non_system = next((item for item in conversations if item["role"] != "system"), None)
    if first_non_system is None or first_non_system["role"] != "user":
        raise ValueError("converted conversation must start with user after system")
    return {"conversations": conversations}


def parse_input_entry(entry: str) -> tuple[str, int | None]:
    path, sep, count_text = entry.rpartition("#")
    if sep and count_text.isdigit():
        return path, int(count_text)
    return entry, None


def load_input_entries(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    entries: list[str] = []
    if args.input:
        entries.extend(args.input)
    if args.input_list:
        with open(args.input_list, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    entries.append(line)
    if not entries:
        raise SystemExit("At least one of --input or --input-list is required.")
    return [parse_input_entry(entry) for entry in entries]


def iter_jsonl(entries: Iterable[tuple[str, int | None]], *, respect_entry_counts: bool):
    for path, entry_limit in entries:
        emitted_for_file = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if respect_entry_counts and entry_limit is not None and emitted_for_file >= entry_limit:
                    break
                if not line.strip():
                    continue
                emitted_for_file += 1
                yield path, line_no, json.loads(line)


def main() -> None:
    args = parse_args()
    input_entries = load_input_entries(args)
    template = load_template(args.template)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")

    written = 0
    skipped = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as out:
            for src_path, line_no, record in iter_jsonl(
                input_entries,
                respect_entry_counts=args.respect_entry_counts,
            ):
                if args.max_records is not None and written >= args.max_records:
                    break
                try:
                    converted = convert_record(
                        record,
                        template,
                        thinking_mode=args.thinking_mode,
                        keep_reasoning=args.keep_reasoning,
                        assistant_content_mode=args.assistant_content_mode,
                    )
                except Exception as exc:
                    if not args.skip_bad_records:
                        raise RuntimeError(f"failed at {src_path}:{line_no}: {exc}") from exc
                    skipped += 1
                    continue
                out.write(json_dumps_compact(converted))
                out.write("\n")
                written += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Wrote {written} records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} malformed records")


if __name__ == "__main__":
    main()
