"""Prepare traceable image-text DSpark supervision windows."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import requests
from transformers import AutoProcessor, AutoTokenizer

from torchspec.data.renderers import RENDERER_REGISTRY
from torchspec.data.utils import extract_media_urls

WINDOW_CACHE_VERSION = "qwen35-full-data-windows-v2"


def _cache_media_reference(
    reference: str,
    media_root: Path,
    cache_dir: Path,
    records: dict[str, dict[str, Any]],
) -> str:
    if reference in records:
        return str(records[reference]["local_uri"])
    if reference.startswith("https://"):
        response = requests.get(reference, timeout=(15, 60))
        response.raise_for_status()
        content = response.content
        digest = hashlib.sha256(content).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(reference.encode()).hexdigest()[:16]
        path = cache_dir / f"{name}-{digest}.image"
        path.write_bytes(content)
    else:
        if "://" in reference and not reference.startswith("file://"):
            raise ValueError(f"Unsupported image reference: {reference}")
        path = (
            Path(unquote(urlsplit(reference).path))
            if reference.startswith("file://")
            else media_root / reference
        ).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records[reference] = {"local_uri": path.as_uri(), "sha256": digest, "frame": 0}
    return path.as_uri()


class _WindowBuilder:
    def __init__(self, tokenizer: Any, processor: Any, renderer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.renderer = renderer
        self.input_limit = max_length - 1
        self.images: dict[str, Any] = {}
        self.image_tokens: dict[str, int] = {}
        self.image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def load_images(self, messages: list[dict[str, Any]]) -> list[Any]:
        from vllm.multimodal.utils import fetch_image

        urls = (extract_media_urls(messages) or {}).get("images", [])
        result = []
        for url in urls:
            if url not in self.images:
                loaded = fetch_image(url)
                image = getattr(loaded, "media", loaded)
                image.load()
                self.images[url] = image
                probe = self.processor(
                    text="<|vision_start|><|image_pad|><|vision_end|>",
                    images=[image],
                    return_tensors="pt",
                )["input_ids"][0].tolist()
                self.image_tokens[url] = probe.count(self.image_pad)
            result.append(self.images[url])
        return result

    def measure(self, row: dict[str, Any], indices: list[int]) -> int:
        messages = [row["messages"][i] for i in indices]
        self.load_images(messages)
        text = self.renderer.format(
            messages, row.get("tools"), generation_config=row.get("generation_config")
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        urls = (extract_media_urls(messages) or {}).get("images", [])
        return len(ids) + sum(self.image_tokens[url] - 1 for url in urls)

    def fit(
        self,
        source: dict[str, Any],
        first: int,
        end: int,
        *,
        shorten: bool = False,
    ) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]] | None:
        row = copy.deepcopy(source)
        indices = list(range(end + 1))
        reductions: list[dict[str, Any]] = []
        if self.measure(row, indices) <= self.input_limit:
            return row, indices, reductions
        users = [i for i in indices if i < first and row["messages"][i]["role"] == "user"]
        if not users:
            raise ValueError("A supervision window requires an original user query")
        latest_user = users[-1]
        removable = [
            i
            for i in indices
            if i < first and i != latest_user and row["messages"][i]["role"] != "system"
        ]
        removed = set(removable)
        minimal = [i for i in indices if i not in removed]
        if self.measure(row, minimal) <= self.input_limit:
            low, high = 0, len(removable)
            while low < high:
                count = (low + high) // 2
                omitted = set(removable[:count])
                kept = [i for i in indices if i not in omitted]
                if self.measure(row, kept) <= self.input_limit:
                    high = count
                else:
                    low = count + 1
            removed = set(removable[:low])
            indices = [i for i in indices if i not in removed]
        else:
            indices = minimal
        reductions.extend(
            {
                "message_index": i,
                "kind": "omitted_context_message",
                "role": source["messages"][i]["role"],
            }
            for i in sorted(removed)
        )
        if self.measure(row, indices) <= self.input_limit:
            return row, indices, reductions
        if not shorten:
            return None
        slots: list[dict[str, Any]] = []
        for i in indices:
            message = row["messages"][i]
            if message["role"] not in ("user", "tool"):
                continue
            content = message.get("content")
            blocks = [{"text": content}] if isinstance(content, str) else content or []
            for block_index, block in enumerate(blocks):
                if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                    continue
                text = block["text"]
                maximum = len(text)
                if i == latest_user and not any(b.get("type") == "image_url" for b in blocks):
                    maximum = max(0, len(text.rstrip()) - 1)
                if maximum:
                    slots.append(
                        {
                            "message": message,
                            "block": block,
                            "text": text,
                            "message_index": i,
                            "content_block": block_index if isinstance(content, list) else None,
                            "maximum": maximum,
                        }
                    )
        total_characters = sum(len(slot["text"]) for slot in slots)
        if not total_characters:
            return None

        def retain(budget: int) -> None:
            for slot in slots:
                text = slot["text"]
                removed_chars = min(
                    slot["maximum"],
                    len(text) - len(text) * budget // total_characters,
                )
                slot["removed_chars"] = removed_chars
                slot["block"]["text"] = text[removed_chars:]
                if slot["content_block"] is None:
                    slot["message"]["content"] = slot["block"]["text"]

        retain(0)
        if self.measure(row, indices) > self.input_limit:
            return None
        low, high = 0, total_characters
        while low < high:
            budget = (low + high + 1) // 2
            retain(budget)
            if self.measure(row, indices) <= self.input_limit:
                low = budget
            else:
                high = budget - 1
        retain(low)
        reductions.extend(
            {
                "message_index": slot["message_index"],
                "kind": "removed_text_prefix",
                "content_block": slot["content_block"],
                "removed_char_range": [0, slot["removed_chars"]],
            }
            for slot in slots
            if slot["removed_chars"]
        )
        return row, indices, reductions

    def build(
        self,
        source: dict[str, Any],
        source_index: int,
        min_loss_tokens: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assistants = [i for i, m in enumerate(source["messages"]) if m["role"] == "assistant"]
        if not assistants:
            raise ValueError(f"Source {source_index}: no assistant supervision segments")
        source_id = f"image_text_source_{source_index}"
        rows: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(assistants):
            first = assistants[cursor]
            fit = self.fit(source, first, assistants[-1])
            last = len(assistants) - 1
            if fit is None:
                low, high = cursor, len(assistants) - 1
                last = cursor - 1
                while low <= high:
                    middle = (low + high) // 2
                    candidate = self.fit(source, first, assistants[middle])
                    if candidate is not None:
                        fit, last = candidate, middle
                        low = middle + 1
                    else:
                        high = middle - 1
            if fit is None:
                last = len(assistants) - 1
                fit = self.fit(source, first, assistants[last], shorten=True)
                if fit is None:
                    low, high = cursor, last
                    last = cursor - 1
                    while low <= high:
                        middle = (low + high) // 2
                        candidate = self.fit(source, first, assistants[middle], shorten=True)
                        if candidate is not None:
                            fit, last = candidate, middle
                            low = middle + 1
                        else:
                            high = middle - 1
            if fit is None:
                raise ValueError(
                    f"Source {source_index}, assistant message {first}: cannot fit an intact "
                    f"supervision segment within {self.input_limit} expanded input tokens"
                )
            row, indices, reductions = fit
            messages = [row["messages"][i] for i in indices]
            text = self.renderer.format(
                messages, row.get("tools"), generation_config=row.get("generation_config")
            )
            images = self.load_images(messages)
            ids = self.processor(text=text, images=images or None, return_tensors="pt")[
                "input_ids"
            ][0].tolist()
            if len(ids) > self.input_limit:
                raise ValueError(
                    f"Source {source_index}: actual expansion exceeded the input limit"
                )
            supervised = assistants[cursor : last + 1]
            retained_assistants = [
                i for i in indices if source["messages"][i]["role"] == "assistant"
            ]
            ordinals = [retained_assistants.index(i) for i in supervised]
            mask = self.renderer.compute_loss_mask(
                ids,
                supervision_window={
                    "assistant_count": len(retained_assistants),
                    "supervised_assistant_ordinals": ordinals,
                },
            )
            if sum(mask) < min_loss_tokens or not any(a and b for a, b in zip(mask, mask[1:])):
                raise ValueError(f"Source {source_index}: no eligible DSpark supervision")
            window_id = f"{source_id}_window_{len(rows):04d}"
            selection = {
                "source_id": source_id,
                "source_index": source_index,
                "window_id": window_id,
                "message_indices": indices,
                "supervised_message_indices": supervised,
                "supervised_assistant_ordinals": ordinals,
                "assistant_count": len(retained_assistants),
                "context_reductions": reductions,
                "expanded_tokens": len(ids),
                "supervised_tokens": sum(mask),
                "image_count": len(images),
            }
            row["messages"] = messages
            row["data_id"] = window_id
            row["metadata"] = {"supervision_window": selection}
            rows.append(row)
            cursor = last + 1
        summary = {
            "source_id": source_id,
            "source_index": source_index,
            "assistant_message_indices": assistants,
            "window_ids": [row["data_id"] for row in rows],
        }
        return rows, summary


def prepare_full_data(args: Any) -> dict[str, Any]:
    """Prepare every source into complete bounded assistant supervision windows."""
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    renderer = RENDERER_REGISTRY.create("qwen3.5-full-data", tokenizer)
    builder = _WindowBuilder(tokenizer, processor, renderer, args.max_length)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    media_records: dict[str, dict[str, Any]] = {}
    with source.open() as input_file:
        for index, line in enumerate(input_file):
            row = json.loads(line)
            row.pop("id", None)
            try:
                for message in row["messages"]:
                    if isinstance(message.get("content"), list):
                        for item in message["content"]:
                            if item.get("type") == "image_url":
                                item["image_url"]["url"] = _cache_media_reference(
                                    item["image_url"]["url"],
                                    Path(args.media_root),
                                    output / "media",
                                    media_records,
                                )
                prepared, summary = builder.build(row, index, args.min_loss_tokens)
            except Exception as error:
                (output / "preparation_failure.json").write_text(
                    json.dumps(
                        {"status": "incomplete", "source_index": index, "error": str(error)},
                        indent=2,
                    )
                    + "\n"
                )
                raise ValueError(f"Source {index}: {error}") from error
            rows.extend(prepared)
            sources.append(summary)
            print(f"Prepared source {index}: {len(prepared)} supervision windows", flush=True)
    if len(sources) != args.expected_sources:
        raise ValueError(f"Expected {args.expected_sources} sources, found {len(sources)}")
    windows = [row["metadata"]["supervision_window"] for row in rows]
    for record in media_records.values():
        image = builder.images[record["local_uri"]]
        record.update(width=image.width, height=image.height, mode=image.mode)
    report = {
        "cache_version": WINDOW_CACHE_VERSION,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": str(Path(args.model).resolve()),
        "media_root": str(Path(args.media_root).resolve()),
        "max_length": args.max_length,
        "min_loss_tokens": args.min_loss_tokens,
        "source_count": len(sources),
        "window_count": len(windows),
        "max_images": max(window["image_count"] for window in windows),
        "media": media_records,
        "sources": sources,
        "windows": windows,
    }
    for filename in ("train.jsonl", "prefill.jsonl"):
        with (output / filename).open("w") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_count": len(sources), "window_count": len(windows)}))
    return report
