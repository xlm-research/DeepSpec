"""Prepare a small, traceable image-text smoke dataset without changing its source."""

import argparse
import copy
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from torchspec.data.renderers import RENDERER_REGISTRY
from torchspec.data.utils import extract_media_urls


def prepare(args):
    indices = [int(value) for value in args.indices.split(",")]
    selected = {}
    with open(args.source) as source:
        for index, line in enumerate(source):
            if index in indices:
                selected[index] = json.loads(line)
            if len(selected) == len(indices):
                break
    if len(selected) != len(indices):
        raise ValueError("Requested source indices were not found")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    renderer = RENDERER_REGISTRY.create("qwen3.5-multimodal", tokenizer)
    rows, summaries = [], []
    for index in indices:
        row = copy.deepcopy(selected[index])
        row.pop("id", None)
        row["data_id"] = f"image_text_source_{index}"
        for message in row["messages"]:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if "://" in url:
                            raise ValueError("First smoke dataset must use local images")
                        item["image_url"]["url"] = (Path(args.media_root) / url).resolve().as_uri()
        text = renderer.format(
            row["messages"], row.get("tools"), generation_config=row.get("generation_config")
        )
        media = extract_media_urls(row["messages"])
        images = []
        for url in media["images"]:
            with Image.open(url.removeprefix("file://")) as image:
                images.append(image.convert("RGB"))
        before = tokenizer.encode(text, add_special_tokens=False)
        expanded = processor(text=text, images=images, return_tensors="pt")
        ids = expanded["input_ids"][0].tolist()
        if len(ids) + 1 > args.max_length:
            raise ValueError(
                f"Source {index}: {len(ids)} expanded tokens exceed reserved context limit"
            )
        mask = renderer.compute_loss_mask(ids)
        supervised_text = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])
        assistants = [message for message in row["messages"] if message["role"] == "assistant"]
        for message in assistants:
            reasoning = (message.get("reasoning_content") or "").strip()
            if reasoning and reasoning not in supervised_text:
                raise ValueError(f"Source {index}: historical reasoning was lost from supervision")
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or call)["name"]
                if f"<function={name}>" not in supervised_text:
                    raise ValueError(f"Source {index}: tool call was lost from supervision")
        eligible = sum(a and b for a, b in zip(mask, mask[1:]))
        if sum(mask) < 14 or eligible == 0:
            raise ValueError(f"Source {index}: no eligible DSpark supervision")
        pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if before.count(pad) != len(images) or ids.count(pad) <= len(images):
            raise ValueError(f"Source {index}: image placeholders were not expanded")
        rows.append(row)
        summaries.append(
            {
                "source_index": index,
                "messages": len(row["messages"]),
                "assistant_turns": len(assistants),
                "images": len(images),
                "tokens_before_expansion": len(before),
                "expanded_tokens": len(ids),
                "supervised_tokens": sum(mask),
                "eligible_anchors": eligible,
                "image_tokens": ids.count(pad),
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, count in (("prefill.jsonl", len(rows)), ("train.jsonl", args.rows)):
        with (output / filename).open("w") as destination:
            for sequence in range(count):
                row = copy.deepcopy(rows[sequence % len(rows)])
                row["data_id"] += f"_repeat_{sequence:04d}"
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source": str(Path(args.source).resolve()),
        "media_root": args.media_root,
        "indices": indices,
        "training_rows": args.rows,
        "global_batch": 16,
        "purpose": "pipeline smoke validation; repeated samples are not a quality evaluation",
        "samples": summaries,
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mydata/image-text-327.jsonl")
    parser.add_argument("--media-root", default="/mnt/afs_toolcall/juxiaolong1/Data/draftData")
    parser.add_argument(
        "--model",
        default="/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp",
    )
    parser.add_argument("--indices", default="114,217,231")
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--output", default="cache/qwen3.8-27b-image-text-1node/data")
    prepare(parser.parse_args())
