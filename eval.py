from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import (
    Gemma4DSparkEvaluator,
    Qwen3DSparkEvaluator,
    Qwen3_6DSparkEvaluator,
    Qwen3_8DSparkEvaluator,
    Qwen3_8DFlash2Evaluator,
)
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.data.parser import parse_media_uri_map_entries
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Qwen3_6DSparkModel": Qwen3_6DSparkEvaluator,
    "Qwen3_8DSparkModel": Qwen3_8DSparkEvaluator,
    "DFlash2DraftModel": Qwen3_8DFlash2Evaluator,
    "Qwen3_8DFlash2Model": Qwen3_8DFlash2Evaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, got {value!r}."
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, got {value!r}."
        )
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative number, got {value!r}."
        ) from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative number, got {value!r}."
        )
    return parsed


def _top_p_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected top-p in (0, 1], got {value!r}."
        ) from exc
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Expected top-p in (0, 1], got {value!r}."
        )
    return parsed


def _closed_unit_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a number in [0, 1], got {value!r}."
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Expected a number in [0, 1], got {value!r}."
        )
    return parsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=_non_negative_int, default=2048)
    parser.add_argument("--temperature", type=_non_negative_float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=_non_negative_int,
        default=0,
        help="Target-distribution top-k filter; 0 keeps the full vocabulary.",
    )
    parser.add_argument(
        "--top-p",
        type=_top_p_float,
        default=1.0,
        help="Target-distribution nucleus filter; 1.0 disables filtering.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=_closed_unit_float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=("static", "hardware-aware"),
        default="static",
        help=(
            "Use the diagnostic static confidence threshold or the DSpark "
            "hardware-aware prefix scheduler."
        ),
    )
    parser.add_argument(
        "--confidence-calibration-json",
        type=str,
        default=None,
        help="Sequential Temperature Scaling artifact for confidence scheduling.",
    )
    parser.add_argument(
        "--sps-profile-json",
        type=str,
        default=None,
        help="Hardware SPS(B) profile required by --scheduler-mode hardware-aware.",
    )
    parser.add_argument(
        "--confidence-observations-jsonl",
        type=str,
        default=None,
        help=(
            "Optional raw conditional-logit/prefix-label output for fitting "
            "STS; requires static scheduling with threshold 0."
        ),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--dataset-root", type=str, default="./eval_datasets")
    parser.add_argument(
        "--media-root",
        type=str,
        default=None,
        help="Root used to resolve relative image/video paths in evaluation JSONL.",
    )
    parser.add_argument(
        "--media-uri-map",
        action="append",
        default=[],
        metavar="SOURCE_PREFIX=REPLACEMENT_PREFIX",
        help="Rewrite media URI prefixes before loading; repeat as needed.",
    )
    parser.add_argument("--chat-template", type=str, default="qwen")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Evaluation task as NAME or NAME:MAX_SAMPLES. Repeat as needed.",
    )
    args = parser.parse_args()
    try:
        args.media_uri_map = parse_media_uri_map_entries(args.media_uri_map)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.tasks = [_parse_task(task) for task in args.task] or list(TASKS)
    return args


def _parse_task(value: str):
    name, separator, raw_max_samples = value.partition(":")
    if not name:
        raise argparse.ArgumentTypeError("Task name cannot be empty.")
    if not separator:
        return name, None
    try:
        max_samples = int(raw_max_samples)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Task max samples must be an integer: {value}"
        ) from exc
    if max_samples <= 0:
        raise argparse.ArgumentTypeError(
            f"Task max samples must be positive: {value}"
        )
    return name, max_samples


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
