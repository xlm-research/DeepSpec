from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import (
    Gemma4DSparkEvaluator,
    Qwen3DSparkEvaluator,
    Qwen3_6DSparkEvaluator,
    Qwen3_8DFlash2Evaluator,
)
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.data.parser import parse_media_uri_map_entries
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Qwen3_6DSparkModel": Qwen3_6DSparkEvaluator,
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Optional path for machine-readable acceptance metrics.",
    )
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
