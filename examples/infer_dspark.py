"""Run the local Qwen/SenseNova target with a trained DSpark draft.

From the TorchSpec root:
    source ./h800conda-tmp.sh
    CUDA_VISIBLE_DEVICES=1 python examples/infer_dspark.py --prompt '你好，请介绍投机解码。'
"""

import argparse
import os
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-model",
        default=os.environ.get(
            "TARGET_MODEL_PATH",
            "/mnt/afs_agents/hongjiawei/delivery/sensenova-flash-lite-v42-20260830-dpo-mtp",
        ),
    )
    parser.add_argument("--draft-model", default=str(root / "outputs/qwen3_8_27b-dspark-128k-hr"))
    parser.add_argument(
        "--prompt", default="请用三句话解释什么是投机解码，以及它为什么能加速大模型推理。"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    args = parser.parse_args()

    # DSpark is implemented by this checkout's V2 GPU model runner.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter

    print(f"Target: {args.target_model}\nDSpark draft: {args.draft_model}", flush=True)
    llm = LLM(
        model=args.target_model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=42,
        speculative_config={
            "method": "dspark",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": "greedy",
            "attention_backend": "FLASH_ATTN",
        },
    )
    outputs = llm.chat(
        [{"role": "user", "content": args.prompt}],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    result = outputs[0].outputs[0]
    print(f"\n{result.text}\n", flush=True)
    print(f"Generated tokens: {len(result.token_ids)}; finish reason: {result.finish_reason}")

    # Nonzero draft counters confirm that generation actually used DSpark.
    counters = {
        metric.name: metric.value for metric in llm.get_metrics() if isinstance(metric, Counter)
    }
    drafts = counters.get("vllm:spec_decode_num_drafts", 0)
    proposed = counters.get("vllm:spec_decode_num_draft_tokens", 0)
    accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
    print(f"DSpark: drafts={drafts}, proposed_tokens={proposed}, accepted_tokens={accepted}")
    if proposed:
        print(f"Acceptance rate: {accepted / proposed:.2%}")
        print(f"Mean acceptance length (including bonus token): {1 + accepted / drafts:.2f}")


if __name__ == "__main__":
    main()
