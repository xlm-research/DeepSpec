"""Text or exact-length token generation with the TP=8 Loop checkpoint."""

import argparse
import json
from pathlib import Path

import yaml

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the checkpoint path in config.yaml")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--prompt-tokens", type=int, help="Repeat text to this token count")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    config_path = Path(__file__).with_name("config.yaml")
    kwargs = {
        key.replace("-", "_"): value
        for key, value in yaml.safe_load(config_path.read_text()).items()
    }
    if args.model:
        kwargs["model"] = args.model
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if args.prompt_tokens is not None:
        if not 0 < args.prompt_tokens < kwargs["max_model_len"]:
            parser.error("--prompt-tokens must be between 1 and 8191")
        prompt = (prompt * ((args.prompt_tokens + len(prompt) - 1) // len(prompt)))[
            : args.prompt_tokens
        ]
    outputs = llm.generate(
        {"prompt_token_ids": prompt},
        SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=args.ignore_eos),
        use_tqdm=False,
    )
    for output in outputs:
        result = output.outputs[0]
        print(
            json.dumps(
                {
                    "request_id": output.request_id,
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(result.token_ids),
                    "finished": output.finished,
                    "finish_reason": result.finish_reason,
                    "stop_reason": result.stop_reason,
                    "token_ids": result.token_ids,
                    "text": result.text,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
