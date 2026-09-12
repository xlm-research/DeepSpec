"""Keep vLLM multiprocessing spawn from re-entering the training stack."""

if __name__ == "__main__":
    import sys

    from deepspec.trainer.qwen3_8_vllm import worker_main

    worker_main(sys.argv[1])
