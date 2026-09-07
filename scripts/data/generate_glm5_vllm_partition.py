"""Keep multiprocessing spawn from importing the training stack in TP workers."""

if __name__ == "__main__":
    import sys

    from deepspec.trainer.glm5_vllm import worker_main

    worker_main(sys.argv[1])
