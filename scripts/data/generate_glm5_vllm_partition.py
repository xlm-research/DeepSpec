"""Keep multiprocessing spawn from importing the training stack in TP workers."""

if __name__ == "__main__":
    import os
    import sys

    from deepspec.trainer import glm5_vllm

    if os.environ.get("DEEPSPEC_BENCHMARK_TIMING") == "1":
        import json
        from pathlib import Path
        import time

        worker_started = time.time()
        original_generate = glm5_vllm.generate_job

        def timed_generate(job, *, extract):
            records = []
            ready_unix = time.time()
            started = time.perf_counter()

            def timed_extract(batch):
                sample_started = time.perf_counter()
                features = extract(batch)
                records.append(
                    {
                        "tokens": batch["input_ids"].numel(),
                        "seconds": time.perf_counter() - sample_started,
                    }
                )
                return features

            original_generate(job, extract=timed_extract)
            result = {
                "owner_ranks": job["owner_ranks"],
                "worker_started_unix": worker_started,
                "llm_ready_unix": ready_unix,
                "generate_job_seconds": time.perf_counter() - started,
                "samples": records,
            }
            Path(sys.argv[1] + ".timing.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print("[deepspec-target-benchmark] " + json.dumps(result), flush=True)

        glm5_vllm.generate_job = timed_generate

    glm5_vllm.worker_main(sys.argv[1])
