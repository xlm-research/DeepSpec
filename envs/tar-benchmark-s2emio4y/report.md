# Conda environment archive benchmark

The test reused the tar.gz archive created by the existing packaging process. Archive creation and validation are excluded from the measured total. All destinations were initially empty in the recorded runs.

| Method | Elapsed seconds |
| --- | ---: |
| Previous cp -a directory copy | 521.257 |
| Previous rsync -a directory copy | 508.492 |
| Copy tar.gz archive from shared disk to /tmp | 22.395 |
| Extract local tar.gz archive | 80.673 |
| Archive transfer plus extraction | 103.068 |

Archive transfer plus extraction used 19.77% of the previous cp time and 20.27% of the previous rsync time (speed ratios: 5.06x and 4.93x).

- Shared archive: `/mnt/afs-agentpro/lezewei/DeepSpec/envs/deepspec_vllm_torchtitan_envs.tar.gz`
- Archive size: 4,102,897,342 bytes (4.103 GB)
- Local archive: `/tmp/deepspec-tar-benchmark-0bcnfhg3/deepspec_vllm_torchtitan_envs.tar.gz`
- Extracted environment: `/tmp/deepspec-tar-benchmark-0bcnfhg3/extracted/deepspec_vllm_torchtitan_envs`
- Regular files: 91,151
- Regular file bytes: 9,634,612,359
- Validation: all paths, entry types, regular file sizes, and symbolic link targets matched the source manifest. No missing, extra, or mismatched entries. Full content hashes and environment runtime behavior were not tested.

Commands timed separately:

```bash
cp -a -- "/mnt/afs-agentpro/lezewei/DeepSpec/envs/deepspec_vllm_torchtitan_envs.tar.gz" "/tmp/deepspec-tar-benchmark-0bcnfhg3/deepspec_vllm_torchtitan_envs.tar.gz"
tar -xzpf "/tmp/deepspec-tar-benchmark-0bcnfhg3/deepspec_vllm_torchtitan_envs.tar.gz" -C "/tmp/deepspec-tar-benchmark-0bcnfhg3/extracted"
```

Limitations: one run per method, different execution times, uncontrolled shared-disk load and caches. The archive had just been created, and its local copy had just been written before extraction. These results do not establish cold-cache or cross-machine performance. Timings end when each command returns; no explicit sync is included.
