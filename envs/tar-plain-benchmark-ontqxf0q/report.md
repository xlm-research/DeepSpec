# Plain tar versus tar.gz environment transfer

Plain tar completed transfer plus extraction in **63.343 seconds**, versus **103.068 seconds** for tar.gz: **39.724 seconds less (38.54% lower elapsed time)**.

| Method | Archive size (decimal GB) | Transfer seconds | Extraction seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| tar | 9.785 | 52.421 | 10.922 | 63.343 |
| tar.gz | 4.103 | 22.395 | 80.673 | 103.068 |
| Earlier cp -a directory copy | — | — | — | 521.257 |
| Earlier rsync -a directory copy | — | — | — | 508.492 |

The plain tar was produced by removing the gzip compression layer from the previously tested local tar.gz archive and writing the exact resulting tar stream to the shared disk. This ensures identical archive contents. Preparation took 273.277 seconds and is excluded from the comparison, as is validation. Plain tar is an archive without compression.

- Shared tar: `/mnt/afs-agentpro/lezewei/DeepSpec/envs/deepspec_vllm_torchtitan_envs.tar`
- Local tar: `/tmp/deepspec-plain-tar-benchmark-xyodbn1x/deepspec_vllm_torchtitan_envs.tar`
- New extracted environment: `/tmp/deepspec-plain-tar-benchmark-xyodbn1x/extracted/deepspec_vllm_torchtitan_envs`
- Regular files: 91,151
- Regular file data: 9,634,612,359 bytes
- Validation: paths, entry types, regular file sizes and symbolic link targets match the source manifest; no missing, extra or different entries. Full content hashes and environment runtime behavior were not tested.

Commands timed separately:

```bash
cp -a -- "/mnt/afs-agentpro/lezewei/DeepSpec/envs/deepspec_vllm_torchtitan_envs.tar" "/tmp/deepspec-plain-tar-benchmark-xyodbn1x/deepspec_vllm_torchtitan_envs.tar"
tar -xpf "/tmp/deepspec-plain-tar-benchmark-xyodbn1x/deepspec_vllm_torchtitan_envs.tar" -C "/tmp/deepspec-plain-tar-benchmark-xyodbn1x/extracted"
```

In this measurement, plain tar transferred 30.027 seconds more slowly but extracted 69.751 seconds faster. Effective archive transfer speeds were 186.65 MB/s for tar and 183.21 MB/s for tar.gz.

Limitations: single runs at different times, uncontrolled system load and caches. Both archives had recently been created on the shared disk, and local extraction followed immediately after copying. No explicit sync was included. These are not cold-cache or cross-machine benchmarks. All extraction destinations were newly created and separate from previous test directories.
