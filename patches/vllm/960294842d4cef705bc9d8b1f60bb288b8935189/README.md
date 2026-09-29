# Local vLLM changes

Base commit: `960294842d4cef705bc9d8b1f60bb288b8935189`.

The `vllm` submodule uses the existing fork at
`git@gitlab.sh.sensetime.com:xlm/vllm.git`; access to that repository is required.
After cloning TorchSpec, restore the local vLLM changes from the repository root:

```bash
git submodule update --init vllm
git -C vllm apply ../patches/vllm/960294842d4cef705bc9d8b1f60bb288b8935189/local-changes.patch
```

The patch contains all nine locally modified tracked files, including tests.
