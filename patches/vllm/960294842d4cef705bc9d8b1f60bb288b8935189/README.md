# Local vLLM changes

Base commit: `960294842d4cef705bc9d8b1f60bb288b8935189`.

The `vllm` submodule uses the existing fork at
`git@gitlab.sh.sensetime.com:xlm/vllm.git`; access to that repository is required.
The submodule now points to the `dev/vllm_spec` branch's committed changes.
After cloning TorchSpec, initialize it from the repository root:

```bash
git submodule update --init vllm
```

The patch is an archive of the original nine locally modified tracked files,
including tests, against the base commit above. Do not apply it to the current
submodule checkout: those changes are already committed, along with formatting
and type-check fixes and an accelerator synchronization API update.
