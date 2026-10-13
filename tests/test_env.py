"""Tests for the default environment forwarded to TorchSpec Ray actors."""

import pytest

from torchspec.utils.env import get_torchspec_env_vars


def test_vllm_v2_model_runner_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    env = get_torchspec_env_vars()

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_explicit_vllm_model_runner_value_is_forwarded(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    env = get_torchspec_env_vars()

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


@pytest.mark.parametrize("backend", ["torch", "triton", None])
def test_qwen_loop_mhc_backend_is_forwarded_only_when_selected(monkeypatch, backend):
    if backend is None:
        monkeypatch.delenv("VLLM_QWEN3_5_MHC_BACKEND", raising=False)
    else:
        monkeypatch.setenv("VLLM_QWEN3_5_MHC_BACKEND", backend)

    env = get_torchspec_env_vars()

    if backend is None:
        assert "VLLM_QWEN3_5_MHC_BACKEND" not in env
    else:
        assert env["VLLM_QWEN3_5_MHC_BACKEND"] == backend
