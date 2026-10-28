"""Regression tests for vLLM engine actor startup and port negotiation."""

import sys
import types
from argparse import Namespace
from unittest.mock import MagicMock, patch


def test_vllm_port_probe_uses_engine_init_timeout():
    """A slow Ray worker must not hit the old hard-coded 30s probe timeout."""
    args = Namespace(
        inference_num_gpus=4,
        inference_num_gpus_per_engine=4,
        inference_num_gpus_per_node=8,
        vllm_nnodes=1,
        vllm_init_timeout=45,
        vllm_pp_size=1,
    )
    pg = (MagicMock(name="pg"), [0, 1, 2, 3], [0, 2, 4, 6])

    engine = MagicMock(name="vllm_engine")
    engine.find_free_port.remote.return_value = MagicMock(name="port_ref")
    engine.init.remote.return_value = MagicMock(name="init_ref")

    ray_actor = MagicMock(name="VllmRayActor")
    ray_actor.options.return_value.remote.return_value = engine

    observed_timeouts = []

    def fake_get(_ref, *, timeout=None):
        observed_timeouts.append(timeout)
        if timeout < 45:
            raise TimeoutError("simulated slow Ray worker startup")
        return 10000

    vllm_engine_stub = types.ModuleType("torchspec.inference.engine.vllm_engine")
    vllm_engine_stub.VllmEngine = object
    with (
        patch("torchspec.inference.factory.ray") as mock_ray,
        patch("torchspec.inference.factory.get_torchspec_env_vars", return_value={}),
        patch.dict(sys.modules, {"torchspec.inference.engine.vllm_engine": vllm_engine_stub}),
    ):
        mock_ray.remote.return_value = ray_actor
        mock_ray.get.side_effect = fake_get

        from torchspec.inference.factory import _prepare_vllm_engines

        engines, init_refs = _prepare_vllm_engines(args, pg, mooncake_config=None)

    assert engines == [engine]
    assert init_refs == [engine.init.remote.return_value]
    assert observed_timeouts == [45.0]
    engine.find_free_port.remote.assert_called_once_with(start_port=10000, consecutive=2)
    # A TP group may occupy non-contiguous physical devices on a shared node.
    assert ray_actor.options.return_value.remote.call_args.kwargs["gpu_ids"] == [0, 2, 4, 6]
