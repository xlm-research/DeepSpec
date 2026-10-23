"""Loop auxiliary-output and checkpoint compatibility contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [7, 8192])
@pytest.mark.parametrize("fusion", ["none", "all"])
def test_loop_router_fusion_preserves_branch_post_and_mix(monkeypatch, tokens, fusion):
    from vllm.model_executor.models.qwen3_5_loop import Qwen3_5MHCRouter

    monkeypatch.setenv("VLLM_QWEN3_5_MHC_BACKEND", "torch")
    monkeypatch.setenv("VLLM_QWEN3_5_MHC_FUSION", "none")
    reference = Qwen3_5MHCRouter(128, 4, 20).cuda()
    generator = torch.Generator(device="cuda").manual_seed(29)
    with torch.no_grad():
        reference.weight.normal_(std=0.03, generator=generator)
        reference.bias.normal_(std=0.03, generator=generator)
    monkeypatch.setenv("VLLM_QWEN3_5_MHC_BACKEND", "triton")
    monkeypatch.setenv("VLLM_QWEN3_5_MHC_FUSION", fusion)
    candidate = Qwen3_5MHCRouter(128, 4, 20).cuda()
    candidate.load_state_dict(reference.state_dict())
    residual = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    saved = residual.clone()
    with torch.inference_mode():
        expected = reference(residual)
        repeated = reference(residual)
        actual = candidate(residual)
    for before, again, after in zip(expected, repeated, actual, strict=True):
        torch.testing.assert_close(before, again, rtol=0, atol=0)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.equal(residual, saved)


def test_loop_forward_returns_boundary_outputs_and_actual_lm_head_input():
    from vllm.model_executor.models.qwen3_5_loop import Qwen3_5LoopModel

    class Layer(torch.nn.Module):
        def forward(self, positions, hidden):
            return hidden + 1

    class Norm(torch.nn.Module):
        def forward(self, hidden):
            return hidden * 2

    model = Qwen3_5LoopModel.__new__(Qwen3_5LoopModel)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList([Layer() for _ in range(80)])
    model.capture_boundary_slots = {4, 28, 52, 76, 79}
    model.norm = Norm()
    model._set_aux_hidden_state_layers((4, 28, 52, 76, 79))
    inputs = torch.zeros(3, 4)
    final, auxiliary = model(None, torch.arange(3), inputs_embeds=inputs)
    for value, expected in zip(auxiliary, [4, 28, 52, 76, 79], strict=True):
        torch.testing.assert_close(value, torch.full_like(inputs, expected))
    torch.testing.assert_close(final, torch.full_like(inputs, 160))
    model._set_aux_hidden_state_layers(())
    torch.testing.assert_close(model(None, torch.arange(3), inputs_embeds=inputs), final)
    for invalid in [(4, 80), (4, 5), (28, 4), (4, 4)]:
        with pytest.raises(ValueError):
            model._set_aux_hidden_state_layers(invalid)


@pytest.mark.parametrize("layers", [[3, 27, 51, 75, 78], [-1], [80], [3, 3], [79]])
def test_loop_backend_resolves_execution_layers_without_filtering(monkeypatch, layers):
    import transformers

    from torchspec.inference.engine.vllm_engine import VllmEngine

    cfg = SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        architectures=["Qwen3_5LoopForConditionalGeneration"],
        loop_settings={"start": 4, "end": 28, "eval_count": 3},
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)
    monkeypatch.setattr(VllmEngine, "_init_engine", lambda *a, **kw: None)
    monkeypatch.setattr(VllmEngine, "_get_hidden_size_from_engine", lambda self: 4096)
    engine = VllmEngine.__new__(VllmEngine)
    engine.args = SimpleNamespace(
        target_model_path="target",
        aux_hidden_states_layers=layers,
        vllm_direct_hidden_state_capture=True,
    )
    engine.rank = engine.node_rank = 0
    engine.base_gpu_id = None
    engine.num_gpus_per_engine = 4
    if layers == [3, 27, 51, 75, 78]:
        engine.init()
        assert engine._num_target_layers == 80
        assert engine.aux_hidden_state_layer_ids == [4, 28, 52, 76, 79]
    else:
        with pytest.raises(ValueError):
            engine.init()


@pytest.mark.parametrize("different_capture", [False, True])
def test_resume_checks_json_roundtripped_contract_before_loading_weights(
    tmp_path, monkeypatch, different_capture
):
    import json

    from torchspec.training import checkpoint

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text('{"num_hidden_layers":32}')
    draft = tmp_path / "draft.json"
    draft.write_text('{"hidden_size":4096}')
    args = SimpleNamespace(
        vllm_direct_hidden_state_capture=True,
        target_model_path=str(target),
        draft_model_config=str(draft),
        draft_model_config_obj=SimpleNamespace(
            to_dict=lambda: {"hidden_size": 4096, "id2label": {0: "LABEL_0"}}
        ),
        aux_hidden_states_layers=[3, 27, 51, 75, 78],
        last_hidden_states_prenorm=False,
        last_turn_loss_only=True,
        chat_template="qwen",
        train_with_decode=False,
    )
    ckpt_dir = tmp_path / "iter_0000004"
    ckpt_dir.mkdir()
    (ckpt_dir / "meta.json").write_text(
        json.dumps({"training_contract": checkpoint.training_contract(args)})
    )
    if different_capture:
        args.last_hidden_states_prenorm = True
    monkeypatch.setattr(checkpoint, "resolve_resume_model_dir", lambda _: ckpt_dir / "model")
    monkeypatch.setattr(checkpoint, "get_gloo_group", lambda: None)
    load = MagicMock()
    monkeypatch.setattr(checkpoint.dcp, "load", load)
    actor = SimpleNamespace(args=args, model=object())
    if different_capture:
        with pytest.raises(ValueError, match="training contract differs"):
            checkpoint.load(actor)
        load.assert_not_called()
    else:
        assert checkpoint.load(actor)["iteration"] == 4
        load.assert_called_once()


@pytest.mark.parametrize("invalid", [None, "missing_metadata", "missing_rows", "extra_token"])
def test_direct_prefill_preserves_input_and_mask_or_rejects_incomplete_capture(invalid):
    from torchspec.inference.engine.vllm_engine import VllmEngine

    tokens = [1, 2, 3]
    metadata = {
        "input_ids_list": tokens,
        "tensor_shapes": {
            "input_ids": (3,),
            "hidden_states": (3, 20),
            "last_hidden_states": (3, 4),
        },
    }
    if invalid == "missing_metadata":
        metadata = None
    elif invalid == "missing_rows":
        metadata["tensor_shapes"]["hidden_states"] = (2, 20)
    elif invalid == "extra_token":
        metadata["input_ids_list"] = tokens + [4]
    engine = VllmEngine.__new__(VllmEngine)
    engine.args = SimpleNamespace(vllm_direct_hidden_state_capture=True)
    engine.rank = 0
    engine._hidden_size = 4
    engine.aux_hidden_state_layer_ids = [4, 28, 52, 76, 79]
    engine._engine = MagicMock()
    engine._engine.generate.return_value = [
        SimpleNamespace(prompt_token_ids=tokens, kv_transfer_params=metadata)
    ]

    def generate():
        return engine.generate(
            data_id=["original"],
            input_ids_ref=[torch.tensor(tokens)],
            packed_loss_mask_list=["original-mask"],
        )

    if invalid:
        with pytest.raises(RuntimeError, match="Direct"):
            generate()
    else:
        result = generate()[0]
        assert result["input_ids_list"] == tokens
        assert result["packed_loss_mask"] == "original-mask"
        assert result["completion_tokens"] == 0
