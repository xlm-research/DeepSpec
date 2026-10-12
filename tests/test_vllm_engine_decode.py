from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _args(**overrides):
    values = {
        "train_with_decode": True,
        "target_model_path": "target",
        "decode_speculative_algorithm": "EAGLE3",
        "decode_speculative_draft_model_path": "/tmp/draft",
        "decode_speculative_num_draft_tokens": 5,
        "decode_speculative_num_steps": None,
        "decode_max_new_tokens": 8,
        "decode_min_new_tokens": 2,
        "decode_stop_token_ids": None,
        "decode_temperature": 1.0,
        "decode_top_p": 1.0,
        "decode_top_k": -1,
        "max_seq_length": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine():
    try:
        from torchspec.inference.engine.vllm_engine import VllmEngine
    except ImportError as exc:
        pytest.skip(f"VllmEngine import failed: {exc}")
    engine = VllmEngine.__new__(VllmEngine)
    engine.args = _args()
    engine.rank = 0
    engine._decode_mode = True
    engine._hidden_size = 16
    engine.aux_hidden_state_layer_ids = [4, 32, 60, 64]
    engine._engine = MagicMock()
    return engine


@pytest.fixture
def draft_reload_state(monkeypatch):
    import torch

    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.v1.worker import gpu_model_runner

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DFlashQwen3Model.__new__(DFlashQwen3Model)
            torch.nn.Module.__init__(self.model)
            self.model.hidden_norm = torch.nn.RMSNorm(4)
            attn = torch.nn.Module()
            attn.qkv_proj = torch.nn.Linear(4, 8, bias=False)
            attn.q_size, attn.kv_size = 4, 2
            attn.head_dim, attn.num_kv_heads = 2, 1
            attn.k_norm = torch.nn.RMSNorm(2)
            attn.q_norm = SimpleNamespace(variance_epsilon=1e-6)
            attn.rotary_emb = torch.nn.Module()
            attn.rotary_emb.register_buffer("cos_sin_cache", torch.ones(4, 2))
            attn.rotary_emb.head_size = 2
            attn.rotary_emb.is_neox_style = True
            attn.attn = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = attn
            self.model.layers = torch.nn.ModuleList([layer])
            self.model._build_fused_kv_buffers()

        def load_weights(self, weights):
            loaded = set()
            for name, value in weights:
                param = self.get_parameter(name)
                getattr(param, "weight_loader", default_weight_loader)(param, value)
                loaded.add(name)
            self.model._build_fused_kv_buffers()
            return loaded

    draft = Draft()
    record_metadata_for_reloading(draft)
    weights = [(name, torch.full_like(param, 2)) for name, param in draft.named_parameters()]
    config = SimpleNamespace(model="old", revision=None)
    runner = SimpleNamespace(
        get_draft_model=lambda: draft,
        speculative_config=SimpleNamespace(draft_model_config=config, draft_load_config=None),
        load_config=SimpleNamespace(load_format="auto"),
    )
    loader = SimpleNamespace(get_all_weights=lambda *_: iter(weights))
    monkeypatch.setattr(gpu_model_runner, "get_model_loader", lambda _: loader)
    return runner, draft, loader


def test_draft_reload_refreshes_fused_rope_cache(draft_reload_state, tmp_path):
    """DFlash/DSpark must not retain Meta buffers after layerwise reload."""
    import torch

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner, draft, _ = draft_reload_state
    result = GPUModelRunner.reload_draft_weights(runner, str(tmp_path))

    assert result["success"]
    assert draft.model._rope_cos_sin_cache.device.type == "cpu"
    torch.testing.assert_close(draft.model._rope_cos_sin_cache, torch.ones(4, 2))
    torch.testing.assert_close(draft.model._fused_kv_weight, torch.full((4, 4), 2.0))
    assert runner.speculative_config.draft_model_config.model == "old"


def test_draft_reload_rejects_directory_missing_on_worker(draft_reload_state, tmp_path):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner, draft, _ = draft_reload_state
    original_params = dict(draft.named_parameters())
    result = GPUModelRunner.reload_draft_weights(runner, str(tmp_path / "missing"))

    assert not result["success"]
    assert "directory" in result["message"]
    assert all(draft.get_parameter(name) is param for name, param in original_params.items())
    assert runner.speculative_config.draft_model_config.model == "old"


def test_draft_reload_read_failure_preserves_live_model(draft_reload_state, tmp_path):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner, draft, loader = draft_reload_state
    original_params = dict(draft.named_parameters())
    get_weights = loader.get_all_weights

    def truncated_checkpoint(*args):
        yield next(get_weights(*args))
        raise OSError("checkpoint truncated")

    loader.get_all_weights = truncated_checkpoint
    with pytest.raises(OSError, match="checkpoint truncated"):
        GPUModelRunner.reload_draft_weights(runner, str(tmp_path))

    assert all(draft.get_parameter(name) is param for name, param in original_params.items())
    assert runner.speculative_config.draft_model_config.model == "old"


def test_draft_reload_load_failure_restores_devices_and_caches(draft_reload_state, tmp_path):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner, draft, _ = draft_reload_state
    load_weights = draft.load_weights

    def partial_load(weights):
        load_weights(iter([next(weights)]))
        raise ValueError("invalid checkpoint tensor")

    draft.load_weights = partial_load
    with pytest.raises(ValueError, match="invalid checkpoint tensor"):
        GPUModelRunner.reload_draft_weights(runner, str(tmp_path))

    assert all(param.device.type == "cpu" for param in draft.parameters())
    assert all(buffer.device.type == "cpu" for buffer in draft.buffers())
    assert draft.model._rope_cos_sin_cache.device.type == "cpu"
    assert draft.model._fused_kv_weight.device.type == "cpu"
    assert runner.speculative_config.draft_model_config.model == "old"


def test_native_decode_config_uses_real_eagle3_method():
    engine = _engine()

    config = engine._build_speculative_config()

    assert config["method"] == "eagle3"
    assert config["model"] == "/tmp/draft"
    assert config["num_speculative_tokens"] == 5
    assert config["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [
        4,
        32,
        60,
        64,
    ]


def test_direct_prefill_initializes_without_a_draft_and_keeps_final_separate(monkeypatch):
    import sys
    from types import ModuleType

    engine = _engine()
    engine._decode_mode = False
    engine.args = _args(vllm_direct_hidden_state_capture=True, train_with_decode=False)
    engine._num_target_layers = 80
    engine.aux_hidden_state_layer_ids = [4, 28, 52, 76, 79]
    engine.base_gpu_id = None
    engine.node_rank = 0
    vllm_stub = ModuleType("vllm")
    vllm_stub.LLM = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    engine._init_engine(4, 1, 1, 0.8, None)

    kwargs = vllm_stub.LLM.call_args.kwargs
    assert kwargs["speculative_config"] is None
    assert kwargs["kv_transfer_config"]["kv_connector_extra_config"] == {
        "target_hidden_state_layer_ids": [4, 28, 52, 76, 79],
        "target_num_hidden_layers": 80,
        "final_hidden_states_prenorm": False,
    }


def test_native_decode_config_uses_dspark_layers_without_final_capture_slot():
    engine = _engine()
    engine.args = _args(
        decode_speculative_algorithm="dspark",
        decode_speculative_num_draft_tokens=7,
    )

    config = engine._build_speculative_config()

    assert config["method"] == "dspark"
    assert config["num_speculative_tokens"] == 7
    assert config["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [
        4,
        32,
        60,
    ]


@pytest.mark.parametrize(
    ("target_only", "algorithm"),
    [(True, None), (False, None), (False, "EAGLE3"), (False, "dspark")],
)
def test_only_explicit_target_only_mode_initializes_llm_without_a_draft(
    monkeypatch, target_only, algorithm
):
    """Old decode configs must keep their proposer unless target-only is explicit."""
    import sys
    from types import ModuleType

    engine = _engine()
    engine.args = _args(
        decode_target_only=target_only,
        decode_speculative_algorithm=algorithm,
        decode_speculative_draft_model_path=None if target_only else "/tmp/draft",
        max_seq_length=131072,
    )
    engine.base_gpu_id = None
    engine.node_rank = 0
    llm = MagicMock()
    vllm_stub = ModuleType("vllm")
    vllm_stub.LLM = llm
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    engine._init_engine(8, 1, 1, 0.8, None)

    kwargs = llm.call_args.kwargs
    assert kwargs["model"] == "target"
    if target_only:
        assert kwargs["speculative_config"] is None
        assert kwargs["kv_transfer_config"]["kv_connector_extra_config"] == {
            "target_hidden_state_layer_ids": [4, 32, 60, 64]
        }
    else:
        assert kwargs["speculative_config"]["method"] == (algorithm or "eagle3").lower()
        assert kwargs["speculative_config"]["model"] == "/tmp/draft"
        assert "kv_connector_extra_config" not in kwargs["kv_transfer_config"]
    assert kwargs["max_model_len"] == 131072


def test_prefill_initialization_admits_qwen_128k_cache(monkeypatch):
    """The engine must avoid padding hidden-state pages across target layers."""
    import sys
    from types import ModuleType

    import torch

    from vllm.config import KVTransferConfig
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.kv_cache_interface import FullAttentionSpec, HiddenStateCacheSpec, MambaSpec

    engine = _engine()
    engine.args = _args(train_with_decode=False, max_seq_length=131072)
    engine._decode_mode = False
    engine.base_gpu_id = None
    engine.node_rank = 0
    llm = MagicMock()
    vllm_stub = ModuleType("vllm")
    vllm_stub.LLM = llm
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    engine._init_engine(4, 1, 1, 0.8, None)

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        model_config=SimpleNamespace(max_model_len=llm.call_args.kwargs["max_model_len"]),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, mamba_cache_mode="none"),
        kv_transfer_config=KVTransferConfig(**llm.call_args.kwargs["kv_transfer_config"]),
    )
    attention = FullAttentionSpec(
        block_size=800, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
    )
    mamba = MambaSpec(block_size=16, shapes=((409600,),), dtypes=(torch.bfloat16,))
    hidden = HiddenStateCacheSpec(
        block_size=800, num_kv_heads=6, head_size=5120, dtype=torch.bfloat16
    )
    specs = {f"attn.{i}": attention for i in range(16)}
    specs.update({f"mamba.{i}": mamba for i in range(48)})
    specs["cache_only_layers.64"] = hidden
    groups = kv_cache_utils.get_kv_cache_groups(config, specs)

    required = kv_cache_utils._max_memory_usage_bytes_from_groups(config, groups)
    assert required < 48 * 2**30
    assert groups[-1].kv_cache_spec == hidden


@pytest.mark.parametrize("method", [None, "eagle3", "dspark"])
def test_runner_keeps_draft_loading_and_rejection_sampling_for_speculation(monkeypatch, method):
    """Target-only capture must leave native proposers and their samplers intact."""
    from contextlib import nullcontext

    import torch

    from vllm.model_executor.models.interfaces import SupportsEagle3
    from vllm.v1.worker.gpu import model_runner as mrv2

    class Target(SupportsEagle3):
        pass

    model = Target()
    model.set_aux_hidden_state_layers = MagicMock()
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(logprobs_mode="raw_logprobs", use_fp64_gumbel=False)
    runner.vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(), model_config=runner.model_config, reasoning_config=None
    )
    runner.eplb = MagicMock()
    runner.eplb.maybe_register_model.return_value = False
    runner.eplb.maybe_register_speculator.return_value = False
    runner.lora_config = None
    runner.use_aux_hidden_state_outputs = True
    runner.target_hidden_state_layer_ids = (1, 4) if method is None else ()
    runner.speculative_config = None
    runner.speculator = None
    runner.num_speculative_steps = 0
    layer_ids = [1] if method == "dspark" else [1, 4]
    if method is not None:
        runner.speculative_config = SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=layer_ids)
            ),
        )
        runner.speculator = MagicMock(spec=mrv2.DraftModelSpeculator)
        runner.speculator.model = SimpleNamespace(num_aux_layers=1)
        runner.num_speculative_steps = 5
    runner.encoder_cache = None
    runner.is_pooling_model = False
    runner.is_last_pp_rank = runner.is_first_pp_rank = True
    runner.max_num_reqs = 4
    runner.vocab_size = 16
    runner.req_states = SimpleNamespace()
    monkeypatch.setattr(
        mrv2, "DeviceMemoryProfiler", lambda: nullcontext(SimpleNamespace(consumed_memory=0))
    )
    loader = SimpleNamespace(load_model=MagicMock(return_value=model))
    monkeypatch.setattr(mrv2, "get_model_loader", lambda _: loader)
    monkeypatch.setattr(
        mrv2,
        "init_model_state",
        lambda *_: SimpleNamespace(
            num_new_sampled_tokens_per_step=1, custom_sampler=lambda _: None
        ),
    )
    for name in ("Sampler", "RejectionSampler", "PromptLogprobsWorker", "StructuredOutputsWorker"):
        monkeypatch.setattr(mrv2, name, MagicMock())
    monkeypatch.setattr(mrv2, "get_offloader", MagicMock())

    runner.load_model()

    model.set_aux_hidden_state_layers.assert_called_once_with(tuple(layer_ids))
    mrv2.Sampler.assert_called_once()
    if method is None:
        assert runner.get_draft_model() is None
        assert runner.decode_query_len == 1
        mrv2.RejectionSampler.assert_not_called()
    else:
        runner.speculator.load_model.assert_called_once_with(model)
        assert runner.get_draft_model() is runner.speculator.model
        assert runner.decode_query_len == 6
        mrv2.RejectionSampler.assert_called_once_with(
            runner.sampler, runner.speculative_config, runner.device
        )
        aux_states = [torch.zeros(2, 2) for _ in layer_ids]
        assert len(runner._aux_hidden_states_for_drafter(aux_states)) == 1


@pytest.mark.parametrize("algorithm", [None, "EAGLE3", "dspark"])
def test_decode_returns_completion_only_loss_mask(algorithm):
    engine = _engine()
    engine.args.decode_speculative_algorithm = algorithm
    engine.args.decode_target_only = algorithm is None
    output = SimpleNamespace(
        prompt_token_ids=[10, 11, 12],
        outputs=[SimpleNamespace(token_ids=[20, 21, 22])],
        kv_transfer_params={
            "mooncake_key": "req-0",
            "tensor_shapes": {
                "input_ids": (5,),
                "hidden_states": (5, 48),
                "last_hidden_states": (5, 16),
            },
            "tensor_dtypes": {
                "input_ids": "int64",
                "hidden_states": "bfloat16",
                "last_hidden_states": "bfloat16",
            },
        },
    )
    engine._engine.generate.return_value = [output]

    results = engine.generate_with_decode(data_id=["row-0"], formatted_prompts=["prompt"])

    assert results[0]["mooncake_key"] == "req-0"
    assert results[0]["packed_loss_mask"] == "3,2"
    assert results[0]["prompt_tokens"] == 3
    assert results[0]["completion_tokens"] == 3
    params = engine._engine.generate.call_args.args[1]
    assert params.max_tokens == 8
    assert params.min_tokens == 2


def test_native_decode_drops_one_token_completion():
    engine = _engine()
    engine._engine.generate.return_value = [
        SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=[SimpleNamespace(token_ids=[3])],
            kv_transfer_params={"mooncake_key": "req-0"},
        )
    ]

    assert engine.generate_with_decode(data_id=["row-0"], formatted_prompts=["prompt"]) == [None]
