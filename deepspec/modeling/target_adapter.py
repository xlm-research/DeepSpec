"""Target-model adapters shared by cache generation, training, and evaluation."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    DynamicCache,
)


class TargetModelAdapter:
    """Default adapter for decoder-only text and image-text target models."""

    name = "generic"

    @classmethod
    def matches(cls, config, model_name_or_path: str | None = None) -> bool:
        del config, model_name_or_path
        return True

    def get_text_config(self, config):
        return getattr(config, "text_config", config)

    def is_multimodal(self, config) -> bool:
        return bool(
            getattr(config, "vision_config", None) is not None
            and getattr(config, "text_config", None) is not None
            and (
                getattr(config, "image_token_id", None) is not None
                or getattr(config, "video_token_id", None) is not None
            )
        )

    def get_language_backbone(self, target_model):
        candidates = [target_model]
        if hasattr(target_model, "model"):
            candidates.append(target_model.model)
        for candidate in candidates:
            language_model = getattr(candidate, "language_model", None)
            if language_model is not None:
                return language_model
        for candidate in candidates:
            if hasattr(candidate, "layers"):
                return candidate
            nested = getattr(candidate, "model", None)
            if nested is not None and hasattr(nested, "layers"):
                return nested
        raise AttributeError(
            f"Cannot locate decoder backbone on {type(target_model).__name__}."
        )

    def get_decoder_layers(self, target_model):
        backbone = self.get_language_backbone(target_model)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise AttributeError(
                f"Decoder backbone {type(backbone).__name__} does not expose `layers`."
            )
        return layers

    def get_final_norm(self, target_model):
        backbone = self.get_language_backbone(target_model)
        norm = getattr(backbone, "norm", None)
        if norm is None:
            raise AttributeError(
                f"Decoder backbone {type(backbone).__name__} does not expose `norm`."
            )
        return norm

    def get_hidden_size(self, target_model_or_config) -> int:
        config = getattr(target_model_or_config, "config", target_model_or_config)
        return int(self.get_text_config(config).hidden_size)

    def get_embeddings(self, target_model):
        embed_tokens = target_model.get_input_embeddings()
        if embed_tokens is None:
            embed_tokens = getattr(
                self.get_language_backbone(target_model), "embed_tokens", None
            )
        lm_head = target_model.get_output_embeddings()
        if lm_head is None:
            lm_head = getattr(target_model, "lm_head", None)
        if embed_tokens is None or lm_head is None:
            raise AttributeError(
                "Cannot locate input embeddings and lm_head on "
                f"{type(target_model).__name__}."
            )
        return embed_tokens, lm_head

    def load_cache_model(
        self,
        model_name_or_path,
        *,
        dtype,
        attn_implementation="sdpa",
    ):
        return AutoModel.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    def load_model_with_head(
        self,
        model_name_or_path,
        *,
        dtype,
        attn_implementation: str | None = None,
    ):
        kwargs = {"dtype": dtype}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    def load_processor(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return None, tokenizer

    def create_generation_cache(self, target_model):
        try:
            return DynamicCache(config=target_model.config)
        except (TypeError, ValueError):
            return DynamicCache()

    def build_prefill_inputs(
        self,
        *,
        target_model,
        model_inputs: dict[str, torch.Tensor],
        cache,
    ) -> dict[str, Any]:
        input_ids = model_inputs["input_ids"]
        position_ids = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        return {
            **model_inputs,
            "position_ids": position_ids,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": True,
            "logits_to_keep": 1,
        }

    def build_verify_inputs(
        self,
        *,
        verify_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        cache,
    ) -> dict[str, Any]:
        verify_length = verify_input_ids.shape[1]
        return {
            "input_ids": verify_input_ids,
            "position_ids": position_ids[:, start : start + verify_length],
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": True,
        }

    def reconcile_generation_cache(
        self,
        *,
        target_model,
        cache,
        model_inputs: dict[str, torch.Tensor],
        output_ids: torch.Tensor,
        committed_length: int,
        accepted_draft_tokens: int,
        draft_token_count: int,
    ):
        del target_model, model_inputs, output_ids, accepted_draft_tokens
        del draft_token_count
        cache.crop(int(committed_length))
        return cache


class Qwen3_6TargetAdapter(TargetModelAdapter):
    """Adapter for Qwen3.6/Qwen3.8 checkpoints' Qwen3.5 hybrid architecture."""

    name = "qwen3_6"

    _CACHE_SNAPSHOT_ATTR = "_deepspec_qwen36_verification_snapshot"

    @classmethod
    def matches(cls, config, model_name_or_path: str | None = None) -> bool:
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = [
            str(item).lower() for item in (getattr(config, "architectures", None) or [])
        ]
        normalized_path = str(model_name_or_path or "").lower()
        return bool(
            "qwen3.6" in normalized_path
            or "qwen3_6" in normalized_path
            or model_type in {"qwen3_5", "qwen3_5_text"}
            or any(item.startswith("qwen3_5") for item in architectures)
        )

    def build_prefill_inputs(
        self,
        *,
        target_model,
        model_inputs: dict[str, torch.Tensor],
        cache,
    ) -> dict[str, Any]:
        # ``rope_deltas`` lives on the multimodal backbone and is mutated by a
        # multimodal prefill.  A later text-only prefill does not overwrite it,
        # so incremental decoding would otherwise reuse the previous sample's
        # MRoPE offset.
        candidate = target_model
        for _ in range(3):
            if hasattr(candidate, "rope_deltas"):
                candidate.rope_deltas = None
            nested = getattr(candidate, "model", None)
            if nested is None or nested is candidate:
                break
            candidate = nested
        # Qwen3.6 computes multimodal MRoPE positions from token types and
        # image/video grids. A flat text position tensor would be incorrect.
        return {
            **model_inputs,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": True,
            "logits_to_keep": 1,
        }

    def load_model_with_head(
        self,
        model_name_or_path,
        *,
        dtype,
        attn_implementation: str | None = None,
    ):
        kwargs = {"dtype": dtype}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    def load_processor(self, model_name_or_path):
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return processor, processor.tokenizer

    def build_verify_inputs(
        self,
        *,
        verify_input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        cache,
    ) -> dict[str, Any]:
        del position_ids, start
        self._snapshot_linear_cache(cache)
        # The model reuses rope_deltas computed during multimodal prefill.
        return {
            "input_ids": verify_input_ids,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": True,
        }

    def reconcile_generation_cache(
        self,
        *,
        target_model,
        cache,
        model_inputs: dict[str, torch.Tensor],
        output_ids: torch.Tensor,
        committed_length: int,
        accepted_draft_tokens: int,
        draft_token_count: int,
    ):
        if int(accepted_draft_tokens) == int(draft_token_count):
            self._discard_linear_cache_snapshot(cache)
            return cache

        replay_start = int(committed_length) - int(accepted_draft_tokens) - 1
        if self._restore_verified_prefix(cache, replay_start):
            # Verification advanced every recurrent layer through the complete
            # proposal.  Restore the state before verification, crop only the
            # ordinary attention KV layers, then replay the committed current
            # token and accepted draft prefix.  This is O(block_size), unlike
            # rebuilding the complete prompt and generated prefix.
            replay_input_ids = output_ids[:, replay_start:committed_length]
            target_model(
                input_ids=replay_input_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
                logits_to_keep=1,
            )
            return cache

        # Qwen3.6 interleaves full-attention KV layers with recurrent
        # DeltaNet layers. DynamicCache.crop cannot roll recurrent states back
        # after a rejected proposal.  If the installed cache implementation is
        # not snapshot-compatible, preserve correctness by rebuilding from the
        # committed tokens.
        rebuilt_inputs = {}
        original_length = int(model_inputs["input_ids"].shape[1])
        for key, value in model_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "input_ids":
                rebuilt_inputs[key] = output_ids[:, :committed_length]
            elif key == "attention_mask":
                rebuilt_inputs[key] = torch.ones(
                    (value.shape[0], committed_length),
                    dtype=value.dtype,
                    device=value.device,
                )
            elif key in ("mm_token_type_ids", "token_type_ids"):
                if value.shape[1] != original_length:
                    rebuilt_inputs[key] = value
                    continue
                generated_length = committed_length - original_length
                if generated_length <= 0:
                    rebuilt_inputs[key] = value[:, :committed_length]
                else:
                    generated_types = torch.zeros(
                        (value.shape[0], generated_length),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    rebuilt_inputs[key] = torch.cat([value, generated_types], dim=1)
            else:
                rebuilt_inputs[key] = value

        rebuilt_cache = self.create_generation_cache(target_model)
        rebuild_kwargs = self.build_prefill_inputs(
            target_model=target_model,
            model_inputs=rebuilt_inputs,
            cache=rebuilt_cache,
        )
        rebuild_kwargs["output_hidden_states"] = False
        target_model(**rebuild_kwargs)
        return rebuilt_cache

    @classmethod
    def _discard_linear_cache_snapshot(cls, cache) -> None:
        if hasattr(cache, cls._CACHE_SNAPSHOT_ATTR):
            delattr(cache, cls._CACHE_SNAPSHOT_ATTR)

    @classmethod
    def _snapshot_linear_cache(cls, cache) -> None:
        layers = getattr(cache, "layers", None)
        if layers is None or not hasattr(cache, "get_seq_length"):
            cls._discard_linear_cache_snapshot(cache)
            return

        linear_layers = {}
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, "recurrent_states"):
                continue
            linear_layers[layer_idx] = {
                "conv_states": {
                    state_idx: (
                        state.detach().clone()
                        if isinstance(state, torch.Tensor)
                        else None
                    )
                    for state_idx, state in layer.conv_states.items()
                },
                "recurrent_states": {
                    state_idx: (
                        state.detach().clone()
                        if isinstance(state, torch.Tensor)
                        else None
                    )
                    for state_idx, state in layer.recurrent_states.items()
                },
                "has_previous_state": dict(layer.has_previous_state),
                "is_conv_states_initialized": dict(
                    layer.is_conv_states_initialized
                ),
                "is_recurrent_states_initialized": dict(
                    layer.is_recurrent_states_initialized
                ),
            }

        setattr(
            cache,
            cls._CACHE_SNAPSHOT_ATTR,
            {
                "prefix_length": int(cache.get_seq_length()),
                "linear_layers": linear_layers,
            },
        )

    @classmethod
    def _restore_verified_prefix(cls, cache, prefix_length: int) -> bool:
        snapshot = getattr(cache, cls._CACHE_SNAPSHOT_ATTR, None)
        cls._discard_linear_cache_snapshot(cache)
        if snapshot is None or int(snapshot["prefix_length"]) != int(prefix_length):
            return False

        layers = getattr(cache, "layers", None)
        linear_layers = snapshot["linear_layers"]
        if layers is None or not linear_layers:
            return False

        # Validate the complete operation before mutating any cache layer.  An
        # unfamiliar future Transformers cache safely falls back to rebuilding.
        for layer_idx, layer in enumerate(layers):
            if layer_idx in linear_layers:
                saved = linear_layers[layer_idx]
                if not all(hasattr(layer, field) for field in saved):
                    return False
                continue
            if not hasattr(layer, "get_seq_length") or not hasattr(layer, "crop"):
                return False
            if int(layer.get_seq_length()) < int(prefix_length):
                return False

        for layer_idx, layer in enumerate(layers):
            if layer_idx in linear_layers:
                saved = linear_layers[layer_idx]
                for field in ("conv_states", "recurrent_states"):
                    current_states = getattr(layer, field)
                    for state_idx, saved_state in saved[field].items():
                        current_state = current_states.get(state_idx)
                        if (
                            isinstance(current_state, torch.Tensor)
                            and isinstance(saved_state, torch.Tensor)
                            and current_state.shape == saved_state.shape
                        ):
                            current_state.copy_(saved_state)
                        else:
                            current_states[state_idx] = saved_state
                layer.has_previous_state = dict(saved["has_previous_state"])
                layer.is_conv_states_initialized = dict(
                    saved["is_conv_states_initialized"]
                )
                layer.is_recurrent_states_initialized = dict(
                    saved["is_recurrent_states_initialized"]
                )
                continue

            tokens_to_remove = int(layer.get_seq_length()) - int(prefix_length)
            if tokens_to_remove > 0:
                layer.crop(-tokens_to_remove)
        return True


_TARGET_ADAPTERS = [Qwen3_6TargetAdapter, TargetModelAdapter]


def get_target_adapter(
    target_model_or_config,
    model_name_or_path: str | None = None,
) -> TargetModelAdapter:
    config = getattr(target_model_or_config, "config", target_model_or_config)
    if isinstance(config, (str, bytes)):
        model_name_or_path = str(config)
        config = AutoConfig.from_pretrained(config)
    for adapter_cls in _TARGET_ADAPTERS:
        if adapter_cls.matches(config, model_name_or_path):
            return adapter_cls()
    raise AssertionError("The generic target adapter must match every target model.")


def get_text_config(config):
    return get_target_adapter(config).get_text_config(config)


def is_multimodal_config(config) -> bool:
    return get_target_adapter(config).is_multimodal(config)


def get_language_backbone(target_model):
    return get_target_adapter(target_model).get_language_backbone(target_model)


def get_decoder_layers(target_model):
    return get_target_adapter(target_model).get_decoder_layers(target_model)


def get_final_norm(target_model):
    return get_target_adapter(target_model).get_final_norm(target_model)


def get_target_hidden_size(target_model_or_config) -> int:
    return get_target_adapter(target_model_or_config).get_hidden_size(
        target_model_or_config
    )


def get_target_embeddings(target_model):
    return get_target_adapter(target_model).get_embeddings(target_model)


def load_target_cache_model(
    model_name_or_path,
    *,
    dtype,
    attn_implementation="sdpa",
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return get_target_adapter(config, model_name_or_path).load_cache_model(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def load_target_model_with_head(
    model_name_or_path,
    *,
    dtype,
    attn_implementation: str | None = None,
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return get_target_adapter(config, model_name_or_path).load_model_with_head(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


__all__ = [
    "Qwen3_6TargetAdapter",
    "TargetModelAdapter",
    "get_decoder_layers",
    "get_final_norm",
    "get_language_backbone",
    "get_target_adapter",
    "get_target_embeddings",
    "get_target_hidden_size",
    "get_text_config",
    "is_multimodal_config",
    "load_target_cache_model",
    "load_target_model_with_head",
]
