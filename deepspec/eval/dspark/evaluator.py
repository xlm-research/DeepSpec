from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    VerificationResult,
    assert_no_final_target_layer,
    generate_decoding_sample,
)
from deepspec.eval.dspark.confidence_head import ConfidenceHeadRecorder
from deepspec.eval.dspark.draft_ops import (
    DSparkDraftProposal,
    build_dspark_proposal,
    forward_dspark_draft_block,
)
from deepspec.eval.dspark.scheduler import (
    HardwareAwarePrefixScheduler,
    SPSProfile,
    SequentialTemperatureScaler,
)
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3_6 import Qwen3_6DSparkModel
from deepspec.modeling.dspark.qwen3_8 import Qwen3_8DSparkModel
from deepspec.modeling.dflash2 import Qwen3_8DFlash2Model
from deepspec.modeling.target_adapter import (
    get_target_adapter,
    get_target_embeddings,
)
from deepspec.utils import jsonable


CONFIDENCE_NUM_BINS = 20
CONFIDENCE_NUM_FINE_BINS = 1000


def _read_json_object(path: str, *, artifact_name: str) -> dict:
    artifact_path = Path(path).expanduser().resolve()
    try:
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read {artifact_name} JSON {artifact_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_name} JSON must contain an object.")
    return payload


def _load_confidence_scaler(
    path: str,
    *,
    block_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    target_model: str,
    draft_model: str,
) -> SequentialTemperatureScaler:
    payload = _read_json_object(path, artifact_name="confidence calibration")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Confidence calibration schema_version must be 1.")
    if payload.get("method") != "sequential_temperature_scaling":
        raise ValueError(
            "Confidence calibration method must be "
            "'sequential_temperature_scaling'."
        )
    expected_models = {
        "target_model": str(target_model).rstrip("/"),
        "draft_model": str(draft_model).rstrip("/"),
    }
    for name, expected in expected_models.items():
        actual = payload.get(name)
        if not isinstance(actual, str) or actual.rstrip("/") != expected:
            raise ValueError(
                f"Confidence calibration {name}={actual!r} does not match "
                f"evaluation {expected!r}."
            )
    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("Confidence calibration must record its sampling policy.")
    expected_sampling = {
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    for name, expected in expected_sampling.items():
        if name not in sampling:
            raise ValueError(f"Confidence calibration sampling is missing {name}.")
        actual = sampling[name]
        try:
            if name == "top_k":
                actual_float = float(actual)
                matches = actual_float.is_integer() and int(actual_float) == expected
            else:
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                "Confidence calibration sampling mismatch: "
                f"{name}={actual!r}, evaluation uses {expected!r}."
            )
    raw_temperatures = payload.get("temperatures")
    if not isinstance(raw_temperatures, list) or not raw_temperatures:
        raise ValueError(
            "Confidence calibration temperatures must be a non-empty list."
        )
    try:
        scaler = SequentialTemperatureScaler(
            temperatures=torch.tensor(raw_temperatures, dtype=torch.float64),
            num_bins=int(payload.get("num_bins", 15)),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid confidence calibration: {exc}") from exc
    if scaler.block_size != int(block_size):
        raise ValueError(
            "Confidence calibration block size mismatch: "
            f"{scaler.block_size} != {int(block_size)}."
        )
    return scaler


def _load_sps_profile(
    path: str,
    *,
    minimum_batch_size: int,
    maximum_batch_size: int,
) -> SPSProfile:
    payload = _read_json_object(path, artifact_name="SPS profile")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("SPS profile schema_version must be 1.")
    raw_profile = payload.get("steps_per_second")
    if not isinstance(raw_profile, dict) or not raw_profile:
        raise ValueError(
            "SPS profile must contain a non-empty steps_per_second object."
        )
    profile_values: dict[int, float] = {}
    for raw_batch_size, raw_rate in raw_profile.items():
        try:
            batch_size = int(raw_batch_size)
            rate = float(raw_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SPS profile keys and values must be numeric."
            ) from exc
        if batch_size in profile_values:
            raise ValueError(f"Duplicate SPS batch size {batch_size}.")
        profile_values[batch_size] = rate
    profile = SPSProfile.from_mapping(profile_values)
    profile.require_range(int(minimum_batch_size), int(maximum_batch_size))
    return profile


class Qwen3DSparkEvaluator(BaseEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    draft_model_cls = Qwen3DSparkModel

    def __init__(self, local_rank: int, args):
        self.confidence_scaler: SequentialTemperatureScaler | None = None
        self.prefix_scheduler: HardwareAwarePrefixScheduler | None = None
        super().__init__(local_rank, args)
        self._configure_confidence_scheduling()
        self.confidence_head_recorder = self._build_confidence_head_recorder()

    def _configure_confidence_scheduling(self) -> None:
        self.confidence_scaler = None
        self.prefix_scheduler = None
        mode = str(getattr(self.args, "scheduler_mode", "static"))
        threshold = float(self.args.confidence_threshold)
        temperature = float(self.args.temperature)
        top_k = int(getattr(self.args, "top_k", 0))
        top_p = float(getattr(self.args, "top_p", 1.0))
        calibration_path = getattr(
            self.args,
            "confidence_calibration_json",
            None,
        )
        sps_path = getattr(self.args, "sps_profile_json", None)
        if mode not in {"static", "hardware-aware"}:
            raise ValueError(f"Unsupported scheduler_mode {mode!r}.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        scheduling_requested = threshold > 0.0 or mode == "hardware-aware"
        if scheduling_requested and self.draft_model.confidence_head is None:
            raise ValueError(
                "Confidence scheduling requires a draft confidence head."
            )
        if calibration_path is not None:
            if self.draft_model.confidence_head is None:
                raise ValueError(
                    "A confidence calibration cannot be used without a "
                    "draft confidence head."
                )
            self.confidence_scaler = _load_confidence_scaler(
                calibration_path,
                block_size=self.max_proposal_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                target_model=self.args.target_name_or_path,
                draft_model=self.args.draft_name_or_path,
            )

        non_training_sampling_policy = (
            not math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=1e-12)
            or top_k > 0
            or not math.isclose(top_p, 1.0, rel_tol=0.0, abs_tol=1e-12)
        )
        if (
            threshold > 0.0
            and non_training_sampling_policy
            and self.confidence_scaler is None
        ):
            raise ValueError(
                "Static confidence scheduling outside the training sampling "
                "policy (temperature=1, full vocabulary) requires a matching "
                "Sequential Temperature Scaling artifact."
            )

        if mode == "hardware-aware":
            if threshold != 0.0:
                raise ValueError(
                    "confidence_threshold must be 0 with hardware-aware scheduling."
                )
            if self.confidence_scaler is None:
                raise ValueError(
                    "Hardware-aware scheduling requires "
                    "--confidence-calibration-json."
                )
            if sps_path is None:
                raise ValueError(
                    "Hardware-aware scheduling requires --sps-profile-json."
                )
            profile = _load_sps_profile(
                sps_path,
                minimum_batch_size=1,
                maximum_batch_size=1 + self.max_proposal_tokens,
            )
            self.prefix_scheduler = HardwareAwarePrefixScheduler(profile)
        elif sps_path is not None:
            raise ValueError(
                "--sps-profile-json is only valid with hardware-aware scheduling."
            )

        observation_path = getattr(
            self.args,
            "confidence_observations_jsonl",
            None,
        )
        if observation_path is not None:
            if self.draft_model.confidence_head is None:
                raise ValueError(
                    "Confidence observations require a draft confidence head."
                )
            if mode != "static" or threshold != 0.0:
                raise ValueError(
                    "Confidence observations require static scheduling with "
                    "confidence_threshold=0."
                )

        self.args.confidence_temperatures = (
            self.confidence_scaler.temperatures.tolist()
            if self.confidence_scaler is not None
            else None
        )
        self.args.sps_profile = (
            {
                str(batch_size): rate
                for batch_size, rate in zip(
                    self.prefix_scheduler.sps_profile.batch_sizes,
                    self.prefix_scheduler.sps_profile.steps_per_second,
                )
            }
            if self.prefix_scheduler is not None
            else None
        )

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.block_size)

    def _build_confidence_head_recorder(self) -> ConfidenceHeadRecorder | None:
        if self.draft_model.confidence_head is None:
            return None
        if float(self.args.confidence_threshold) != 0.0:
            return None
        if str(getattr(self.args, "scheduler_mode", "static")) != "static":
            return None

        artifact_root = None
        if self.args.tensorboard_dir is not None:
            artifact_root = (
                Path(self.args.tensorboard_dir)
                / "artifacts"
                / f"step_{self.args.step}"
            )
        return ConfidenceHeadRecorder(
            device=self.device,
            max_proposal_tokens=self.max_proposal_tokens,
            num_bins=CONFIDENCE_NUM_BINS,
            num_fine_bins=CONFIDENCE_NUM_FINE_BINS,
            draft_name_or_path=self.args.draft_name_or_path,
            tensorboard_dir=self.args.tensorboard_dir,
            step=self.args.step,
            artifact_root=artifact_root,
            observation_path=getattr(
                self.args,
                "confidence_observations_jsonl",
                None,
            ),
            observation_metadata={
                "target_model": self.args.target_name_or_path,
                "draft_model": self.args.draft_name_or_path,
                "sampling": {
                    "temperature": float(self.args.temperature),
                    "top_k": int(getattr(self.args, "top_k", 0)),
                    "top_p": float(getattr(self.args, "top_p", 1.0)),
                },
                "block_size": self.max_proposal_tokens,
            },
            rank=self.global_rank,
            world_size=self.world_size,
        )

    def build_models(self) -> tuple[object, Qwen3DSparkModel, AutoTokenizer]:
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
        assert 0.0 <= float(self.args.confidence_threshold) <= 1.0
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        self.target_adapter = get_target_adapter(
            target_model,
            self.args.target_name_or_path,
        )
        return target_model, draft_model, tokenizer

    def _init_context(
        self,
        *,
        initial_output,
        **kwargs,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            past_key_values_draft=DynamicCache(),
            target_hidden_states=extract_context_feature(
                initial_output.hidden_states,
                self.draft_model.target_layer_ids,
            ),
        )

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        model = self.draft_model
        draft_input_ids = torch.full(
            (output_ids.size(0), self.max_proposal_tokens),
            int(model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]
        block_hidden = forward_dspark_draft_block(
            model,
            draft_input_ids=draft_input_ids,
            position_ids=position_ids,
            past_key_values_draft=context.past_key_values_draft,
            target_hidden_states=context.target_hidden_states,
            start=start,
            block_size=self.max_proposal_tokens,
        )
        return build_dspark_proposal(
            model=model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
            confidence_scaler=self.confidence_scaler,
            prefix_scheduler=self.prefix_scheduler,
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        verified_target_hidden = extract_context_feature(
            verification.target_output.hidden_states,
            self.draft_model.target_layer_ids,
        )
        context.target_hidden_states = verified_target_hidden[
            :,
            : verification.accepted_draft_tokens + 1,
            :,
        ]

    def _post_verify(
        self,
        proposal: DraftProposal,
        verification: VerificationResult,
    ) -> None:
        if self.confidence_head_recorder is None:
            return
        assert isinstance(proposal, DSparkDraftProposal)
        self.confidence_head_recorder.observe(
            proposal=proposal,
            verification=verification,
        )

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
        model_inputs: dict[str, torch.Tensor] | None = None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
            post_verify=self._post_verify,
            model_inputs=model_inputs,
            target_adapter=self.target_adapter,
            top_k=int(getattr(self.args, "top_k", 0)),
            top_p=float(getattr(self.args, "top_p", 1.0)),
        )

    def evaluate(self) -> None:
        try:
            for dataset_name, max_samples in self.tasks:
                if self.confidence_head_recorder is not None:
                    self.confidence_head_recorder.start(dataset_name)
                responses = self.run_dataset(
                    dataset_name=dataset_name,
                    max_samples=max_samples,
                )
                metric_summary = self.allreduce_response_metrics(responses)
                confidence_row = (
                    self.confidence_head_recorder.finish(
                        dataset_name=dataset_name,
                        metric_summary=metric_summary,
                    )
                    if self.confidence_head_recorder is not None
                    else None
                )

                metrics_row = self.record_dataset_metrics(
                    dataset_name=dataset_name,
                    metric_summary=metric_summary,
                )
                if metrics_row is not None and confidence_row is not None:
                    self.confidence_head_recorder.report_dataset(
                        metrics_row=metrics_row,
                        confidence_row=confidence_row,
                        args_payload=jsonable(vars(self.args)),
                        tasks=self.tasks,
                    )
        finally:
            if self.confidence_head_recorder is not None:
                self.confidence_head_recorder.close()

        self.report_results()

    def log_tensorboard(self) -> None:
        super().log_tensorboard()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.log_tensorboard()

    def print_results(self) -> None:
        super().print_results()
        if self.confidence_head_recorder is not None:
            self.confidence_head_recorder.print_results()


class Gemma4DSparkEvaluator(Qwen3DSparkEvaluator):
    draft_model_cls = Gemma4DSparkModel


class Qwen3_6DSparkEvaluator(Qwen3DSparkEvaluator):
    draft_model_cls = Qwen3_6DSparkModel

    def build_models(self):
        target_config = AutoConfig.from_pretrained(self.args.target_name_or_path)
        self.target_adapter = get_target_adapter(
            target_config,
            self.args.target_name_or_path,
        )
        target_model = self.target_adapter.load_model_with_head(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()
        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
        assert 0.0 <= float(self.args.confidence_threshold) <= 1.0
        self.processor, tokenizer = self.target_adapter.load_processor(
            self.args.target_name_or_path
        )
        if self.processor is None:
            raise ValueError(
                "Qwen3.5-hybrid evaluation requires the multimodal Qwen processor."
            )
        return target_model, draft_model, tokenizer
class Qwen3_8DFlash2Evaluator(Qwen3_6DSparkEvaluator):
    draft_model_cls = Qwen3_8DFlash2Model

    def build_models(self):
        target_model, draft_model, tokenizer = super().build_models()
        target_embed_tokens, target_lm_head = get_target_embeddings(target_model)
        draft_model.initialize_embeddings_and_head(
            embed_tokens=target_embed_tokens,
            lm_head=target_lm_head,
            freeze=True,
        )
        return target_model, draft_model, tokenizer


class Qwen3_8DSparkEvaluator(Qwen3_6DSparkEvaluator):
    draft_model_cls = Qwen3_8DSparkModel
