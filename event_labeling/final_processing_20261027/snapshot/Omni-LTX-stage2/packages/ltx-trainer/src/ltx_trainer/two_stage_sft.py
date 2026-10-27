"""Two-stage latent-upsample inference for Qwen-conditioned full-SFT checkpoints.

The trainer full-SFT checkpoint contains only native transformer weights. The
official development checkpoint remains the source of model configuration and
all non-transformer components.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning import ConditioningItem, VideoConditionByLatentIndex
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader.fuse_loras import apply_loras
from ltx_core.loader.helpers import load_state_dict
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps, LoraStateDictWithStrength, StateDict
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP, SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader, SafetensorsStateDictLoader
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer import LTXModel, LTXModelConfigurator, Modality
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.types import Audio, LatentState
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
    timesteps_from_mask,
)
from ltx_pipelines.utils.types import DenoisedLatentResult, ModalitySpec, OffloadMode
from safetensors import safe_open

from ltx_trainer import logger


@dataclass(frozen=True)
class SFTArchitecture:
    """Conditioning dimensions inferred from a trainer full-SFT checkpoint."""

    video_qwen_context_dim: int | None
    audio_qwen_context_dim: int | None
    first_frame_in_channels: int | None
    use_qwen_layer_adapters: bool = False
    qwen_layer_adapter_num_layers: int | None = None
    qwen_layer_adapter_input_dim: int | None = None
    video_omni_context_dim: int | None = None
    audio_omni_context_dim: int | None = None
    omni_layer_adapter_num_layers: int | None = None
    omni_layer_adapter_input_dim: int | None = None
    omni_field_resampler_input_dim: int | None = None
    omni_field_resampler_output_dim: int | None = None
    listener_timbre_dim: int | None = None


@dataclass(frozen=True)
class GemmaConditioning:
    """Positive and CFG-negative Gemma contexts for one sample."""

    video_positive: torch.Tensor
    audio_positive: torch.Tensor
    video_negative: torch.Tensor
    audio_negative: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> GemmaConditioning:
        return GemmaConditioning(
            video_positive=self.video_positive.to(device=device, dtype=dtype),
            audio_positive=self.audio_positive.to(device=device, dtype=dtype),
            video_negative=self.video_negative.to(device=device, dtype=dtype),
            audio_negative=self.audio_negative.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class TwoStageLatents:
    """Final unpatchified video and audio latents."""

    video: torch.Tensor
    audio: torch.Tensor

    def to_cpu(self) -> TwoStageLatents:
        return TwoStageLatents(video=self.video.cpu(), audio=self.audio.cpu())


def _inference_iterator(iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    with torch.inference_mode():
        yield from iterator


class BaseConfigSFTStateDictLoader:
    """Read config metadata from the official checkpoint and weights from SFT."""

    def __init__(
        self,
        base_checkpoint: str,
        *,
        metadata_loader: SafetensorsModelStateDictLoader | None = None,
        weights_loader: SafetensorsStateDictLoader | None = None,
    ) -> None:
        self.base_checkpoint = base_checkpoint
        self.metadata_loader = metadata_loader or SafetensorsModelStateDictLoader()
        self.weights_loader = weights_loader or SafetensorsStateDictLoader()

    def metadata(self, _path: str) -> dict:
        return self.metadata_loader.metadata(self.base_checkpoint)

    def load(
        self,
        path: str | list[str],
        sd_ops: SDOps | None = None,
        device: torch.device | None = None,
    ) -> StateDict:
        return self.weights_loader.load(path, sd_ops=sd_ops, device=device)


@dataclass(frozen=True)
class StrictSFTModelBuilder(SingleGPUModelBuilder[LTXModel]):
    """Single-GPU builder that rejects every missing or unexpected SFT key.

    Optional LoRAs (e.g. official distilled LoRA) are fused after the strict SFT
    key check, matching the official two-stage Stage-2 path.
    """

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **_kwargs: object,
    ) -> LTXModel:
        target = torch.device("cuda") if device is None else device
        model = self.meta_model(self.model_config(), self.module_ops)
        loaded = load_state_dict(
            self.model_path,
            self.model_loader,
            self.registry,
            target,
            self.model_sd_ops,
        )
        state_dict = loaded.sd
        if dtype is not None:
            state_dict = {
                key: value.to(dtype=dtype) if value.is_floating_point() else value
                for key, value in state_dict.items()
            }

        model_keys = {name for name, _ in (*model.named_parameters(), *model.named_buffers())}
        missing_keys = sorted(model_keys - state_dict.keys())
        unexpected_keys = sorted(state_dict.keys() - model_keys)
        if missing_keys or unexpected_keys:
            missing = ", ".join(missing_keys[:20]) or "none"
            unexpected = ", ".join(unexpected_keys[:20]) or "none"
            raise RuntimeError(
                "Full-SFT checkpoint does not exactly match the configured transformer. "
                f"Missing ({len(missing_keys)}): {missing}. "
                f"Unexpected ({len(unexpected_keys)}): {unexpected}."
            )

        final_sd = state_dict
        if self.loras:
            strengths = [lora.strength for lora in self.loras]
            if min(strengths) != 0 or max(strengths) != 0:
                lora_state_dicts = [
                    load_state_dict(
                        [lora.path],
                        self.model_loader,
                        self.registry,
                        self.lora_load_device,
                        lora.sd_ops,
                    )
                    for lora in self.loras
                ]
                fused = apply_loras(
                    model_sd=StateDict(
                        sd=state_dict,
                        device=loaded.device,
                        size=loaded.size,
                        dtype=loaded.dtype,
                    ),
                    lora_sd_and_strengths=[
                        LoraStateDictWithStrength(sd, strength)
                        for sd, strength in zip(lora_state_dicts, strengths, strict=True)
                    ],
                    dtype=dtype,
                    destination_sd=None,
                )
                final_sd = fused.sd
                logger.info(
                    "Fused "
                    + ", ".join(f"{Path(lora.path).name}@{lora.strength:g}" for lora in self.loras)
                    + " into the full-SFT transformer"
                )

        model.load_state_dict(final_sd, strict=False, assign=True)
        meta_names = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.is_meta]
        if meta_names:
            raise RuntimeError(f"Full-SFT load left {len(meta_names)} tensors on meta: {meta_names[:20]}")
        return model.to(target)


def inspect_sft_architecture(checkpoint_path: str | Path) -> SFTArchitecture:
    """Infer custom branch dimensions without materializing checkpoint tensors."""

    path = str(checkpoint_path)
    qwen_video_key = "transformer_blocks.0.qwen_attn2.to_k.weight"
    audio_key = "transformer_blocks.0.audio_qwen_attn2.to_k.weight"
    omni_video_key = "transformer_blocks.0.omni_attn2.to_k.weight"
    omni_audio_key = "transformer_blocks.0.audio_omni_attn2.to_k.weight"
    timbre_key = "transformer_blocks.0.audio_timbre_attn2.to_k.weight"
    first_frame_key = "audio_first_frame_patch_embedding.weight"
    with safe_open(path, framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
        if qwen_video_key not in keys and omni_video_key not in keys:
            raise ValueError(
                "SFT checkpoint has neither a video Qwen nor an Omni conditioning branch"
            )
        video_shape = checkpoint.get_slice(qwen_video_key).get_shape() if qwen_video_key in keys else None
        audio_shape = checkpoint.get_slice(audio_key).get_shape() if audio_key in keys else None
        omni_video_shape = (
            checkpoint.get_slice(omni_video_key).get_shape()
            if omni_video_key in keys
            else None
        )
        omni_audio_shape = (
            checkpoint.get_slice(omni_audio_key).get_shape()
            if omni_audio_key in keys
            else None
        )
        timbre_shape = checkpoint.get_slice(timbre_key).get_shape() if timbre_key in keys else None
        first_frame_shape = checkpoint.get_slice(first_frame_key).get_shape() if first_frame_key in keys else None
        adapter_key = "video_qwen_layer_adapter.proj.weight"
        use_layer_adapters = adapter_key in keys
        adapter_num_layers = None
        adapter_input_dim = None
        if use_layer_adapters:
            adapter_shape = checkpoint.get_slice(adapter_key).get_shape()
            layer_logits_shape = checkpoint.get_slice("video_qwen_layer_adapter.layer_logits").get_shape()
            adapter_input_dim = int(adapter_shape[1])
            adapter_num_layers = int(layer_logits_shape[0])
        omni_adapter_key = "video_omni_layer_adapter.proj.weight"
        omni_adapter_input_dim = None
        omni_adapter_num_layers = None
        if omni_adapter_key in keys:
            omni_adapter_shape = checkpoint.get_slice(omni_adapter_key).get_shape()
            omni_logits_shape = checkpoint.get_slice(
                "video_omni_layer_adapter.layer_logits"
            ).get_shape()
            omni_adapter_input_dim = int(omni_adapter_shape[1])
            omni_adapter_num_layers = int(omni_logits_shape[0])
        omni_resampler_key = "omni_field_resampler.proj_in.weight"
        omni_resampler_input_dim = None
        omni_resampler_output_dim = None
        if omni_resampler_key in keys:
            omni_resampler_shape = checkpoint.get_slice(
                omni_resampler_key
            ).get_shape()
            omni_resampler_output_dim = int(omni_resampler_shape[0])
            omni_resampler_input_dim = int(omni_resampler_shape[1])

    return SFTArchitecture(
        video_qwen_context_dim=int(video_shape[1]) if video_shape is not None else None,
        audio_qwen_context_dim=int(audio_shape[1]) if audio_shape is not None else None,
        first_frame_in_channels=int(first_frame_shape[1]) if first_frame_shape is not None else None,
        use_qwen_layer_adapters=use_layer_adapters,
        qwen_layer_adapter_num_layers=adapter_num_layers,
        qwen_layer_adapter_input_dim=adapter_input_dim,
        video_omni_context_dim=(
            int(omni_video_shape[1]) if omni_video_shape is not None else None
        ),
        audio_omni_context_dim=(
            int(omni_audio_shape[1]) if omni_audio_shape is not None else None
        ),
        omni_layer_adapter_num_layers=omni_adapter_num_layers,
        omni_layer_adapter_input_dim=omni_adapter_input_dim,
        omni_field_resampler_input_dim=omni_resampler_input_dim,
        omni_field_resampler_output_dim=omni_resampler_output_dim,
        listener_timbre_dim=int(timbre_shape[1]) if timbre_shape is not None else None,
    )


def create_sft_transformer_builder(
    base_checkpoint: str | Path,
    sft_checkpoint: str | Path,
    *,
    registry: Registry | None = None,
    loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
) -> tuple[StrictSFTModelBuilder, SFTArchitecture]:
    """Create a strict builder with official config and trainer SFT weights."""

    architecture = inspect_sft_architecture(sft_checkpoint)

    def enable_sft_conditioning(model: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(model, LTXModel):
            raise TypeError(f"Expected LTXModel, got {type(model).__name__}")
        if architecture.video_qwen_context_dim is not None:
            model.enable_qwen_conditioning(
                qwen_context_dim=architecture.video_qwen_context_dim,
                audio_qwen_context_dim=architecture.audio_qwen_context_dim,
                enable_audio=architecture.audio_qwen_context_dim is not None,
            )
        if architecture.use_qwen_layer_adapters:
            if architecture.qwen_layer_adapter_num_layers is None or architecture.qwen_layer_adapter_input_dim is None:
                raise ValueError("SFT checkpoint has incomplete Qwen layer adapter metadata")
            model.enable_qwen_layer_adapters(
                num_layers=architecture.qwen_layer_adapter_num_layers,
                input_dim=architecture.qwen_layer_adapter_input_dim,
                video_output_dim=architecture.video_qwen_context_dim,
                audio_output_dim=architecture.audio_qwen_context_dim,
                enable_audio=architecture.audio_qwen_context_dim is not None,
            )
        if architecture.video_omni_context_dim is not None:
            model.enable_omni_conditioning(
                omni_context_dim=architecture.video_omni_context_dim,
                audio_omni_context_dim=architecture.audio_omni_context_dim,
                enable_audio=architecture.audio_omni_context_dim is not None,
            )
            if architecture.omni_layer_adapter_num_layers is not None:
                if architecture.omni_layer_adapter_input_dim is None:
                    raise ValueError(
                        "SFT checkpoint has incomplete Omni layer adapter metadata"
                    )
                model.enable_omni_layer_adapters(
                    num_layers=architecture.omni_layer_adapter_num_layers,
                    input_dim=architecture.omni_layer_adapter_input_dim,
                    video_output_dim=architecture.video_omni_context_dim,
                    audio_output_dim=architecture.audio_omni_context_dim,
                    enable_audio=architecture.audio_omni_context_dim is not None,
                )
                if architecture.omni_field_resampler_input_dim is not None:
                    if architecture.omni_field_resampler_output_dim != architecture.omni_layer_adapter_input_dim:
                        raise ValueError(
                            "SFT Omni Resampler output disagrees with layer adapter input"
                        )
                    model.enable_omni_field_resampler(
                        num_layers=architecture.omni_layer_adapter_num_layers,
                        input_dim=architecture.omni_field_resampler_input_dim,
                        output_dim=architecture.omni_field_resampler_output_dim,
                        num_heads=8,
                    )
        if architecture.listener_timbre_dim is not None:
            model.enable_listener_timbre_conditioning(
                listener_timbre_context_dim=architecture.listener_timbre_dim
            )
        if architecture.first_frame_in_channels is not None:
            model.enable_first_frame_audio_conditioning(
                first_frame_in_channels=architecture.first_frame_in_channels,
            )
        return model

    builder = StrictSFTModelBuilder(
        model_path=str(sft_checkpoint),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=None,
        module_ops=(
            ModuleOps(
                name="enable_qwen_and_first_frame_sft_conditioning",
                matcher=lambda model: isinstance(model, LTXModel),
                mutator=enable_sft_conditioning,
            ),
        ),
        model_loader=BaseConfigSFTStateDictLoader(str(base_checkpoint)),
        registry=registry or DummyRegistry(),
        loras=loras,
    )
    return builder, architecture


def distilled_lora_ops(
    path: str | Path,
    strength: float = 0.8,
) -> LoraPathStrengthAndSDOps:
    """Official distilled LoRA handle used by Stage 2 of the two-stage pipeline."""

    return LoraPathStrengthAndSDOps(str(path), strength, LTXV_LORA_COMFY_RENAMING_MAP)


def make_stage_2_sigmas(
    sigma_start: float,
    steps: int,
    *,
    dense_steps: int = 1000,
    scheduler: LTX2Scheduler | None = None,
) -> torch.Tensor:
    """Truncate and resample a normal LTX2 schedule for non-distilled refinement."""

    if not 0.0 < sigma_start <= 1.0:
        raise ValueError(f"sigma_start must be in (0, 1], got {sigma_start}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if dense_steps < 2:
        raise ValueError(f"dense_steps must be >= 2, got {dense_steps}")

    dense = (scheduler or LTX2Scheduler()).execute(steps=max(dense_steps, steps * 8)).to(torch.float32)
    tail = dense[dense < sigma_start]
    if tail.numel() == 0 or not torch.isclose(tail[-1], torch.tensor(0.0)):
        tail = torch.cat((tail, torch.zeros(1, dtype=torch.float32)))
    samples = torch.cat((torch.tensor([sigma_start], dtype=torch.float32), tail))

    positions = torch.linspace(0, samples.numel() - 1, steps + 1, dtype=torch.float32)
    lower = positions.floor().to(torch.long)
    upper = positions.ceil().to(torch.long)
    fraction = positions - lower
    sigmas = samples[lower] * (1.0 - fraction) + samples[upper] * fraction
    sigmas[0] = float(sigma_start)
    sigmas[-1] = 0.0

    if not torch.all(sigmas[:-1] > sigmas[1:]):
        raise RuntimeError(f"Stage-2 sigma schedule is not strictly decreasing: {sigmas.tolist()}")
    return sigmas


def two_stage_pixel_sizes(height: int, width: int) -> tuple[int, int, int, int]:
    """Map CLI ``--height/--width`` to Stage-1 (training) and Stage-2 (2x) sizes.

    Official ``ti2vid_two_stages`` treats CLI size as the *final target* and runs
    Stage 1 at half. Qwen DIY two-stage does the opposite so Stage 1 matches
    training (e.g. 1088x640 -> Stage 2 2176x1280).
    """
    if height < 1 or width < 1:
        raise ValueError(f"height and width must be positive, got {width}x{height}")
    return height, width, height * 2, width * 2


def frame_zero_latent(conditionings: Sequence[ConditioningItem], *, stage: str) -> torch.Tensor:
    frame_zero = [
        item
        for item in conditionings
        if isinstance(item, VideoConditionByLatentIndex) and item.latent_idx == 0
    ]
    if len(frame_zero) != 1:
        raise ValueError(f"Exactly one frame-0 image conditioning is required for {stage}")
    return frame_zero[0].latent


class SFTGuidedDenoiser:
    """Trainer-equivalent sequential CFG/STG denoiser for Qwen full-SFT."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        gemma: GemmaConditioning,
        qwen_context: torch.Tensor,
        qwen_context_mask: torch.Tensor,
        first_frame_latent: torch.Tensor | None,
        audio_qwen_context: torch.Tensor | None = None,
        audio_qwen_context_mask: torch.Tensor | None = None,
        omni_context: torch.Tensor | None = None,
        omni_context_mask: torch.Tensor | None = None,
        audio_omni_context: torch.Tensor | None = None,
        audio_omni_context_mask: torch.Tensor | None = None,
        listener_timbre_context: torch.Tensor | None = None,
        guidance_scale: float = 4.0,
        stg_scale: float = 1.0,
        stg_blocks: Sequence[int] = (29,),
        stg_mode: str = "stg_av",
    ) -> None:
        if stg_mode not in {"stg_av", "stg_v"}:
            raise ValueError(f"stg_mode must be 'stg_av' or 'stg_v', got {stg_mode!r}")
        self.gemma = gemma
        self.qwen_context = qwen_context
        self.qwen_context_mask = qwen_context_mask
        self.audio_qwen_context = (
            audio_qwen_context if audio_qwen_context is not None else qwen_context
        )
        self.audio_qwen_context_mask = (
            audio_qwen_context_mask
            if audio_qwen_context_mask is not None
            else qwen_context_mask
        )
        self.first_frame_latent = first_frame_latent
        self.omni_context = omni_context
        self.omni_context_mask = omni_context_mask
        self.audio_omni_context = (
            audio_omni_context if audio_omni_context is not None else omni_context
        )
        self.audio_omni_context_mask = (
            audio_omni_context_mask
            if audio_omni_context_mask is not None
            else omni_context_mask
        )
        self.listener_timbre_context = listener_timbre_context
        self.guidance_scale = float(guidance_scale)
        self.stg_scale = float(stg_scale)
        self.stg_blocks = list(stg_blocks)
        self.stg_mode = stg_mode

    @staticmethod
    def _modality(  # noqa: PLR0913
        state: LatentState,
        context: torch.Tensor,
        sigma: torch.Tensor,
        *,
        qwen_context: torch.Tensor | None,
        qwen_context_mask: torch.Tensor | None,
        omni_context: torch.Tensor | None = None,
        omni_context_mask: torch.Tensor | None = None,
        listener_timbre_context: torch.Tensor | None = None,
        first_frame_latent: torch.Tensor | None = None,
    ) -> Modality:
        sigma_batch = sigma.expand(state.latent.shape[0])
        return Modality(
            enabled=True,
            latent=state.latent,
            sigma=sigma_batch,
            timesteps=timesteps_from_mask(state.denoise_mask, sigma_batch),
            positions=state.positions,
            context=context,
            context_mask=None,
            attention_mask=state.attention_mask,
            qwen_context=qwen_context,
            qwen_context_mask=qwen_context_mask,
            omni_context=omni_context,
            omni_context_mask=omni_context_mask,
            listener_timbre_context=listener_timbre_context,
            first_frame_latent=first_frame_latent,
        )

    def _stg_perturbations(self, batch_size: int) -> BatchedPerturbationConfig:
        perturbations = [
            Perturbation(
                type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=self.stg_blocks,
            )
        ]
        if self.stg_mode == "stg_av":
            perturbations.append(
                Perturbation(
                    type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    blocks=self.stg_blocks,
                )
            )
        return BatchedPerturbationConfig(
            perturbations=[PerturbationConfig(list(perturbations)) for _ in range(batch_size)]
        )

    @staticmethod
    def _result(
        positive: torch.Tensor | None,
        negative: torch.Tensor | None,
        perturbed: torch.Tensor | None,
        guidance_scale: float,
        stg_scale: float,
    ) -> DenoisedLatentResult | None:
        if positive is None:
            return None
        denoised = positive
        if negative is not None:
            denoised = denoised + (guidance_scale - 1.0) * (positive - negative)
        if perturbed is not None:
            denoised = denoised + stg_scale * (positive - perturbed)
        return DenoisedLatentResult(
            denoised=denoised,
            cond=positive,
            uncond=negative,
            ptb=perturbed,
        )

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        if video_state is None or audio_state is None:
            raise ValueError("Qwen full-SFT inference requires both video and audio states")
        sigma = sigmas[step_index]
        # Match ValidationSampler: PEFT LoRA + mixed activations need autocast.
        device_type = video_state.latent.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            first_frame = (
                self.first_frame_latent.to(
                    device=video_state.latent.device,
                    dtype=(
                        torch.bfloat16
                        if device_type == "cuda"
                        else self.first_frame_latent.dtype
                    ),
                )
                if self.first_frame_latent is not None
                else None
            )
            video_positive = self._modality(
                video_state,
                self.gemma.video_positive,
                sigma,
                qwen_context=self.qwen_context,
                qwen_context_mask=self.qwen_context_mask,
                omni_context=self.omni_context,
                omni_context_mask=self.omni_context_mask,
            )
            audio_positive = self._modality(
                audio_state,
                self.gemma.audio_positive,
                sigma,
                qwen_context=self.audio_qwen_context,
                qwen_context_mask=self.audio_qwen_context_mask,
                omni_context=self.audio_omni_context,
                omni_context_mask=self.audio_omni_context_mask,
                listener_timbre_context=self.listener_timbre_context,
                first_frame_latent=first_frame,
            )
            positive_video, positive_audio = transformer(
                video=video_positive,
                audio=audio_positive,
                perturbations=None,
            )

            negative_video: torch.Tensor | None = None
            negative_audio: torch.Tensor | None = None
            if self.guidance_scale != 1.0:
                video_negative = self._modality(
                    video_state,
                    self.gemma.video_negative,
                    sigma,
                    qwen_context=None,
                    qwen_context_mask=None,
                    omni_context=None,
                    omni_context_mask=None,
                )
                audio_negative = self._modality(
                    audio_state,
                    self.gemma.audio_negative,
                    sigma,
                    qwen_context=None,
                    qwen_context_mask=None,
                    omni_context=None,
                    omni_context_mask=None,
                    listener_timbre_context=self.listener_timbre_context,
                    first_frame_latent=first_frame,
                )
                negative_video, negative_audio = transformer(
                    video=video_negative,
                    audio=audio_negative,
                    perturbations=None,
                )

            perturbed_video: torch.Tensor | None = None
            perturbed_audio: torch.Tensor | None = None
            if self.stg_scale != 0.0:
                perturbations = self._stg_perturbations(video_state.latent.shape[0])
                perturbed_video, perturbed_audio = transformer(
                    video=video_positive,
                    audio=audio_positive,
                    perturbations=perturbations,
                )

        return (
            self._result(
                positive_video,
                negative_video,
                perturbed_video,
                self.guidance_scale,
                self.stg_scale,
            ),
            self._result(
                positive_audio,
                negative_audio,
                perturbed_audio,
                self.guidance_scale,
                self.stg_scale,
            ),
        )


class TwoStageSFTPipeline:
    """DIY full-SFT Stage 1 at training resolution -> official x2 upsampler -> Stage 2.

    ``height``/``width`` are Stage 1 (training) pixels. Stage 2 is 2x.

    Default (``official_stage2=True``): Stage 2 is the **official** LTX-2.3-dev
    transformer + official distilled LoRA + ``SimpleDenoiser`` /
    ``STAGE_2_DISTILLED_SIGMAS`` (no SFT weights, no Qwen).

    Legacy (``official_stage2=False``): Stage 2 reuses the SFT transformer and
    optionally fuses distilled LoRA into those DIY weights (CFG/STG refine).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        checkpoint_path: str,
        sft_checkpoint: str,
        gemma_root: str,
        spatial_upsampler_path: str | None = None,
        device: torch.device | None = None,
        registry: Registry | None = None,
        distilled_lora: LoraPathStrengthAndSDOps | None = None,
        official_stage2: bool = True,
    ) -> None:
        if distilled_lora is not None and spatial_upsampler_path is None:
            raise ValueError("distilled_lora requires spatial_upsampler_path for Stage 2")
        if official_stage2 and spatial_upsampler_path is not None and distilled_lora is None:
            raise ValueError(
                "official_stage2=True requires distilled_lora "
                "(Stage 2 = official transformer + official distilled LoRA)"
            )

        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.scheduler = LTX2Scheduler()
        self.distilled_lora = distilled_lora
        self.official_stage2 = official_stage2
        builder, self.sft_architecture = create_sft_transformer_builder(
            checkpoint_path,
            sft_checkpoint,
            registry=registry,
        )

        self.prompt_encoder = PromptEncoder(
            checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=OffloadMode.NONE,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
        )
        self.upsampler = (
            VideoUpsampler(
                checkpoint_path,
                spatial_upsampler_path,
                self.dtype,
                self.device,
                registry=registry,
            )
            if spatial_upsampler_path is not None
            else None
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
        )
        self.stage = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            torch_compile=False,
            offload_mode=OffloadMode.NONE,
            transformer_builder=builder,
        )
        if spatial_upsampler_path is None:
            # Stage-1-only callers never touch stage_2.
            self.stage_2 = self.stage
        elif official_stage2:
            # Official base + distilled only (no DIY / Qwen builder).
            assert distilled_lora is not None
            self.stage_2 = DiffusionStage(
                checkpoint_path,
                self.dtype,
                self.device,
                loras=(distilled_lora,),
                registry=registry,
                torch_compile=False,
                offload_mode=OffloadMode.NONE,
            )
        else:
            stage_2_loras = (distilled_lora,) if distilled_lora is not None else ()
            stage_2_builder, _ = create_sft_transformer_builder(
                checkpoint_path,
                sft_checkpoint,
                registry=registry,
                loras=stage_2_loras,
            )
            self.stage_2 = (
                DiffusionStage(
                    checkpoint_path,
                    self.dtype,
                    self.device,
                    registry=registry,
                    torch_compile=False,
                    offload_mode=OffloadMode.NONE,
                    transformer_builder=stage_2_builder,
                )
                if distilled_lora is not None
                else self.stage
            )

    def transformer_context(self) -> AbstractContextManager[torch.nn.Module]:
        """Keep one plain-SFT transformer resident across multiple generation calls."""

        return self.stage.model_context()

    def stage_2_transformer_context(self) -> AbstractContextManager[torch.nn.Module]:
        """Keep the Stage-2 transformer (optionally distilled) resident."""

        return self.stage_2.model_context()

    def encode_prompt_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[GemmaConditioning]:
        """Encode all unique positive/negative prompts in one Gemma lifecycle."""

        unique_prompts = list(dict.fromkeys(prompt for pair in pairs for prompt in pair))
        logger.info(f"Encoding {len(unique_prompts)} unique prompt(s) with Gemma")
        outputs = self.prompt_encoder(unique_prompts)
        by_prompt = dict(zip(unique_prompts, outputs, strict=True))

        result: list[GemmaConditioning] = []
        for positive, negative in pairs:
            positive_output = by_prompt[positive]
            negative_output = by_prompt[negative]
            if positive_output.audio_encoding is None or negative_output.audio_encoding is None:
                raise ValueError("The official checkpoint does not provide audio prompt encodings")
            result.append(
                GemmaConditioning(
                    video_positive=positive_output.video_encoding.detach().cpu(),
                    audio_positive=positive_output.audio_encoding.detach().cpu(),
                    video_negative=negative_output.video_encoding.detach().cpu(),
                    audio_negative=negative_output.audio_encoding.detach().cpu(),
                )
            )
        return result

    def _image_conditionings(
        self,
        image: ImageConditioningInput,
        *,
        stage_1_height: int,
        stage_1_width: int,
        stage_2_height: int,
        stage_2_width: int,
    ) -> tuple[list[ConditioningItem], list[ConditioningItem], torch.Tensor, torch.Tensor]:
        def encode(
            encoder: VideoEncoder,
        ) -> tuple[list[ConditioningItem], list[ConditioningItem], torch.Tensor, torch.Tensor]:
            images = [image]
            stage_1 = combined_image_conditionings(
                images=images,
                height=stage_1_height,
                width=stage_1_width,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
            )
            stage_2 = combined_image_conditionings(
                images=images,
                height=stage_2_height,
                width=stage_2_width,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
            )
            return (
                stage_1,
                stage_2,
                frame_zero_latent(stage_1, stage="stage 1"),
                frame_zero_latent(stage_2, stage="stage 2"),
            )

        return self.image_conditioner(encode)

    def _denoiser(  # noqa: PLR0913
        self,
        *,
        gemma: GemmaConditioning,
        qwen_context: torch.Tensor,
        qwen_context_mask: torch.Tensor,
        first_frame_latent: torch.Tensor | None,
        guidance_scale: float,
        stg_scale: float,
        stg_blocks: Sequence[int],
        stg_mode: str,
    ) -> SFTGuidedDenoiser:
        return SFTGuidedDenoiser(
            gemma=gemma,
            qwen_context=qwen_context,
            qwen_context_mask=qwen_context_mask,
            audio_qwen_context=qwen_context if self.sft_architecture.audio_qwen_context_dim is not None else None,
            audio_qwen_context_mask=(
                qwen_context_mask if self.sft_architecture.audio_qwen_context_dim is not None else None
            ),
            first_frame_latent=(
                first_frame_latent if self.sft_architecture.first_frame_in_channels is not None else None
            ),
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks,
            stg_mode=stg_mode,
        )

    @torch.inference_mode()
    def generate_latents(  # noqa: PLR0913
        self,
        *,
        gemma: GemmaConditioning,
        qwen_context: torch.Tensor,
        qwen_context_mask: torch.Tensor,
        image_path: str,
        image_strength: float,
        image_crf: int,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        stage_1_steps: int,
        stage_2_steps: int,
        stage_2_start_sigma: float,
        guidance_scale: float,
        stg_scale: float,
        stg_blocks: Sequence[int],
        stg_mode: str,
        stage_2_guidance_scale: float | None = None,
        stage_2_stg_scale: float | None = None,
        reload_transformer_between_stages: bool = False,
        transformer: torch.nn.Module | None = None,
        transformer_stage_2: torch.nn.Module | None = None,
        stage_1_only: bool = False,
    ) -> TwoStageLatents:
        """Generate unpatchified latents.

        Default two-stage path: Stage 1 at training resolution (``width`` x
        ``height``), then upsample + Stage 2 at 2x. With ``stage_1_only=True``,
        run a single diffusion pass at ``width`` x ``height`` and decode
        directly — no upsampler / Stage 2 / distilled LoRA.
        """

        if num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 1, got {num_frames}")
        if stage_1_steps < 1:
            raise ValueError(f"stage_1_steps must be >= 1, got {stage_1_steps}")
        if stage_1_only and self.distilled_lora is not None:
            raise ValueError("stage_1_only cannot be combined with distilled_lora")
        if not stage_1_only and self.upsampler is None:
            raise ValueError("spatial_upsampler_path is required unless stage_1_only=True")
        if qwen_context.shape[0] != 1 or qwen_context.ndim not in (3, 4):
            raise ValueError(f"qwen_context must have shape [1, seq, hidden] or [1, layers, seq, hidden], got {qwen_context.shape}")
        if self.sft_architecture.use_qwen_layer_adapters:
            if qwen_context.ndim != 4:
                raise ValueError(f"SFT checkpoint expects stacked Qwen layer features, got {qwen_context.shape}")
            if qwen_context.shape[1] != self.sft_architecture.qwen_layer_adapter_num_layers:
                raise ValueError(
                    f"Qwen layer count {qwen_context.shape[1]} does not match SFT adapter "
                    f"count {self.sft_architecture.qwen_layer_adapter_num_layers}"
                )
            if qwen_context.shape[-1] != self.sft_architecture.qwen_layer_adapter_input_dim:
                raise ValueError(
                    f"Qwen hidden size {qwen_context.shape[-1]} does not match SFT adapter "
                    f"input dimension {self.sft_architecture.qwen_layer_adapter_input_dim}"
                )
            expected_mask_shape = (qwen_context.shape[0], qwen_context.shape[2])
        else:
            if qwen_context.ndim != 3:
                raise ValueError(f"SFT checkpoint expects last-layer Qwen features, got {qwen_context.shape}")
            if qwen_context.shape[-1] != self.sft_architecture.video_qwen_context_dim:
                raise ValueError(
                    f"Qwen hidden size {qwen_context.shape[-1]} does not match SFT "
                    f"dimension {self.sft_architecture.video_qwen_context_dim}"
                )
            if (
                self.sft_architecture.audio_qwen_context_dim is not None
                and qwen_context.shape[-1] != self.sft_architecture.audio_qwen_context_dim
            ):
                raise ValueError(
                    f"Qwen hidden size {qwen_context.shape[-1]} does not match SFT audio "
                    f"dimension {self.sft_architecture.audio_qwen_context_dim}"
                )
            expected_mask_shape = qwen_context.shape[:2]
        if qwen_context_mask.shape != expected_mask_shape:
            raise ValueError(
                f"qwen_context_mask shape {qwen_context_mask.shape} does not match {expected_mask_shape}"
            )

        gemma = gemma.to(self.device, self.dtype)
        qwen_context = qwen_context.to(device=self.device, dtype=self.dtype)
        qwen_context_mask = qwen_context_mask.to(device=self.device, dtype=torch.long)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stage_1_sigmas = self.scheduler.execute(steps=stage_1_steps).to(
            device=self.device,
            dtype=torch.float32,
        )
        image = ImageConditioningInput(image_path, 0, image_strength, image_crf)

        if stage_1_only:
            # Match training resolution: one-stage diffusion at full HxW.
            assert_resolution(height=height, width=width, is_two_stage=False)

            def encode_full(encoder: VideoEncoder) -> tuple[list[ConditioningItem], torch.Tensor]:
                conditionings = combined_image_conditionings(
                    images=[image],
                    height=height,
                    width=width,
                    video_encoder=encoder,
                    dtype=self.dtype,
                    device=self.device,
                )
                frame_zero = [
                    item
                    for item in conditionings
                    if isinstance(item, VideoConditionByLatentIndex) and item.latent_idx == 0
                ]
                if len(frame_zero) != 1:
                    raise ValueError("Exactly one frame-0 image conditioning is required")
                return conditionings, frame_zero[0].latent

            full_conditionings, first_frame_latent = self.image_conditioner(encode_full)
            denoiser = self._denoiser(
                gemma=gemma,
                qwen_context=qwen_context,
                qwen_context_mask=qwen_context_mask,
                first_frame_latent=first_frame_latent,
                guidance_scale=guidance_scale,
                stg_scale=stg_scale,
                stg_blocks=stg_blocks,
                stg_mode=stg_mode,
            )

            def run_full(transformer: torch.nn.Module) -> tuple[LatentState, LatentState]:
                video_state, audio_state = self.stage.run(
                    transformer,
                    denoiser,
                    stage_1_sigmas,
                    noiser,
                    width,
                    height,
                    num_frames,
                    frame_rate,
                    ModalitySpec(
                        context=gemma.video_positive,
                        conditionings=full_conditionings,
                    ),
                    ModalitySpec(context=gemma.audio_positive),
                    max_batch_size=1,
                )
                if video_state is None or audio_state is None:
                    raise RuntimeError("Stage-1-only pass did not return both video and audio states")
                return video_state, audio_state

            if transformer is not None:
                video_state, audio_state = run_full(transformer)
            else:
                with self.stage.model_context() as stage_transformer:
                    video_state, audio_state = run_full(stage_transformer)
            return TwoStageLatents(video=video_state.latent, audio=audio_state.latent)

        stage_1_height, stage_1_width, stage_2_height, stage_2_width = two_stage_pixel_sizes(
            height, width
        )
        assert_resolution(height=stage_1_height, width=stage_1_width, is_two_stage=False)
        assert_resolution(height=stage_2_height, width=stage_2_width, is_two_stage=True)
        stage_1_conditionings, stage_2_conditionings, stage_1_first_frame, stage_2_first_frame = (
            self._image_conditionings(
                image,
                stage_1_height=stage_1_height,
                stage_1_width=stage_1_width,
                stage_2_height=stage_2_height,
                stage_2_width=stage_2_width,
            )
        )
        stage_1_denoiser = self._denoiser(
            gemma=gemma,
            qwen_context=qwen_context,
            qwen_context_mask=qwen_context_mask,
            first_frame_latent=stage_1_first_frame,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks,
            stg_mode=stg_mode,
        )

        def run_stage_1(transformer: torch.nn.Module) -> tuple[LatentState, LatentState]:
            video_state, audio_state = self.stage.run(
                transformer,
                stage_1_denoiser,
                stage_1_sigmas,
                noiser,
                stage_1_width,
                stage_1_height,
                num_frames,
                frame_rate,
                ModalitySpec(
                    context=gemma.video_positive,
                    conditionings=stage_1_conditionings,
                ),
                ModalitySpec(context=gemma.audio_positive),
                max_batch_size=1,
            )
            if video_state is None or audio_state is None:
                raise RuntimeError("Stage 1 did not return both video and audio states")
            return video_state, audio_state

        if self.official_stage2:
            # Match official TI2Vid Stage 2: distilled short schedule, no CFG/STG/Qwen.
            stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(device=self.device, dtype=torch.float32)
            stage_2_denoiser: object = SimpleDenoiser(
                v_context=gemma.video_positive,
                a_context=gemma.audio_positive,
            )
        else:
            stage_2_sigmas = make_stage_2_sigmas(stage_2_start_sigma, stage_2_steps).to(
                device=self.device,
                dtype=torch.float32,
            )
            stage_2_denoiser = self._denoiser(
                gemma=gemma,
                qwen_context=qwen_context,
                qwen_context_mask=qwen_context_mask,
                first_frame_latent=stage_2_first_frame,
                guidance_scale=stage_2_guidance_scale if stage_2_guidance_scale is not None else guidance_scale,
                stg_scale=stage_2_stg_scale if stage_2_stg_scale is not None else stg_scale,
                stg_blocks=stg_blocks,
                stg_mode=stg_mode,
            )
        stage_2_noise_scale = float(stage_2_sigmas[0].item())

        def run_stage_2(
            transformer: torch.nn.Module,
            upscaled_video: torch.Tensor,
            stage_1_audio: torch.Tensor,
        ) -> tuple[LatentState, LatentState]:
            video_state, audio_state = self.stage_2.run(
                transformer,
                stage_2_denoiser,  # type: ignore[arg-type]
                stage_2_sigmas,
                noiser,
                stage_2_width,
                stage_2_height,
                num_frames,
                frame_rate,
                ModalitySpec(
                    context=gemma.video_positive,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_noise_scale,
                    initial_latent=upscaled_video,
                ),
                ModalitySpec(
                    context=gemma.audio_positive,
                    noise_scale=stage_2_noise_scale,
                    initial_latent=stage_1_audio,
                ),
                max_batch_size=1,
            )
            if video_state is None or audio_state is None:
                raise RuntimeError("Stage 2 did not return both video and audio states")
            return video_state, audio_state

        assert self.upsampler is not None
        # Hybrid and distilled-DIY Stage 2 always use different weights than Stage 1.
        split_stages = (
            self.official_stage2
            or reload_transformer_between_stages
            or self.distilled_lora is not None
            or self.stage_2 is not self.stage
        )
        stage_2_module = transformer_stage_2 if transformer_stage_2 is not None else transformer
        if transformer is not None:
            if reload_transformer_between_stages:
                raise ValueError(
                    "reload_transformer_between_stages cannot be used with a caller-provided transformer"
                )
            if split_stages and stage_2_module is None:
                raise ValueError("transformer_stage_2 is required when Stage 2 uses a different builder")
            if stage_2_module is None:
                stage_2_module = transformer
            stage_1_video, stage_1_audio = run_stage_1(transformer)
            upscaled_video = self.upsampler(stage_1_video.latent[:1])
            final_video, final_audio = run_stage_2(
                stage_2_module,
                upscaled_video,
                stage_1_audio.latent,
            )
        elif split_stages:
            with self.stage.model_context() as stage_1_transformer:
                stage_1_video, stage_1_audio = run_stage_1(stage_1_transformer)
            upscaled_video = self.upsampler(stage_1_video.latent[:1])
            with self.stage_2.model_context() as stage_2_transformer:
                final_video, final_audio = run_stage_2(
                    stage_2_transformer,
                    upscaled_video,
                    stage_1_audio.latent,
                )
        else:
            with self.stage.model_context() as stage_1_transformer:
                stage_1_video, stage_1_audio = run_stage_1(stage_1_transformer)
                upscaled_video = self.upsampler(stage_1_video.latent[:1])
                final_video, final_audio = run_stage_2(
                    stage_1_transformer,
                    upscaled_video,
                    stage_1_audio.latent,
                )

        return TwoStageLatents(video=final_video.latent, audio=final_audio.latent)

    @torch.inference_mode()
    def decode(
        self,
        latents: TwoStageLatents,
        *,
        seed: int,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        """Decode final video/audio latents after the SFT transformer is released."""

        generator = torch.Generator(device=self.device).manual_seed(seed)
        audio = self.audio_decoder(latents.audio.to(device=self.device, dtype=self.dtype))
        video = _inference_iterator(
            self.video_decoder(
                latents.video.to(device=self.device, dtype=self.dtype),
                tiling_config,
                generator,
            )
        )
        return video, audio

