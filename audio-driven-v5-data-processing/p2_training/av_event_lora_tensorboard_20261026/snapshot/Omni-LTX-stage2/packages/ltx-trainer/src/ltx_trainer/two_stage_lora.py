"""Two-stage latent-upsample inference for Qwen-conditioned movie LoRA checkpoints.

Default hybrid recipe: Stage 1 = official base + Qwen/first-frame + movie LoRA
(CFG/STG); Stage 2 = official LTX-2.3-dev + official distilled LoRA + SimpleDenoiser
(no movie LoRA, no Qwen). Legacy DIY Stage 2 keeps movie LoRA on both stages.
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
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer import (
    LTXModel,
    LTXModelConfigurator,
    LTXV_MODEL_COMFY_RENAMING_MAP,
)
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
)
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

from ltx_trainer import logger
from ltx_trainer.lora_checkpoint import (
    apply_movie_lora,
    checkpoint_uses_first_frame_audio_conditioning,
    inspect_qwen_layer_adapter_architecture,
)
from ltx_trainer.two_stage_sft import (
    GemmaConditioning,
    SFTArchitecture,
    SFTGuidedDenoiser,
    TwoStageLatents,
    frame_zero_latent,
    make_stage_2_sigmas,
    two_stage_pixel_sizes,
)


def _inference_iterator(iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    with torch.inference_mode():
        yield from iterator


@dataclass(frozen=True)
class PeftLoraTransformerBuilder(SingleGPUModelBuilder[LTXModel]):
    """Build base LTX (+ optional fused distilled) then attach movie PEFT LoRA."""

    movie_lora_path: str = ""
    qwen_context_dim: int = 4096
    audio_qwen_context_dim: int = 4096
    first_frame_audio_cond: bool = False

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> LTXModel:
        target_device = torch.device("cuda") if device is None else device
        target_dtype = torch.bfloat16 if dtype is None else dtype
        model = SingleGPUModelBuilder.build(
            self, device=target_device, dtype=target_dtype, **kwargs
        )
        if not isinstance(model, LTXModel):
            # PEFT may wrap later; base build must return LTXModel.
            raise TypeError(
                f"Expected LTXModel from base build, got {type(model).__name__}"
            )
        model.enable_qwen_conditioning(
            qwen_context_dim=self.qwen_context_dim,
            audio_qwen_context_dim=self.audio_qwen_context_dim,
        )
        if self.first_frame_audio_cond:
            model.enable_first_frame_audio_conditioning()
        # Newly created Qwen / first-frame modules may default to float32.
        model = model.to(device=target_device, dtype=target_dtype)
        if not self.movie_lora_path:
            raise ValueError("movie_lora_path is required")
        return apply_movie_lora(model, self.movie_lora_path)  # type: ignore[return-value]


class _GemmaFromPretrainedBuilder:
    """Load Gemma with transformers from_pretrained, skipping the pipeline meta-builder.

    transformers 5.x flattened SiglipVisionModel / Gemma3 RoPE, so
    ``create_and_populate`` crashes. Training already uses from_pretrained
    (8-bit). Inference uses bf16 so ``gpu_model`` can move the encoder to
    meta afterwards; bitsandbytes Int8Params cannot.
    """

    def __init__(self, gemma_root: str) -> None:
        self.gemma_root = gemma_root

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> torch.nn.Module:
        del kwargs
        from transformers import Gemma3ForConditionalGeneration

        from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_trainer.gemma_8bit import _find_gemma_subpath

        target_dtype = torch.bfloat16 if dtype is None else dtype
        target_device = torch.device("cuda") if device is None else device
        gemma_path = _find_gemma_subpath(self.gemma_root, "model*.safetensors")
        tokenizer_path = _find_gemma_subpath(self.gemma_root, "tokenizer.model")
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_path,
            torch_dtype=target_dtype,
            device_map={"": target_device},
            local_files_only=True,
        )
        tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
        return GemmaTextEncoder(
            tokenizer=tokenizer, model=gemma_model, dtype=target_dtype
        )


def create_lora_transformer_builder(
    base_checkpoint: str | Path,
    movie_lora_path: str | Path,
    *,
    qwen_context_dim: int = 4096,
    audio_qwen_context_dim: int = 4096,
    registry: Registry | None = None,
    loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
) -> tuple[PeftLoraTransformerBuilder, SFTArchitecture]:
    """Create builder: official base (+ optional distilled fuse) + movie PEFT LoRA."""

    layer_adapter = inspect_qwen_layer_adapter_architecture(movie_lora_path)
    first_frame_audio_cond = checkpoint_uses_first_frame_audio_conditioning(
        movie_lora_path
    )
    architecture = SFTArchitecture(
        video_qwen_context_dim=qwen_context_dim,
        audio_qwen_context_dim=audio_qwen_context_dim,
        first_frame_in_channels=(
            0 if first_frame_audio_cond else None
        ),  # 0 is resolved inside enable_first_frame_audio_conditioning
        use_qwen_layer_adapters=layer_adapter is not None,
        qwen_layer_adapter_num_layers=layer_adapter[0]
        if layer_adapter is not None
        else None,
        qwen_layer_adapter_input_dim=layer_adapter[1]
        if layer_adapter is not None
        else None,
    )
    builder = PeftLoraTransformerBuilder(
        model_path=str(base_checkpoint),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        module_ops=(),
        loras=loras,
        registry=registry or DummyRegistry(),
        movie_lora_path=str(movie_lora_path),
        qwen_context_dim=qwen_context_dim,
        audio_qwen_context_dim=audio_qwen_context_dim,
        first_frame_audio_cond=first_frame_audio_cond,
    )
    return builder, architecture


class TwoStageLoraPipeline:
    """Movie-LoRA Stage 1 at training resolution -> official x2 upsampler -> Stage 2.

    ``height``/``width`` are Stage 1 (training) pixels. Stage 2 is 2x.

    Default (``official_stage2=True``): Stage 2 is official LTX-2.3-dev +
    official distilled LoRA + ``SimpleDenoiser`` / ``STAGE_2_DISTILLED_SIGMAS``
    (no movie LoRA, no Qwen).

    Legacy (``official_stage2=False``): Stage 2 reuses movie-LoRA weights and
    optionally fuses distilled LoRA into those DIY weights (CFG/STG refine).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        checkpoint_path: str,
        movie_lora_path: str,
        gemma_root: str,
        spatial_upsampler_path: str | None = None,
        device: torch.device | None = None,
        registry: Registry | None = None,
        distilled_lora: LoraPathStrengthAndSDOps | None = None,
        fuse_distilled_into_stage1: bool = False,
        official_stage2: bool = True,
        qwen_context_dim: int = 4096,
        audio_qwen_context_dim: int = 4096,
    ) -> None:
        if fuse_distilled_into_stage1 and distilled_lora is None:
            raise ValueError("fuse_distilled_into_stage1 requires distilled_lora")
        if (
            fuse_distilled_into_stage1
            and official_stage2
            and spatial_upsampler_path is not None
        ):
            raise ValueError(
                "fuse_distilled_into_stage1 is only for stage-1-only; "
                "hybrid two-stage already uses official distilled LoRA on Stage 2"
            )
        if (
            distilled_lora is not None
            and spatial_upsampler_path is None
            and not fuse_distilled_into_stage1
        ):
            raise ValueError(
                "distilled_lora requires spatial_upsampler_path unless fuse_distilled_into_stage1"
            )
        if (
            official_stage2
            and spatial_upsampler_path is not None
            and distilled_lora is None
        ):
            raise ValueError(
                "official_stage2=True requires distilled_lora "
                "(Stage 2 = official transformer + official distilled LoRA)"
            )

        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.scheduler = LTX2Scheduler()
        self.distilled_lora = distilled_lora
        self.fuse_distilled_into_stage1 = fuse_distilled_into_stage1
        self.official_stage2 = official_stage2

        stage1_loras = (
            (distilled_lora,)
            if (fuse_distilled_into_stage1 and distilled_lora is not None)
            else ()
        )
        builder, self.sft_architecture = create_lora_transformer_builder(
            checkpoint_path,
            movie_lora_path,
            qwen_context_dim=qwen_context_dim,
            audio_qwen_context_dim=audio_qwen_context_dim,
            registry=registry,
            loras=stage1_loras,
        )

        self.prompt_encoder = PromptEncoder(
            checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=OffloadMode.NONE,
            # transformers 5.x flattened SiglipVisionModel / Gemma3 RoPE, so the
            # pipeline meta-builder's create_and_populate crashes. Reuse the
            # trainer's 8-bit from_pretrained path, which already runs in training.
            text_encoder_builder=_GemmaFromPretrainedBuilder(gemma_root),
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, self.device, registry=registry
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
            checkpoint_path, self.dtype, self.device, registry=registry
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path, self.dtype, self.device, registry=registry
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
        if spatial_upsampler_path is None or fuse_distilled_into_stage1:
            # Stage-1-only callers never touch a separate Stage 2 builder.
            self.stage_2 = self.stage
        elif official_stage2:
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
            stage_2_builder, _ = create_lora_transformer_builder(
                checkpoint_path,
                movie_lora_path,
                qwen_context_dim=qwen_context_dim,
                audio_qwen_context_dim=audio_qwen_context_dim,
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
        return self.stage.model_context()

    def stage_2_transformer_context(self) -> AbstractContextManager[torch.nn.Module]:
        return self.stage_2.model_context()

    def encode_prompt_pairs(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[GemmaConditioning]:
        unique_prompts = list(
            dict.fromkeys(prompt for pair in pairs for prompt in pair)
        )
        logger.info(f"Encoding {len(unique_prompts)} unique prompt(s) with Gemma")
        outputs = self.prompt_encoder(unique_prompts)
        by_prompt = dict(zip(unique_prompts, outputs, strict=True))
        result: list[GemmaConditioning] = []
        for positive, negative in pairs:
            positive_output = by_prompt[positive]
            negative_output = by_prompt[negative]
            if (
                positive_output.audio_encoding is None
                or negative_output.audio_encoding is None
            ):
                raise ValueError(
                    "The official checkpoint does not provide audio prompt encodings"
                )
            result.append(
                GemmaConditioning(
                    video_positive=positive_output.video_encoding,
                    audio_positive=positive_output.audio_encoding,
                    video_negative=negative_output.video_encoding,
                    audio_negative=negative_output.audio_encoding,
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
    ) -> tuple[
        list[ConditioningItem], list[ConditioningItem], torch.Tensor, torch.Tensor
    ]:
        def encode(
            encoder: VideoEncoder,
        ) -> tuple[
            list[ConditioningItem], list[ConditioningItem], torch.Tensor, torch.Tensor
        ]:
            stage_1 = combined_image_conditionings(
                images=[image],
                height=stage_1_height,
                width=stage_1_width,
                video_encoder=encoder,
                dtype=self.dtype,
                device=self.device,
            )
            stage_2 = combined_image_conditionings(
                images=[image],
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
            first_frame_latent=(
                first_frame_latent
                if self.sft_architecture.first_frame_in_channels is not None
                else None
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
        if num_frames % 8 != 1:
            raise ValueError(
                f"num_frames must satisfy num_frames % 8 == 1, got {num_frames}"
            )
        if stage_1_steps < 1:
            raise ValueError(f"stage_1_steps must be >= 1, got {stage_1_steps}")
        if (
            stage_1_only
            and self.distilled_lora is not None
            and not self.fuse_distilled_into_stage1
        ):
            raise ValueError(
                "stage_1_only with distilled_lora requires fuse_distilled_into_stage1=True"
            )
        if not stage_1_only and self.upsampler is None:
            raise ValueError(
                "spatial_upsampler_path is required unless stage_1_only=True"
            )
        expected_ndim = 4 if self.sft_architecture.use_qwen_layer_adapters else 3
        if qwen_context.ndim != expected_ndim or qwen_context.shape[0] != 1:
            expected = (
                "[1, layers, seq, hidden]" if expected_ndim == 4 else "[1, seq, hidden]"
            )
            raise ValueError(
                f"qwen_context must have shape {expected}, got {qwen_context.shape}"
            )
        if self.sft_architecture.use_qwen_layer_adapters and (
            qwen_context.shape[1] != self.sft_architecture.qwen_layer_adapter_num_layers
        ):
            raise ValueError(
                f"Qwen layer count {qwen_context.shape[1]} does not match "
                f"{self.sft_architecture.qwen_layer_adapter_num_layers}"
            )
        if qwen_context.shape[-1] != self.sft_architecture.video_qwen_context_dim:
            raise ValueError(
                f"Qwen hidden size {qwen_context.shape[-1]} does not match "
                f"{self.sft_architecture.video_qwen_context_dim}"
            )
        expected_mask_shape = (qwen_context.shape[0], qwen_context.shape[-2])
        if tuple(qwen_context_mask.shape) != expected_mask_shape:
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
            assert_resolution(height=height, width=width, is_two_stage=False)

            def encode_full(
                encoder: VideoEncoder,
            ) -> tuple[list[ConditioningItem], torch.Tensor]:
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
                    if isinstance(item, VideoConditionByLatentIndex)
                    and item.latent_idx == 0
                ]
                if len(frame_zero) != 1:
                    raise ValueError(
                        "Exactly one frame-0 image conditioning is required"
                    )
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

            def run_full(
                transformer: torch.nn.Module,
            ) -> tuple[LatentState, LatentState]:
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
                        context=gemma.video_positive, conditionings=full_conditionings
                    ),
                    ModalitySpec(context=gemma.audio_positive),
                    max_batch_size=1,
                )
                if video_state is None or audio_state is None:
                    raise RuntimeError(
                        "Stage-1-only pass did not return both video and audio states"
                    )
                return video_state, audio_state

            if transformer is not None:
                video_state, audio_state = run_full(transformer)
            else:
                with self.stage.model_context() as stage_transformer:
                    video_state, audio_state = run_full(stage_transformer)
            return TwoStageLatents(video=video_state.latent, audio=audio_state.latent)

        stage_1_height, stage_1_width, stage_2_height, stage_2_width = (
            two_stage_pixel_sizes(height, width)
        )
        assert_resolution(
            height=stage_1_height, width=stage_1_width, is_two_stage=False
        )
        assert_resolution(height=stage_2_height, width=stage_2_width, is_two_stage=True)
        (
            stage_1_conditionings,
            stage_2_conditionings,
            stage_1_first_frame,
            stage_2_first_frame,
        ) = self._image_conditionings(
            image,
            stage_1_height=stage_1_height,
            stage_1_width=stage_1_width,
            stage_2_height=stage_2_height,
            stage_2_width=stage_2_width,
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

        def run_stage_1(
            transformer: torch.nn.Module,
        ) -> tuple[LatentState, LatentState]:
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
                    context=gemma.video_positive, conditionings=stage_1_conditionings
                ),
                ModalitySpec(context=gemma.audio_positive),
                max_batch_size=1,
            )
            if video_state is None or audio_state is None:
                raise RuntimeError("Stage 1 did not return both video and audio states")
            return video_state, audio_state

        if self.official_stage2:
            stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(
                device=self.device, dtype=torch.float32
            )
            stage_2_denoiser: object = SimpleDenoiser(
                v_context=gemma.video_positive,
                a_context=gemma.audio_positive,
            )
        else:
            stage_2_sigmas = make_stage_2_sigmas(stage_2_start_sigma, stage_2_steps).to(
                device=self.device, dtype=torch.float32
            )
            stage_2_denoiser = self._denoiser(
                gemma=gemma,
                qwen_context=qwen_context,
                qwen_context_mask=qwen_context_mask,
                first_frame_latent=stage_2_first_frame,
                guidance_scale=stage_2_guidance_scale
                if stage_2_guidance_scale is not None
                else guidance_scale,
                stg_scale=stage_2_stg_scale
                if stage_2_stg_scale is not None
                else stg_scale,
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
        split_stages = (
            self.official_stage2
            or reload_transformer_between_stages
            or (self.distilled_lora is not None and not self.fuse_distilled_into_stage1)
            or self.stage_2 is not self.stage
        )
        stage_2_module = (
            transformer_stage_2 if transformer_stage_2 is not None else transformer
        )
        if transformer is not None:
            if reload_transformer_between_stages:
                raise ValueError(
                    "reload_transformer_between_stages cannot be used with a caller-provided transformer"
                )
            if split_stages and stage_2_module is None:
                raise ValueError(
                    "transformer_stage_2 is required when Stage 2 uses a different builder"
                )
            if stage_2_module is None:
                stage_2_module = transformer
            stage_1_video, stage_1_audio = run_stage_1(transformer)
            upscaled_video = self.upsampler(stage_1_video.latent[:1])
            final_video, final_audio = run_stage_2(
                stage_2_module, upscaled_video, stage_1_audio.latent
            )
        elif split_stages:
            with self.stage.model_context() as stage_1_transformer:
                stage_1_video, stage_1_audio = run_stage_1(stage_1_transformer)
            upscaled_video = self.upsampler(stage_1_video.latent[:1])
            with self.stage_2.model_context() as stage_2_transformer:
                final_video, final_audio = run_stage_2(
                    stage_2_transformer, upscaled_video, stage_1_audio.latent
                )
        else:
            with self.stage.model_context() as stage_1_transformer:
                stage_1_video, stage_1_audio = run_stage_1(stage_1_transformer)
                upscaled_video = self.upsampler(stage_1_video.latent[:1])
                final_video, final_audio = run_stage_2(
                    stage_1_transformer, upscaled_video, stage_1_audio.latent
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
        generator = torch.Generator(device=self.device).manual_seed(seed)
        audio = self.audio_decoder(
            latents.audio.to(device=self.device, dtype=self.dtype)
        )
        video = _inference_iterator(
            self.video_decoder(
                latents.video.to(device=self.device, dtype=self.dtype),
                tiling_config,
                generator,
            )
        )
        return video, audio
