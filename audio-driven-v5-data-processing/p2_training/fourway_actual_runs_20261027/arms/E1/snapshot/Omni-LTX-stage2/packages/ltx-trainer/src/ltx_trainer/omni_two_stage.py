"""Standalone x2 latent super-resolution; independent of training validation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import torch
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS

from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler


def validate_two_stage(config: GenerationConfig, sigmas: torch.Tensor) -> None:
    if config.height < 64 or config.width < 64 or config.height % 64 or config.width % 64:
        raise ValueError("x2 two-stage output height/width must be multiples of 64; no implicit resize is performed")
    if config.num_frames < 1 or config.num_frames % 8 != 1:
        raise ValueError("num_frames must be positive and satisfy F % 8 == 1")
    if not math.isfinite(config.frame_rate) or config.frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive")
    if config.num_inference_steps < 1 or not math.isfinite(config.guidance_scale) or config.guidance_scale < 1:
        raise ValueError("Invalid first-stage steps or CFG")
    if not math.isfinite(config.stg_scale) or config.stg_scale < 0:
        raise ValueError("Invalid first-stage STG")
    if config.qwen_context is not None or config.audio_qwen_context is not None:
        raise ValueError("This standalone pipeline supports Omni conditions only; Qwen-VL is not wired to refinement")
    if config.reference_video is not None:
        raise ValueError(
            "Two-stage Omni inference currently supports text/first-frame conditioning, not reference_video"
        )
    if sigmas.ndim != 1 or len(sigmas) < 2 or not torch.isfinite(sigmas).all():
        raise ValueError("Refinement sigmas must be a finite 1-D sequence")
    if not (0 < float(sigmas[0]) < 1) or float(sigmas[-1]) != 0 or not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("Refinement sigmas must decrease strictly from (0,1) to zero")


class OmniTwoStageSampler(ValidationSampler):
    """Reuse Stage-1 sampling semantics; refine at x2 with all Omni conditions intact.

    replace_transformer is called AFTER low-resolution sampling and before upscale.
    It must return a freshly loaded base + distilled LoRA + same movie LoRA model.
    It must not mutate a model owned by a running trainer.
    """

    def __init__(
        self,
        *,
        upsample: Callable,
        replace_transformer: Callable,
        refinement_sigmas: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._upsample = upsample
        self._replace_transformer = replace_transformer
        self._refinement_sigmas = (
            STAGE_2_DISTILLED_SIGMAS.clone() if refinement_sigmas is None else refinement_sigmas.detach().clone()
        )

    @torch.no_grad()
    def generate(self, config: GenerationConfig, device: torch.device | str):
        device = torch.device(device)
        self._validate_config(config)
        validate_two_stage(config, self._refinement_sigmas)
        low = replace(config, height=config.height // 2, width=config.width // 2)
        embeddings = self._get_prompt_embeddings(config, device)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        noiser = GaussianNoiser(generator)
        video_tools = self._create_video_latent_tools(low)
        audio_tools = self._create_audio_latent_tools(low) if config.generate_audio else None
        vc = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        ac = audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        first = None
        if config.condition_image is not None:
            vc, encoded = self._apply_image_conditioning(vc, config.condition_image, low, device)
            first = encoded if config.first_frame_audio_cond else None
        print(f"Stage 1: {low.width}x{low.height}, {config.num_inference_steps} steps", flush=True)
        video, audio = self._run_denoising(
            config=low,
            video_state=noiser(vc),
            audio_state=noiser(ac) if ac is not None else None,
            video_clean_state=vc,
            audio_clean_state=ac,
            v_ctx_pos=embeddings[0],
            a_ctx_pos=embeddings[1],
            v_ctx_neg=embeddings[2],
            a_ctx_neg=embeddings[3],
            device=device,
            first_frame_latent=first,
        )
        low_latent = video_tools.unpatchify(video_tools.clear_conditioning(video)).latent
        audio_latent = (
            audio_tools.unpatchify(audio_tools.clear_conditioning(audio)).latent if audio is not None else None
        )
        # Drop the low-res model before constructing the distilled variant.
        self._transformer = self._replace_transformer(self._transformer)
        upscaled = self._upsample(low_latent)
        high_tools = self._create_video_latent_tools(config)
        expected = high_tools.target_shape.to_torch_shape()
        if tuple(upscaled.shape) != tuple(expected) or not torch.isfinite(upscaled).all():
            raise ValueError(f"Upsampler output must be finite {tuple(expected)}, got {tuple(upscaled.shape)}")
        vc = high_tools.create_initial_state(
            device=device, dtype=torch.bfloat16, initial_latent=upscaled.to(torch.bfloat16)
        )
        ac = (
            audio_tools.create_initial_state(
                device=device, dtype=torch.bfloat16, initial_latent=audio_latent.to(torch.bfloat16)
            )
            if audio_latent is not None
            else None
        )
        first = None
        if config.condition_image is not None:
            vc, encoded = self._apply_image_conditioning(vc, config.condition_image, config, device)
            first = encoded if config.first_frame_audio_cond else None
        sigmas = self._refinement_sigmas.to(device=device, dtype=torch.float32)
        print(f"Stage 2: {config.width}x{config.height}, {len(sigmas) - 1} distilled refinement steps", flush=True)
        video, audio = self._refine(
            config,
            noiser(vc, float(sigmas[0])),
            noiser(ac, float(sigmas[0])) if ac is not None else None,
            vc,
            ac,
            embeddings[0],
            embeddings[1],
            first,
            sigmas,
            device,
        )
        decoded = self._decode_video(
            high_tools.unpatchify(high_tools.clear_conditioning(video)), device, config.tiled_decoding
        )
        decoded_audio = (
            self._decode_audio(audio_tools.unpatchify(audio_tools.clear_conditioning(audio)), device)
            if audio is not None
            else None
        )
        return decoded, decoded_audio

    def _refine(self, config, vs, aus, vc, ac, video_text, audio_text, first, sigmas, device):
        contexts = self._move_omni_conditions(
            config,
            device=device,
            video_dtype=vs.latent.dtype,
            audio_dtype=aus.latent.dtype if aus is not None else None,
        )
        vctx, vmask, vfields, actx, amask, afields = contexts
        model = X0Model(self._transformer.to(device).eval())
        stepper = EulerDiffusionStep()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i, sigma in enumerate(sigmas[:-1]):
                v = Modality(
                    enabled=True,
                    latent=vs.latent,
                    sigma=sigma.repeat(vs.latent.shape[0]),
                    timesteps=sigma * vs.denoise_mask,
                    positions=vs.positions,
                    context=video_text,
                    context_mask=None,
                    omni_context=vctx,
                    omni_context_mask=vmask,
                    omni_field_mask=vfields,
                )
                a = None
                if aus is not None:
                    a = Modality(
                        enabled=True,
                        latent=aus.latent,
                        sigma=sigma.repeat(aus.latent.shape[0]),
                        timesteps=sigma * aus.denoise_mask,
                        positions=aus.positions,
                        context=audio_text,
                        context_mask=None,
                        omni_context=actx,
                        omni_context_mask=amask,
                        omni_field_mask=afields,
                        listener_timbre_context=config.listener_timbre_context.to(device)
                        if config.listener_timbre_context is not None
                        else None,
                        first_frame_latent=first,
                    )
                pv, pa = model(video=v, audio=a, perturbations=None)
                pv = pv * vs.denoise_mask + vc.latent.float() * (1 - vs.denoise_mask)
                vs = replace(vs, latent=stepper.step(sample=v.latent, denoised_sample=pv, sigmas=sigmas, step_index=i))
                if aus is not None:
                    pa = pa * aus.denoise_mask + ac.latent.float() * (1 - aus.denoise_mask)
                    aus = replace(
                        aus, latent=stepper.step(sample=a.latent, denoised_sample=pa, sigmas=sigmas, step_index=i)
                    )
        return vs, aus
