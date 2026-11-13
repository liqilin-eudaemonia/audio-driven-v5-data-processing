"""Text-to-video training strategy.
This strategy implements standard text-to-video generation training where:
- Only target latents are used (no reference videos)
- Standard noise application and loss computation
- Supports first frame conditioning
- Optionally supports joint audio-video training
"""

from typing import Any, Literal
import os

import torch
import torch.distributed as dist
from pydantic import Field, model_validator
from torch import Tensor

from ltx_core.distributed.comm import broadcast_tensor
from ltx_core.distributed.parallel_state import (
    get_sp_group,
    get_sp_rank,
    get_sp_world_size,
    is_sp_enabled,
)
from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    LossBreakdown,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToVideoConfig(TrainingStrategyConfigBase):
    """Configuration for text-to-video training strategy."""

    name: Literal["text_to_video"] = "text_to_video"

    i2av_only: bool = Field(
        default=False,
        description="Require every training sample to use video first-frame conditioning. "
        "The separate audio first-frame branch remains optional.",
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    with_audio: bool = Field(
        default=False,
        description="Whether to include audio in training (joint audio-video generation)",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True",
    )

    use_qwen_cond: bool = Field(
        default=False,
        description="Whether to load cached Qwen/Qwen-VL condition tokens from a separate precompute directory",
    )

    qwen_conditions_dir: str = Field(
        default="qwen_conditions",
        description="Directory name for cached Qwen/Qwen-VL condition tensors when use_qwen_cond is True",
    )

    qwen_cond_for_audio: bool = Field(
        default=True,
        description="Whether to also feed Qwen/Qwen-VL condition tokens to the audio transformer branch",
    )

    qwen_cross_attention_dim: int = Field(
        default=2560,
        description="Feature dimension of cached Qwen/Qwen-VL condition tokens",
        ge=1,
    )

    audio_qwen_cross_attention_dim: int | None = Field(
        default=None,
        description="Optional feature dimension for audio Qwen/Qwen-VL condition tokens. Defaults to qwen_cross_attention_dim.",
    )

    # --- Online condition encoding (Gemma + Qwen fully online, no conditions/ or
    # qwen_conditions/ caches). Requires data.metadata_jsonls. ---

    online_conditioning: bool = Field(
        default=False,
        description="Legacy default for online Gemma text and Qwen first-frame encoding. "
        "The per-encoder online_*_conditioning options override this value when set.",
    )

    online_text_conditioning: bool | None = Field(
        default=None,
        description="Override online_conditioning for Gemma text encoding. False loads "
        "precomputed conditions/ features and keeps only the embedding connectors resident.",
    )

    online_qwen_conditioning: bool | None = Field(
        default=None,
        description="Override online_conditioning for Qwen first-frame encoding. False loads "
        "precomputed qwen_conditions/ features when use_qwen_cond is enabled.",
    )

    qwen_model_path: str | None = Field(
        default=None,
        description="Qwen3-VL checkpoint directory for online condition extraction. "
        "Required when online_conditioning and use_qwen_cond are both true.",
    )

    qwen_max_length: int = Field(
        default=2560,
        description="Fixed Qwen sequence length for online extraction (right truncate / right pad).",
        ge=1,
    )

    qwen_image_width: int = Field(
        default=1088,
        description="Qwen first-frame canvas width. Should equal the VAE input width so the "
        "condition is pixel-aligned with the encoded latents.",
        ge=32,
    )

    qwen_image_height: int = Field(
        default=640,
        description="Qwen first-frame canvas height. Should equal the VAE input height.",
        ge=32,
    )

    qwen_dropout: float = Field(
        default=0.1,
        description="Probability of dropping the Qwen condition for a whole training micro-batch "
        "(branch skipped entirely, matching the CFG negative pass at inference).",
        ge=0.0,
        le=1.0,
    )

    qwen_feature_layer_mode: Literal["last", "all"] = Field(
        default="last",
        description="Which Qwen decoder layer features to extract online. 'all' returns "
        "[B, layers, seq, hidden] and is intended for trainable Qwen layer adapters.",
    )

    use_qwen_layer_adapters: bool = Field(
        default=False,
        description="Fuse all online Qwen decoder layers through separate trainable video/audio adapters "
        "before the existing Qwen cross-attention branches.",
    )

    qwen_layer_adapter_num_layers: int = Field(
        default=36,
        description="Number of Qwen decoder layers expected by the layer adapters.",
        ge=1,
    )

    qwen_layer_adapter_input_dim: int | None = Field(
        default=None,
        description="Input hidden size of raw Qwen layer features. Defaults to qwen_cross_attention_dim.",
    )

    use_omni_cond: bool = Field(
        default=False,
        description="Inject offline Qwen3-Omni planner hidden states through independent video/audio branches.",
    )

    omni_conditions_dir: str = Field(
        default="omni_conditions",
        description="Directory containing versioned offline Omni planner condition packages.",
    )
    omni_expected_route: Literal["route_a_no_context", "route_b_context"] = Field(
        default="route_a_no_context",
        description="Planner route required for every offline Omni condition package.",
    )

    omni_cond_for_audio: bool = Field(
        default=True,
        description="Also inject Omni planner hidden states into the audio tower.",
    )

    omni_cross_attention_dim: int = Field(
        default=1024,
        description="Hidden dimension of normalized Omni plan tokens.",
        ge=1,
    )

    audio_omni_cross_attention_dim: int | None = Field(
        default=None,
        description="Optional audio Omni context dimension; defaults to omni_cross_attention_dim.",
    )

    use_omni_layer_adapters: bool = Field(
        default=False,
        description="Use independent video/audio adapters to fuse the Omni depth axis.",
    )

    omni_layer_adapter_num_layers: int = Field(default=4, ge=1)
    omni_layer_adapter_input_dim: int = Field(default=1024, ge=1)
    use_omni_field_resampler: bool = Field(
        default=False,
        description="Train a field-aware Resampler on raw multi-layer Omni Plan tokens.",
    )
    omni_field_resampler_input_dim: int = Field(default=2048, ge=1)
    omni_field_resampler_num_heads: int = Field(default=8, ge=1)

    omni_dropout: float = Field(
        default=0.1,
        description="Whole-micro-batch semantic Omni dropout; identity/timbre conditions are retained.",
        ge=0.0,
        le=1.0,
    )
    omni_monitor_interval: int = Field(
        default=20,
        ge=0,
        description="Optimization-step interval for Omni layer-weight, gradient, and "
        "representative-block residual monitoring; 0 disables it.",
    )
    omni_residual_monitor_blocks: list[int] = Field(
        default_factory=lambda: [0, 15, 31, 47],
        description="Transformer block indices sampled for low-overhead Omni residual ratios.",
    )

    use_listener_timbre_cond: bool = Field(
        default=False,
        description="Inject a precomputed listener voice embedding into the audio tower only.",
    )

    listener_timbre_embeddings_dir: str = Field(
        default="listener_timbre_embeddings",
        description="Directory containing precomputed listener timbre embedding packages.",
    )

    listener_timbre_dim: int = Field(
        default=192,
        description="Input dimension of the configured speaker/timbre encoder.",
        ge=1,
    )

    use_first_frame_audio_cond: bool = Field(
        default=False,
        description="Add a first-frame (video latent) cross-attention condition branch on the "
        "audio stream, mirroring UniAV's 3D-VAE first-frame conditioning of the audio tower.",
    )

    first_frame_dropout: float = Field(
        default=0.0,
        description="Probability of dropping the first-frame audio condition for a whole "
        "training micro-batch. Inference keeps the condition in both CFG passes, so 0 is the default.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_i2av_only(self) -> "TextToVideoConfig":
        if any(index < 0 for index in self.omni_residual_monitor_blocks):
            raise ValueError("omni_residual_monitor_blocks must contain non-negative indices")
        if len(set(self.omni_residual_monitor_blocks)) != len(
            self.omni_residual_monitor_blocks
        ):
            raise ValueError("omni_residual_monitor_blocks must not contain duplicates")
        if self.use_omni_cond and not uses_online_text_conditioning(self):
            raise ValueError(
                "use_omni_cond=true requires online_text_conditioning=true so "
                "audio_prompt/video_prompt use the native modality-specific connectors"
            )
        if self.use_omni_layer_adapters and not self.use_omni_cond:
            raise ValueError(
                "use_omni_layer_adapters=true requires use_omni_cond=true"
            )
        if self.use_omni_field_resampler and not self.use_omni_layer_adapters:
            raise ValueError(
                "use_omni_field_resampler=true requires use_omni_layer_adapters=true"
            )
        if (
            self.use_omni_field_resampler
            and self.omni_layer_adapter_input_dim
            % self.omni_field_resampler_num_heads
        ):
            raise ValueError(
                "omni_layer_adapter_input_dim must be divisible by "
                "omni_field_resampler_num_heads"
            )
        if (
            self.use_omni_field_resampler
            and self.omni_field_resampler_num_heads != 8
        ):
            raise ValueError(
                "Checkpoint-compatible Omni field Resampler currently requires 8 heads"
            )
        if self.use_listener_timbre_cond and not self.with_audio:
            raise ValueError(
                "use_listener_timbre_cond=true requires with_audio=true"
            )
        if not self.i2av_only:
            return self
        invalid = []
        if self.first_frame_conditioning_p != 1.0:
            invalid.append("first_frame_conditioning_p must be 1.0")
        if not self.with_audio:
            invalid.append("with_audio must be true")
        if self.use_first_frame_audio_cond and self.first_frame_dropout != 0.0:
            invalid.append("first_frame_dropout must be 0.0")
        if invalid:
            raise ValueError("i2av_only=true requires: " + "; ".join(invalid))
        return self


def uses_online_text_conditioning(config: Any) -> bool:
    """Resolve the Gemma online mode while preserving legacy config behavior."""
    override = getattr(config, "online_text_conditioning", None)
    if override is not None:
        return bool(override)
    return bool(getattr(config, "online_conditioning", False))


def uses_online_qwen_conditioning(config: Any) -> bool:
    """Resolve the Qwen online mode while preserving legacy config behavior."""
    override = getattr(config, "online_qwen_conditioning", None)
    if override is not None:
        return bool(override)
    return bool(getattr(config, "online_conditioning", False))


class TextToVideoStrategy(TrainingStrategy):
    """Text-to-video training strategy.
    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Supports first frame conditioning
    - Optionally supports joint audio-video training when with_audio=True
    """

    config: TextToVideoConfig

    def __init__(self, config: TextToVideoConfig):
        """Initialize strategy with configuration.
        Args:
            config: Text-to-video configuration
        """
        super().__init__(config)

    @property
    def requires_audio(self) -> bool:
        """Whether this training strategy requires audio components."""
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        """
        Text-to-video training requires latents and text conditions.
        When with_audio is True, also requires audio latents.
        """
        sources = {
            "latents": "latents",
            "conditions": "conditions",
        }

        if uses_online_text_conditioning(self.config):
            # Gemma text conditions are computed online in the trainer.
            sources.pop("conditions")

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        if self.config.use_qwen_cond and not uses_online_qwen_conditioning(self.config):
            sources[self.config.qwen_conditions_dir] = "qwen_conditions"

        if self.config.use_omni_cond:
            sources[self.config.omni_conditions_dir] = "omni_conditions"

        if self.config.use_listener_timbre_cond:
            if not self.config.with_audio:
                raise ValueError("listener timbre conditioning requires with_audio=true")
            sources[self.config.listener_timbre_embeddings_dir] = (
                "listener_timbre_embedding"
            )

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Prepare inputs for text-to-video training."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["latents"]
        video_latents = latents["latents"]

        # Get video dimensions (assume same for all batch elements)
        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # First-frame audio conditioning: the first latent time-slice IS the encoded
        # first frame (num_frames % 8 == 1), so no extra VAE pass is needed.
        first_frame_latent = None
        if self.config.with_audio and self.config.use_first_frame_audio_cond:
            drop = False
            if self.config.first_frame_dropout > 0:
                drop_draw = torch.rand(1, device=video_latents.device)
                if is_sp_enabled():
                    broadcast_tensor(drop_draw, group=get_sp_group())
                drop = bool(
                    drop_draw.item() < self.config.first_frame_dropout
                )
            if not drop:
                first_frame_latent = video_latents[:, :, 0:1].clone()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        audio_prompt_embeds = conditions["audio_prompt_embeds"]
        video_prompt_attention_mask = conditions.get(
            "video_prompt_attention_mask", conditions["prompt_attention_mask"]
        )
        audio_prompt_attention_mask = conditions.get(
            "audio_prompt_attention_mask", conditions["prompt_attention_mask"]
        )

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        (
            qwen_context,
            qwen_context_mask,
            audio_qwen_context,
            audio_qwen_context_mask,
        ) = self._prepare_qwen_conditions(
            batch=batch,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        (
            omni_context,
            omni_context_mask,
            omni_field_mask,
            audio_omni_context,
            audio_omni_context_mask,
            audio_omni_field_mask,
        ) = self._prepare_omni_conditions(
            batch=batch,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        listener_timbre_context = self._prepare_listener_timbre_condition(
            batch=batch,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self.config.first_frame_conditioning_p,
        )
        if is_sp_enabled():
            broadcast_tensor(video_conditioning_mask, group=get_sp_group())
        if self.config.i2av_only:
            first_frame_token_count = height * width
            if not torch.all(video_conditioning_mask[:, :first_frame_token_count]):
                raise RuntimeError(
                    "i2av_only invariant violated: every sample must keep video first-frame conditioning"
                )
            if self.config.use_first_frame_audio_cond and first_frame_latent is None:
                raise RuntimeError(
                    "first-frame audio conditioning is enabled but its latent is missing"
                )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)
        if is_sp_enabled():
            broadcast_tensor(sigmas, group=get_sp_group())
            broadcast_tensor(video_noise, group=get_sp_group())
        if os.environ.get("LTX_CP_FINITE_DEBUG") == "1":
            for name, value in (
                ("clean video latent", video_latents),
                ("sigma", sigmas),
                ("video noise", video_noise),
            ):
                if not torch.isfinite(value).all():
                    raise FloatingPointError(
                        f"Non-finite {name} before noising: "
                        f"{int((~torch.isfinite(value)).sum().item())}/{value.numel()}"
                    )

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (
            1 - sigmas_expanded
        ) * video_latents + sigmas_expanded * video_noise
        if (
            os.environ.get("LTX_CP_FINITE_DEBUG") == "1"
            and not torch.isfinite(noisy_video).all()
        ):
            raise FloatingPointError(
                "Non-finite video latent immediately after noising: "
                f"{int((~torch.isfinite(noisy_video)).sum().item())}/{noisy_video.numel()}; "
                f"sigma={sigmas.detach().float().cpu().tolist()}"
            )

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(
            conditioning_mask_expanded, video_latents, noisy_video
        )

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._create_per_token_timesteps(
            video_conditioning_mask, sigmas.squeeze()
        )

        # Generate video positions using ltx_core's native implementation
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask
        video_attention_mask = None

        if is_sp_enabled():
            sp_size = get_sp_world_size()
            sp_rank = get_sp_rank()
            pad_tokens = (-video_seq_len) % sp_size
            padded_seq_len = video_seq_len + pad_tokens
            if pad_tokens:
                noisy_video = torch.nn.functional.pad(
                    noisy_video, (0, 0, 0, pad_tokens)
                )
                video_targets = torch.nn.functional.pad(
                    video_targets, (0, 0, 0, pad_tokens)
                )
                video_timesteps = torch.nn.functional.pad(
                    video_timesteps, (0, pad_tokens)
                )
                video_positions = torch.nn.functional.pad(
                    video_positions, (0, 0, 0, pad_tokens)
                )
                video_loss_mask = torch.nn.functional.pad(
                    video_loss_mask, (0, pad_tokens), value=False
                )
                video_attention_mask = torch.nn.functional.pad(
                    torch.ones(
                        (batch_size, video_seq_len),
                        dtype=torch.bool,
                        device=device,
                    ),
                    (0, pad_tokens),
                    value=False,
                )

            chunk_size = padded_seq_len // sp_size
            start = sp_rank * chunk_size
            end = (sp_rank + 1) * chunk_size

            noisy_video = noisy_video[:, start:end, :].contiguous()
            video_targets = video_targets[:, start:end, :].contiguous()
            video_timesteps = video_timesteps[:, start:end].contiguous()
            video_positions = video_positions[:, :, start:end, :].contiguous()
            video_loss_mask = video_loss_mask[:, start:end].contiguous()

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            sigma=sigmas,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=video_prompt_attention_mask,
            attention_mask=video_attention_mask,
            qwen_context=qwen_context,
            qwen_context_mask=qwen_context_mask,
            omni_context=omni_context,
            omni_context_mask=omni_context_mask,
            omni_field_mask=omni_field_mask,
        )

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=audio_prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                qwen_context=audio_qwen_context,
                qwen_context_mask=audio_qwen_context_mask,
                omni_context=audio_omni_context,
                omni_context_mask=audio_omni_context_mask,
                omni_field_mask=audio_omni_field_mask,
                listener_timbre_context=listener_timbre_context,
                first_frame_latent=first_frame_latent,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        sigmas: Tensor,
        audio_prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        qwen_context: Tensor | None = None,
        qwen_context_mask: Tensor | None = None,
        omni_context: Tensor | None = None,
        omni_context_mask: Tensor | None = None,
        omni_field_mask: Tensor | None = None,
        listener_timbre_context: Tensor | None = None,
        first_frame_latent: Tensor | None = None,
    ) -> tuple[Modality, Tensor, Tensor]:
        """Prepare audio inputs for joint audio-video training.
        Args:
            batch: Raw batch data containing audio_latents
            sigmas: Sampled sigma values (same as video)
            audio_prompt_embeds: Audio context embeddings
            prompt_attention_mask: Attention mask for context
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
        Returns:
            Tuple of (audio_modality, audio_targets, audio_loss_mask)
        """
        # Get audio latents - dataset provides uniform non-patchified format [B, C, T, F]
        audio_data = batch["audio_latents"]
        audio_latents = audio_data["latents"]

        # Patchify audio latents: [B, C, T, F] -> [B, T, C*F]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        audio_seq_len = audio_latents.shape[1]

        # Sample audio noise
        audio_noise = torch.randn_like(audio_latents)
        if is_sp_enabled():
            broadcast_tensor(audio_noise, group=get_sp_group())

        # Apply noise to audio (same sigma as video)
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_audio = (
            1 - sigmas_expanded
        ) * audio_latents + sigmas_expanded * audio_noise

        # Compute audio targets
        audio_targets = audio_noise - audio_latents

        # Audio timesteps: all tokens use the sampled sigma (no conditioning mask)
        audio_timesteps = sigmas.view(-1, 1).expand(-1, audio_seq_len)

        # Generate audio positions
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        # Create audio Modality
        audio_modality = Modality(
            enabled=True,
            latent=noisy_audio,
            sigma=sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_prompt_embeds,
            context_mask=prompt_attention_mask,
            qwen_context=qwen_context,
            qwen_context_mask=qwen_context_mask,
            omni_context=omni_context,
            omni_context_mask=omni_context_mask,
            omni_field_mask=omni_field_mask,
            listener_timbre_context=listener_timbre_context,
            first_frame_latent=(
                first_frame_latent.to(device=device, dtype=dtype)
                if first_frame_latent is not None
                else None
            ),
        )

        # Audio loss mask: all tokens contribute to loss (no conditioning)
        audio_loss_mask = torch.ones(
            batch_size, audio_seq_len, dtype=torch.bool, device=device
        )

        return audio_modality, audio_targets, audio_loss_mask

    def _prepare_qwen_conditions(
        self,
        batch: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Load optional cached Qwen/Qwen-VL contexts for the independent cross-attention branch."""
        if not self.config.use_qwen_cond:
            return None, None, None, None

        if "qwen_conditions" not in batch:
            if uses_online_qwen_conditioning(self.config):
                # Online mode: the trainer omits qwen_conditions when the whole micro-batch
                # is dropped (qwen_dropout), matching the CFG negative pass at inference.
                return None, None, None, None
            raise ValueError(
                "use_qwen_cond=True requires a qwen_conditions data source"
            )

        qwen_conditions = batch["qwen_conditions"]
        qwen_context = self._select_qwen_tensor(
            qwen_conditions,
            ("qwen_video_context", "qwen_context", "qwen_hidden_states"),
        )
        if qwen_context is None:
            raise ValueError(
                "qwen_conditions must contain one of: qwen_video_context, qwen_context, qwen_hidden_states"
            )

        qwen_context = self._normalize_qwen_context(
            qwen_context, batch_size, device, dtype, "qwen_context"
        )
        qwen_context_mask = self._normalize_qwen_mask(
            self._select_qwen_tensor(
                qwen_conditions, ("qwen_attention_mask", "qwen_context_mask")
            ),
            batch_size=batch_size,
            seq_len=qwen_context.shape[-2],
            device=device,
            name="qwen_attention_mask",
        )

        audio_qwen_context = None
        audio_qwen_context_mask = None
        if self.config.with_audio and self.config.qwen_cond_for_audio:
            audio_qwen_context = self._select_qwen_tensor(
                qwen_conditions, ("qwen_audio_context",)
            )
            if audio_qwen_context is None:
                audio_qwen_context = qwen_context
                audio_qwen_context_mask = qwen_context_mask
            else:
                audio_qwen_context = self._normalize_qwen_context(
                    audio_qwen_context,
                    batch_size,
                    device,
                    dtype,
                    "qwen_audio_context",
                )
                audio_qwen_context_mask = self._normalize_qwen_mask(
                    self._select_qwen_tensor(
                        qwen_conditions,
                        ("qwen_audio_attention_mask", "qwen_audio_context_mask"),
                    ),
                    batch_size=batch_size,
                    seq_len=audio_qwen_context.shape[-2],
                    device=device,
                    name="qwen_audio_attention_mask",
                )

        return (
            qwen_context,
            qwen_context_mask,
            audio_qwen_context,
            audio_qwen_context_mask,
        )

    def _prepare_omni_conditions(
        self,
        batch: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        """Prepare versioned offline Omni plan tokens for the two towers."""
        if not self.config.use_omni_cond:
            return None, None, None, None, None, None
        conditions = batch.get("omni_conditions")
        if not isinstance(conditions, dict):
            raise ValueError(
                "use_omni_cond=True requires a validated omni_conditions package"
            )
        if self.config.omni_dropout > 0:
            drop_draw = torch.rand(1, device=device)
            if is_sp_enabled():
                broadcast_tensor(drop_draw, group=get_sp_group())
            if drop_draw.item() < self.config.omni_dropout:
                return None, None, None, None, None, None

        context = conditions.get("plan_latent")
        if not isinstance(context, Tensor):
            raise ValueError("omni_conditions.plan_latent must be a tensor")
        # Loader normalizes each sample to [layers, seq, hidden]; collate adds B.
        if context.ndim != 4:
            raise ValueError(
                "Omni plan_latent must have shape [B,layers,seq,hidden], got "
                f"{tuple(context.shape)}"
            )
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Omni batch mismatch: got {context.shape[0]}, expected {batch_size}"
            )
        expected_hidden = (
            self.config.omni_field_resampler_input_dim
            if self.config.use_omni_field_resampler
            else self.config.omni_layer_adapter_input_dim
        )
        if context.shape[-1] != expected_hidden:
            raise ValueError(
                "Omni hidden size mismatch: got "
                f"{context.shape[-1]}, expected {expected_hidden}"
            )
        if context.shape[1] != self.config.omni_layer_adapter_num_layers:
            raise ValueError(
                "Omni layer count mismatch: got "
                f"{context.shape[1]}, expected {self.config.omni_layer_adapter_num_layers}"
            )
        context = context.to(device=device, dtype=dtype)
        mask = conditions.get(
            "omni_attention_mask", conditions.get("attention_mask")
        )
        if mask is not None and not isinstance(mask, Tensor):
            raise ValueError("omni_conditions.attention_mask must be a tensor")
        mask = self._normalize_qwen_mask(
            mask,
            batch_size=batch_size,
            seq_len=context.shape[-2],
            device=device,
            name="omni_attention_mask",
        )
        field_mask = conditions.get("plan_field_mask")
        if self.config.use_omni_field_resampler:
            if not isinstance(field_mask, Tensor):
                raise ValueError(
                    "use_omni_field_resampler=True requires plan_field_mask"
                )
            if field_mask.shape != (batch_size, 3, context.shape[-2]):
                raise ValueError(
                    "Omni plan_field_mask must have shape [B,3,T], got "
                    f"{tuple(field_mask.shape)}"
                )
            field_mask = field_mask.to(device=device, dtype=torch.bool)
        elif field_mask is not None:
            raise ValueError(
                "Raw Omni field masks require use_omni_field_resampler=True"
            )
        if self.config.with_audio and self.config.omni_cond_for_audio:
            return context, mask, field_mask, context, mask, field_mask
        return context, mask, field_mask, None, None, None

    def _prepare_listener_timbre_condition(
        self,
        batch: dict[str, Any],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Normalize a precomputed voice embedding without exposing it to video."""
        if not self.config.use_listener_timbre_cond:
            return None
        value: Any = batch.get("listener_timbre_embedding")
        if isinstance(value, dict):
            value = value.get("listener_timbre_embedding", value.get("embedding"))
        if not isinstance(value, Tensor):
            raise ValueError(
                "use_listener_timbre_cond=True requires listener_timbre_embedding"
            )
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[0] != batch_size:
            raise ValueError(
                "listener timbre must have shape [B,D] or [B,T,D], got "
                f"{tuple(value.shape)}"
            )
        if value.shape[-1] != self.config.listener_timbre_dim:
            raise ValueError(
                "listener timbre dimension mismatch: got "
                f"{value.shape[-1]}, expected {self.config.listener_timbre_dim}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("listener timbre embedding contains NaN/Inf")
        return value.to(device=device, dtype=dtype)

    @staticmethod
    def _select_qwen_tensor(
        data: dict[str, Any], keys: tuple[str, ...]
    ) -> Tensor | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, Tensor):
                return value
        return None

    @staticmethod
    def _normalize_qwen_context(
        context: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Tensor:
        if context.ndim == 4:
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"{name} batch size mismatch: got {context.shape[0]}, expected {batch_size}"
                )
            return context.to(device=device, dtype=dtype)
        if context.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, seq, hidden] or [B, layers, seq, hidden], got {context.shape}"
            )
        if context.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size mismatch: got {context.shape[0]}, expected {batch_size}"
            )
        return context.to(device=device, dtype=dtype)

    @staticmethod
    def _normalize_qwen_mask(
        mask: Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        name: str,
    ) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        if mask.ndim == 3:
            mask = mask[:, -1]
        if mask.ndim != 2:
            raise ValueError(
                f"{name} must have shape [B, seq] or [B, layers, seq], got {mask.shape}"
            )
        if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
            raise ValueError(
                f"{name} shape mismatch: got {mask.shape}, expected [{batch_size}, {seq_len}]"
            )
        return mask.to(device=device, dtype=torch.long)

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> LossBreakdown:
        """Compute masked MSE loss for video and optionally audio."""
        # Video loss: each CP rank owns a sequence shard. Normalize local sums by
        # the full-sequence valid count and multiply by SP size because FSDP later
        # averages gradients across the duplicated CP ranks.
        video_error = (video_pred - inputs.video_targets).pow(2)
        video_loss_mask = inputs.video_loss_mask.unsqueeze(-1)
        masked = torch.where(
            video_loss_mask, video_error, torch.zeros_like(video_error)
        )
        local_sum = masked.sum(dim=[-2, -1])
        valid_elements = (
            inputs.video_loss_mask.sum(dim=-1).to(local_sum.dtype)
            * video_error.shape[-1]
        )
        if is_sp_enabled():
            dist.all_reduce(valid_elements, group=get_sp_group())
            video_loss = (
                local_sum
                / valid_elements.clamp(min=1)
                * get_sp_world_size()
            )
        else:
            video_loss = local_sum / valid_elements.clamp(min=1)

        # If no audio, return video loss only
        if (
            not self.config.with_audio
            or audio_pred is None
            or inputs.audio_targets is None
        ):
            return LossBreakdown(loss=video_loss, components={"video_loss": video_loss})

        # Audio loss: per-element mean over (seq, channels), [B,]
        audio_loss = (audio_pred - inputs.audio_targets).pow(2).mean(dim=[-2, -1])

        # Combined loss [B,]
        return LossBreakdown(
            loss=video_loss + audio_loss,
            components={
                "video_loss": video_loss,
                "audio_loss": audio_loss,
            },
        )
