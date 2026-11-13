from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B* is
            the batch size, *T* is the total number of tokens (noisy +
            conditioning), and *D* is the input dimension.
        timesteps: Per-token timestep embeddings, shape ``(B, T)``.
        positions: Positional coordinates, shape ``(B, 3, T)`` for video
            (time, height, width) or ``(B, 1, T)`` for audio.
        context: Text conditioning embeddings from the prompt encoder.
        qwen_context: Optional cached Qwen/Qwen-VL condition tokens for an
            independent cross-attention branch.
        qwen_context_mask: Optional mask for the Qwen condition tokens.
        omni_context: Optional Qwen3-Omni planner tokens for independent
            video/audio cross-attention branches.
        omni_context_mask: Optional mask for the Omni planner tokens.
        omni_field_mask: Optional ``[B, 3, T]`` local token masks for the
            trainable field-aware Omni resampler.
        listener_timbre_context: Optional listener timbre tokens. These are
            consumed only by the audio stream.
        enabled: Whether this modality is active in the current forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional 2-D self-attention mask, shape ``(B, T, T)``.
            Values in ``[0, 1]`` where ``1`` = full attention and ``0`` = no
            attention. ``None`` means unrestricted (full) attention between
            all tokens. Built incrementally by conditioning items; see
            :class:`~ltx_core.conditioning.types.attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper`.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    sigma: torch.Tensor  # Shape: (B,). Current sigma value, used for cross-attention timestep calculation.
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    qwen_context: torch.Tensor | None = None
    qwen_context_mask: torch.Tensor | None = None
    # Whisper W1: seconds relative to the common clip origin, (B,1,S,2).
    qwen_time_positions: torch.Tensor | None = None
    omni_context: torch.Tensor | None = None
    omni_context_mask: torch.Tensor | None = None
    omni_field_mask: torch.Tensor | None = None
    listener_timbre_context: torch.Tensor | None = None
    # Optional raw first-frame video latent, shape (B, C, 1, H, W). Only used by the
    # audio modality when first-frame audio conditioning is enabled: the model
    # patchifies it into context tokens for an independent cross-attention branch.
    first_frame_latent: torch.Tensor | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
