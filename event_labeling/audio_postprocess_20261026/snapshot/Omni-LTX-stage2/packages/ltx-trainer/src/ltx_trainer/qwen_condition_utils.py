"""Helpers for loading / encoding Qwen3-VL condition tensors at inference time."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.utils import open_image_as_srgb

QWEN_CONTEXT_KEYS = ("qwen_video_context", "qwen_context", "qwen_hidden_states")
QWEN_MASK_KEYS = ("qwen_attention_mask", "qwen_context_mask")
QWEN_AUDIO_CONTEXT_KEYS = ("qwen_audio_context",)
QWEN_AUDIO_MASK_KEYS = ("qwen_audio_attention_mask", "qwen_audio_context_mask")


@dataclass(frozen=True)
class QwenConditionBundle:
    """Normalized Qwen contexts ready for ValidationSampler / GenerationConfig."""

    video_context: Tensor  # [1, seq, hidden]
    video_context_mask: Tensor  # [1, seq]
    audio_context: Tensor | None = None  # [1, seq, hidden]
    audio_context_mask: Tensor | None = None  # [1, seq]


def select_qwen_tensor(data: dict[str, Any], keys: tuple[str, ...]) -> Tensor | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, Tensor):
            return value
    return None


def normalize_qwen_context(context: Tensor, name: str, *, preserve_layers: bool = False) -> Tensor:
    """Normalize cached Qwen context, optionally retaining stacked decoder layers."""
    if preserve_layers:
        if context.ndim == 4 and context.shape[0] == 1:
            return context.contiguous()
        if context.ndim == 3:
            return context.unsqueeze(0).contiguous()
        raise ValueError(
            f"{name} must have shape [layers, seq, hidden] or [1, layers, seq, hidden], got {context.shape}"
        )
    if context.ndim == 4:
        # [B, layers, seq, hidden] -> final stored layer
        context = context[:, -1]
    if context.ndim == 3:
        # [layers, seq, hidden] or [1, seq, hidden] -> [seq, hidden]
        context = context[-1]
    if context.ndim != 2:
        raise ValueError(f"{name} must have shape [seq, hidden] or [layers, seq, hidden], got {context.shape}")
    return context.unsqueeze(0).contiguous()


def normalize_qwen_mask(mask: Tensor | None, seq_len: int, name: str) -> Tensor:
    if mask is None:
        return torch.ones(1, seq_len, dtype=torch.long)
    if mask.ndim == 3:
        mask = mask[:, -1]
    if mask.ndim == 2:
        mask = mask[-1]
    if mask.ndim != 1 or mask.shape[0] != seq_len:
        raise ValueError(f"{name} must have shape [seq] or [layers, seq], got {mask.shape}; expected seq={seq_len}")
    return mask.to(dtype=torch.long).unsqueeze(0).contiguous()


def load_qwen_condition_file(path: str | Path, *, preserve_layers: bool = False) -> QwenConditionBundle:
    """Load a process_qwen_conditions .pt cache into a sampler-ready bundle."""
    condition_path = Path(path).expanduser()
    if not condition_path.is_file():
        raise FileNotFoundError(f"Qwen condition file not found: {condition_path}")

    data = torch.load(condition_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {condition_path}, got {type(data).__name__}")

    video_context = select_qwen_tensor(data, QWEN_CONTEXT_KEYS)
    if video_context is None:
        raise ValueError(
            f"Qwen condition file {condition_path} must contain one of: {', '.join(QWEN_CONTEXT_KEYS)}"
        )

    video_context = normalize_qwen_context(
        video_context, "qwen_video_context", preserve_layers=preserve_layers
    )
    video_mask = normalize_qwen_mask(
        select_qwen_tensor(data, QWEN_MASK_KEYS),
        seq_len=video_context.shape[-2],
        name="qwen_attention_mask",
    )

    audio_context = select_qwen_tensor(data, QWEN_AUDIO_CONTEXT_KEYS)
    audio_mask = None
    if audio_context is not None:
        audio_context = normalize_qwen_context(
            audio_context, "qwen_audio_context", preserve_layers=preserve_layers
        )
        audio_mask = normalize_qwen_mask(
            select_qwen_tensor(data, QWEN_AUDIO_MASK_KEYS),
            seq_len=audio_context.shape[-2],
            name="qwen_audio_attention_mask",
        )

    return QwenConditionBundle(
        video_context=video_context,
        video_context_mask=video_mask,
        audio_context=audio_context,
        audio_context_mask=audio_mask,
    )


def _import_process_qwen_conditions() -> Any:
    """Import scripts/process_qwen_conditions.py without installing it as a package."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "process_qwen_conditions.py"
    module_name = "ltx_trainer_process_qwen_conditions"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if not script_path.is_file():
        raise FileNotFoundError(f"Cannot find Qwen extraction script: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _TrainingAlignedQwenInferenceAdapter:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_length: int,
        dtype: torch.dtype,
        qwen_lora_path: str | Path,
    ) -> None:
        from ltx_trainer.online_conditioning import OnlineQwenEncoder

        self._encoder = OnlineQwenEncoder(
            model_path=model_path,
            device=device,
            max_length=max_length,
            image_width=1088,
            image_height=640,
            dtype=dtype,
            feature_layer_mode="all",
            lora_checkpoint=qwen_lora_path,
        )
        self.model = self._encoder.model

    def encode(
        self,
        video_path: str,
        caption: str,
        image: Any | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        pil_image = image if image is not None else open_image_as_srgb(video_path)
        encoded = self._encoder.encode_batch([pil_image], [caption], require_grad=False)
        return (
            {
                "qwen_video_context": encoded.context.detach().cpu(),
                "qwen_attention_mask": encoded.attention_mask.detach().cpu(),
            },
            {},
        )


def build_online_qwen_encoder(  # noqa: PLR0913
    model_path: str,
    device: str,
    *,
    max_length: int = 3072,
    max_pixels: int = 921600,
    dtype: torch.dtype = torch.bfloat16,
    feature_layer_mode: str = "last",
    qwen_lora_path: str | Path | None = None,
) -> Any:
    """Construct a frozen Stage-1 or Stage-2 Qwen3-VL feature encoder."""
    if qwen_lora_path is not None:
        if feature_layer_mode != "all":
            raise ValueError("Stage-2 Qwen LoRA inference requires all-layer Qwen features")
        return _TrainingAlignedQwenInferenceAdapter(
            model_path=model_path,
            device=device,
            max_length=max_length,
            dtype=dtype,
            qwen_lora_path=qwen_lora_path,
        )
    module = _import_process_qwen_conditions()
    return module.Qwen3VLConditionEncoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        attn_implementation=None,
        max_pixels=max_pixels,
        min_pixels=None,
        instruction=module.DEFAULT_QWEN_INSTRUCTION,
        max_length=max_length,
        layer_mode=(module.LayerMode.ALL if feature_layer_mode == "all" else module.LayerMode.LAST),
        selected_layers=None,
        layer_aggregation=module.LayerAggregation.MEAN,
        save_padding=module.SavePadding.NONE,
        load_model=True,
    )


def encode_qwen_condition_from_image(
    encoder: Any,
    *,
    image_path: str | Path,
    caption: str,
    preserve_layers: bool = False,
) -> QwenConditionBundle:
    """Online extract last-layer Qwen features from an I2V first frame + sd_text caption."""
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Qwen online encode requires an existing image: {image_path}")

    pil_image = open_image_as_srgb(image_path)
    payload, _stats = encoder.encode(
        video_path=str(image_path),
        caption=caption,
        image=pil_image,
    )
    video_context = normalize_qwen_context(
        payload["qwen_video_context"],
        "qwen_video_context",
        preserve_layers=preserve_layers,
    )
    video_mask = normalize_qwen_mask(
        payload.get("qwen_attention_mask"),
        seq_len=video_context.shape[-2],
        name="qwen_attention_mask",
    )
    return QwenConditionBundle(
        video_context=video_context,
        video_context_mask=video_mask,
        audio_context=None,
        audio_context_mask=None,
    )
