"""Helpers for loading trainer movie LoRA checkpoints at inference time."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors import safe_open
from safetensors.torch import load_file

from ltx_trainer import logger
from ltx_trainer.stage2_lora import apply_stage2_injection_lora

_LORA_TARGET_RE = re.compile(r"(.+)\.lora_[AB]\.")
_FULL_MODULE_NAMES = (
    "qwen_attn2",
    "audio_qwen_attn2",
    "video_qwen_layer_adapter",
    "audio_qwen_layer_adapter",
    "omni_attn2",
    "audio_omni_attn2",
    "video_omni_layer_adapter",
    "audio_omni_layer_adapter",
    "omni_field_resampler",
    "audio_timbre_attn2",
    "audio_ff_attn2",
    "audio_first_frame_patch_embedding",
)


def _normalize_stage2_base_key(key: str) -> str:
    return key.replace(".base_layer.", ".", 1)


def _normalize_conditioning_key(key: str) -> str:
    return _normalize_stage2_base_key(
        key.replace("diffusion_model.", "", 1).replace(
            ".modules_to_save.default.", ".", 1
        )
    )


def _require_complete_module_state(
    base_model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    module_names: tuple[str, ...],
) -> None:
    """Reject partially persisted conditioning modules before loading weights."""
    normalized_keys = {_normalize_conditioning_key(key) for key in state_dict}
    expected = [
        key
        for key in base_model.state_dict()
        if any(
            key.startswith(f"{name}.") or f".{name}." in key
            for name in module_names
        )
    ]
    missing = [
        key
        for key in expected
        if not any(
            candidate == key or candidate.endswith(f".{key}")
            for candidate in normalized_keys
        )
    ]
    if missing:
        raise ValueError(
            "Checkpoint is missing persisted conditioning tensors: "
            f"{missing[:8]}"
        )


def load_movie_lora_tensors(
    transformer: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Restore both native PEFT and trainer prefix-stripped A/B keys strictly."""
    expected = {
        key: value for key, value in get_peft_model_state_dict(transformer).items()
        if "lora_" in key and ".parametrizations." not in key
    }
    def canonical(key: str) -> str:
        return key.removeprefix("base_model.model.")

    supplied = {canonical(key): value for key, value in state_dict.items()}
    expected_keys = {canonical(key) for key in expected}
    if len(supplied) != len(state_dict) or set(supplied) != expected_keys:
        raise ValueError(
            "LoRA A/B checkpoint keys mismatch: "
            f"missing={sorted(expected_keys - supplied.keys())[:5]}, "
            f"unexpected={sorted(supplied.keys() - expected_keys)[:5]}"
        )
    normalized = {key: supplied[canonical(key)] for key in expected}
    incompatible = set_peft_model_state_dict(transformer, normalized)
    missing_lora = [
        key
        for key in incompatible.missing_keys
        if "lora_" in key and ".parametrizations." not in key
    ]
    if missing_lora or incompatible.unexpected_keys:
        raise ValueError(f"LoRA weights were not fully restored: {missing_lora}, {incompatible.unexpected_keys}")


def checkpoint_uses_stage2_injection_lora(lora_path: str | Path) -> bool:
    """Return whether a trainer checkpoint contains Stage-2 injection deltas."""
    with safe_open(str(Path(lora_path)), framework="pt") as checkpoint:
        return any(
            "stage2_lora_A" in key or "stage2_lora_B" in key
            for key in checkpoint.keys()
        )


def checkpoint_uses_first_frame_audio_conditioning(
    lora_path: str | Path,
) -> bool:
    """Return whether a checkpoint contains the optional audio first-frame branch."""
    with safe_open(str(Path(lora_path)), framework="pt") as checkpoint:
        return any(
            "audio_first_frame_patch_embedding" in key or "audio_ff_attn2" in key
            for key in checkpoint.keys()
        )


def resolve_first_frame_audio_conditioning(
    lora_path: str | Path | None,
    *,
    manually_enabled: bool = False,
) -> bool:
    """Resolve the optional branch from checkpoint structure, with a no-LoRA fallback."""
    if lora_path is None:
        return manually_enabled
    detected = checkpoint_uses_first_frame_audio_conditioning(lora_path)
    if manually_enabled and not detected:
        raise ValueError(
            "first-frame audio conditioning was manually enabled, but the model "
            "checkpoint has no audio_first_frame_patch_embedding/audio_ff_attn2 tensors"
        )
    return detected


def inspect_omni_conditioning(
    checkpoint_path: str | Path,
) -> dict[str, int | bool]:
    """Infer optional Omni/timbre branches before loading checkpoint tensors."""
    with safe_open(str(Path(checkpoint_path)), framework="pt") as checkpoint:
        keys = list(checkpoint.keys())
        shapes = {key: checkpoint.get_slice(key).get_shape() for key in keys}

    def find_shape(suffix: str) -> list[int] | None:
        key = next((key for key in keys if key.endswith(suffix)), None)
        return list(shapes[key]) if key is not None else None

    layer_logits = find_shape("video_omni_layer_adapter.layer_logits")
    adapter_weight = find_shape("video_omni_layer_adapter.proj.weight")
    audio_adapter_weight = find_shape("audio_omni_layer_adapter.proj.weight")
    resampler_weight = find_shape("omni_field_resampler.proj_in.weight")
    resampler_stage2_lora = False
    if resampler_weight is None:
        resampler_weight = find_shape(
            "omni_field_resampler.proj_in.parametrizations.weight.original"
        )
        resampler_stage2_lora = resampler_weight is not None
    timbre_weight = find_shape("audio_timbre_attn2.to_k.weight")
    has_omni = any(".omni_attn2." in key or key.startswith("omni_attn2.") for key in keys)
    has_audio_omni = any("audio_omni_attn2." in key for key in keys)
    has_timbre = timbre_weight is not None
    adapter_parts = (layer_logits, adapter_weight, audio_adapter_weight)
    if any(part is not None for part in adapter_parts):
        if not has_omni:
            raise ValueError(
                "Checkpoint has Omni adapters but no video Omni attention"
            )
        if layer_logits is None or adapter_weight is None:
            raise ValueError("Checkpoint has an incomplete video Omni adapter")
        if has_audio_omni != (audio_adapter_weight is not None):
            raise ValueError(
                "Audio Omni attention and audio Omni adapter must be persisted together"
            )
        video_attention_weight = find_shape(
            "transformer_blocks.0.omni_attn2.to_k.weight"
        )
        if (
            video_attention_weight is None
            or adapter_weight[0] != video_attention_weight[1]
        ):
            raise ValueError(
                "Video Omni adapter output dimension disagrees with "
                "video Omni attention context dimension"
            )
        audio_attention_weight = find_shape(
            "transformer_blocks.0.audio_omni_attn2.to_k.weight"
        )
        if (
            audio_adapter_weight is not None
            and audio_attention_weight is not None
            and audio_adapter_weight[0] != audio_attention_weight[1]
        ):
            raise ValueError(
                "Audio Omni adapter output dimension disagrees with "
                "audio Omni attention context dimension"
            )
    if resampler_weight is not None:
        if adapter_weight is None:
            raise ValueError(
                "Omni field Resampler requires a persisted Omni layer adapter"
            )
        if resampler_weight[0] != adapter_weight[1]:
            raise ValueError(
                "Omni Resampler output dimension disagrees with layer adapter input"
            )
    return {
        "enabled": has_omni,
        "audio_enabled": has_audio_omni,
        "layer_adapters": adapter_weight is not None,
        "num_layers": layer_logits[0] if layer_logits else 1,
        "input_dim": adapter_weight[1] if adapter_weight else 1024,
        "video_context_dim": adapter_weight[0] if adapter_weight else 1024,
        "audio_context_dim": (
            audio_adapter_weight[0]
            if audio_adapter_weight
            else adapter_weight[0]
            if adapter_weight
            else 1024
        ),
        "field_resampler": resampler_weight is not None,
        "field_resampler_stage2_lora": resampler_stage2_lora,
        "field_resampler_input_dim": (
            resampler_weight[1] if resampler_weight else 0
        ),
        "field_resampler_output_dim": (
            resampler_weight[0] if resampler_weight else 0
        ),
        "field_resampler_num_heads": 8,
        "timbre_enabled": has_timbre,
        "timbre_dim": timbre_weight[1] if timbre_weight else 0,
    }


def resolve_qwen_lora_companion(
    ltx_path: str | Path, qwen_lora_path: str | Path | None = None
) -> Path | None:
    """Resolve and validate Qwen LoRA paired with a LoRA or full-SFT checkpoint."""
    ltx_path = Path(ltx_path)
    if qwen_lora_path is not None:
        explicit = Path(qwen_lora_path)
        if not explicit.is_file():
            raise FileNotFoundError(f"Qwen LoRA checkpoint not found: {explicit}")
        return explicit

    match = re.fullmatch(
        r"(?:lora|model)_weights_step_(\d+)\.safetensors", ltx_path.name
    )
    candidate = (
        ltx_path.with_name(f"qwen_weights_step_{match.group(1)}.safetensors")
        if match
        else None
    )
    if candidate is not None and candidate.is_file():
        return candidate

    with safe_open(str(ltx_path), framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    requires_qwen_lora = (
        metadata.get("qwen_training_mode") == "lora" or "qwen_lora_rank" in metadata
    )
    if requires_qwen_lora:
        expected = candidate or ltx_path.with_name(
            "qwen_weights_<matching-step>.safetensors"
        )
        raise FileNotFoundError(
            f"Stage-2 checkpoint requires its matching Qwen LoRA companion: {expected}"
        )
    return None


def split_lora_and_full_module_state(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Separate PEFT A/B tensors from fully trained conditioning-module tensors."""
    modules_to_save_state: dict[str, torch.Tensor] = {}
    lora_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".modules_to_save.default." in key:
            modules_to_save_state[key.replace(".modules_to_save.default.", ".", 1)] = (
                value
            )
        elif any(
            key.startswith(f"{name}.") or f".{name}." in key
            for name in _FULL_MODULE_NAMES
        ):
            modules_to_save_state[key] = value
        elif "modules_to_save" not in key and "original_module" not in key:
            lora_state[key] = value
    return lora_state, modules_to_save_state


def inspect_qwen_layer_adapter_architecture(
    lora_path: str | Path,
) -> tuple[int, int, int, int] | None:
    """Return (layers, input_dim, video_dim, audio_dim) without loading tensor data."""
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(Path(lora_path)), framework="pt") as checkpoint:
        for key in checkpoint.keys():
            normalized = _normalize_stage2_base_key(
                key.replace("diffusion_model.", "", 1).replace(
                    ".modules_to_save.default.", ".", 1
                )
            )
            if normalized in {
                "video_qwen_layer_adapter.layer_logits",
                "video_qwen_layer_adapter.proj.weight",
                "audio_qwen_layer_adapter.layer_logits",
                "audio_qwen_layer_adapter.proj.weight",
            }:
                shapes[normalized] = tuple(checkpoint.get_slice(key).get_shape())
    if not shapes:
        return None
    required = {
        "video_qwen_layer_adapter.layer_logits",
        "video_qwen_layer_adapter.proj.weight",
        "audio_qwen_layer_adapter.layer_logits",
        "audio_qwen_layer_adapter.proj.weight",
    }
    missing = sorted(required - shapes.keys())
    if missing:
        raise ValueError(
            f"LoRA checkpoint contains incomplete Qwen layer adapters: missing {missing}"
        )
    video_logits = shapes["video_qwen_layer_adapter.layer_logits"]
    audio_logits = shapes["audio_qwen_layer_adapter.layer_logits"]
    video_proj = shapes["video_qwen_layer_adapter.proj.weight"]
    audio_proj = shapes["audio_qwen_layer_adapter.proj.weight"]
    if video_logits != audio_logits or video_proj[1] != audio_proj[1]:
        raise ValueError(
            "Video/audio Qwen layer-adapter checkpoint shapes are inconsistent"
        )
    return video_logits[0], video_proj[1], video_proj[0], audio_proj[0]


def checkpoint_uses_qwen_layer_adapters(lora_path: str | Path) -> bool:
    """Return whether a trainer LoRA contains the per-stream all-layer Qwen adapters."""
    return inspect_qwen_layer_adapter_architecture(lora_path) is not None


def enable_qwen_layer_adapters_from_state_dict(
    transformer: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """Create per-stream Qwen layer adapters before their full weights are loaded."""

    def find(suffix: str) -> torch.Tensor | None:
        for key, value in state_dict.items():
            normalized = _normalize_conditioning_key(key)
            if normalized == suffix:
                return value
        return None

    video_logits = find("video_qwen_layer_adapter.layer_logits")
    video_proj = find("video_qwen_layer_adapter.proj.weight")
    audio_logits = find("audio_qwen_layer_adapter.layer_logits")
    audio_proj = find("audio_qwen_layer_adapter.proj.weight")
    present = [
        value is not None
        for value in (video_logits, video_proj, audio_logits, audio_proj)
    ]
    if not any(present):
        return False
    if not all(present):
        raise ValueError(
            "LoRA checkpoint contains an incomplete video/audio Qwen layer-adapter pair"
        )
    if video_logits.shape != audio_logits.shape:
        raise ValueError(
            "Video/audio Qwen layer adapters disagree on layer count: "
            f"{video_logits.shape} vs {audio_logits.shape}"
        )
    if video_proj.shape[1] != audio_proj.shape[1]:
        raise ValueError(
            "Video/audio Qwen layer adapters disagree on input dim: "
            f"{video_proj.shape} vs {audio_proj.shape}"
        )
    base_model = (
        transformer.get_base_model()
        if hasattr(transformer, "get_base_model")
        else transformer
    )
    if not hasattr(base_model, "enable_qwen_layer_adapters"):
        raise TypeError("Transformer does not support Qwen layer adapters")
    base_model.enable_qwen_layer_adapters(
        num_layers=int(video_logits.shape[0]),
        input_dim=int(video_proj.shape[1]),
        video_output_dim=int(video_proj.shape[0]),
        audio_output_dim=int(audio_proj.shape[0]),
        enable_audio=True,
    )
    return True


def enable_omni_conditioning_from_state_dict(
    transformer: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    """Create Omni and timbre modules before loading their persisted tensors."""

    def find(suffix: str) -> torch.Tensor | None:
        for key, value in state_dict.items():
            normalized = _normalize_stage2_base_key(
                key.replace("diffusion_model.", "", 1).replace(
                    ".modules_to_save.default.", ".", 1
                )
            )
            if normalized == suffix or normalized.endswith(f".{suffix}"):
                return value
        return None

    base_model = (
        transformer.get_base_model()
        if hasattr(transformer, "get_base_model")
        else transformer
    )
    video_key = find("transformer_blocks.0.omni_attn2.to_k.weight")
    audio_key = find("transformer_blocks.0.audio_omni_attn2.to_k.weight")
    video_logits = find("video_omni_layer_adapter.layer_logits")
    video_proj = find("video_omni_layer_adapter.proj.weight")
    audio_logits = find("audio_omni_layer_adapter.layer_logits")
    audio_proj = find("audio_omni_layer_adapter.proj.weight")
    resampler_proj = find("omni_field_resampler.proj_in.weight")
    resampler_lora_a = None
    if resampler_proj is None:
        resampler_proj = find(
            "omni_field_resampler.proj_in.parametrizations.weight.original"
        )
        resampler_lora_a = find(
            "omni_field_resampler.proj_in.parametrizations.weight.0.lora_A"
        )
    timbre_key = find("transformer_blocks.0.audio_timbre_attn2.to_k.weight")

    if video_key is None and any(
        value is not None
        for value in (
            audio_key,
            video_logits,
            video_proj,
            audio_logits,
            audio_proj,
            resampler_proj,
        )
    ):
        raise ValueError("Checkpoint contains incomplete Omni conditioning modules")
    if video_key is not None:
        base_model.enable_omni_conditioning(
            omni_context_dim=int(video_key.shape[1]),
            audio_omni_context_dim=(
                int(audio_key.shape[1]) if audio_key is not None else None
            ),
            enable_audio=audio_key is not None,
        )
        adapter_values = (video_logits, video_proj, audio_logits, audio_proj)
        if any(value is not None for value in adapter_values):
            if video_logits is None or video_proj is None:
                raise ValueError("Checkpoint has an incomplete video Omni adapter")
            if (audio_logits is None) != (audio_proj is None):
                raise ValueError("Checkpoint has an incomplete audio Omni adapter")
            if audio_key is not None and audio_logits is None:
                raise ValueError(
                    "Audio Omni attention requires a persisted audio Omni adapter "
                    "when stacked-layer adapters are enabled"
                )
            if audio_key is None and audio_logits is not None:
                raise ValueError(
                    "Checkpoint has an audio Omni adapter but no audio Omni attention"
                )
            if audio_logits is not None and video_logits.shape != audio_logits.shape:
                raise ValueError("Video/audio Omni adapters disagree on layer count")
            if int(video_proj.shape[0]) != int(video_key.shape[1]):
                raise ValueError(
                    "Video Omni adapter output dimension disagrees with "
                    "video Omni attention context dimension"
                )
            if (
                audio_proj is not None
                and audio_key is not None
                and int(audio_proj.shape[0]) != int(audio_key.shape[1])
            ):
                raise ValueError(
                    "Audio Omni adapter output dimension disagrees with "
                    "audio Omni attention context dimension"
                )
            base_model.enable_omni_layer_adapters(
                num_layers=int(video_logits.shape[0]),
                input_dim=int(video_proj.shape[1]),
                video_output_dim=int(video_proj.shape[0]),
                audio_output_dim=(
                    int(audio_proj.shape[0]) if audio_proj is not None else None
                ),
                enable_audio=audio_proj is not None,
            )
        if resampler_proj is not None:
            if video_logits is None or video_proj is None:
                raise ValueError(
                    "Omni field Resampler requires a persisted Omni layer adapter"
                )
            if int(resampler_proj.shape[0]) != int(video_proj.shape[1]):
                raise ValueError(
                    "Omni Resampler output dimension disagrees with layer adapter input"
                )
            base_model.enable_omni_field_resampler(
                num_layers=int(video_logits.shape[0]),
                input_dim=int(resampler_proj.shape[1]),
                output_dim=int(resampler_proj.shape[0]),
                num_heads=8,
            )
            if resampler_lora_a is not None:
                rank = int(resampler_lora_a.shape[0])
                base_model.omni_field_resampler.enable_stage2_lora(
                    rank=rank,
                    alpha=rank,
                )
    if timbre_key is not None:
        base_model.enable_listener_timbre_conditioning(
            listener_timbre_context_dim=int(timbre_key.shape[1])
        )
    required_modules: list[str] = []
    if video_key is not None:
        required_modules.append("omni_attn2")
    if audio_key is not None:
        required_modules.append("audio_omni_attn2")
    if video_logits is not None:
        required_modules.append("video_omni_layer_adapter")
    if audio_logits is not None:
        required_modules.append("audio_omni_layer_adapter")
    if resampler_proj is not None:
        required_modules.append("omni_field_resampler")
    if timbre_key is not None:
        required_modules.append("audio_timbre_attn2")
    if required_modules:
        _require_complete_module_state(
            base_model, state_dict, tuple(required_modules)
        )
    return video_key is not None or timbre_key is not None


def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Extract PEFT target module names from LoRA A/B keys."""
    targets: set[str] = set()
    for key in state_dict:
        match = _LORA_TARGET_RE.match(key)
        if match:
            target = match.group(1)
            for prefix in ("base_model.model.", "model.", "module."):
                if target.startswith(prefix):
                    target = target.removeprefix(prefix)
            targets.add(target)
    return sorted(targets)


def apply_movie_lora(
    transformer: torch.nn.Module, lora_path: str | Path
) -> torch.nn.Module:
    """Wrap ``transformer`` with PEFT and load a trainer movie-LoRA safetensors."""
    lora_path = Path(lora_path)
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

    logger.info(f"Loading movie LoRA weights from {lora_path}")
    ref_param = next(transformer.parameters())
    target_device = ref_param.device
    target_dtype = ref_param.dtype
    state_dict = load_file(str(lora_path))
    state_dict = {
        k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()
    }
    enable_omni_conditioning_from_state_dict(transformer, state_dict)
    enable_qwen_layer_adapters_from_state_dict(transformer, state_dict)

    lora_state, modules_to_save_state = split_lora_and_full_module_state(state_dict)

    # Cast on CPU before load — avoids a full-model .to(dtype) copy that can OOM,
    # and keeps PEFT A/B weights matched to the bf16 base transformer.
    lora_state = {
        key: value.to(dtype=target_dtype) if value.is_floating_point() else value
        for key, value in lora_state.items()
    }
    modules_to_save_state = {
        key: value.to(dtype=target_dtype) if value.is_floating_point() else value
        for key, value in modules_to_save_state.items()
    }

    target_modules = extract_lora_target_modules(lora_state)
    if not target_modules:
        raise ValueError(
            f"Could not extract target modules from LoRA checkpoint: {lora_path}"
        )

    lora_rank = None
    for key, value in lora_state.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = int(value.shape[0])
            break
    if lora_rank is None:
        raise ValueError("Could not auto-detect LoRA rank from weights")

    with safe_open(str(lora_path), framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    metadata_rank = metadata.get("lora_rank")
    if metadata_rank is not None and int(metadata_rank) != lora_rank:
        raise ValueError(
            "LoRA rank metadata disagrees with tensor shape: "
            f"metadata={metadata_rank}, tensor={lora_rank}"
        )
    lora_alpha = int(metadata.get("lora_alpha", lora_rank))
    if lora_alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {lora_alpha}")
    if "lora_alpha" not in metadata:
        logger.warning(
            "Checkpoint has no lora_alpha metadata; falling back to alpha=rank "
            "for legacy compatibility"
        )

    logger.info(
        f"  Detected {len(target_modules)} target modules, "
        f"rank={lora_rank}, alpha={lora_alpha}"
    )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)
    base_model = transformer.get_base_model()
    load_movie_lora_tensors(transformer, lora_state)

    if modules_to_save_state:
        stage2_state = {
            key: value
            for key, value in modules_to_save_state.items()
            if "stage2_lora_" in key
        }
        learned_base_state = {
            _normalize_stage2_base_key(key): value
            for key, value in modules_to_save_state.items()
            if "stage2_lora_" not in key
        }
        incompatible = base_model.load_state_dict(learned_base_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"Full-weight LoRA tensors do not match the model: {incompatible.unexpected_keys[:5]}. "
                "Did you forget to enable first-frame audio conditioning / Qwen branches first?"
            )
        logger.info(f"  Loaded {len(learned_base_state)} full-weight tensors")

        if stage2_state:
            injection_rank = next(
                int(value.shape[0])
                for key, value in stage2_state.items()
                if "stage2_lora_A" in key
            )
            injection_modules = ["qwen_attn2", "audio_qwen_attn2"]
            if any("audio_ff_attn2" in key for key in stage2_state):
                injection_modules.append("audio_ff_attn2")
            full_trainable_modules = (
                ["audio_first_frame_patch_embedding"]
                if any(
                    "audio_first_frame_patch_embedding" in key
                    for key in learned_base_state
                )
                else []
            )
            targets = apply_stage2_injection_lora(
                transformer,
                injection_modules=injection_modules,
                adapter_modules=[
                    "video_qwen_layer_adapter",
                    "audio_qwen_layer_adapter",
                ],
                rank=injection_rank,
                alpha=injection_rank,
                dropout=0.0,
                train_adapter_norm_and_logits=False,
                full_trainable_modules=full_trainable_modules,
            )
            incompatible = base_model.load_state_dict(stage2_state, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    "Stage-2 injection LoRA tensors do not match the model: "
                    f"{incompatible.unexpected_keys[:5]}"
                )
            logger.info(
                f"  Loaded {len(stage2_state)} Stage-2 injection tensors into "
                f"{len(targets)} Linear modules"
            )

    # Cast only stray float32 PEFT params in-place (no full-module copy).
    with torch.no_grad():
        for param in transformer.parameters():
            if param.is_floating_point() and param.dtype != target_dtype:
                param.data = param.data.to(device=target_device, dtype=target_dtype)
        for buffer in transformer.buffers():
            if buffer.is_floating_point() and buffer.dtype != target_dtype:
                buffer.data = buffer.data.to(device=target_device, dtype=target_dtype)

    logger.info("Movie LoRA loaded successfully")
    return transformer
