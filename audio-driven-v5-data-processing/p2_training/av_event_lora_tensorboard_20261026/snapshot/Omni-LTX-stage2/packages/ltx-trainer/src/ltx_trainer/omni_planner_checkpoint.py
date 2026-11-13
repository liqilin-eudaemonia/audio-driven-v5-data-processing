"""Versioned PEFT checkpoints shared by Planner SFT and joint Stage 2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ltx_trainer.omni_stage2 import (
    CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT,
    OmniJointConfig,
)

PLANNER_SFT_FORMAT = "omni_planner_sft_lora_v1"
PLANNER_JOINT_FORMAT = "omni_joint_stage2_lora_v2"
PLANNER_INPUT_CONTRACTS = {
    "audio_frame_timing": "speaker_frame_timing_v3",
    "audio_timing": "speaker_audio_timing_sft_v1",
}

_TOPOLOGY_FIELDS = (
    "attention_rank",
    "attention_alpha",
    "expert_rank",
    "expert_alpha",
    "lora_dropout",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_config_sha256(model_path: str | Path) -> str:
    config = Path(model_path) / "config.json"
    if not config.is_file():
        return "missing"
    return file_sha256(config)


def planner_topology(config: OmniJointConfig) -> dict[str, Any]:
    return {
        "base_model_path": str(Path(config.model_path).resolve()),
        "base_config_sha256": _base_config_sha256(config.model_path),
        "lora": {field: getattr(config, field) for field in _TOPOLOGY_FIELDS},
    }


def planner_topology_json(config: OmniJointConfig) -> str:
    return json.dumps(planner_topology(config), sort_keys=True, separators=(",", ":"))


def save_planner_adapter(
    model: Any,
    path: str | Path,
    config: OmniJointConfig,
    *,
    checkpoint_format: str,
    extra_metadata: dict[str, str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    metadata = {
        "format": checkpoint_format,
        "planner_topology": planner_topology_json(config),
        "planner_input_mode": config.planner_input_mode,
        "input_contract": PLANNER_INPUT_CONTRACTS[config.planner_input_mode],
        "system_prompt_sha256": hashlib.sha256(
            CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT.encode()
        ).hexdigest(),
    }
    metadata.update(extra_metadata or {})
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in get_peft_model_state_dict(model).items()
    }
    save_file(state, str(temporary), metadata=metadata)
    os.replace(temporary, target)
    checksum = target.with_suffix(target.suffix + ".sha256")
    checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
    checksum_tmp.write_text(file_sha256(target) + "\n")
    os.replace(checksum_tmp, checksum)
    return target


def inspect_planner_adapter(
    path: str | Path,
    config: OmniJointConfig,
    *,
    accepted_formats: Iterable[str],
    accepted_input_modes: Iterable[str] | None = None,
) -> dict[str, str]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Planner adapter not found: {checkpoint}")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") not in set(accepted_formats):
        raise ValueError(f"Unsupported Planner adapter format: {metadata.get('format')!r}")
    if metadata.get("planner_topology") != planner_topology_json(config):
        raise ValueError("Planner adapter base model or LoRA topology differs")
    input_modes = set(accepted_input_modes or (config.planner_input_mode,))
    if metadata.get("planner_input_mode") not in input_modes:
        raise ValueError(
            f"Planner adapter input mode {metadata.get('planner_input_mode')!r} "
            f"is not accepted; expected one of {sorted(input_modes)}"
        )
    mode = metadata["planner_input_mode"]
    if metadata.get("input_contract") != PLANNER_INPUT_CONTRACTS.get(mode):
        raise ValueError("Planner adapter input contract metadata is inconsistent")
    expected_prompt_hash = hashlib.sha256(
        CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT.encode()
    ).hexdigest()
    if metadata.get("system_prompt_sha256") != expected_prompt_hash:
        raise ValueError("Planner adapter Route-A system prompt differs")
    if metadata.get("format") == PLANNER_SFT_FORMAT:
        checksum = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
        if not checksum.is_file():
            raise FileNotFoundError(f"Planner SFT adapter checksum is missing: {checksum}")
        expected_sha256 = checksum.read_text().strip()
        actual_sha256 = file_sha256(checkpoint)
        if expected_sha256 != actual_sha256:
            raise ValueError("Planner SFT adapter SHA256 mismatch")
    return metadata


def load_planner_adapter(
    model: Any,
    path: str | Path,
    config: OmniJointConfig,
    *,
    accepted_formats: Iterable[str],
    accepted_input_modes: Iterable[str] | None = None,
) -> dict[str, str]:
    metadata = inspect_planner_adapter(
        path,
        config,
        accepted_formats=accepted_formats,
        accepted_input_modes=accepted_input_modes,
    )
    state = load_file(str(path), device="cpu")
    expected = get_peft_model_state_dict(model)
    if state.keys() != expected.keys() or any(
        state[key].shape != expected[key].shape for key in expected
    ):
        raise ValueError("Planner adapter tensor keys/shapes do not match LoRA architecture")
    set_peft_model_state_dict(model, state)
    return metadata
