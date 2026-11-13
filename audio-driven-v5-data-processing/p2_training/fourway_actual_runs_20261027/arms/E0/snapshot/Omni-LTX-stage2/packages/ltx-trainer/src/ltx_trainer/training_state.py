from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class ConfigFingerprint(BaseModel):
    optimizer_type: str
    scheduler_type: str
    training_mode: str
    lora_rank: int | None = None
    stage2_enabled: bool | None = None
    stage2_injection_lora_enabled: bool | None = None
    stage2_injection_rank: int | None = None
    qwen_training_mode: str | None = None
    qwen_rank: int | None = None
    gradient_accumulation_steps: int | None = None
    weight_decay: float | None = None
    batch_size: int | None = None
    seed: int | None = None
    world_size: int | None = None
    context_parallel_size: int | None = None
    training_data_fingerprint: str | None = None


class RngStates(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    torch_state: torch.Tensor
    cuda_state: torch.Tensor | None = None


class TrainingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    global_step: int
    config_fingerprint: ConfigFingerprint
    rng_states: RngStates
    rng_states_by_rank: list[RngStates] | None = None
    lr_scheduler_state_dict: dict[str, Any] | None = None
    optimizer_state_dict: dict[str, Any] | None = None
    optimizer_state_path: str | None = None
    optimizer_state_format: str | None = None
    wandb_run_id: str | None = None

    def to_save_dict(self) -> dict[str, Any]:
        """Build dict suitable for torch.save -- recurses BaseModel sub-models, passes tensors/dicts through."""

        def _convert(value: object) -> object:
            if isinstance(value, BaseModel):
                return {k: _convert(v) for k, v in value if v is not None}
            if isinstance(value, list):
                return [_convert(item) for item in value]
            if isinstance(value, dict):
                return {key: _convert(item) for key, item in value.items()}
            return value

        return {k: _convert(v) for k, v in self if v is not None}

    @classmethod
    def from_save_dict(cls, data: dict[str, Any]) -> TrainingState:
        """Construct from torch.load output with Pydantic validation."""
        return cls(
            global_step=data["global_step"],
            config_fingerprint=ConfigFingerprint(**data["config_fingerprint"]),
            rng_states=RngStates(**data["rng_states"]),
            rng_states_by_rank=[
                RngStates(**state) for state in data["rng_states_by_rank"]
            ]
            if data.get("rng_states_by_rank") is not None
            else None,
            lr_scheduler_state_dict=data.get("lr_scheduler_state_dict"),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            optimizer_state_path=data.get("optimizer_state_path"),
            optimizer_state_format=data.get("optimizer_state_format"),
            wandb_run_id=data.get("wandb_run_id"),
        )
