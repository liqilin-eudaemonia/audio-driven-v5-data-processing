"""Automatic selection of complete local training checkpoints, never weight-only fallback."""

from __future__ import annotations

import re
from pathlib import Path


def select_resume_checkpoint(output_dir: Path, training_mode: str) -> Path | None:
    prefix = "lora" if training_mode == "lora" else "model"
    root = output_dir / "checkpoints"
    candidates = []
    for path in root.glob(f"{prefix}_weights_step_*.safetensors"):
        match = re.fullmatch(rf"{prefix}_weights_step_(\d+)\.safetensors", path.name)
        if match:
            candidates.append((int(match[1]), path, match[1]))
    if not candidates:
        return None
    for _, path, suffix in sorted(candidates, reverse=True):
        state = path.with_name(f"training_state_step_{suffix}.pt")
        if path.stat().st_size and state.is_file() and state.stat().st_size:
            return path
    raise ValueError(f"No complete weights + training-state pair in {root}; refusing a fresh restart")


def require_full_resume_state(state: object, checkpoint: Path, world_size: int) -> None:
    match = re.search(r"step_(\d+)", checkpoint.name)
    if not match or state.global_step != int(match[1]):
        raise ValueError("Checkpoint step differs from training state")
    if state.optimizer_state_dict is None and state.optimizer_state_path is None:
        raise ValueError("Automatic resume requires full optimizer state")
    scheduler_type = getattr(getattr(state, "config_fingerprint", None), "scheduler_type", None)
    # _init_scheduler returns None for constant LR; no scheduler state exists.
    if state.lr_scheduler_state_dict is None and scheduler_type != "constant":
        raise ValueError("Automatic resume requires scheduler state for a non-constant schedule")
    if world_size > 1 and (state.rng_states_by_rank is None or len(state.rng_states_by_rank) != world_size):
        raise ValueError("Automatic resume requires matching per-rank RNG states")
