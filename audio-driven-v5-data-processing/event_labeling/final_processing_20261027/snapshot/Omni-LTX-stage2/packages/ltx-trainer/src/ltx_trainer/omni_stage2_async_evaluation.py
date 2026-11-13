"""Durable handoff of paired Route-A Stage-2 checkpoints to an evaluator."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    if stat.st_size <= 0:
        raise ValueError(f"Checkpoint artifact is empty: {path}")
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _pin(source: Path, destination: Path, expected: dict[str, int]) -> None:
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, temporary)
        if file_stamp(source) != expected or file_stamp(temporary) != expected:
            raise ValueError(f"Checkpoint changed while publishing: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_fsdp_optimizer_state(state_path: Path, world_size: int) -> None:
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer = state.get("optimizer_state_path")
    if not isinstance(optimizer, str) or not optimizer:
        raise ValueError("FSDP full training state has no optimizer shard directory")
    directory = state_path.parent / optimizer
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("FSDP optimizer manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("world_size") != world_size:
        raise ValueError("FSDP optimizer world size does not match checkpoint publication")
    ranks = sorted(directory.glob("rank_*.pt"))
    if len(ranks) != world_size or any(path.stat().st_size == 0 for path in ranks):
        raise ValueError("FSDP optimizer shard directory is incomplete")


def publish_checkpoint(
    weights: Path,
    *,
    companion: Path,
    step: int,
    queue: Path,
    config: dict[str, Any],
    fsdp_world_size: int | None,
) -> Path:
    """Pin an immutable LTX/Omni pair only after the full save is complete."""
    weights = Path(weights)
    companion = Path(companion)
    if config.get("model", {}).get("training_mode") != "lora":
        raise ValueError("Stage-2 async evaluation requires LoRA checkpoints")
    if config.get("checkpoints", {}).get("save_training_state") != "full":
        raise ValueError("Stage-2 async evaluation requires full training state")
    if not isinstance(config.get("omni_joint"), dict):
        raise ValueError("Stage-2 async evaluation requires the Omni joint configuration")
    state = weights.with_name(f"training_state_step_{step:05d}.pt")
    if not weights.is_file() or not companion.is_file() or not state.is_file():
        raise ValueError("Cannot publish incomplete paired checkpoint")
    weight_stamp, companion_stamp, state_stamp = (
        file_stamp(weights),
        file_stamp(companion),
        file_stamp(state),
    )
    if fsdp_world_size is not None:
        _validate_fsdp_optimizer_state(state, fsdp_world_size)

    job = Path(queue) / f"step_{step:08d}"
    job.mkdir(parents=True, exist_ok=True)
    ready = job / "ready.json"
    config_hash = digest(config)
    if ready.exists():
        previous = json.loads(ready.read_text())
        if (
            previous.get("config_hash") != config_hash
            or previous.get("weights_stamp") != weight_stamp
            or previous.get("omni_weights_stamp") != companion_stamp
        ):
            raise ValueError(f"Conflicting already-published Stage-2 step {step}")
        validate_job(ready)
        return ready

    pinned_weights = job / weights.name
    pinned_companion = job / companion.name
    _pin(weights, pinned_weights, weight_stamp)
    try:
        _pin(companion, pinned_companion, companion_stamp)
        atomic_json(job / "training_config.json", config)
        atomic_json(
            ready,
            {
                "format": "omni_stage2_async_checkpoint_v1",
                "step": step,
                "weights": pinned_weights.name,
                "weights_stamp": weight_stamp,
                "omni_weights": pinned_companion.name,
                "omni_weights_stamp": companion_stamp,
                "config_hash": config_hash,
                "full_state_saved_at": str(state.resolve()),
                "full_state_stamp": state_stamp,
                "fsdp_world_size": fsdp_world_size,
            },
        )
    except Exception:
        pinned_weights.unlink(missing_ok=True)
        pinned_companion.unlink(missing_ok=True)
        raise
    return ready


def validate_job(ready: Path) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    ready = Path(ready)
    record = json.loads(ready.read_text())
    if record.get("format") != "omni_stage2_async_checkpoint_v1":
        raise ValueError("Unknown Stage-2 ready marker format")
    names = (record.get("weights"), record.get("omni_weights"))
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in names):
        raise ValueError("Unsafe paired checkpoint name")
    weights, companion = (ready.parent / name for name in names)
    if file_stamp(weights) != record.get("weights_stamp"):
        raise ValueError("Published LTX checkpoint changed")
    if file_stamp(companion) != record.get("omni_weights_stamp"):
        raise ValueError("Published Omni companion changed")
    config = json.loads((ready.parent / "training_config.json").read_text())
    if digest(config) != record.get("config_hash") or not isinstance(config.get("omni_joint"), dict):
        raise ValueError("Published Stage-2 configuration changed")
    return record, weights, companion, config


def jobs(queue: Path) -> list[Path]:
    return sorted(Path(queue).glob("step_*/ready.json"), key=lambda path: int(path.parent.name.split("_")[1]))


def make_eval_config(
    training: dict[str, Any], protocol: dict[str, Any], *, weights: Path, output: Path
) -> dict[str, Any]:
    result = copy.deepcopy(training)
    result["model"]["load_checkpoint"] = str(weights.resolve())
    result["checkpoints"]["no_resume"] = True
    validation = copy.deepcopy(protocol)
    omni_root = validation.pop("omni_conditions_dir", None)
    if omni_root is not None:
        result["training_strategy"]["omni_conditions_dir"] = str(omni_root)
        # Generated-Plan evaluation must override the training-time validation
        # condition root as well; otherwise OmniStage2Trainer restores GT data.
        if "omni_joint" in result:
            result["omni_joint"]["validation_conditions_dir"] = str(omni_root)
    result["validation"] = validation
    result["validation"].update(
        run_only=True,
        skip_initial_validation=False,
        interval=1,
        metric_failure_policy="error",
    )
    result["acceleration"]["context_parallel_size"] = 1
    result["output_dir"] = str(output.resolve())
    result.pop("cluster", None)
    return result
