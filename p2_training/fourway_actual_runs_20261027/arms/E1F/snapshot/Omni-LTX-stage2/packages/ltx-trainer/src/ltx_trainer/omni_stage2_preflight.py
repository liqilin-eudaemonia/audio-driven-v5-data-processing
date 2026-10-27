"""Read-only per-node checks before any Stage-2 distributed process is launched."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]


def require_explicit_checkpoint(value: str | None) -> Path:
    """Return the selected Stage-1/Stage-2 checkpoint or fail before data checks."""
    if not value or value == "auto":
        raise ValueError("Select an explicit Stage-1 initialization or Stage-2 resume checkpoint")
    checkpoint = Path(value)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Stage-2 requires an explicit existing checkpoint: {checkpoint}. "
            "Set model.load_checkpoint to the audited Stage-1 weights or a paired Stage-2 checkpoint."
        )
    return checkpoint


def check_configuration(path: Path) -> tuple[Any, Any]:
    from safetensors import safe_open

    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.omni_planner_checkpoint import (
        PLANNER_SFT_FORMAT,
        inspect_planner_adapter,
    )
    from ltx_trainer.omni_stage2 import OmniJointConfig
    from ltx_trainer.omni_stage2_trainer import (
        OMNI_STAGE2_ARCHITECTURE,
        checkpoint_omni_config_matches,
        companion_path,
        validate_joint_configuration,
    )
    from ltx_trainer.stage1_readiness import require_stage1_audit

    raw = yaml.safe_load(path.read_text())
    joint = OmniJointConfig(**raw.pop("omni_joint"))
    config = LtxTrainerConfig(**raw)
    validate_joint_configuration(config, joint)
    checkpoint = require_explicit_checkpoint(config.model.load_checkpoint)
    required = [
        config.model.model_path,
        config.model.text_encoder_path,
        joint.model_path,
        config.data.preprocessed_data_root,
        config.training_strategy.omni_conditions_dir,
        *config.data.metadata_jsonls,
        *joint.manifests,
    ]
    required += [
        p
        for p in (
            joint.validation_conditions_dir,
            joint.planner_init_checkpoint,
            config.validation.metadata_jsonl,
            config.validation.extended_metadata_jsonl,
        )
        if p
    ]
    missing = [str(p) for p in required if not p or not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing Stage-2 inputs: " + ", ".join(missing))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("omni_joint_stage2") == "true":
        if metadata.get("omni_stage2_architecture") != OMNI_STAGE2_ARCHITECTURE:
            raise ValueError(
                "Stage-2 v2 cannot resume a v1 checkpoint; select the audited Stage-1 initialization"
            )
        mode = "fsdp" if os.environ.get("FSDP_ENABLE") == "1" else "replicated"
        if not config.checkpoints.no_resume and metadata.get("omni_parallel_mode", "replicated") != mode:
            raise ValueError("Stage-2 optimizer resume cannot change DDP/FSDP mode")
        paired = companion_path(checkpoint)
        with safe_open(str(paired), framework="pt", device="cpu") as handle:
            paired_metadata = handle.metadata() or {}
        if (
            paired_metadata.get("ltx_checkpoint") != checkpoint.name
            or not checkpoint_omni_config_matches(
                paired_metadata.get("omni_config", ""), joint
            )
            or paired_metadata.get("format") != "omni_joint_stage2_lora_v2"
        ):
            raise ValueError("Omni companion/config mismatch")
    elif not config.checkpoints.no_resume:
        raise ValueError("Stage-1 initialization requires no_resume=true")
    elif joint.planner_init_checkpoint:
        inspect_planner_adapter(
            joint.planner_init_checkpoint,
            joint,
            accepted_formats=(PLANNER_SFT_FORMAT,),
            accepted_input_modes=("audio_timing",),
        )
    if not config.checkpoints.no_resume:
        step = checkpoint.stem.split("step_")[-1]
        if not checkpoint.with_name(f"training_state_step_{step}.pt").is_file():
            raise FileNotFoundError("Stage-2 resume requires complete optimizer/RNG training state")
    require_stage1_audit(raw, oracle_prefix="oracle_stage2")
    if config.validation.metrics:
        validation = config.validation
        dependencies = [
            validation.id_python,
            validation.id_model_root,
            validation.benchmark_root,
            validation.avbench_python,
            validation.vocal_sync_python,
            validation.syncnet_root,
        ]
        if any(not p or not Path(p).exists() for p in dependencies):
            raise FileNotFoundError("Validation metric dependencies are missing on this node")
    return config, joint


def available_gpu_uuids(devices: str, gpu_csv: str, process_csv: str) -> list[str]:
    """Resolve selected physical GPUs and refuse to share them with existing jobs."""
    by_id = {}
    for line in gpu_csv.splitlines():
        if line.strip():
            index, gpu_uuid, memory = [part.strip() for part in line.split(",")]
            by_id[index] = by_id[gpu_uuid] = (gpu_uuid, int(memory))
    busy = {line.split(",")[0].strip() for line in process_csv.splitlines() if line.strip()}
    result = []
    for device in devices.split(","):
        if device not in by_id:
            raise ValueError(f"GPU {device} is not available on this node")
        gpu_uuid, memory = by_id[device]
        if gpu_uuid in busy or memory > 2048:
            raise RuntimeError(f"GPU {device} is occupied ({memory} MiB); refusing to affect existing training")
        result.append(gpu_uuid)
    return result


def check_node(config_path: Path, nproc: int, env: dict[str, str]) -> dict[str, Any]:
    os.environ.clear()
    os.environ.update(env)
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise ValueError("This node is not using the independent Stage-2 virtual environment")
    for name in ("ltx_trainer", "ltx_core", "ltx_pipelines"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise ValueError(f"{name} imported outside Stage-2 worktree: {module.__file__}")
    import qwen_omni_utils  # noqa: F401
    import torch

    check_configuration(config_path)
    if torch.cuda.device_count() != nproc:
        raise ValueError(f"Expected {nproc} visible CUDA devices; found {torch.cuda.device_count()}")
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    process_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    selected = available_gpu_uuids(env["CUDA_VISIBLE_DEVICES"], gpu_csv, process_csv)
    digest = hashlib.sha256()
    for package in ("ltx-trainer", "ltx-core", "ltx-pipelines"):
        for source in sorted((ROOT / f"packages/{package}/src").rglob("*.py")):
            digest.update(str(source.relative_to(ROOT)).encode())
            digest.update(source.read_bytes())
    return {
        "gpu_uuids": selected,
        "software": {
            "source_sha256": digest.hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "peft", "qwen-omni-utils", "safetensors")
            },
        },
    }
