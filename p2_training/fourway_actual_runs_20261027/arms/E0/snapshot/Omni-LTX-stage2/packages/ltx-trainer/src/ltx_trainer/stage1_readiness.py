"""Stage-1 audit provenance and preflight checks (CPU / filesystem only)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

AUDIT_VERSION = 1


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if path.suffix == ".json" else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected object rows")
    return rows


def row_id(row: dict[str, Any]) -> str:
    sid = row.get("sample_id") or row.get("clip_id")
    if not isinstance(sid, str) or not sid or Path(sid).name != sid or sid in (".", ".."):
        raise ValueError(f"Invalid sample ID: {sid!r}")
    return sid


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        sid = row_id(row)
        if sid in indexed:
            raise ValueError(f"Duplicate sample ID: {sid}")
        indexed[sid] = row
    return indexed


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_snapshot(path: Path, *, content: bool = False) -> dict[str, Any]:
    """Hash small manifests/GT; stat large immutable tensor caches to avoid TiB rereads."""
    path = path.absolute()
    before = path.stat()
    result = {
        "path": str(path),
        "resolved": str(path.resolve()),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
    }
    if content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"File changed during audit: {path}")
    return result


def verify_snapshots(records: list[dict[str, Any]]) -> None:
    for record in records:
        actual = file_snapshot(Path(record["path"]), content="sha256" in record)
        if actual != record:
            raise ValueError(f"Audit is stale; input changed: {record['path']}")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_stage1_audit(config: dict[str, Any], *, oracle_prefix: str = "oracle_stage1") -> str | None:
    """Verify the audited dataset before launcher SSH or direct train.py model loading."""
    strategy = config.get("training_strategy") or {}
    if not (
        strategy.get("use_omni_cond")
        and strategy.get("use_omni_field_resampler")
        and strategy.get("omni_expected_route", "route_a_no_context") == "route_a_no_context"
    ):
        return None
    data = config["data"]
    root = Path(data["preprocessed_data_root"]).resolve()
    path = root / "full_training_contract_audit.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("audit_version") != AUDIT_VERSION:
        raise ValueError(f"A current passed Stage-1 audit is required: {path}")
    fingerprint = report.get("fingerprint")
    if fingerprint != json_digest({k: v for k, v in report.items() if k != "fingerprint"}):
        raise ValueError("Audit report fingerprint is inconsistent")
    if report.get("omni_route") != "route_a_no_context" or report.get("omni_version") != "teacher_forcing_raw":
        raise ValueError("Audit route/version does not match Stage 1")
    if report.get("input_contract") != "speaker_frame_timing_v3":
        raise ValueError("Stage 1 requires an audited v3 audio contract")
    if report.get("latent_root") != str(root) or report.get("omni_root") != str(
        Path(strategy["omni_conditions_dir"]).resolve()
    ):
        raise ValueError("Audit roots do not match training config")
    train_paths = [str(Path(p).resolve()) for p in data.get("metadata_jsonls", [])]
    if train_paths != [report["metadata_paths"]["train"]]:
        raise ValueError("Training metadata differs from audited train split")
    if data.get("preprocessed_data_roots") or data.get("precomputed_sample_index_files"):
        raise ValueError("Stage-1 audit currently supports the single-root metadata dataset")
    if data.get("max_training_frames") is not None:
        raise ValueError("Stage 1 requires full-length targets")
    validation = config.get("validation") or {}
    allowed = report["evaluation_manifests"]
    if validation.get("from_train"):
        raise ValueError("Formal Stage-1 validation must use the frozen Oracle manifest")
    for field in ("metadata_jsonl", "extended_metadata_jsonl"):
        configured = validation.get(field)
        if configured and str(Path(configured).resolve()) not in allowed:
            raise ValueError(f"{field} is not an audited Oracle evaluation manifest")
    if not str(validation.get("metric_prefix", "")).startswith(oracle_prefix):
        raise ValueError(f"GT-conditioned metrics must use an {oracle_prefix} prefix")
    verify_snapshots(report["files"])
    return fingerprint


def validate_duration(
    source: dict[str, Any], metadata: dict[str, Any], video_shape: tuple[int, ...], tolerance_ms: float
) -> float:
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("duration_tolerance_ms must be a finite non-negative explicit value")
    fps = float(metadata["fps"])
    frames = metadata["target_frames"]
    height, width = metadata["target_height"], metadata["target_width"]
    if fps != 30.0 or not isinstance(frames, int) or frames < 1 or frames % 8 != 1:
        raise ValueError("Expected 30 FPS and F % 8 == 1")
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("Target resolution must be positive and divisible by 32")
    expected = (128, (frames - 1) // 8 + 1, height // 32, width // 32)
    if video_shape != expected:
        raise ValueError(f"Video latent shape {video_shape} != {expected}")
    duration = float(source["turn_duration_s"])
    speaker = float(source["speaker_speech_duration_s"])
    if not math.isfinite(duration) or not math.isfinite(speaker) or not 0 < speaker <= duration:
        raise ValueError("Invalid source timing")
    # Frame-count duration, not timestamp of the last frame (F-1)/fps.
    delta_ms = 1000.0 * (frames / fps - duration)
    if abs(delta_ms) > tolerance_ms + 1e-6:
        raise ValueError(f"VAE/Manifest duration delta {delta_ms:.6f} ms exceeds {tolerance_ms} ms")
    return delta_ms
