"""Strict contracts for autonomous Generated-Plan end-to-end evaluation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENERATED_EVALUATION_FORMAT = "omni_stage2_generated_e2e_v1"


@dataclass(frozen=True)
class GeneratedPlanSettings:
    max_new_tokens: int
    max_planned_duration_ms: int
    validation_mode: str


def split_generated_protocol(
    raw: dict[str, Any],
) -> tuple[GeneratedPlanSettings, dict[str, Any]]:
    """Separate Planner generation controls from the downstream AV protocol."""
    protocol = copy.deepcopy(raw)
    generated = protocol.pop("generated_plan_evaluation", None)
    if not isinstance(generated, dict):
        raise ValueError("generated_plan_evaluation must be an object")
    expected = {"format", "max_new_tokens", "max_planned_duration_ms", "validation_mode"}
    required = expected - {"validation_mode"}
    if set(generated) - expected or not required.issubset(generated):
        raise ValueError(
            "generated_plan_evaluation keys must be exactly "
            f"{sorted(expected)}, got {sorted(generated)}"
        )
    if generated["format"] != GENERATED_EVALUATION_FORMAT:
        raise ValueError(
            f"generated_plan_evaluation.format must be {GENERATED_EVALUATION_FORMAT!r}"
        )
    max_new_tokens = generated["max_new_tokens"]
    max_duration = generated["max_planned_duration_ms"]
    validation_mode = generated.get("validation_mode", "formal")
    if validation_mode not in {"formal", "preview"}:
        raise ValueError("validation_mode must be 'formal' or 'preview'")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        isinstance(max_duration, bool)
        or not isinstance(max_duration, int)
        or max_duration < 1
    ):
        raise ValueError("max_planned_duration_ms must be a positive integer")
    if "omni_conditions_dir" in protocol or "omni_condition_version" in protocol:
        raise ValueError(
            "Generated evaluation creates its own conditions; "
            "the protocol must not supply Omni conditions or a version"
        )
    return GeneratedPlanSettings(max_new_tokens, max_duration, validation_mode), protocol


def fixed_probe_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the exact ordered probe set used for autonomous generation."""
    manifest = Path(protocol["metadata_jsonl"]).expanduser()
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requested = protocol.get("jsonl_max_samples")
    if not isinstance(requested, int) or requested < 1:
        raise ValueError("Generated evaluation requires a positive jsonl_max_samples")
    rows = rows[:requested]
    sample_ids = [row.get("sample_id") for row in rows]
    if (
        len(rows) != requested
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != requested
    ):
        raise ValueError(
            f"Generated evaluation requires exactly {requested} unique non-empty sample IDs"
        )
    return rows


def shard_sample_ids(sample_ids: list[str], nproc: int) -> list[list[str]]:
    """Round-robin unique samples across at most one Planner process per GPU."""
    if nproc < 1 or not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Planner sharding requires positive nproc and unique samples")
    worker_count = min(nproc, len(sample_ids))
    shards = [sample_ids[index::worker_count] for index in range(worker_count)]
    if any(not shard for shard in shards):
        raise RuntimeError("Planner sharding created an empty worker")
    if sorted(sample for shard in shards for sample in shard) != sorted(sample_ids):
        raise RuntimeError("Planner sharding lost or duplicated samples")
    return shards


def physical_cuda_devices(nproc: int, visible: str | None) -> list[str]:
    """Map watcher-local ranks back to physical CUDA IDs for child processes."""
    devices = (
        [item.strip() for item in visible.split(",") if item.strip()]
        if visible
        else [str(index) for index in range(nproc)]
    )
    if len(devices) != nproc:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain exactly nproc devices for "
            "Generated-Plan evaluation"
        )
    return devices


def gpu_uuids_for_devices(
    devices: list[str], inventory: list[tuple[str, str]]
) -> set[str]:
    """Resolve physical CUDA indices or UUIDs to the assigned GPU UUIDs."""
    by_index = {index: uuid for index, uuid in inventory}
    known_uuids = set(by_index.values())
    resolved = {
        by_index[device] if device in by_index else device
        for device in devices
    }
    unknown = resolved - known_uuids
    if unknown:
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains unknown GPUs: {sorted(unknown)}")
    return resolved


def validate_generation_results(
    results: Any,
    *,
    requested_ids: list[str],
    condition_dir: Path,
) -> tuple[list[str], dict[str, int], dict[str, dict[str, Any]]]:
    """Validate complete Planner accounting and return accepted IDs in probe order."""
    if not isinstance(results, list):
        raise ValueError("Planner results must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("Every Planner result must be an object")
        sample_id = row.get("sample_id")
        if sample_id not in requested_ids or sample_id in by_id:
            raise ValueError(f"Unexpected or duplicate Planner result: {sample_id!r}")
        status = row.get("status")
        if status not in {"passed", "rejected"}:
            raise ValueError(f"Invalid Planner status for {sample_id}: {status!r}")
        if status == "rejected" and not isinstance(row.get("reason"), str):
            raise ValueError(f"Rejected Planner result lacks a reason: {sample_id}")
        by_id[sample_id] = row
    if set(by_id) != set(requested_ids):
        missing = sorted(set(requested_ids) - set(by_id))
        raise ValueError(f"Planner results are incomplete; missing {missing}")
    accepted = [
        sample_id
        for sample_id in requested_ids
        if by_id[sample_id]["status"] == "passed"
    ]
    for sample_id in accepted:
        if not (condition_dir / f"{sample_id}.pt").is_file():
            raise ValueError(f"Accepted Planner result has no generated_raw package: {sample_id}")
        row = by_id[sample_id]
        duration = row.get("planned_duration_ms")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError(f"Accepted Planner result has invalid duration: {sample_id}")
        for key in ("video_prompt", "audio_prompt"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Accepted Planner result has invalid {key}: {sample_id}")
    reasons: dict[str, int] = {}
    for sample_id in requested_ids:
        row = by_id[sample_id]
        if row["status"] == "rejected":
            reason = row["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    return accepted, reasons, {sample_id: by_id[sample_id] for sample_id in accepted}


def write_accepted_metadata(
    rows: list[dict[str, Any]],
    accepted_ids: list[str],
    generated: dict[str, dict[str, Any]],
    caption_field: str,
    output: Path,
) -> None:
    """Write accepted rows with generated text/duration and no GT future caption."""
    accepted = set(accepted_ids)
    selected = []
    for original in rows:
        sample_id = original["sample_id"]
        if sample_id not in accepted:
            continue
        row = copy.deepcopy(original)
        result = generated[sample_id]
        fps = row.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or fps <= 0
        ):
            raise ValueError(f"Probe row has invalid fps: {sample_id}")
        duration_ms = result["planned_duration_ms"]
        frames = max(1, math.ceil(duration_ms * float(fps) / 1000))
        frames += (1 - frames) % 8
        row[caption_field] = result["video_prompt"]
        row["target_frames"] = frames
        row["duration_s"] = duration_ms / 1000
        row["generated_plan_duration_ms"] = duration_ms
        row["evaluation_condition_source"] = "generated_plan"
        response_text = result.get("response_text", "")
        if not isinstance(response_text, str):
            raise ValueError(f"Generated response_text must be a string: {sample_id}")
        response_text = " ".join(response_text.split())
        row["qwenasr"] = response_text
        row["_syncnet_applicable"] = bool(response_text)
        row["_validation_category"] = (
            "single_speaker" if response_text else "generated_plan_silence"
        )
        row["_visible_speaker_count"] = 1
        row["_dialogue_speaker_ids"] = ["listener"] if response_text else []
        selected.append(row)
    if [row["sample_id"] for row in selected] != accepted_ids:
        raise ValueError("Accepted sample order disagrees with the fixed probe order")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )


def generated_av_protocol(
    protocol: dict[str, Any],
    *,
    metadata_jsonl: Path,
    condition_dir: Path,
    accepted_count: int,
) -> dict[str, Any]:
    """Bind AV generation strictly to generated_raw packages, never GT fallback."""
    if accepted_count < 1:
        raise ValueError("Cannot build an AV protocol without an accepted generated Plan")
    result = copy.deepcopy(protocol)
    result["metadata_jsonl"] = str(metadata_jsonl.resolve())
    result["jsonl_max_samples"] = accepted_count
    result["omni_conditions_dir"] = str(condition_dir.resolve())
    result["omni_condition_version"] = "generated_raw"
    result["metric_prefix"] = f"{result['metric_prefix']}.generated_plan"
    if accepted_count < 8:
        # Preview evaluation still reports every metric, while explicitly
        # opting out of the production protocol's eight-sample stability gate.
        result["allow_small_sample_metrics"] = True
    return result


def end_to_end_summary(
    *,
    requested_count: int,
    accepted_count: int,
    rejection_reasons: dict[str, int],
    av_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Report validity and unconditional end-to-end success without hiding rejects."""
    if requested_count < 1 or not 0 <= accepted_count <= requested_count:
        raise ValueError("Invalid Generated-Plan evaluation counts")
    av_count = 0
    if av_metrics is not None:
        sample_count = av_metrics.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or not 0 <= sample_count <= accepted_count
        ):
            raise ValueError("AV metrics contain an invalid sample_count")
        av_count = sample_count
    return {
        "format": GENERATED_EVALUATION_FORMAT,
        "requested_count": requested_count,
        "plan_accepted_count": accepted_count,
        "plan_rejected_count": requested_count - accepted_count,
        "plan_validity_rate": accepted_count / requested_count,
        "rejection_reasons": rejection_reasons,
        "av_completed_count": av_count,
        "conditional_av_success_rate": av_count / accepted_count
        if accepted_count
        else None,
        "end_to_end_success_rate": av_count / requested_count,
        "av_metrics_on_accepted_plans": av_metrics,
        "gt_plan_fallback_count": 0,
    }
