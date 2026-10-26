"""Strict loading and normalization for offline Route-A Omni conditions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

OmniConditionVersion = Literal[
    "teacher_forcing_raw",
    "generated_raw",
    "teacher_forcing",
    "generated",
]
ExpectedOmniConditionVersion = (
    OmniConditionVersion | tuple[OmniConditionVersion, ...] | None
)

ROUTE_A = "route_a_no_context"
ROUTE_B = "route_b_context"
VALID_PLANNER_ROUTES = (ROUTE_A, ROUTE_B)
PLAN_LAYERS = (12, 24, 36, 48)
PLAN_SEQUENCE_LENGTH = 44
PLAN_HIDDEN_SIZE = 1024
TEACHER_FORCING_FORMAT = "multi_depth_per_layer_v1"
RAW_PLAN_HIDDEN_SIZE = 2048
RAW_TEACHER_FORCING_FORMAT = "raw_multi_depth_plan_tokens_v2"
PLAN_FIELD_NAMES = ("semantics", "response", "events")
OMNI_SCHEMA_VERSION = 2
ALLOWED_OMNI_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "plan_latent",
        "plan_hidden_states",
        "plan_token_mask",
        "plan_field_mask",
        "plan_field_names",
        "plan_latent_layers",
        "plan_latent_format",
        "plan_hidden_state_format",
        "hard_controls",
        "audio_prompt",
        "video_prompt",
        "raw_plan_json",
        "planner_route",
        "causal_validated",
        "source",
        # Audited export metadata. These fields remain metric/debug-only.
        "metadata",
        "speaker_audio_tail_ms",
        "resampler_seed",
        "gen_time_s",
        "prompt_format",
    }
)


@dataclass(frozen=True)
class OmniConditionBundle:
    """Validated Omni condition data before optional trainable resampling."""

    plan_latent: Tensor
    omni_attention_mask: Tensor
    plan_field_mask: Tensor | None
    audio_prompt: str
    video_prompt: str
    raw_plan_json: dict[str, Any]
    planner_route: str
    causal_validated: bool
    source: str
    version: OmniConditionVersion
    sample_id: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return only conditions consumed by Stage-1 training."""
        return {
            "plan_latent": self.plan_latent,
            "omni_attention_mask": self.omni_attention_mask,
            "plan_field_mask": self.plan_field_mask,
            "audio_prompt": self.audio_prompt,
            "video_prompt": self.video_prompt,
            "raw_plan_json": self.raw_plan_json,
            "planner_route": self.planner_route,
            "causal_validated": self.causal_validated,
            "source": self.source,
            "version": self.version,
            "sample_id": self.sample_id,
        }


def _require_nonempty_string(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return value


def _normalize_plan_latent(
    payload: dict[str, Any],
    path: Path,
    expected_version: ExpectedOmniConditionVersion,
) -> tuple[Tensor, OmniConditionVersion]:
    raw_hidden_states = payload.get("plan_hidden_states")
    if isinstance(raw_hidden_states, Tensor):
        if (
            raw_hidden_states.ndim != 3
            or raw_hidden_states.shape[0] != len(PLAN_LAYERS)
            or raw_hidden_states.shape[1] < 1
            or raw_hidden_states.shape[2] != RAW_PLAN_HIDDEN_SIZE
        ):
            raise ValueError(
                f"{path}: plan_hidden_states must have shape [4,T,{RAW_PLAN_HIDDEN_SIZE}]"
            )
        if (
            not raw_hidden_states.is_floating_point()
            or not bool(torch.isfinite(raw_hidden_states).all())
        ):
            raise ValueError(f"{path}: plan_hidden_states must be finite floating point")
        if payload.get("plan_latent_layers") != list(PLAN_LAYERS):
            raise ValueError(f"{path}: plan_latent_layers must be {list(PLAN_LAYERS)}")
        if payload.get("plan_hidden_state_format") != RAW_TEACHER_FORCING_FORMAT:
            raise ValueError(
                f"{path}: plan_hidden_state_format must be {RAW_TEACHER_FORCING_FORMAT!r}"
            )
        version: OmniConditionVersion = (
            "generated_raw"
            if payload.get("source") == "generated"
            else "teacher_forcing_raw"
        )
        accepted_versions = (
            expected_version
            if isinstance(expected_version, tuple)
            else (expected_version,)
        )
        if expected_version is not None and version not in accepted_versions:
            raise ValueError(
                f"{path}: expected Omni version {expected_version!r}, found {version!r}"
            )
        return raw_hidden_states.contiguous(), version

    latent = payload.get("plan_latent")
    if not isinstance(latent, Tensor):
        raise ValueError(f"{path}: plan_latent must be a tensor")
    if not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
        raise ValueError(f"{path}: plan_latent must be a finite floating-point tensor")

    shape = tuple(latent.shape)
    if shape == (len(PLAN_LAYERS), PLAN_SEQUENCE_LENGTH, PLAN_HIDDEN_SIZE):
        version: OmniConditionVersion = "teacher_forcing"
        if payload.get("plan_latent_layers") != list(PLAN_LAYERS):
            raise ValueError(f"{path}: teacher_forcing plan_latent_layers must be {list(PLAN_LAYERS)}")
        if payload.get("plan_latent_format") != TEACHER_FORCING_FORMAT:
            raise ValueError(
                f"{path}: teacher_forcing plan_latent_format must be {TEACHER_FORCING_FORMAT!r}"
            )
    elif shape == (PLAN_SEQUENCE_LENGTH, PLAN_HIDDEN_SIZE):
        version = "generated"
        if payload.get("plan_latent_layers") not in (None, []):
            raise ValueError(f"{path}: generated plan_latent must not declare teacher-forcing layers")
        if payload.get("plan_latent_format") == TEACHER_FORCING_FORMAT:
            raise ValueError(f"{path}: generated plan_latent must not use the teacher-forcing format")
        latent = latent.unsqueeze(0)
    else:
        raise ValueError(
            f"{path}: plan_latent must have shape [4, 44, 1024] (teacher_forcing) "
            f"or [44, 1024] (generated), got {shape}"
        )

    accepted_versions = (
        expected_version if isinstance(expected_version, tuple) else (expected_version,)
    )
    if expected_version is not None and version not in accepted_versions:
        raise ValueError(f"{path}: expected Omni version {expected_version!r}, found {version!r}")
    return latent.contiguous(), version


def load_omni_condition_file(
    path: str | Path,
    *,
    expected_version: ExpectedOmniConditionVersion = None,
    expected_sample_id: str | None = None,
    expected_route: str | None = None,
) -> OmniConditionBundle:
    """Load and strictly validate one offline Route-A or Route-B ``.pt`` condition package."""
    condition_path = Path(path).expanduser()
    if not condition_path.is_file():
        raise FileNotFoundError(f"Omni condition file not found: {condition_path}")

    payload = torch.load(condition_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {condition_path}, got {type(payload).__name__}")
    unknown_keys = set(payload) - ALLOWED_OMNI_KEYS
    if unknown_keys:
        raise ValueError(
            f"{condition_path}: unknown Omni schema keys: {sorted(unknown_keys)}"
        )
    schema_version = payload.get("schema_version", 1)
    expected_schema = (
        OMNI_SCHEMA_VERSION if "plan_hidden_states" in payload else 1
    )
    if schema_version != expected_schema:
        raise ValueError(
            f"{condition_path}: expected schema_version={expected_schema}, "
            f"got {schema_version!r}"
        )

    latent, version = _normalize_plan_latent(payload, condition_path, expected_version)
    plan_field_mask: Tensor | None = None
    if version in ("teacher_forcing_raw", "generated_raw"):
        token_mask = payload.get("plan_token_mask")
        plan_field_mask = payload.get("plan_field_mask")
        if (
            not isinstance(token_mask, Tensor)
            or token_mask.dtype != torch.bool
            or tuple(token_mask.shape) != (latent.shape[1],)
            or not bool(token_mask.all())
        ):
            raise ValueError(
                f"{condition_path}: plan_token_mask must be all-true bool [T]"
            )
        if (
            not isinstance(plan_field_mask, Tensor)
            or plan_field_mask.dtype != torch.bool
            or tuple(plan_field_mask.shape)
            != (len(PLAN_FIELD_NAMES), latent.shape[1])
            or not bool(plan_field_mask.any(dim=1).all())
        ):
            raise ValueError(
                f"{condition_path}: plan_field_mask must be non-empty bool [3,T]"
            )
        if payload.get("plan_field_names") != list(PLAN_FIELD_NAMES):
            raise ValueError(
                f"{condition_path}: plan_field_names must be {list(PLAN_FIELD_NAMES)}"
            )
        metadata = payload.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("input_contract_version") != "speaker_frame_timing_v3"
            or set(metadata.get("planner_input_metadata", {}))
            != {"speaker_start_ms", "speaker_end_ms"}
            or not isinstance(metadata.get("speaker_audio_tail_ms"), int)
            or metadata["speaker_audio_tail_ms"] < 0
        ):
            raise ValueError(
                f"{condition_path}: raw features require speaker_frame_timing_v3 "
                "with a non-negative integer speaker_audio_tail_ms"
            )
    controls = payload.get("hard_controls")
    if version not in ("teacher_forcing_raw", "generated_raw") or controls is not None:
        if (
            not isinstance(controls, Tensor)
            or controls.ndim != 2
            or controls.shape[0] < 1
            or controls.shape[1] != 16
            or not controls.is_floating_point()
            or not bool(torch.isfinite(controls).all())
        ):
            shape = tuple(controls.shape) if isinstance(controls, Tensor) else None
            raise ValueError(
                f"{condition_path}: legacy hard_controls must be a finite "
                f"floating tensor [T, 16], got {shape}"
            )
    route = _require_nonempty_string(payload, "planner_route", condition_path)
    if expected_route is not None:
        if route != expected_route:
            raise ValueError(f"{condition_path}: expected planner_route {expected_route!r}, got {route!r}")
    elif route not in VALID_PLANNER_ROUTES:
        raise ValueError(
            f"{condition_path}: planner_route must be one of {VALID_PLANNER_ROUTES!r}, got {route!r}"
        )
    if payload.get("causal_validated") is not True:
        raise ValueError(f"{condition_path}: causal_validated must be exactly True")

    raw_plan = payload.get("raw_plan_json")
    if not isinstance(raw_plan, dict):
        raise ValueError(f"{condition_path}: raw_plan_json must be a dict")
    if version in ("teacher_forcing_raw", "generated_raw"):
        plan = raw_plan.get("avatar_plan")
        if not isinstance(plan, dict):
            plan = raw_plan
        timing = plan.get("timing_anchors")
        planner_input_metadata = payload["metadata"]["planner_input_metadata"]
        if (
            not isinstance(timing, dict)
            or timing.get("speaker_start_ms")
            != planner_input_metadata["speaker_start_ms"]
            or timing.get("speaker_end_ms")
            != planner_input_metadata["speaker_end_ms"]
            or timing.get("source") != "LOCKED_METADATA"
            or timing.get("locked") is not True
        ):
            raise ValueError(
                f"{condition_path}: raw_plan_json timing_anchors must exactly "
                "match metadata.planner_input_metadata"
            )

    sample_id_value = payload.get("sample_id")
    if sample_id_value is not None and (not isinstance(sample_id_value, str) or not sample_id_value.strip()):
        raise ValueError(f"{condition_path}: sample_id must be a non-empty string when present")
    if expected_sample_id is not None and sample_id_value != expected_sample_id:
        raise ValueError(
            f"{condition_path}: sample_id mismatch, expected {expected_sample_id!r}, got {sample_id_value!r}"
        )

    return OmniConditionBundle(
        plan_latent=latent,
        omni_attention_mask=torch.ones(latent.shape[:2], dtype=torch.bool),
        plan_field_mask=(
            plan_field_mask.contiguous() if plan_field_mask is not None else None
        ),
        audio_prompt=_require_nonempty_string(payload, "audio_prompt", condition_path),
        video_prompt=_require_nonempty_string(payload, "video_prompt", condition_path),
        raw_plan_json=raw_plan,
        planner_route=route,
        causal_validated=True,
        # Early Route-A exports predate the explicit source field. Their
        # tensor-format validation above is sufficient to derive an unambiguous
        # legacy source without confusing teacher-forcing and generated data.
        source=(
            _require_nonempty_string(payload, "source", condition_path)
            if "source" in payload
            else f"legacy_{version}"
        ),
        version=version,
        sample_id=sample_id_value,
    )


def load_listener_timbre_embedding(path: str | Path) -> Tensor:
    """Load a listener embedding and normalize ``[D]``/``[1, D]`` to ``[D]``."""
    embedding_path = Path(path).expanduser()
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Listener timbre embedding not found: {embedding_path}")
    value = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get("listener_timbre_embedding")
    if not isinstance(value, Tensor):
        raise ValueError(f"{embedding_path}: listener_timbre_embedding must be a tensor")
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{embedding_path}: listener_timbre_embedding must have shape [D] or [1, D]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{embedding_path}: listener_timbre_embedding must be finite and floating-point")
    return value.contiguous()
