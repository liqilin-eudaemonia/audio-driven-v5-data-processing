"""Generated Plan checks adapted to the actual Stage-1 GT schema.

Source: dialogue_context_tagging/scripts/extract_avatar_plan_features.py
SHA256: 66898e9ada293dc2f57747677045c30f40ed7e7ec01d72b085d45719694f6546
Adaptations: normalized avatar_plan-only wrapper, interview-two-stem version,
and locked source field; no invented annotation_quality or repaired JSON.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from ltx_trainer.omni_stage2_contract import PlanCompiler

AVATAR_PLAN_VERSION = "1.3-interview-two-stem"
AVATAR_PLAN_POLICY_ID = "LISTENER_ONLY_STRICT_V1"

SEMANTIC_ENUMS = {
    "dialogue_act": {
        "EMPATHIZE",
        "ACKNOWLEDGE",
        "AGREE",
        "ANSWER",
        "CLARIFY",
        "INFORM",
        "OTHER",
    },
    "tone": {
        "GENTLE",
        "SERIOUS",
        "ENTHUSIASTIC",
        "PLAYFUL",
        "CASUAL",
        "THOUGHTFUL",
    },
    "primary_emotion": {
        "NEUTRAL",
        "CONCERN",
        "HAPPINESS",
        "AMUSEMENT",
        "SADNESS",
        "SURPRISE",
    },
}

FLOOR_STATES = frozenset(PlanCompiler.FLOOR_ENUM_MAP)

VOCAL_STATES = frozenset({"SILENCE", "BACKCHANNEL", "PARALINGUISTIC", "REPLY_SPEECH"})

VISUAL_EVENT_TYPES = frozenset(PlanCompiler.VISUAL_EVENT_MAP)


def derive_planned_duration_ms(
    avatar_plan: dict[str, Any],
    *,
    speaker_end_ms: int,
) -> int:
    """Validate the duration autonomously selected by the generated Planner."""
    timebase = avatar_plan.get("timebase")
    if not isinstance(timebase, dict) or timebase.get("unit") != "ms":
        raise ValueError("avatar_plan.timebase.unit must be 'ms'")
    duration = timebase.get("duration")
    if isinstance(duration, bool):
        raise ValueError("avatar_plan.timebase.duration must be an integer")
    try:
        duration_ms = int(duration)
    except (TypeError, ValueError) as exc:
        raise ValueError("avatar_plan.timebase.duration must be an integer") from exc
    if duration_ms != duration or duration_ms < speaker_end_ms:
        raise ValueError(
            "avatar_plan.timebase.duration must be an integer >= speaker_end_ms; "
            f"got duration={duration!r}, speaker_end_ms={speaker_end_ms}"
        )
    return duration_ms


def validate_generated_avatar_plan(
    parsed: Any,
    *,
    speaker_start_ms: int,
    speaker_end_ms: int,
    max_planned_duration_ms: int,
) -> tuple[dict[str, Any], int]:
    """Validate generated output without repairing or clipping the Plan."""
    # Stage-2 supervision uses this exact normalized wrapper (no teacher audit output).
    if not isinstance(parsed, dict) or set(parsed) != {"avatar_plan"}:
        raise ValueError("output must contain exactly avatar_plan")
    avatar_plan = parsed["avatar_plan"]
    if not isinstance(avatar_plan, dict):
        raise ValueError("avatar_plan must be an object")

    if avatar_plan.get("avatar_plan_version") != AVATAR_PLAN_VERSION:
        raise ValueError(f"avatar_plan_version must be {AVATAR_PLAN_VERSION!r}")
    if avatar_plan.get("policy_id") != AVATAR_PLAN_POLICY_ID:
        raise ValueError(f"policy_id must be {AVATAR_PLAN_POLICY_ID!r}")

    anchors = avatar_plan.get("timing_anchors")
    if not isinstance(anchors, dict):
        raise ValueError("timing_anchors must be an object")
    expected_anchors = {
        "speaker_start_ms": speaker_start_ms,
        "speaker_end_ms": speaker_end_ms,
        "locked": True,
        "source": "LOCKED_METADATA",
    }
    if anchors != expected_anchors:
        raise ValueError(
            "timing_anchors must exactly copy CAUSAL_TIMING and locked=true; "
            f"got {anchors!r}, expected {expected_anchors!r}"
        )

    duration_ms = derive_planned_duration_ms(
        avatar_plan,
        speaker_end_ms=speaker_end_ms,
    )
    if duration_ms > max_planned_duration_ms:
        raise ValueError(f"planned duration {duration_ms} exceeds safety limit {max_planned_duration_ms} ms")

    semantics = avatar_plan.get("semantics")
    if not isinstance(semantics, dict):
        raise ValueError("semantics must be an object")
    for field, allowed in SEMANTIC_ENUMS.items():
        value = semantics.get(field)
        if value not in allowed:
            raise ValueError(f"semantics.{field} has invalid value {value!r}")
    intensity = semantics.get("expressive_intensity")
    if (
        isinstance(intensity, bool)
        or not isinstance(intensity, (int, float))
        or not math.isfinite(float(intensity))
        or not 0.0 <= float(intensity) <= 1.0
    ):
        raise ValueError("semantics.expressive_intensity must be finite in [0, 1]")

    response = avatar_plan.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("present"), bool):
        raise ValueError("response.present must be boolean")
    response_present = response["present"]
    response_text = response.get("text")
    if not isinstance(response_text, str):
        raise ValueError("response.text must be a string")
    if response_present:
        if response.get("id") != "main_reply" or not response_text.strip():
            raise ValueError("present response requires id='main_reply' and non-empty text")
    elif response.get("id") is not None or response_text:
        raise ValueError("absent response requires id=null and empty text")
    if response.get("language") != "en":
        raise ValueError("response.language must be 'en'")
    prosody = response.get("prosody")
    if not isinstance(prosody, dict):
        raise ValueError("response.prosody must be an object")
    for field, allowed in {
        "pace": {"SLOW", "MEDIUM", "FAST"},
        "energy": {"LOW", "MEDIUM", "HIGH"},
        "pitch": {"LOW", "MEDIUM", "HIGH"},
        "volume": {"SOFT", "MEDIUM", "LOUD"},
    }.items():
        if prosody.get(field) not in allowed:
            raise ValueError(f"response.prosody.{field} has invalid value {prosody.get(field)!r}")

    def integer_time(value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        return value

    def validate_covering_track(
        name: str,
        *,
        state_key: str | None,
        allowed_states: frozenset[str] | None,
    ) -> list[dict[str, Any]]:
        track = avatar_plan.get(name)
        if not isinstance(track, list) or not track:
            raise ValueError(f"{name} must be a non-empty list")
        cursor = 0
        for index, segment in enumerate(track):
            path = f"{name}[{index}]"
            if not isinstance(segment, dict):
                raise ValueError(f"{path} must be an object")
            start = integer_time(segment.get("start"), f"{path}.start")
            end = integer_time(segment.get("end"), f"{path}.end")
            if start != cursor or end <= start or end > duration_ms:
                raise ValueError(
                    f"{path} must form a contiguous positive interval; "
                    f"got [{start}, {end}), expected start={cursor}, "
                    f"duration={duration_ms}"
                )
            if state_key is not None:
                state = segment.get(state_key)
                if state not in allowed_states:
                    raise ValueError(f"{path}.{state_key} has invalid value {state!r}")
            cursor = end
        if cursor != duration_ms:
            raise ValueError(f"{name} ends at {cursor}, expected duration {duration_ms}")
        return track

    floor_track = validate_covering_track("floor_track", state_key="state", allowed_states=FLOOR_STATES)
    vocal_track = validate_covering_track("vocal_track", state_key="state", allowed_states=VOCAL_STATES)
    validate_covering_track("visual_state_track", state_key=None, allowed_states=None)

    for index, segment in enumerate(floor_track):
        if segment["start"] < speaker_end_ms and segment["state"] != "LISTEN":
            raise ValueError(f"floor_track[{index}] must remain LISTEN before speaker_end_ms")
        if segment["state"] == "REPLY" and segment["start"] < speaker_end_ms:
            raise ValueError(f"floor_track[{index}] starts REPLY before speaker_end_ms")
    reply_floor = [seg for seg in floor_track if seg["state"] == "REPLY"]
    reply_vocal = [seg for seg in vocal_track if seg["state"] == "REPLY_SPEECH"]
    for index, segment in enumerate(reply_vocal):
        if segment["start"] < speaker_end_ms:
            raise ValueError(f"vocal_track[{index}] starts REPLY_SPEECH before speaker_end_ms")
    if response_present and (not reply_floor or not reply_vocal):
        raise ValueError("present response requires REPLY and REPLY_SPEECH intervals")
    if not response_present and (reply_floor or reply_vocal):
        raise ValueError("absent response cannot contain REPLY or REPLY_SPEECH intervals")

    events = avatar_plan.get("visual_events")
    if not isinstance(events, list):
        raise ValueError("visual_events must be a list")
    for index, event in enumerate(events):
        path = f"visual_events[{index}]"
        if not isinstance(event, dict):
            raise ValueError(f"{path} must be an object")
        start = integer_time(event.get("start"), f"{path}.start")
        end = integer_time(event.get("end"), f"{path}.end")
        if start < 0 or end <= start or end > duration_ms:
            raise ValueError(f"{path} interval [{start}, {end}) is outside Plan duration")
        if event.get("type") not in VISUAL_EVENT_TYPES:
            raise ValueError(f"{path}.type has invalid value {event.get('type')!r}")

    return avatar_plan, duration_ms


def validate_generated_avatar_plan_preview(
    parsed: Any,
    *,
    speaker_start_ms: int,
    speaker_end_ms: int,
    max_planned_duration_ms: int,
) -> tuple[dict[str, Any], int, list[str]]:
    """Apply hard causal/time gates and deterministic soft-field defaults for preview."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("avatar_plan"), dict):
        raise ValueError("output must contain an avatar_plan object")
    plan = copy.deepcopy(parsed["avatar_plan"])
    warnings: list[str] = []
    expected_anchors = {
        "speaker_start_ms": speaker_start_ms,
        "speaker_end_ms": speaker_end_ms,
        "locked": True,
        "source": "LOCKED_METADATA",
    }
    if plan.get("timing_anchors") != expected_anchors:
        raise ValueError("timing_anchors disagree with causal speaker timing")
    duration = derive_planned_duration_ms(plan, speaker_end_ms=speaker_end_ms)
    if duration > max_planned_duration_ms:
        raise ValueError(f"planned duration {duration} exceeds safety limit {max_planned_duration_ms} ms")

    semantics = plan.get("semantics") if isinstance(plan.get("semantics"), dict) else {}
    defaults = {"dialogue_act": "ACKNOWLEDGE", "tone": "GENTLE", "primary_emotion": "NEUTRAL"}
    for field, default in defaults.items():
        if semantics.get(field) not in SEMANTIC_ENUMS[field]:
            semantics[field] = default
            warnings.append(f"defaulted semantics.{field}")
    intensity = semantics.get("expressive_intensity")
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not math.isfinite(float(intensity)) or not 0 <= float(intensity) <= 1:
        semantics["expressive_intensity"] = 0.5
        warnings.append("defaulted semantics.expressive_intensity")
    plan["semantics"] = semantics

    response = plan.get("response") if isinstance(plan.get("response"), dict) else {}
    present = response.get("present") is True and isinstance(response.get("text"), str) and bool(response["text"].strip())
    if not present:
        response = {"present": False, "id": None, "text": "", "language": "en", "prosody": {}}
        warnings.append("defaulted response to absent")
    else:
        response.setdefault("id", "main_reply")
        response.setdefault("language", "en")
        prosody = response.get("prosody") if isinstance(response.get("prosody"), dict) else {}
        for field, default in {"pace": "MEDIUM", "energy": "LOW", "pitch": "MEDIUM", "volume": "SOFT"}.items():
            if field not in prosody:
                prosody[field] = default
                warnings.append(f"defaulted response.prosody.{field}")
        response["prosody"] = prosody
    plan["response"] = response

    def covering_track(name: str, default_state: str, allowed_states: frozenset[str]) -> list[dict[str, Any]]:
        value = plan.get(name)
        valid = isinstance(value, list)
        cursor = 0
        if valid:
            for segment in value:
                if not isinstance(segment, dict) or segment.get("start") != cursor or not isinstance(segment.get("end"), int) or segment["end"] <= cursor or segment["end"] > duration or segment.get("state") not in allowed_states:
                    valid = False
                    break
                cursor = segment["end"]
        if not valid or cursor != duration:
            warnings.append(f"defaulted {name}")
            return [{"start": 0, "end": duration, "state": default_state}]
        return value

    plan["floor_track"] = covering_track("floor_track", "LISTEN", FLOOR_STATES)
    plan["vocal_track"] = covering_track("vocal_track", "SILENCE", VOCAL_STATES)
    visual = plan.get("visual_state_track")
    if not isinstance(visual, list) or not visual or visual[0].get("start") != 0 or visual[-1].get("end") != duration:
        plan["visual_state_track"] = [{"start": 0, "end": duration, "expression": "ATTENTIVE"}]
        warnings.append("defaulted visual_state_track")
    events = plan.get("visual_events")
    if not isinstance(events, list):
        events = []
        warnings.append("defaulted visual_events")
    plan["visual_events"] = [event for event in events if isinstance(event, dict) and isinstance(event.get("start"), int) and isinstance(event.get("end"), int) and 0 <= event["start"] < event["end"] <= duration and event.get("type") in VISUAL_EVENT_TYPES]
    return plan, duration, warnings
