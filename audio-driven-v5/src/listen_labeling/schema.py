"""Strict model responses and explicit conversion to the existing P2 contract."""
import copy
import json
import math
from .legacy_contract import EVENTS, validate, intersection_length
from . import VERSION

CLASSES = {"visual": ["head_nod"], "audio": ["listener_vocalization"], "omni": list(EVENTS)}


def parse(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate JSON field: " + k)
            out[k] = v
        return out
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def interval(pair, duration):
    if (not isinstance(pair, list) or len(pair) != 2 or
        any(type(x) not in (int, float) or not math.isfinite(x) for x in pair) or
        not 0 <= pair[0] < pair[1] <= duration):
        raise ValueError("Invalid/out-of-range interval")


def validate_discovery(value, branch, duration):
    modalities = ["visual", "audio"] if branch == "omni" else [branch]
    if not isinstance(value, dict) or set(value) != {"reviewed", "unknown", "events", "notes"}:
        raise ValueError("Expected discovery reviewed/unknown/events/notes")
    for field in ("reviewed", "unknown"):
        if not isinstance(value[field], dict) or set(value[field]) != set(modalities):
            raise ValueError("Discovery requires per-modality support, not class coverage")
        for spans in value[field].values():
            if not isinstance(spans, list):
                raise ValueError("Intervals must be lists")
            for span in spans:
                interval(span, duration)
    if not isinstance(value["notes"], str) or not isinstance(value["events"], list):
        raise ValueError("Invalid discovery notes/events")
    required = {"start", "end", "modality", "owner", "behavior", "evidence", "uncertainty", "boundary_censored"}
    for event in value["events"]:
        if not isinstance(event, dict) or not required <= set(event):
            raise ValueError("Missing open event observation/evidence fields")
        interval([event["start"], event["end"]], duration)
        if event["modality"] not in modalities or event["owner"] not in ("listener", "uncertain"):
            raise ValueError("Discovery modality/owner mismatch")
        if any(not isinstance(event[k], str) or not event[k].strip() for k in ("behavior", "evidence")):
            raise ValueError("Free-text behavior and direct evidence must be nonempty")
        if (not isinstance(event["uncertainty"], str) or type(event["boundary_censored"]) is not bool or
            (event["owner"] == "uncertain" and not event["uncertainty"].strip())):
            raise ValueError("Explicit uncertainty/censoring required")
        if "interpretation" in event and not isinstance(event["interpretation"], str):
            raise ValueError("Interpretation must remain a separate hypothesis string")
        if "confidence" in event and (type(event["confidence"]) not in (float, int) or
                                      not math.isfinite(event["confidence"]) or not 0 <= event["confidence"] <= 1):
            raise ValueError("Invalid optional confidence")
        if intersection_length(value["reviewed"][event["modality"]], event["start"], event["end"]) < event["end"] - event["start"] - 1e-6:
            raise ValueError("Open event outside inspected media support")
    return value


def validate_response(value, branch, duration, stage="audit"):
    if stage == "discovery":
        return validate_discovery(value, branch, duration)
    if stage != "audit":
        raise ValueError("Unknown annotation stage")
    if not isinstance(value, dict) or set(value) != {"reviewed", "unknown", "events", "notes"}:
        raise ValueError("Expected reviewed/unknown/events/notes response")
    for field in ("reviewed", "unknown"):
        if not isinstance(value[field], dict) or set(value[field]) != set(CLASSES[branch]):
            raise ValueError("Explicit per-class reviewed/unknown intervals required")
        for spans in value[field].values():
            if not isinstance(spans, list):
                raise ValueError("Intervals must be lists")
            for span in spans:
                interval(span, duration)
    if not isinstance(value["notes"], str) or not isinstance(value["events"], list):
        raise ValueError("Invalid notes/events")
    allowed = set(CLASSES[branch])
    for event in value["events"]:
        if not isinstance(event, dict) or not {"type", "start", "end", "owner", "description", "confidence", "boundary_censored"} <= set(event):
            raise ValueError("Event fields missing")
        if event["type"] not in allowed or event["owner"] != "listener":
            raise ValueError("Invalid event type or unresolved owner; use unknown intervals")
        interval([event["start"], event["end"]], duration)
        if (type(event["confidence"]) not in (float, int) or not math.isfinite(event["confidence"]) or
            not 0 <= event["confidence"] <= 1 or type(event["boundary_censored"]) is not bool or
            not isinstance(event["description"], str)):
            raise ValueError("Invalid confidence/description/censoring")
    # Also require every primary event to lie within explicitly reviewed support.
    canonical(value, branch, "validation", duration, {})
    return value


def canonical(value, branch, sid, duration, provenance):
    label = dict(schema="listen-events-v1", sample_id=sid, time_origin="clip_start_seconds",
                 duration_s=duration, taxonomy=list(EVENTS), coverage={k: [] for k in EVENTS},
                 unknown={k: [[0, duration]] for k in EVENTS}, events=[], provenance=provenance or {"validation": True})
    for name in CLASSES[branch]:
        label["coverage"][name] = copy.deepcopy(value["reviewed"][name])
        label["unknown"][name] = copy.deepcopy(value["unknown"][name])
    label["events"] = [dict(e, verification="automatic") for e in value["events"] if e["type"] in EVENTS]
    return validate(label, {"sample_id": sid, "requested_duration_s": duration})


def fuse(row, results, mode):
    provenance = dict(human_verified=False, media_hashes=row["media_hashes"],
                      annotator_version=VERSION, mapping_version="independent-primary-audit-v2",
                      mode=mode, branch_fingerprints={b: {s: v["fingerprint"] for s, v in r.items()} for b, r in results.items()})
    if "video_transform" in row:
        provenance["video_transform"] = copy.deepcopy(row["video_transform"])
    provenance["audio_role"] = row["audio_role"]
    provenance["training_admitted"] = row.get("training_admitted", True)
    if "source_reference" in row:
        provenance["source_reference"] = copy.deepcopy(row["source_reference"])
    branches = {"split": ["visual", "audio"], "omni": ["omni"], "visual": ["visual"]}[mode]
    if set(results) != set(branches) or any(set(r) != {"discovery", "audit"} for r in results.values()):
        raise ValueError("Missing annotation branch; cannot emit a complete draft")
    label = canonical(results[branches[0]]["audit"]["parsed"], branches[0], row["sample_id"], row["duration_s"], provenance)
    for branch in branches[1:]:
        part = canonical(results[branch]["audit"]["parsed"], branch, row["sample_id"], row["duration_s"], provenance)
        for name in CLASSES[branch]:
            label["coverage"][name] = part["coverage"][name]
            label["unknown"][name] = part["unknown"][name]
        label["events"].extend(part["events"])
    label["events"].sort(key=lambda e: (e["start"], e["type"]))
    validate(label, {"sample_id": row["sample_id"], "requested_duration_s": row["duration_s"]})
    return label


def review_reasons(label, results):
    reasons = ["human_review_pending"]
    if any(label["unknown"].values()):
        reasons.append("unknown_intervals")
    if any(e.get("boundary_censored") for e in label["events"]):
        reasons.append("boundary_censored")
    if any(e.get("confidence", 1) < .7 for r in results.values() for s in r.values() for e in s["parsed"]["events"]):
        reasons.append("low_self_reported_confidence")
    if any(e["uncertainty"].strip() for r in results.values() for e in r["discovery"]["parsed"]["events"]):
        reasons.append("open_event_uncertainty")
    if any(any(r["discovery"]["parsed"]["unknown"].values()) for r in results.values()):
        reasons.append("open_discovery_unknown")
    if not label["events"]:
        reasons.append("no_primary_events_check_misses")
    return reasons
