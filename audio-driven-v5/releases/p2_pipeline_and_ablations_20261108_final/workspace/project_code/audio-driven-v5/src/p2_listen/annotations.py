"""Interval labels retain unknown support; no implicit negatives from omissions."""
import math
import torch
from .contracts import EVENTS, LABEL_SCHEMA, TYPE_TO_MODALITY, class_subset, positive, safe_id


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intersection_length(intervals, a, b):
    return sum(max(0., min(b, y) - max(a, x)) for x, y in union(intervals))


def validate(label, obs, require_admitted=True):
    allowed = {"schema", "sample_id", "time_origin", "duration_s", "taxonomy", "coverage", "unknown", "events", "provenance", "training_admitted"}
    if set(label) != allowed or label["schema"] != LABEL_SCHEMA or label["sample_id"] != obs["sample_id"]:
        raise ValueError("Label schema/identity mismatch")
    safe_id(label["sample_id"])
    if type(label["training_admitted"]) is not bool or require_admitted and not label["training_admitted"]:
        raise ValueError("Candidate labels have not been admitted")
    if label["time_origin"] != "clip_start_seconds" or label["taxonomy"] != list(EVENTS):
        raise ValueError("Label time origin/taxonomy mismatch")
    duration = positive(label["duration_s"], "label duration")
    if abs(duration - obs["requested_duration_s"]) > 1e-6:
        raise ValueError("Label duration mismatch")
    def interval(pair):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(type(t) in (float, int) and math.isfinite(t) for t in pair):
            raise ValueError("Invalid interval")
        a, b = pair
        if not 0 <= a < b <= duration:
            raise ValueError("Interval outside real support")
    for key in ("coverage", "unknown"):
        if set(label[key]) != set(EVENTS):
            raise ValueError("Explicit per-class coverage/unknown required")
        for values in label[key].values():
            for value in values:
                interval(value)
    for e in label["events"]:
        if e["type"] not in EVENTS:
            raise ValueError("Unknown event type")
        interval([e["start"], e["end"]])
        if intersection_length(label["coverage"][e["type"]], e["start"], e["end"]) < e["end"] - e["start"] - 1e-6:
            raise ValueError("Event outside annotated coverage")
        safe_id(e.get("event_id"))
        if not isinstance(e.get("description"), str) or not isinstance(e.get("source"), dict) or not e["source"]:
            raise ValueError("Original description and event source required")
        if e.get("verification") not in ("human", "weak_accepted"):
            raise ValueError("Event requires explicit human/weak acceptance")
        if e.get("boundary_status") not in ("verified", "weak_accepted", "censored"):
            raise ValueError("Explicit boundary status required")
        if e.get("boundary_censored", False) != (e["boundary_status"] == "censored"):
            raise ValueError("Inconsistent censored boundary")
    ids = [e["event_id"] for e in label["events"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate training event ID")
    p = label["provenance"]
    for key in ("mapping_version", "admission_policy"):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ValueError(f"Annotation {key} required")
    sources = p.get("coverage_sources", {})
    if set(sources) != set(EVENTS):
        raise ValueError("Per-class coverage sources required")
    for k in EVENTS:
        if label["coverage"][k] and (not isinstance(sources[k], dict) or not sources[k].get("reference") or
                                    sources[k].get("verification") not in ("human", "weak_accepted")):
            raise ValueError("Known coverage requires explicit reviewed evidence")
    if label["training_admitted"] and (not isinstance(p.get("admission_record"), dict) or not p["admission_record"].get("reference")):
        raise ValueError("Admission requires a recorded decision, not just a boolean")
    return label


def rasterize(label, layout, k_train=EVENTS):
    enabled = class_subset(k_train)
    result = {}
    for name in EVENTS:
        mod = TYPE_TO_MODALITY[name]
        events = [e for e in label["events"] if e["type"] == name]
        unknown = label["unknown"][name] + [[e["start"], e["end"]] for e in events if e.get("boundary_censored", False)]
        spans = [[e["start"], e["end"]] for e in events if not e.get("boundary_censored", False)]
        bins = layout[mod + "_intervals"]
        y = torch.zeros(bins.shape[:2], device=bins.device)
        mask = torch.zeros(bins.shape[:2], device=bins.device, dtype=torch.bool)
        supported = torch.zeros(bins.shape[:2], device=bins.device)
        valid = layout[mod + "_valid"] & layout.get(mod + "_target_valid", layout[mod + "_valid"])
        supported.scatter_add_(1, layout[mod + "_groups"], valid.to(supported.dtype))
        for b in range(bins.shape[0]):
            for i, (a, z) in enumerate(bins[b].tolist()):
                known = (name in enabled and label.get("training_admitted") is True and supported[b, i] > 0 and 0 <= a < z and
                         z <= label["duration_s"] + 1e-6 and
                         intersection_length(label["coverage"][name], a, z) >= z - a - 1e-6 and
                         intersection_length(unknown, a, z) < 1e-8)
                if known:
                    mask[b, i] = True
                    y[b, i] = intersection_length(spans, a, z) / (z - a)
        result[name] = dict(y=y, known=mask)
    return result


def class_statistics(all_labels, cap=10, k_train=EVENTS):
    enabled = class_subset(k_train)
    result = {}
    for name in EVENTS:
        pos = sum(float(x[name]["y"][x[name]["known"]].sum()) for x in all_labels)
        count = sum(int(x[name]["known"].sum()) for x in all_labels)
        reason = ("not_requested" if name not in enabled else "no_known_support" if count == 0 else
                  "no_positive_support" if pos <= 0 else "no_negative_support" if pos >= count else None)
        result[name] = dict(positive=pos, negative=count-pos, count=count,
                            enabled=reason is None, disabled_reason=reason,
                            pos_weight=min((count-pos) / max(pos, 1.), cap) if reason is None else 1.)
    return result


def mask_classes(labels, k_train):
    """Apply the train-frozen effective set without mutating the cached labels."""
    enabled = class_subset(k_train)
    return {k: dict(y=labels[k]["y"], known=labels[k]["known"] if k in enabled else torch.zeros_like(labels[k]["known"]))
            for k in EVENTS}


def validate_contract(label, cfg):
    for key in ("mapping_version", "admission_policy"):
        if label["provenance"][key] != cfg["events"][key]:
            raise ValueError(f"Label/config {key} mismatch")


def validate_reference(label):
    if label["provenance"].get("human_verified") is not True:
        raise ValueError("Dev/test references require human verification")
    if any(e["verification"] != "human" or e["boundary_status"] == "weak_accepted" for e in label["events"]):
        raise ValueError("Dev/test events and boundaries require human verification")
    if any(label["coverage"][k] and label["provenance"]["coverage_sources"][k].get("verification") != "human" for k in EVENTS):
        raise ValueError("Dev/test known coverage requires human verification")
