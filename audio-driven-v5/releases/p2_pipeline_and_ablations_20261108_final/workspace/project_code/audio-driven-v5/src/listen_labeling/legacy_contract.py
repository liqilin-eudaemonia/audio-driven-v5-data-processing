"""Frozen v1 label validation/rasterization for the historical annotation client.

Copied from pre-migration p2_listen/annotations.py on 2026-10-25.
These labels cannot be imported as v2 seven-class training targets.
"""
import math
import torch
from p2_listen.contracts import positive

EVENTS = ("listener_vocalization", "head_nod")


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


def validate(label, obs):
    allowed = {"schema", "sample_id", "time_origin", "duration_s", "taxonomy", "coverage", "unknown", "events", "provenance"}
    if set(label) != allowed or label["schema"] != "listen-events-v1" or label["sample_id"] != obs["sample_id"]:
        raise ValueError("Label schema/identity mismatch")
    if label["time_origin"] != "clip_start_seconds" or label["taxonomy"] != list(EVENTS):
        raise ValueError("Label time origin/taxonomy mismatch")
    duration = positive(label["duration_s"], "label duration")
    if abs(duration - obs["requested_duration_s"]) > 1e-6:
        raise ValueError("Label duration mismatch")
    def interval(pair):
        if len(pair) != 2 or not all(isinstance(t, (float, int)) and math.isfinite(t) for t in pair):
            raise ValueError("Invalid interval")
        a, b = pair
        if not 0 <= a < b <= duration + 1e-6:
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
    if not label["provenance"]:
        raise ValueError("Annotation provenance required")
    return label


def rasterize(label, layout):
    result = {}
    for name, mod in zip(EVENTS, ("audio", "video")):
        events = [e for e in label["events"] if e["type"] == name]
        unknown = label["unknown"][name] + [[e["start"], e["end"]] for e in events if e.get("boundary_censored", False)]
        spans = [[e["start"], e["end"]] for e in events if not e.get("boundary_censored", False)]
        bins = layout[mod + "_intervals"]
        y, mask = torch.zeros(bins.shape[:2]), torch.zeros(bins.shape[:2], dtype=torch.bool)
        for b in range(bins.shape[0]):
            for i, (a, z) in enumerate(bins[b].tolist()):
                known = (z <= label["duration_s"] + 1e-6 and
                         intersection_length(label["coverage"][name], a, z) >= z - a - 1e-6 and
                         intersection_length(unknown, a, z) < 1e-8)
                if known:
                    mask[b, i] = True
                    y[b, i] = intersection_length(spans, a, z) / (z - a)
        result[name] = dict(y=y, known=mask)
    return result


def class_statistics(all_labels, cap=10):
    result = {}
    for name in EVENTS:
        pos = sum(float(x[name]["y"][x[name]["known"]].sum()) for x in all_labels)
        count = sum(int(x[name]["known"].sum()) for x in all_labels)
        if count == 0 or pos <= 0 or pos >= count:
            raise ValueError(f"Need known positive and negative training support for {name}")
        result[name] = dict(positive=pos, negative=count-pos, count=count,
                            pos_weight=min((count-pos) / pos, cap))
    return result
