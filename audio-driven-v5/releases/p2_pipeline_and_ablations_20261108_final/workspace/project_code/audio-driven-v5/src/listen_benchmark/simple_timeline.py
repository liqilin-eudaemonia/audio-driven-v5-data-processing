"""Minimal event timeline contract for the user-requested concise prompt experiment."""
import math


def response_schema(duration):
    time = dict(type="number", minimum=0, maximum=duration)
    event = dict(type="object", properties=dict(start=time, end=time,
                 modality=dict(type="string", enum=["visual", "audio"]), behavior=dict(type="string")),
                 required=["start", "end", "modality", "behavior"], additionalProperties=False)
    return dict(type="object", properties=dict(events=dict(type="array", items=event)),
                required=["events"], additionalProperties=False)


def validate_timeline(value, duration):
    if not isinstance(value, dict) or set(value) != {"events"} or not isinstance(value["events"], list):
        raise ValueError("Expected an events array")
    for event in value["events"]:
        if not isinstance(event, dict) or set(event) != {"start", "end", "modality", "behavior"}:
            raise ValueError("Expected exactly start, end, modality, behavior")
        if any(type(event[k]) not in (int, float) or not math.isfinite(event[k]) for k in ("start", "end")):
            raise ValueError("Event times must be finite numbers")
        # Window subtraction may produce 2.7799999999999994 for a 2.78 s end.
        # Accept numerical roundoff only; never clamp or rewrite model times.
        if not (0 <= event["start"] < event["end"] and event["end"] <= duration + 1e-9):
            raise ValueError("Invalid event interval")
        if event["modality"] not in ("visual", "audio") or not isinstance(event["behavior"], str) or not event["behavior"].strip():
            raise ValueError("Expected a modality and nonempty description")
    # Preserve the actual model response; do not fabricate inspected coverage,
    # uncertainty, source ownership, or event evidence absent from this contract.
    return value
