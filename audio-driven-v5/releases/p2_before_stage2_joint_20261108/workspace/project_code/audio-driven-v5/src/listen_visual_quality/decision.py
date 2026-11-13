"""Latest OR rule plus source-bound reviewed evidence; no hidden ID exceptions."""
from .io import canonical_hash, safe_id

ALLOWED_EXCLUDE_REASONS = {"sideways_video", "clip_face_detection_failed", "clip_face_measurement_failed", "user_rejected_visual_clarity"}


def evidence_index(value):
    result = {}
    for item in value.get("records", []):
        sid = safe_id(item["sample_id"])
        if sid in result:
            raise ValueError("Duplicate reviewed evidence")
        if not item.get("source_sha256") or not item.get("provenance"):
            raise ValueError("Reviewed evidence must be source-bound and attributable")
        if not item.get("exclude_reasons") or not set(item["exclude_reasons"]) <= ALLOWED_EXCLUDE_REASONS:
            raise ValueError("Unsupported reviewed evidence")
        result[sid] = item
    return result


def decide(record, policy, evidence=None):
    sid = record["id"]
    item = (evidence or {}).get(sid)
    reasons, review_reasons = [], []
    if item:
        if canonical_hash(record["source"]) != item["source_sha256"]:
            raise ValueError("Reviewed evidence belongs to a different source/crop/time: " + sid)
        reasons.extend(item["exclude_reasons"])
    if record["status"] != "ok":
        review_reasons.append("measurement_execution_error_retry_required")
    else:
        q = record["focus"]
        if q["longest_low_s"] >= policy["min_continuous_low_s"] - 1e-9:
            reasons.append("continuous_low_clarity_at_least_2s")
        if q["low_fraction"] >= policy["min_low_fraction"] - 1e-9:
            reasons.append("cumulative_low_clarity_at_least_25pct")
        detected = sum(bool(x["detected"]) for x in record["raw_measurements"])
        if not detected:
            reasons.append("clip_face_detection_failed")
        elif q["measured_fraction"] < policy["minimum_measurement_support"]:
            # An uncalibrated global 80% exclusion would contradict accepted partial-occlusion examples.
            review_reasons.append("partial_measurement_requires_review")
    decision = "exclude" if reasons else "review" if review_reasons else "retain"
    return {"sample_id": sid, "decision": decision, "reasons": sorted(set(reasons)),
            "review_reasons": review_reasons, "source_sha256": canonical_hash(record["source"]),
            "training_admitted": False, "production_applied": False}
