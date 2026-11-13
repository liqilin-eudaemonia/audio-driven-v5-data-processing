"""Human-GT event evaluation. Failed requests remain in recall denominators."""
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from listen_labeling.schema import CLASSES, validate_response
from p2_listen.contracts import read


def subtract(spans, excluded):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for left, right in excluded:
        pieces = []
        for start, end in merged:
            if right <= start or left >= end:
                pieces.append([start, end])
            else:
                if start < left:
                    pieces.append([start, left])
                if right < end:
                    pieces.append([right, end])
        merged = pieces
    return merged


def overlap(spans, start, end):
    return sum(max(0, min(b, end) - max(a, start)) for a, b in subtract(spans, []))


def supported(event, spans):
    return overlap(spans, event["start"], event["end"]) >= event["end"] - event["start"] - 1e-6


def match_events(predictions, truths, threshold):
    if not predictions or not truths:
        return []
    scores = np.zeros((len(predictions), len(truths)))
    for i, prediction in enumerate(predictions):
        for j, truth in enumerate(truths):
            intersection = max(0, min(prediction["end"], truth["end"]) - max(prediction["start"], truth["start"]))
            union = max(prediction["end"], truth["end"]) - min(prediction["start"], truth["start"])
            iou = intersection / union
            if intersection > 0 and iou >= threshold:
                # Maximize cardinality first, then summed IoU; never double-count a detection.
                scores[i, j] = min(scores.shape) + 1 + iou
    rows, cols = linear_sum_assignment(-scores)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if scores[i, j] > 0]


def validate_gt(ground_truth, samples, partition):
    source = {row["sample_id"]: row for row in samples}
    selected, seen = [], set()
    for row in ground_truth:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate GT sample")
        seen.add(row["sample_id"])
        if row.get("evaluation_partition") != partition:
            continue
        if row.get("human_verified") is not True or not row.get("reviewer", "").strip() or not row.get("reviewed_at"):
            raise ValueError("GT requires human_verified, reviewer and reviewed_at; empty template is not GT")
        sid = row["sample_id"]
        if sid not in source or row["media_hashes"] != source[sid]["media_hashes"]:
            raise ValueError("GT media identity differs from benchmark")
        if row.get("video_transform") != source[sid].get("video_transform"):
            raise ValueError("GT listener crop differs from benchmark")
        if row["duration_s"] != source[sid]["duration_s"]:
            raise ValueError("GT duration mismatch")
        validate_response(row["label"], "omni", row["duration_s"], "audit")
        if not any(row["label"]["reviewed"].values()):
            raise ValueError("GT has no inspected support")
        selected.append(row)
    if not selected:
        raise ValueError("No verified GT in requested partition")
    return selected


def evaluate_system(summary, inputs, ground_truth, partition="heldout", thresholds=(0.1, 0.3, 0.5)):
    if summary["auxiliary"]:
        raise ValueError("Auxiliary descriptions/transcripts/geometry are not event annotations")
    if summary["run_id"] != inputs["run_id"]:
        raise ValueError("Summary and input identity differ")
    selected = validate_gt(ground_truth, inputs["samples"], partition)
    results = {}
    for result in summary["results"]:
        if result["stage"] == "audit":
            sid = result["sample_id"]
            if sid in results:
                raise ValueError("Duplicate audit result")
            results[sid] = result
    groups = {"all": selected}
    for stratum in sorted({row.get("stratum", "unassigned") for row in selected}):
        groups["stratum:" + stratum] = [row for row in selected if row.get("stratum", "unassigned") == stratum]
    reports = {}
    for group_name, rows in groups.items():
        group = {"samples": len(rows), "classes": {}}
        for name in CLASSES[summary["branch"]]:
            report = {"gt_supported_seconds": 0., "predicted_supported_seconds": 0.,
                      "audit_failed_or_missing": 0, "ignored_predictions_outside_gt_support": 0,
                      "boundary_censored_predictions": 0, "metrics": {}}
            pairs = []
            for row in rows:
                label = row["label"]
                # Clip-edge GT events are excluded from the primary boundary metric.
                censored = [[e["start"], e["end"]] for e in label["events"] if e["type"] == name and e["boundary_censored"]]
                support = subtract(label["reviewed"][name], label["unknown"][name] + censored)
                report["gt_supported_seconds"] += sum(b - a for a, b in support)
                truth = [e for e in label["events"] if e["type"] == name and not e["boundary_censored"] and supported(e, support)]
                result = results.get(row["sample_id"])
                predicted = []
                if result and result["status"] == "schema_pass":
                    prediction = validate_response(result["parsed"], summary["branch"], row["duration_s"], "audit")
                    pred_support = subtract(prediction["reviewed"][name], prediction["unknown"][name])
                    report["predicted_supported_seconds"] += sum(overlap(pred_support, a, b) for a, b in support)
                    for event in prediction["events"]:
                        if event["type"] != name:
                            continue
                        if supported(event, support):
                            # A model cannot evade false positives by self-marking as censored.
                            predicted.append(event)
                            report["boundary_censored_predictions"] += int(event["boundary_censored"])
                        else:
                            report["ignored_predictions_outside_gt_support"] += 1
                else:
                    report["audit_failed_or_missing"] += 1
                pairs.append((predicted, truth))
            for threshold in thresholds:
                tp = fp = fn = 0
                onset, offset = [], []
                for predicted, truth in pairs:
                    matches = match_events(predicted, truth, threshold)
                    tp += len(matches); fp += len(predicted) - len(matches); fn += len(truth) - len(matches)
                    for i, j in matches:
                        onset.append(abs(predicted[i]["start"] - truth[j]["start"]))
                        offset.append(abs(predicted[i]["end"] - truth[j]["end"]))
                def errors(values):
                    return {"median_s": float(np.median(values)) if values else None,
                            "p90_s": float(np.quantile(values, .9)) if values else None}
                report["metrics"][str(threshold)] = {
                    "tp": tp, "fp": fp, "fn": fn, "gt_events": tp + fn,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                    "matched_onset_error": errors(onset), "matched_offset_error": errors(offset)}
            duration = report["gt_supported_seconds"]
            report["supported_fraction"] = report["predicted_supported_seconds"] / duration if duration else None
            group["classes"][name] = report
        reports[group_name] = group
    return {"system": summary["system"], "run_id": summary["run_id"], "partition": partition,
            "matching": "maximum-cardinality one-to-one temporal IoU; all GT positives count including failed/abstained requests",
            "groups": reports, "training_admitted": False,
            "limitations": ["Open discovery and ownership errors require separate human review",
                            "GT boundary-censored spans excluded from primary metrics",
                            "Model selection must use tuning split; heldout is for frozen final comparison"]}


def summarize(root):
    systems = []
    for path in sorted(Path(root).glob("*/summary.json")):
        report = read(path)
        results = report["results"]
        systems.append({"system": report["system"], "auxiliary": report["auxiliary"],
            "expected": report["expected_requests"], "completed": len(results),
            "first_attempt_schema_pass": sum(r["first_attempt_status"] == "schema_pass" for r in results),
            "latest_schema_pass": sum(r["status"] == "schema_pass" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "attempts": sum(r["attempts"] for r in results),
            "latest_inference_wall_seconds": sum(r.get("inference_wall_seconds", 0) for r in results),
            "not_accuracy": True})
    return {"systems": systems, "human_ground_truth": False, "training_admitted": False}


def evaluate_pair(visual_directory, audio_directory, ground_truth, partition="heldout"):
    directories = [Path(visual_directory), Path(audio_directory)]
    inputs = [read(path / "inputs.json") for path in directories]
    if inputs[0]["samples"] != inputs[1]["samples"]:
        raise ValueError("Split systems must use the exact same sample cohort and media identities")
    reports = []
    for directory, identity, branch in zip(directories, inputs, ("visual", "audio")):
        summary = read(directory / "summary.json")
        if summary["branch"] != branch:
            raise ValueError("Expected a pure visual system and a pure audio system")
        reports.append(evaluate_system(summary, identity, ground_truth, partition))
    groups = {}
    for name, group in reports[0]["groups"].items():
        groups[name] = dict(group, classes=dict(group["classes"], **reports[1]["groups"][name]["classes"]))
    return {"scheme": "independent_visual_audio", "systems": [r["system"] for r in reports],
            "partition": partition, "groups": groups, "training_admitted": False,
            "notes": "Each class uses its own branch; a failed branch does not erase the other branch"}
