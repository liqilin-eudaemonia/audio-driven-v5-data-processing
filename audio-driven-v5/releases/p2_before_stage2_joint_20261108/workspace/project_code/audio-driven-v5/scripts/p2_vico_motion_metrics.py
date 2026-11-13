"""Auditable reproduction of published motion metric definitions, not author code.

All inputs are raw Deep3DFaceRecon coefficients on the same 30 Hz frame grid.
Correlation choices are explicitly documented in the adjacent protocol.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = {"exp": slice(0, 64), "angle": slice(64, 67),
          "trans": slice(67, 70), "pose": slice(64, 70)}


def frechet(x, y):
    """Gaussian Wasserstein squared distance; unbiased sample covariances."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("FID requires two matrices with >=2 observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite features")
    if np.array_equal(x, y):
        return 0.0
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.atleast_2d(np.cov(x, rowvar=False)), np.atleast_2d(np.cov(y, rowvar=False))
    ev, v = np.linalg.eigh((cx + cx.T) / 2)
    root = (v * np.sqrt(np.maximum(ev, 0))) @ v.T
    product = root @ cy @ root
    cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh((product + product.T) / 2), 0)).sum()
    value = float((mx-my) @ (mx-my) + np.trace(cx) + np.trace(cy) - 2*cross_trace)
    if value < -1e-7 * max(1, np.trace(cx) + np.trace(cy)):
        raise ValueError(f"Numerically invalid FID: {value}")
    return max(value, 0.0)


def tlcc(speaker, listener, max_lag=60):
    """Per-coordinate Pearson correlation. Positive lag = listener follows speaker.

    Undefined constant-coordinate correlations are set to zero and counted.
    Only overlapping frames participate; each overlap is centered separately.
    """
    speaker, listener = np.asarray(speaker, np.float64), np.asarray(listener, np.float64)
    if speaker.shape != listener.shape or len(speaker) <= max_lag + 1:
        raise ValueError("Unaligned or too-short TLCC sequence")
    rows, undefined = [], 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            s, l = speaker[:-lag], listener[lag:]
        elif lag < 0:
            s, l = speaker[-lag:], listener[:lag]
        else:
            s, l = speaker, listener
        s, l = s-s.mean(0), l-l.mean(0)
        den = np.sqrt(np.sum(s*s, 0) * np.sum(l*l, 0))
        valid = den > 1e-12
        rows.append(np.divide(np.sum(s*l, 0), den, out=np.zeros_like(den), where=valid))
        undefined += int((~valid).sum())
    return np.stack(rows), undefined


def score_clip(speaker, gt, pred, max_lag=60, window=120, stride=30):
    if speaker.shape != gt.shape or gt.shape != pred.shape or gt.shape[1] != 70:
        raise ValueError(f"Expected equal T x 70 arrays: {speaker.shape}, {gt.shape}, {pred.shape}")
    if not all(np.isfinite(x).all() for x in (speaker, gt, pred)):
        raise ValueError("Nonfinite coefficients")
    starts = list(range(0, len(gt)-window+1, stride))
    result = {"FD": {}, "RTLCC": {}, "RWTLCC": {}, "FID_fm": {},
              "FID_delta_fm": {}, "SND": {}, "V_D": {}, "V_D_GT": {}}
    audit = {"window_starts_frames": starts, "window_count": len(starts),
             "window_uncovered_tail_frames": len(gt)-(starts[-1]+window) if starts else len(gt),
             "rtlcc_supported": len(gt) > max_lag+1, "rwtlcc_supported": bool(starts),
             "unsupported_reason": "N/A when too short for complete configured lag range/window; no padding or adaptive shortening",
             "undefined_correlation_coordinates": {}}
    for name in ("exp", "angle", "trans"):
        sl = GROUPS[name]
        result["FD"][name] = float(np.abs(pred[:, sl]-gt[:, sl]).mean())
    for name in ("exp", "pose"):
        sl = GROUPS[name]
        s, g, p = speaker[:, sl], gt[:, sl], pred[:, sl]
        gu = pu = 0
        result["RTLCC"][name] = None
        if len(gt) > max_lag+1:
            gc, gu = tlcc(s, g, max_lag)
            pc, pu = tlcc(s, p, max_lag)
            result["RTLCC"][name] = float(np.abs(gc-pc).mean())
        wvalues, wu = [], 0
        for start in starts:
            wcg, ug = tlcc(s[start:start+window], g[start:start+window], max_lag)
            wcp, up = tlcc(s[start:start+window], p[start:start+window], max_lag)
            wvalues.append(float(np.abs(wcg-wcp).mean()))
            wu += ug+up
        result["RWTLCC"][name] = float(np.mean(wvalues)) if wvalues else None
        result["FID_fm"][name] = frechet(g, p)
        result["FID_delta_fm"][name] = frechet(np.diff(g, axis=0), np.diff(p, axis=0))
        result["SND"][name] = result["FID_fm"][name]+result["FID_delta_fm"][name]
        result["V_D"][name] = float(np.var(p, axis=0, ddof=0).mean())
        result["V_D_GT"][name] = float(np.var(g, axis=0, ddof=0).mean())
        audit["undefined_correlation_coordinates"][name] = {
            "whole_clip_gt": gu, "whole_clip_prediction": pu, "windows_both": wu}
    return result, audit


def run(work):
    protocol = json.loads((work / "protocol.json").read_text())
    import hashlib
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    scoring_protocol = json.loads((work/"SCORING_PROTOCOL.json").read_text())
    if scoring_protocol["extraction_protocol_sha256"] != digest(work/"protocol.json"):
        raise ValueError("Scoring protocol refers to different extraction inputs")
    if scoring_protocol["implementation_sha256"][str(Path(__file__).resolve())] != digest(Path(__file__)):
        raise ValueError("Scorer changed after scoring protocol freeze")
    rows = []
    for sample in protocol["samples"]:
        uid = sample["id"]
        streams = {role: np.load(work / "features" / role / (uid+".npz"))["motion"].astype(np.float64)
                   for role in ("speaker", "gt", "prediction")}
        score, audit = score_clip(streams["speaker"], streams["gt"], streams["prediction"],
                                 **protocol["metric_parameters"])
        null, _ = score_clip(streams["speaker"], streams["gt"], streams["gt"],
                             **protocol["metric_parameters"])
        null_errors = {k: max((x for x in v.values() if x is not None), default=0.)
                       for k, v in null.items() if k not in ("V_D", "V_D_GT")}
        if max(null_errors.values()) > 1e-12:
            raise AssertionError(f"GT self-comparison failed: {uid}: {null_errors}")
        rows.append({"id": uid, "frames": len(streams["gt"]), "metrics_raw": score,
                     "timing_audit": audit, "gt_self_distance_max": max(null_errors.values())})
    raw = {k: {g: float(np.mean([row["metrics_raw"][k][g] for row in rows if row["metrics_raw"][k][g] is not None]))
               for g in rows[0]["metrics_raw"][k]} for k in rows[0]["metrics_raw"]}
    coverage = {k: {g: sum(row["metrics_raw"][k][g] is not None for row in rows)
                    for g in rows[0]["metrics_raw"][k]} for k in rows[0]["metrics_raw"]}
    displayed = {k: {g: value * (100 if k in ("FD", "FID_delta_fm") else 1)
                     for g, value in values.items()} for k, values in raw.items()}
    result = {"dataset": "ViCo paper test43", "model": "AV+Event+Feature+Whisper step8000",
              "implementation": "reproduction of public definitions with user-approved explicit parameters",
              "strict_customlistener_table_reproduction": False, "clips": len(rows),
              "extraction_protocol_sha256": digest(work/"protocol.json"),
              "scoring_protocol_sha256": digest(work/"SCORING_PROTOCOL.json"),
              "frames_per_stream": sum(row["frames"] for row in rows),
              "aggregation": "equal-weight mean of per-clip metrics; equal-weight complete windows within each clip",
              "metrics_raw": raw, "metrics_table_scale": displayed, "coverage_clips": coverage,
              "table_scale": {"FD": 100, "FID_delta_fm": 100, "others": 1},
              "per_clip": rows}
    (work/"RESULTS.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    import csv
    with (work/"summary.csv").open("w") as f:
        writer = csv.writer(f);writer.writerow(["metric", "group", "raw", "table_scale_value"])
        for key, groups in raw.items():
            for group, value in groups.items():
                writer.writerow([key, group, value, displayed[key][group]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    output = run(args.work.resolve())
    print(json.dumps({k: v for k, v in output.items() if k != "per_clip"}, indent=2))
