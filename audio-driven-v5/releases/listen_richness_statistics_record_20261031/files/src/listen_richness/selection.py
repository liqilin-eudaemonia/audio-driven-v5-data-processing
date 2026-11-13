"""Deterministic, duration-aware core selection plus open-behavior coverage."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import math
import re

from .common import digest, number

DEFAULT_POLICY = dict(schema="listen-richness-selection-v1", hours=50, core_fraction=.9,
                      core_face_floor=224, rare_face_floor=160, max_per_pair=8,
                      min_face_sample_fraction=.98, min_clarity_sample_fraction=.8,
                      target_pairs_per_behavior=3, quiet_fraction=.05,
                      duration_bin_s=2, density_floor_s=5,
                      weights=dict(face=.45, diversity=.30, change=.20, nod_support=.05),
                      status="provisional_pilot_parameters")


def person_keys(ident):
    v, s, _, person = ident.split("_")
    m = re.fullmatch(r"(P\d+)[A-Za-z]*", person)
    if not m: raise ValueError("Malformed participant: " + ident)
    return ["session:" + v + "_" + s, "person:" + v + "_" + m[1]]


def audit_catalog(catalog):
    seen = {}; pairs = {}
    for r in catalog["rows"]:
        pair = r["pair"]; split = r["split"]
        if pair in pairs or split not in ("train", "dev", "test") or not r["group_id"]:
            raise ValueError("Invalid/duplicate catalog pair")
        keys = ["pair:" + pair, "group:" + r["group_id"]]
        for role in ("a", "b"): keys.extend(person_keys(r[role]["id"]))
        for key in keys:
            if seen.setdefault(key, split) != split:
                raise ValueError("Cross-split identity: " + key)
        pairs[pair] = r
    return pairs


def compact(snapshot, summary):
    s = snapshot["source"]
    return dict(sample_id=snapshot["sample_id"], pair=s["pair"], split=s["split"], group_id=s["group_id"],
                listener_id=s["listener_id"], speaker_id=s["speaker_id"],
                source_start_s=s["source_start_s"], source_end_s=s["source_end_s"], duration_s=s["duration_s"],
                **snapshot["face"], eligible=snapshot["eligible_for_extraction"],
                consistency_flags=summary["consistency_flags"],
                behaviors=sorted(g["key"] for g in summary["groups"]),
                behavior_type_count=summary["behavior_type_count"], change_score=summary["capped_change_score"],
                detector_only_nod_candidate=summary["detector_only_nod_candidate"],
                nod_support=bool(snapshot["nod"]["candidate_interval_count"]),
                nod_candidate_interval_count=snapshot["nod"]["candidate_interval_count"],
                official_core_s=snapshot["audio"]["official_core_s"],
                official_bout_count=snapshot["audio"]["official_bout_count"],
                waveform_full_silence=snapshot["audio"]["waveform_full_silence"],
                quiet_candidate=summary["quiet_candidate"], crop_revision=snapshot["crop_revision"])


def check_policy(p):
    if set(p) != set(DEFAULT_POLICY) or p["schema"] != DEFAULT_POLICY["schema"]:
        raise ValueError("Unsupported selection policy")
    for k in ("hours", "core_fraction", "core_face_floor", "rare_face_floor", "min_face_sample_fraction",
              "min_clarity_sample_fraction", "quiet_fraction", "duration_bin_s", "density_floor_s"):
        number(p[k])
    if (p["hours"] <= 0 or not 0 < p["core_fraction"] < 1 or not 0 <= p["quiet_fraction"] < p["core_fraction"]
            or not 0 < p["min_face_sample_fraction"] <= 1 or not 0 < p["min_clarity_sample_fraction"] <= 1
            or not 160 <= p["rare_face_floor"] <= p["core_face_floor"] or p["duration_bin_s"] <= 0 or p["density_floor_s"] <= 0):
        raise ValueError("Invalid thresholds; no implicit relaxation")
    for k in ("max_per_pair", "target_pairs_per_behavior"):
        if type(p[k]) is not int or p[k] < 1: raise ValueError("Invalid count policy")
    w = p["weights"]
    if set(w) != {"face", "diversity", "change", "nod_support"} or any(number(v) < 0 for v in w.values()) or abs(sum(w.values())-1) > 1e-9:
        raise ValueError("Invalid score weights")


def select(rows, catalog, policy):
    check_policy(policy); pairs = audit_catalog(catalog)
    pool = []; excluded = Counter(); ids = set()
    for r0 in rows:
        r = dict(r0); sid = r["sample_id"]
        if sid in ids: raise ValueError("Duplicate sample ID")
        ids.add(sid)
        cat = pairs.get(r["pair"])
        if cat is None:
            excluded["missing_catalog_pair"] += 1; continue
        if (cat["split"] != r["split"] or cat["group_id"] != r["group_id"] or
                {cat["a"]["id"], cat["b"]["id"]} != {r["listener_id"], r["speaker_id"]}):
            raise ValueError("Sample/catalog identity mismatch")
        if r["split"] != "train": excluded["non_train"] += 1; continue
        if not r["eligible"] or r["consistency_flags"]: excluded["unresolved_evidence"] += 1; continue
        if r["face_short_side_p10"] is None or r["face_short_side_p10"] < policy["rare_face_floor"]:
            excluded["small_or_unknown_face"] += 1; continue
        if r["face_sample_detection_fraction"] < policy["min_face_sample_fraction"] or r["clarity_measured_fraction"] < policy["min_clarity_sample_fraction"]:
            excluded["measurement_coverage"] += 1; continue
        a, b, d = number(r["source_start_s"]), number(r["source_end_s"]), number(r["duration_s"])
        if a < 0 or d <= 0 or abs(b-a-d) > 1e-7: raise ValueError("Invalid source span")
        r["duration_bin"] = math.floor(d/policy["duration_bin_s"])
        r["change_density_proxy"] = r["change_score"] * 10 / max(d, policy["density_floor_s"])
        pool.append(r)
    buckets = defaultdict(lambda: {"diversity": [], "change": []})
    for r in pool:
        b = buckets[r["duration_bin"]]; b["diversity"].append(r["behavior_type_count"]); b["change"].append(r["change_density_proxy"])
    for b in buckets.values():
        for values in b.values(): values.sort()
    def rank(values, v):
        # Constant/zero-only buckets must not receive an artificial richness bonus.
        if not values or v <= 0 or values[-1] == values[0]: return 0.
        return (bisect_left(values, v) + bisect_right(values, v)-1) / (2*max(1, len(values)-1))
    w = policy["weights"]
    for r in pool:
        bucket = buckets[r["duration_bin"]]
        r["score_components"] = dict(face=min(r["face_short_side_p10"]/384, 1),
                                     diversity=rank(bucket["diversity"], r["behavior_type_count"]),
                                     change=rank(bucket["change"], r["change_density_proxy"]),
                                     nod_support=float(r["nod_support"]))
        r["selection_score"] = sum(w[k]*v for k,v in r["score_components"].items())
    ranked = sorted(pool, key=lambda r: (-r["selection_score"], -r["face_short_side_p10"], r["sample_id"]))
    available = defaultdict(set); by_type = defaultdict(list)
    for r in ranked:
        for kind in r["behaviors"]:
            available[kind].add(r["pair"]); by_type[kind].append(r)
    chosen = []; selected = set(); spans = defaultdict(list); coverage = defaultdict(set)
    seconds = 0.; quiet_seconds = 0.; skipped = Counter()
    target = policy["hours"] * 3600
    def add(r, reason):
        nonlocal seconds, quiet_seconds
        if r["sample_id"] in selected: return False
        source = spans[r["pair"]]
        if len(source) >= policy["max_per_pair"]: skipped["pair_cap"] += 1; return False
        a, b = r["source_start_s"], r["source_end_s"]
        if any(min(b, y)-max(a, x) > 1e-7 for x,y in source):
            skipped["source_overlap"] += 1; return False
        if seconds >= target: return False
        source.append((a,b)); selected.add(r["sample_id"])
        chosen.append(dict(r, selection_reason=reason)); seconds += r["duration_s"]
        if r["quiet_candidate"]: quiet_seconds += r["duration_s"]
        for kind in r["behaviors"]: coverage[kind].add(r["pair"])
        return True
    core = [r for r in ranked if r["face_short_side_p10"] >= policy["core_face_floor"]]
    for r in core:
        if quiet_seconds >= target*policy["quiet_fraction"]: break
        if r["quiet_candidate"]: add(r, "core:quiet_or_sustained")
    for r in core:
        if seconds >= target*policy["core_fraction"]: break
        add(r, "core:quality_and_richness")
    core_seconds = seconds
    # Rarest types first. Within a deficit, favor candidates filling several deficits.
    for kind in sorted(available, key=lambda k: (len(available[k]), k)):
        goal = min(policy["target_pairs_per_behavior"], len(available[kind]))
        if len(coverage[kind]) >= goal or seconds >= target: continue
        options = sorted(by_type[kind], key=lambda r: (
            -sum(len(coverage[k]) < min(policy["target_pairs_per_behavior"], len(available[k]))
                 and r["pair"] not in coverage[k] for k in r["behaviors"]),
            -r["face_short_side_p10"], -r["selection_score"], r["sample_id"]))
        for r in options:
            if len(coverage[kind]) >= goal or seconds >= target: break
            if r["pair"] not in coverage[kind]: add(r, "supplement:" + kind)
    for r in core:
        if seconds >= target: break
        add(r, "core:fill")
    gaps = {k: dict(available_pairs=len(available[k]), selected_pairs=len(coverage[k]),
                   target_pairs=min(policy["target_pairs_per_behavior"], len(available[k])))
            for k in available if len(coverage[k]) < min(policy["target_pairs_per_behavior"], len(available[k]))}
    report = dict(schema="listen-richness-selection-report-v1", policy=policy, policy_sha256=digest(policy),
                  input_count=len(ids), eligible_count=len(pool), selected_count=len(chosen),
                  selected_seconds=seconds, target_seconds=target, shortfall_seconds=max(0, target-seconds),
                  core_seconds=core_seconds, quiet_or_sustained_seconds=quiet_seconds,
                  quiet_target_seconds=target*policy["quiet_fraction"],
                  coverage={k: len(coverage[k]) for k in sorted(available)}, coverage_gaps=gaps,
                  exclusions=dict(excluded), constraint_skips=dict(skipped),
                  selection_complete=(seconds >= target and core_seconds >= target*policy["core_fraction"]),
                  split_isolation="full_supplied_catalog_audited; selected_rows_train_only",
                  training_admitted=False, human_ground_truth=False,
                  note="Full clips; final clip may exceed target. No threshold auto-relaxation. Coverage is caption-derived.")
    return chosen, report
