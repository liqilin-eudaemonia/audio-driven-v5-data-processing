"""Versioned event/background weak labels from completed listener annotation receipts.

No model inference, text reclassification, human-ground-truth claims, or mutation of
source labels. v1 conservatively requires a successful full view per modality.
"""
import copy
import json
import math
from pathlib import Path
import shutil

from .annotations import intersection_length, union, validate
from .candidates import DECISION_SCHEMA, convert_sample, valid_time
from .contracts import EVENTS, VISUAL_TYPES, atomic_json, digest, read, rows, sha

POLICY = "event-interval-weak-background-v1"
# These are model/boundary provenance flags, explicitly accepted as weak labels;
# they are NOT evidence of human review. All other flags remain unresolved.
WEAK_FLAGS = {"raw_model_unverified", "normalization_unverified",
              "inherited_interval_union_requires_boundary_review"}


def close(a, b):
    return (type(a) in (int, float) and type(b) in (int, float) and
            math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-6)


def full_receipt(records, branch, sid, duration, media, target):
    """Return coverage evidence or a recorded exclusion; never infer success."""
    candidates = [r for r in records if r.get("sample_id") == sid and r.get("view") == "full"]
    good = [r for r in candidates if r.get("status") == "schema_pass"]
    if not good:
        return None, "missing_or_failed_full_view"
    if len(good) != 1:
        raise ValueError(f"Ambiguous successful full receipts: {sid}/{branch}")
    r = good[0]
    if (r.get("branch") != branch or r.get("record_kind") != "result" or
            r.get("time_origin") != "window_start_seconds" or
            not close(r.get("window_start_s"), 0) or not close(r.get("window_end_s"), duration) or
            not close(r.get("parent_duration_s"), duration) or
            not isinstance(r.get("parsed", {}).get("events"), list) or
            r.get("raw_response", {}).get("finish_reason") != "stop"):
        return None, "incomplete_or_invalid_full_result"
    source = r.get("source", {})
    if (source.get("sample_id") != sid or not close(source.get("duration_s"), duration) or
            not close(source.get("source_start_s"), media["source_start_s"]) or
            source.get("source") != media["source_video"] or
            not source.get("listener_id") or source.get("listener_id") == source.get("speaker_id")):
        raise ValueError(f"Receipt/source identity differs: {sid}/{branch}")
    evidence = r.get("input_evidence", {})
    if branch == "visual":
        v = evidence.get("video", {})
        frames = v.get("frames", [])
        if (source.get("crop_error") or not media["video_transform"].get("crop") or
                v.get("source") != media["source_video"] or
                v.get("video_transform") != media["video_transform"] or
                media["video_transform"].get("output_size") != [512, 512] or
                not close(v.get("source_time_origin_s"), media["source_start_s"])):
            return None, "invalid_listener_crop_or_video_identity"
        times = [f.get("time_s") for f in frames]
        if (not frames or any(f.get("size") != [512, 512] or not f.get("jpeg_sha256") for f in frames) or
                any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t < duration for t in times) or
                times != sorted(set(times)) or times[0] > .25 or duration-times[-1] > .25 or
                any(b-a > .25+1e-6 for a,b in zip(times, times[1:]))):
            return None, "incomplete_512_video_sampling"
    else:
        a = evidence.get("audio", {})
        s = media["audio_sources"]["listener"]
        if (a.get("role") != "listener" or a.get("isolated") is not True or
                a.get("wav_sha256") != target["media_hashes"]["target_audio"] or
                a.get("source_waveform_sha256_declared") != s["declared_source_sha256"] or
                a.get("source_start_sample") != s["start_sample"] or
                a.get("source_end_sample") != s["end_sample"] or a.get("source_rate") != s["sample_rate"]):
            return None, "invalid_isolated_listener_audio_identity"
    return r, None


def inherited_unknown(raw, duration, replace_placeholder):
    unknown = copy.deepcopy(raw.get("unknown", {k: [] for k in EVENTS}))
    # Only this documented upstream placeholder may be replaced, and only via
    # explicit opt-in. Partial/class-specific unknowns always survive unchanged.
    placeholder = (raw.get("schema") == "listen-open-semantic-events-v1" and
                   raw.get("version") == "open-to-visual6-audio1-semantic-v3" and
                   raw.get("training_admitted") is False and raw.get("K_train") == [] and
                   raw.get("unlabeled_time_policy") == "unknown" and
                   raw.get("coverage") == {k: [] for k in EVENTS} and
                   unknown == {k: [[0, duration]] for k in EVENTS})
    replaced = bool(placeholder and replace_placeholder)
    return ({k: [] for k in EVENTS} if replaced else unknown), replaced


def derive_sample(candidate, raw, cfg, target, receipts, *, debug_only, replace_placeholder=False):
    """Convert accepted intervals + successful coverage into existing BCE labels."""
    if cfg["events"]["admission_policy"] != POLICY:
        raise ValueError(f"Weak derivation requires admission_policy={POLICY}")
    # A caller cannot promote individual debug records through a broader summary.
    debug_only = debug_only or any(
        value.get("debug_only") is True or value.get("formal_training_admitted") is False
        for row in (candidate, raw, target) for value in (row, row.get("provenance", {})))
    sid, duration = candidate["sample_id"], candidate["duration_s"]
    coverage, sources = {}, {}
    unknown, replaced = inherited_unknown(raw, duration, replace_placeholder)
    for k in EVENTS:
        r = receipts["visual" if k in VISUAL_TYPES else "audio"]
        coverage[k] = [[0., duration]] if r else []
        sources[k] = (dict(reference="receipt:" + digest(r), verification="weak_accepted",
                           request_id=r["request_id"], assumption="completed_full_view_with_valid_listener_media") if r else {})
        if not r:
            unknown[k].append([0., duration])
    decisions = []
    for e in candidate["events"]:
        classes = e.get("categories", [])
        k = classes[0] if len(classes) == 1 and classes[0] in EVENTS else None
        flags = e.get("review_flags", [])
        expected = "visual" if k in VISUAL_TYPES else "audio"
        acceptable = (k is not None and e.get("modality") == expected and valid_time(e, duration)
                      and bool(coverage[k]) and not set(flags)-WEAK_FLAGS)
        r = receipts.get(expected) or {}
        boundary = r.get("source", {}).get("boundary_activity", {})
        censored = (e.get("boundary_censored") is True or e.get("boundary_status") == "censored" or
                    e.get("timing_evidence", {}).get("edge") is True or
                    (close(e.get("start"), 0) and boundary.get("onset_crossing") is True) or
                    (close(e.get("end"), duration) and boundary.get("end_crossing") is True))
        if acceptable:
            decisions.append(dict(candidate_event_id=e["event_id"], decision="accepted",
                                  reason="policy_weak_acceptance_not_human_review", verification="weak_accepted",
                                  boundary_status="censored" if censored else "weak_accepted", reviewed_flags=flags))
        else:
            decisions.append(dict(candidate_event_id=e["event_id"], decision="unresolved",
                                  reason="missing_coverage_invalid_event_or_unaccepted_flags"))
    # Unlocated/conflicting audio evidence cannot silently become background.
    fusion = candidate.get("automatic_fusion", {})
    if fusion.get("unresolved_audio_alternatives") or fusion.get("outside_asr_timing_support"):
        unknown["listener_vocalization"].append([0., duration])
    decision = dict(schema=DECISION_SCHEMA, sample_id=sid, candidate_sha256=digest(candidate),
                    coverage=coverage, unknown=unknown, coverage_sources=sources, events=decisions,
                    media_hashes=target["media_hashes"], human_verified=False,
                    admission_record={"reference": POLICY + ":" + sid}, source_review_reference=POLICY)
    label, audit = convert_sample(candidate, cfg, decision, raw)
    # Make censored spans visible in exported unknown as well as rasterizer masks.
    for e in label["events"]:
        if e["boundary_censored"]:
            label["unknown"][e["type"]].append([e["start"], e["end"]])
    label["unknown"] = {k: union(v) for k,v in label["unknown"].items()}
    label["provenance"].update(weak_background_policy=POLICY, debug_only=debug_only,
        formal_training_admitted=not debug_only, placeholder_unknown_replaced=replaced,
        original_unknown=copy.deepcopy(raw.get("unknown", {})),
        background_weight=1., weak_background_is_human_ground_truth=False)
    label["training_admitted"] = any(coverage.values())
    validate(label, dict(sample_id=sid, requested_duration_s=duration), require_admitted=False)
    audit.update(placeholder_unknown_replaced=replaced,
                 censored_event_ids=[e["event_id"] for e in label["events"] if e["boundary_censored"]],
                 audio_conflict_masked=bool(fusion.get("unresolved_audio_alternatives") or fusion.get("outside_asr_timing_support")))
    return label, decision, audit


def background_seconds(label, k):
    excluded = label["unknown"][k] + [[e["start"], e["end"]] for e in label["events"] if e["type"] == k]
    return sum(b-a-intersection_length(excluded, a, b) for a,b in union(label["coverage"][k]))


def export_weak_labels(cfg, candidates_path, review_path, observations_path, visual_path, audio_path,
                       media_path, summary_path, output, replace_placeholder=False):
    from .dataset import load_observations, load_targets
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Weak export requires a new empty output directory")
    paths = {"candidates" + Path(candidates_path).suffix: Path(candidates_path), "review.json": Path(review_path),
             "observations" + Path(observations_path).suffix: Path(observations_path), "visual.jsonl": Path(visual_path),
             "audio.jsonl": Path(audio_path), "media_manifest.json": Path(media_path),
             "source_summary.json": Path(summary_path)}
    # Hash before reading and verify after writing to detect source mutation.
    hashes = {name: sha(p) for name,p in paths.items()}
    candidates = rows(candidates_path)
    raw = {r["sample_id"]: r for r in rows(observations_path)}
    review = read(review_path)["samples"]
    reviews = {r["sample_id"]: r for r in review}
    media = {r["sample_id"]: r for r in rows(media_path)}
    observations = {r["sample_id"]: r for r in load_observations(cfg)}
    ids = {c["sample_id"] for c in candidates}
    if set(raw) != ids or set(reviews) != ids or len(review) != len(ids) or set(media) != ids or set(observations) != ids:
        raise ValueError("Candidate/raw/review/media/prepared sample IDs must match exactly")
    if any(o["experiment_split"] != "train" for o in observations.values()):
        raise ValueError("Automatically derived weak labels are train-only, never dev/test truth")
    targets = load_targets(cfg, list(observations.values()))
    source_summary = read(summary_path)
    if type(source_summary.get("debug_only")) is not bool or type(source_summary.get("formal_training_admitted")) is not bool:
        raise ValueError("Source summary must explicitly declare debug_only/formal_training_admitted")
    debug = source_summary["debug_only"] or not source_summary["formal_training_admitted"]
    records = {branch: [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
               for branch,p in (("visual", visual_path), ("audio", audio_path))}
    indexed = {branch: {} for branch in records}
    for branch, rr in records.items():
        for r in rr:
            indexed[branch].setdefault(r.get("sample_id"), []).append(r)
    labels, decisions, audits = [], [], []
    for c in candidates:
        sid, duration = c["sample_id"], c["duration_s"]
        m, t = media[sid], targets[sid]
        if (not all(close(x, duration) for x in [raw[sid]["duration_s"], reviews[sid]["duration_s"],
                     m["duration_s"], observations[sid]["requested_duration_s"]]) or reviews[sid]["events"] != c["events"]):
            raise ValueError("Candidate/raw/review/media identity differs")
        if (m["sha256"]["video.mp4"] != t["media_hashes"]["target_video"] or
                m["sha256"]["listener.wav"] != t["media_hashes"]["target_audio"] or m["target_geometry"] != [512,512]):
            raise ValueError("Media export does not bind prepared listener targets")
        receipts, exclusions = {}, {}
        for branch in records:
            receipts[branch], exclusions[branch] = full_receipt(indexed[branch].get(sid, []), branch, sid, duration, m, t)
        label, decision, audit = derive_sample(c, raw[sid], cfg, t, receipts,
            debug_only=debug, replace_placeholder=replace_placeholder)
        label["provenance"]["source_files"] = hashes
        audit["coverage_exclusions"] = exclusions
        labels.append(label); decisions.append(decision); audits.append(audit)
    debug = any(l["provenance"]["debug_only"] for l in labels)
    out.mkdir(parents=True, exist_ok=True)
    (out/"sources").mkdir(exist_ok=True)
    for name,p in paths.items():
        shutil.copyfile(p, out/"sources"/name)
        if sha(p) != hashes[name] or sha(out/"sources"/name) != hashes[name]:
            raise ValueError("Source changed during weak export")
    atomic_json(out/"labels.json", labels)
    atomic_json(out/"labels.admitted.json", [l for l in labels if l["training_admitted"]])
    atomic_json(out/"decisions.json", decisions)
    atomic_json(out/"audit.json", audits)
    summary = dict(policy=POLICY, samples=len(labels), admitted_samples=sum(l["training_admitted"] for l in labels),
                   accepted_events=sum(len(l["events"]) for l in labels), debug_only=debug,
                   formal_training_admitted=not debug, human_verified=False, source_files=hashes,
                   per_class={k: dict(events=sum(e["type"]==k for l in labels for e in l["events"]),
                        weak_background_seconds=sum(background_seconds(l,k) for l in labels),
                        unknown_seconds=sum(b-a for l in labels for a,b in l["unknown"][k])) for k in EVENTS},
                   raw_preserved=True, real_training_started=False)
    atomic_json(out/"summary.json", summary)
    return summary
