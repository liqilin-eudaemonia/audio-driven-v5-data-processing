"""CPU-only export of immutable open candidates to explicitly reviewed training labels.

This adapter neither classifies descriptions nor resolves timing/content disputes.
Without a separate decision file it emits unadmitted, entirely unknown drafts.
"""
import copy
import math
from pathlib import Path
import shutil

from .contracts import (EVENTS, VISUAL_TYPES, LABEL_SCHEMA, TYPE_TO_MODALITY,
                        atomic_json, digest, positive, read, rows, safe_id, sha)
from .annotations import validate, union

MAPPING_VERSION = "listen-candidate-training-v2"
CANDIDATE_SCHEMA = "listen-existing-result-fusion-verbatim-v2"
DECISION_SCHEMA = "listen-candidate-decisions-v2"


def valid_time(event, duration):
    a, b = event.get("start"), event.get("end")
    return all(type(v) in (float, int) and math.isfinite(v) for v in (a, b)) and 0 <= a < b <= duration


def uncertain_classes(event):
    classes = [k for k in EVENTS if k in event.get("categories", [])]
    return classes or (list(VISUAL_TYPES) if event.get("modality") == "visual" else
                       ["listener_vocalization"] if event.get("modality") == "audio" else list(EVENTS))


def convert_sample(candidate, cfg, decision=None, raw=None):
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        raise ValueError("Expected current verbatim candidates, not legacy merged labels")
    sid = safe_id(candidate["sample_id"])
    duration = positive(candidate["duration_s"], "candidate duration")
    events = candidate["events"]
    ids = [safe_id(e["event_id"]) for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate event ID")
    if cfg["events"]["mapping_version"] != MAPPING_VERSION:
        raise ValueError("Unsupported candidate adapter mapping version")
    d = decision or {}
    if decision is not None:
        allowed = {"schema", "sample_id", "candidate_sha256", "coverage", "unknown", "coverage_sources",
                   "events", "media_hashes", "admission_record", "human_verified", "source_review_reference"}
        if set(d) != allowed or d["schema"] != DECISION_SCHEMA or d["sample_id"] != sid or d["candidate_sha256"] != digest(candidate):
            raise ValueError("Decision schema or candidate identity differs")
        if not d["source_review_reference"] or not d["admission_record"].get("reference"):
            raise ValueError("Explicit source review and admission record required")
        if set(d["media_hashes"]) != {"target_video", "target_audio"} or not all(
                isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in d["media_hashes"].values()):
            raise ValueError("Decisions must bind target media SHA256 hashes")
        if type(d["human_verified"]) is not bool:
            raise ValueError("human_verified must be boolean")
    decisions = {e["candidate_event_id"]: e for e in d.get("events", [])}
    if len(decisions) != len(d.get("events", [])) or set(decisions) - set(ids):
        raise ValueError("Duplicate or unknown event decision")
    coverage = copy.deepcopy(d.get("coverage", {k: [] for k in EVENTS}))
    unknown = copy.deepcopy(d.get("unknown", {k: [] for k in EVENTS}))
    if set(coverage) != set(EVENTS) or set(unknown) != set(EVENTS):
        raise ValueError("Explicit seven-class coverage/unknown required")
    selected, mapping, excluded = [], [], []

    def mask_event(event):
        span = [event["start"], event["end"]] if valid_time(event, duration) else [0., duration]
        for k in uncertain_classes(event):
            unknown[k].append(span)

    for event in events:
        eid = event["event_id"]
        choice = decisions.get(eid)
        status = "unresolved" if choice is None else choice.get("decision")
        if status not in ("accepted", "rejected", "unresolved"):
            raise ValueError("Unknown candidate decision")
        if choice is not None and not choice.get("reason"):
            raise ValueError("Candidate decision reason required")
        reason = "no_explicit_decision" if choice is None else choice["reason"]
        categories = event.get("categories", [])
        valid_category = len(categories) == 1 and categories[0] in EVENTS
        valid_modality = valid_category and TYPE_TO_MODALITY[categories[0]] == {"visual": "video", "audio": "audio"}.get(event.get("modality"))
        if status == "accepted":
            allowed = {"candidate_event_id", "decision", "reason", "verification", "boundary_status", "reviewed_flags"}
            if set(choice) != allowed or not valid_modality or not valid_time(event, duration):
                raise ValueError("Accepted candidate requires one existing category, matching modality and valid clip-local time")
            if sorted(choice["reviewed_flags"]) != sorted(event.get("review_flags", [])):
                raise ValueError("Every candidate review flag must be explicitly reviewed")
            training_id = "train-" + digest([sid, eid, MAPPING_VERSION])[:24]
            selected.append(dict(event_id=training_id, type=categories[0], start=event["start"], end=event["end"],
                                 description=event["description"], text=event.get("text"),
                                 verification=choice["verification"], boundary_status=choice["boundary_status"],
                                 boundary_censored=choice["boundary_status"] == "censored",
                                 source=copy.deepcopy(event), admission=copy.deepcopy(choice)))
        else:
            training_id = None
            excluded.append(dict(candidate_event_id=eid, status=status, reason=reason))
            if status == "unresolved":
                mask_event(event)
        mapping.append(dict(sample_id=sid, candidate_event_id=eid, training_event_id=training_id,
                            status=status, reason=reason))

    # Unlocated words cannot silently become negative time; keep the entire audio
    # class unknown until a new upstream candidate revision supplies their bounds.
    failures = candidate.get("automatic_fusion", {}).get("alignment_failures", [])
    if failures:
        unknown["listener_vocalization"].append([0., duration])
        excluded.append(dict(status="unresolved", reason="unlocated_words", evidence=copy.deepcopy(failures)))
    raw_events = {e.get("event_id", e.get("id")): e for e in (raw or {}).get("events", [])}
    for pending in candidate.get("unresolved_observations", []):
        if pending.get("reason") != "out_of_scope":
            mask_event(raw_events.get(pending.get("observation_id"), {}))
            excluded.append(dict(status="unresolved", reason="upstream_observation", evidence=copy.deepcopy(pending)))
    # Validate interval data before merging unknown spans: no silent correction.
    label = dict(schema=LABEL_SCHEMA, sample_id=sid, time_origin="clip_start_seconds", duration_s=duration,
                 taxonomy=list(EVENTS), coverage=coverage, unknown=unknown, events=selected,
                 training_admitted=decision is not None,
                 provenance=dict(mapping_version=MAPPING_VERSION, admission_policy=cfg["events"]["admission_policy"],
                     candidate_sha256=digest(candidate), decision_sha256=digest(decision),
                     media_hashes=copy.deepcopy(d.get("media_hashes", {})), human_verified=d.get("human_verified", False),
                     coverage_sources=copy.deepcopy(d.get("coverage_sources", {k: {} for k in EVENTS})),
                     admission_record=copy.deepcopy(d.get("admission_record", {})),
                     source_review_reference=d.get("source_review_reference")))
    validate(label, dict(sample_id=sid, requested_duration_s=duration), require_admitted=False)
    label["unknown"] = {k: union(unknown[k]) for k in EVENTS}
    return label, dict(sample_id=sid, mapping=mapping, excluded=excluded)


def export_candidates(cfg, candidates_path, review_path, observations_path, output, decisions_path=None):
    candidates = rows(candidates_path)
    raw = {r["sample_id"]: r for r in rows(observations_path)}
    review = read(review_path)
    review_rows = {r["sample_id"]: r for r in review["samples"]}
    ids = {r["sample_id"] for r in candidates}
    if set(raw) != ids or set(review_rows) != ids or len(review_rows) != len(review["samples"]):
        raise ValueError("Candidates, raw observations and review must cover identical sample IDs")
    decisions = {} if decisions_path is None else {r["sample_id"]: r for r in rows(decisions_path)}
    if set(decisions) - ids:
        raise ValueError("Decision sample absent from candidates")
    sources = {"candidates" + Path(candidates_path).suffix: Path(candidates_path), "review.json": Path(review_path),
               "observations" + Path(observations_path).suffix: Path(observations_path)}
    if decisions_path is not None:
        sources["decisions" + Path(decisions_path).suffix] = Path(decisions_path)
    hashes = {name: sha(p) for name, p in sources.items()}
    labels, audits = [], []
    for candidate in candidates:
        sid = candidate["sample_id"]
        if raw[sid]["duration_s"] != candidate["duration_s"] or review_rows[sid]["duration_s"] != candidate["duration_s"]:
            raise ValueError("Source durations differ")
        if review_rows[sid]["events"] != candidate["events"]:
            raise ValueError("Review and candidate event identities differ")
        label, audit = convert_sample(candidate, cfg, decisions.get(sid), raw[sid])
        label["provenance"]["source_files"] = hashes
        labels.append(label); audits.append(audit)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Export requires an empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(exist_ok=True)
    for name, p in sources.items():
        shutil.copyfile(p, out / "sources" / name)
        if sha(out / "sources" / name) != hashes[name] or sha(p) != hashes[name]:
            raise ValueError("Source changed during export")
    atomic_json(out / "labels.json", labels)
    atomic_json(out / "labels.admitted.json", [l for l in labels if l["training_admitted"]])
    atomic_json(out / "audit.json", audits)
    summary = dict(schema=LABEL_SCHEMA, mapping_version=MAPPING_VERSION, taxonomy=list(EVENTS), K_train=cfg["events"]["K_train"],
                   source_files=hashes, samples=len(labels), candidate_events=sum(len(c["events"]) for c in candidates),
                   preserved_observations=sum(len(r["events"]) for r in raw.values()),
                   admitted_samples=sum(l["training_admitted"] for l in labels),
                   accepted_events=sum(len(l["events"]) for l in labels),
                   unresolved_words=sum(len(c.get("automatic_fusion", {}).get("alignment_failures", [])) for c in candidates),
                   per_class={k: dict(events=sum(e["type"] == k for l in labels for e in l["events"]),
                                      coverage_seconds=sum(b-a for l in labels for a,b in union(l["coverage"][k]))) for k in EVENTS},
                   raw_preserved=True, real_training_started=False)
    atomic_json(out / "summary.json", summary)
    return summary
