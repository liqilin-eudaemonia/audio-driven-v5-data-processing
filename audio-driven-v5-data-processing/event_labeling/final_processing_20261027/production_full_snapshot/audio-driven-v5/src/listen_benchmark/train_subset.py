"""Deterministic train-only selection: large-face core plus rare-class supplements.

Selection scores do not rewrite events or infer additional occurrences.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time

from .backchannel_face_stats import CATEGORIES, atomic_json, loads, read, sha, SourceReader

VERSION = "p2-train-large-face-core-rare-v1"


def person_keys(ident):
    v, s, _, person = ident.split("_")
    match = re.fullmatch(r"(P\d+)[A-Za-z]*", person)
    if not match:
        raise ValueError(f"Malformed participant: {ident}")
    return v + "_" + s, v + "_" + match[1]


def identities(pair):
    keys = {"pair:" + pair["pair"], "group:" + pair["group_id"]}
    for role in ("a", "b"):
        session, person = person_keys(pair[role]["id"])
        keys.update(("session:" + session, "person:" + person))
    return keys


def audit_catalog(catalog):
    seen, counts = {}, collections.Counter()
    pairs = {}
    for pair in catalog["rows"]:
        split = pair["split"]
        if split not in ("train", "dev", "test") or pair["pair"] in pairs:
            raise ValueError("Invalid catalog split or duplicate pair")
        for key in identities(pair):
            if seen.setdefault(key, split) != split:
                raise ValueError(f"Cross-split identity: {key}")
        counts[split] += 1
        pairs[pair["pair"]] = pair
    return pairs, dict(counts)


def score(row):
    counts = [row[f"{c}_count"] for c in CATEGORIES]
    capped = sum(min(n, 1 if c == "smile" else 3) for c, n in zip(CATEGORIES, counts))
    diversity = sum(n > 0 for n in counts)
    # Do not reward hundreds of descriptions of a sustained smile. Short clips
    # receive a 5-second denominator floor only for ranking, never for timestamps.
    density = capped * 10 / max(row["duration_s"], 5)
    value = .5 * min(row["face_short_side_p10"] / 384, 1) + .3 * diversity / 7 + .2 * min(density / 12, 1)
    return value, capped, density


class Selection:
    def __init__(self, max_per_pair):
        self.rows = []; self.ids = set(); self.intervals = collections.defaultdict(list)
        self.counts = collections.Counter(); self.seconds = 0.0
        self.max_per_pair = max_per_pair

    def reason(self, row):
        if row["sample_id"] in self.ids:
            return "already_selected"
        intervals = self.intervals[row["pair"]]
        if len(intervals) >= self.max_per_pair:
            return "pair_cap"
        a, b = row["source_start_s"], row["source_end_s"]
        if not (math.isfinite(a) and math.isfinite(b) and a < b):
            raise ValueError("Invalid source interval")
        if any(min(b, y) - max(a, x) > 1e-6 for x, y in intervals):
            return "source_overlap"
        return None

    def add(self, row, reason):
        rejection = self.reason(row)
        if rejection:
            return False
        row = dict(row, selection_reason=reason)
        self.rows.append(row); self.ids.add(row["sample_id"])
        self.intervals[row["pair"]].append((row["source_start_s"], row["source_end_s"]))
        self.seconds += row["duration_s"]
        self.counts.update(c for c in CATEGORIES if row[f"{c}_count"] > 0)
        return True


def select_rows(rows, hours=50, core_fraction=.9, class_minimum=2000, max_per_pair=8):
    if hours <= 0 or not 0 < core_fraction < 1 or class_minimum < 1 or max_per_pair < 1:
        raise ValueError("Invalid selection configuration")
    target = hours * 3600
    enriched = []
    for original in rows:
        row = dict(original)
        if row["split"] != "train" or row["label_issue_flag"] or row["face_valid_fraction"] < .98:
            raise ValueError("Ineligible row in selection pool")
        if not row["face_short_side_p10"] >= 160 or row["non_smile_event_count"] < 1:
            raise ValueError("Invalid face/event eligibility")
        row["selection_score"], row["capped_event_count"], row["capped_density_per_10s"] = score(row)
        enriched.append(row)
    ranked = sorted(enriched, key=lambda r: (-r["selection_score"], -r["face_short_side_p10"], r["sample_id"]))
    core = [r for r in ranked if r["face_short_side_p10"] >= 224]
    result = Selection(max_per_pair)
    for row in core:
        if result.seconds >= target*core_fraction:
            break
        result.add(row, "core")
    if result.seconds < target*core_fraction:
        raise ValueError("Insufficient non-overlapping large-face core; no threshold relaxation")
    core_counts = dict(result.counts)
    shake = sorted((r for r in enriched if r["head_shake_count"] > 0),
                   key=lambda r: (-r["face_short_side_p10"], -r["selection_score"], r["sample_id"]))
    shake_skips = collections.Counter()
    for row in shake:
        why = result.reason(row)
        if why:
            shake_skips[why] += 1
            continue
        if result.seconds + row["duration_s"] > target:
            raise ValueError("Rare-class supplement exceeds reserved duration")
        result.add(row, "supplement:head_shake")
    # Fill deficits from rarest first; each sample can satisfy several quotas.
    pools = {c: [r for r in ranked if r["face_short_side_p10"] >= 192 and r[f"{c}_count"] > 0]
             for c in CATEGORIES if c != "head_shake"}
    for cat in sorted(pools, key=lambda c: (len(pools[c]), c)):
        for row in pools[cat]:
            if result.counts[cat] >= class_minimum:
                break
            if result.reason(row):
                continue
            if result.seconds + row["duration_s"] > target:
                raise ValueError(f"Reserved duration insufficient for {cat} quota")
            result.add(row, "supplement:" + cat)
        if result.counts[cat] < class_minimum:
            raise ValueError(f"Unavailable class quota: {cat}; do not fabricate or repeat samples")
    for row in core:
        if result.seconds >= target:
            break
        result.add(row, "core_fill")
    if result.seconds < target or any(result.counts[c] == 0 for c in CATEGORIES):
        raise ValueError("Incomplete duration or taxonomy coverage")
    return result.rows, dict(target_hours=hours, core_fraction=core_fraction, class_minimum=class_minimum,
        max_per_pair=max_per_pair, core_counts_before_supplement=core_counts,
        eligible_head_shake_samples=len(shake), head_shake_skipped=dict(shake_skips),
        score="0.5*min(face_p10/384,1)+0.3*class_count/7+0.2*min(capped_density/12,1)",
        capped_density="10*sum(min(class_count,3), except smile cap=1)/max(duration_s,5)",
        floors=dict(core_face_p10=224, supplement_face_p10=192, head_shake_face_p10=160, valid_fraction=.98))


def summarize(rows):
    import numpy as np
    stages = collections.defaultdict(lambda: dict(samples=0, seconds=0))
    for row in rows:
        stage = stages[row["selection_reason"]]; stage["samples"] += 1; stage["seconds"] += row["duration_s"]
    per_class = {}
    for cat in CATEGORIES:
        values = [r for r in rows if r[f"{cat}_count"] > 0]
        per_class[cat] = dict(samples=len(values), events=sum(r[f"{cat}_count"] for r in values),
            pairs=len({r["pair"] for r in values}),
            face_p10_median=float(np.median([r["face_short_side_p10"] for r in values])))
    quantiles = {}
    for col in ("duration_s", "face_short_side_p10", "face_short_side_p50", "event_count", "non_smile_event_count", "events_per_10s", "category_count"):
        quantiles[col] = {str(q): float(np.percentile([r[col] for r in rows], q)) for q in (0,10,50,90,100)}
    return dict(samples=len(rows), hours=sum(r["duration_s"] for r in rows)/3600,
                mean_duration_s=sum(r["duration_s"] for r in rows)/len(rows),
                events=sum(r["event_count"] for r in rows), per_class=per_class, stages=dict(stages),
                pairs=len({r["pair"] for r in rows}), groups=len({r["group_id"] for r in rows}),
                quantiles=quantiles, below_core_face_floor=sum(r["face_short_side_p10"]<224 for r in rows),
                split="train", intervals_overlap=False)


def prepare(args):
    stats, catalog_path = Path(args.stats).resolve(), Path(args.catalog).resolve()
    if not (stats/"COMPLETE.json").exists():
        raise ValueError("Incomplete statistics")
    if sha(stats/"stats.sqlite") != read(stats/"COMPLETE.json")["files"]["stats.sqlite"]:
        raise ValueError("Statistics changed")
    catalog = read(catalog_path)
    pairs, split_counts = audit_catalog(catalog)
    with sqlite3.connect(f"file:{stats}/stats.sqlite?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        pool = [dict(r) for r in conn.execute("SELECT * FROM samples WHERE split='train' AND label_issue_flag=0 "
            "AND face_valid_fraction>=.98 AND face_short_side_p10>=160 AND non_smile_event_count>=1")]
    for row in pool:
        pair = pairs[row["pair"]]
        if pair["split"] != "train" or pair["group_id"] != row["group_id"]:
            raise ValueError("Statistics/catalog split identity mismatch")
        if row["listener_id"] not in (pair["a"]["id"], pair["b"]["id"]):
            raise ValueError("Catalog listener mismatch")
    selected, policy = select_rows(pool, args.hours, class_minimum=args.class_minimum, max_per_pair=args.max_per_pair)
    return selected, policy, dict(catalog_sha256=sha(catalog_path), retained_catalog_pairs=split_counts,
        identity_intersections=0, identity_keys=["pair", "connected_group", "session", "version+participant_without_AB_suffix"],
        policy="Preserve existing retained train/dev/test; previously excluded eval linked to train stays excluded",
        preexisting_excluded_pairs=(sum(catalog["excluded_pairs"].values())
            if isinstance(catalog.get("excluded_pairs"), dict) else len(catalog.get("excluded_pairs", [])))), pairs


def audit_eval_manifests(directory, pairs, selected):
    selected_keys = set().union(*(identities(pairs[r["pair"]]) for r in selected))
    selected_ids = {r["sample_id"] for r in selected}
    files = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        split = path.name.split("_")[0]
        if split not in ("dev", "test"):
            continue
        h, count = hashlib.sha256(), 0
        with path.open("rb") as f:
            for raw in f:
                h.update(raw); row = loads(raw); count += 1
                pair = pairs[row["pair"]]
                if row["split"] != split or pair["split"] != split or row["group_id"] != pair["group_id"]:
                    raise ValueError("Eval manifest/catalog mismatch")
                if selected_keys & identities(pair) or row["sample_id"] in selected_ids:
                    raise ValueError("Selected train/eval overlap")
                if {row["speaker_id"], row["listener_id"]} != {pair["a"]["id"], pair["b"]["id"]}:
                    raise ValueError("Eval participant mismatch")
        files[str(path.resolve())] = dict(samples=count, sha256=h.hexdigest(), split=split)
    if {v["split"] for v in files.values()} != {"dev", "test"}:
        raise ValueError("Missing dev/test manifests")
    return files


def export_one(args):
    row, feature, output = args
    sid = row["sample_id"]
    receipt_path = feature/"media"/sid/"COMPLETE.json"
    receipt = read(receipt_path)
    obs, target = receipt["observation"], receipt["target"]
    if receipt["source_line_sha256"] != row["source_line_sha256"] or not receipt["media_verified"]:
        raise ValueError("Unverified or mismatched media receipt")
    if obs["sample_id"] != sid or target["sample_id"] != sid or obs["group_id"] != row["group_id"]:
        raise ValueError("Native manifest identity mismatch")
    if obs["official_split"] != "train" or obs["experiment_split"] != "train":
        raise ValueError("Non-train native manifest")
    if abs(obs["requested_duration_s"]-row["duration_s"]) > .0011:
        raise ValueError("Media/statistics duration mismatch")
    for path in [obs["speaker_audio"], obs["listener_image"], target["target_video"], target["target_audio"]]:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError("Missing media")
    canonical = feature/"labels/canonical"/(sid+".json")
    raw = canonical.read_bytes(); label = loads(raw)
    if label["sample_id"] != sid or label["taxonomy"] != list(CATEGORIES):
        raise ValueError("Canonical taxonomy/sample mismatch")
    counts = collections.Counter(e["type"] for e in label["events"])
    if any(counts[c] != row[c+"_count"] for c in CATEGORIES):
        raise ValueError("Canonical label/statistics category count mismatch")
    (output/"labels/canonical"/(sid+".json")).write_bytes(raw)
    cache = {}
    for part in ("input", "teacher", "vae", "events"):
        directory = feature/"cache"/part/sid
        marker = read(directory/"complete.json")
        if marker["contract"]["part"] != part or marker["contract"]["input"]["sample_id"] != sid:
            raise ValueError("Cache identity mismatch")
        if marker["contract"]["input"]["requested_duration_s"] != obs["requested_duration_s"]:
            raise ValueError("Cache duration mismatch")
        payload = directory/"payload.pt"
        if not payload.is_file() or payload.stat().st_size == 0:
            raise ValueError("Cache payload missing")
        cache[part] = dict(path=str(directory), identity=marker["identity"], payload_sha256=marker["payload_sha256"])
    return dict(observation=obs, target=target, reference=dict(sample_id=sid, source_manifest=row["source_manifest"],
        source_line_1based=row["source_line_1based"], source_line_sha256=row["source_line_sha256"],
        canonical_label_sha256=hashlib.sha256(raw).hexdigest(), media_receipt_sha256=sha(receipt_path),
        canonical_label_source=str(canonical), cache=cache))


def build(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output exists; use a new version directory")
    rows, policy, audit, pairs = prepare(args)
    summary = summarize(rows)
    if args.plan_only:
        print(json.dumps(dict(policy=policy, summary=summary, split_audit=audit), ensure_ascii=False, indent=2))
        return
    stats = Path(args.stats).resolve()
    provenance = read(stats/"provenance.json")
    feature = Path(provenance["feature_run"])
    if sha(feature/"index.json") != read(feature/"COMPLETE.json")["index_sha256"]:
        raise ValueError("Feature run completion identity mismatch")
    output.mkdir(parents=True)
    (output/"manifests").mkdir(); (output/"labels/canonical").mkdir(parents=True)
    atomic_json(output/"selection_policy.json", policy)
    atomic_json(output/"summary.json", summary)
    audit["eval_manifests"] = audit_eval_manifests(Path(args.catalog).parent/"manifests", pairs, rows)
    atomic_json(output/"split_audit.json", audit)
    with (output/"samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"sample_ids.txt").write_text("".join(r["sample_id"]+"\n" for r in rows))
    # Export original integrated records byte-for-byte, in shard/offset order.
    reader = SourceReader(feature)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["label_file"]].append(row)
    with (output/"integrated_labels.jsonl").open("wb") as out:
        for name in sorted(grouped):
            with Path(name).open("rb") as f:
                for row in sorted(grouped[name], key=lambda r: r["label_offset"]):
                    f.seek(row["label_offset"]); raw = f.read(row["label_bytes"])
                    if hashlib.sha256(raw).hexdigest() != row["label_sha256"]:
                        raise ValueError("Integrated label changed")
                    candidate = loads(raw)
                    full = reader.get(candidate["source"])
                    if full["sample_id"] != row["sample_id"] or full["split"] != "train":
                        raise ValueError("Export source mismatch")
                    out.write(raw)
    reader.check_files(); reader.close()
    observations, targets, reference_hashes = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (output/"dataset.jsonl").open("w") as out:
        for n, exported in enumerate(pool.map(export_one, ((r,feature,output) for r in rows)), 1):
            observations.append(exported["observation"]); targets.append(exported["target"])
            out.write(json.dumps(exported["reference"], ensure_ascii=False)+"\n")
            reference_hashes.append(exported["reference"]["canonical_label_sha256"])
            if n % 1000 == 0:
                atomic_json(output/"progress.json", dict(exported=n, total=len(rows)))
                print(json.dumps(dict(exported=n,total=len(rows))), flush=True)
    atomic_json(output/"manifests/observations.json", observations)
    atomic_json(output/"manifests/targets.json", targets)
    actual = sum(o["requested_duration_s"] for o in observations)
    if abs(actual-summary["hours"]*3600) > .1:
        raise ValueError("Native duration total mismatch")
    (output/"cache").symlink_to(feature/"cache", target_is_directory=True)
    atomic_json(output/"dataset_contract.json", dict(schema=VERSION, purpose="50-hour train-only rapid experiment subset",
        official_split="train", experiment_split="train", samples=len(rows), duration_s=actual,
        observations="manifests/observations.json", targets="manifests/targets.json", labels="labels/canonical",
        cache_source=str(feature/"cache"), feature_run=str(feature), heldout_policy="Use existing independent dev/test; never randomly split this train subset for final evaluation",
        labels_are_weak=True, training_started=False, input_statistics=str(stats),
        statistics_completion_sha256=sha(stats/"COMPLETE.json"), catalog=str(Path(args.catalog).resolve()),
        catalog_sha256=audit["catalog_sha256"], selector_code_sha256=sha(__file__),
        validation="Source and label hashes checked per selected sample; media/cache presence and receipt identity checked; full cache payload checksums inherited from completed upstream audit"))
    files = ["samples.csv", "sample_ids.txt", "dataset.jsonl", "integrated_labels.jsonl", "manifests/observations.json",
             "manifests/targets.json", "dataset_contract.json", "selection_policy.json", "split_audit.json", "summary.json"]
    atomic_json(output/"COMPLETE.json", dict(schema=VERSION, samples=len(rows), hours=actual/3600,
        files={f:sha(output/f) for f in files}, canonical_labels_sha256=hashlib.sha256("".join(reference_hashes).encode()).hexdigest(),
        canonical_hash_order="dataset.jsonl order", training_started=False))
    atomic_json(output/"progress.json",dict(phase="complete",exported=len(rows),total=len(rows)))
    print(json.dumps(dict(output=str(output),summary=summary),ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True); parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--hours", type=float, default=50)
    parser.add_argument("--class-minimum", type=int, default=2000)
    parser.add_argument("--max-per-pair", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--plan-only", action="store_true")
    build(parser.parse_args())
