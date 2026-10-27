"""Read-only statistics of integrated candidates and native, whole-clip face size.

No inference, label rewriting, event merging, or training admission is performed.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

VERSION = "listen-backchannel-face-stats-v1"
CATEGORIES = ("head_nod", "head_shake", "smile", "laugh", "frown", "surprise", "listener_vocalization")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return loads(Path(path).read_bytes())


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def union_seconds(intervals):
    end, total = -math.inf, 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


class SourceReader:
    """Persistent indexed reads; each worker owns its handles and RO connection."""
    def __init__(self, feature_run):
        self.root = Path(feature_run)
        self.files = read(self.root / "source_index/COMPLETE.json")["files"]
        self.check_files()
        self.handles = {name: Path(name).open("rb") for name in self.files}
        self.db = sqlite3.connect(f"file:{self.root}/source_index/offsets.sqlite?mode=ro", uri=True)

    def check_files(self):
        for name, info in self.files.items():
            st = Path(name).stat()
            if (st.st_size, st.st_mtime_ns) != (info["identity"]["size"], info["identity"]["mtime_ns"]):
                raise ValueError(f"Source manifest changed: {name}")

    def get(self, source):
        name = source["source_manifest"]
        info = self.files[name]
        result = self.db.execute("SELECT offset FROM offsets WHERE file_id=? AND line=?",
                                 (info["file_id"], source["source_line_1based"])).fetchone()
        if result is None:
            raise ValueError("Source offset missing")
        f = self.handles[name]
        f.seek(result[0])
        line = f.readline()
        if hashlib.sha256(line).hexdigest() != source["source_line_sha256"]:
            raise ValueError("Source line hash mismatch")
        full = loads(line)
        for key in ("sample_id", "split", "listener_id", "speaker_id"):
            if full[key] != source[key]:
                raise ValueError(f"Source {key} mismatch")
        if full["listener_id"] == full["speaker_id"]:
            raise ValueError("Listener/speaker collision")
        if full["sources"]["listener"] != source["source"]:
            raise ValueError("Listener video mismatch")
        if full["normalized_audio"]["listener"]["path"] != source["normalized_audio"]["path"]:
            raise ValueError("Listener audio mismatch")
        if abs(full["start"] - source["source_start_s"]) > 1e-6 or abs(full["end"] - source["source_end_s"]) > 1e-6:
            raise ValueError("Source time span mismatch")
        return full

    def close(self):
        self.db.close()
        for f in self.handles.values():
            f.close()


def sample_stats(candidate, full):
    source = candidate["source"]
    if candidate["sample_id"] != full["sample_id"]:
        raise ValueError("Candidate sample mismatch")
    duration = candidate["duration_s"]
    if not finite(duration) or duration <= 0:
        raise ValueError("Invalid sample duration")
    # Fused duration is rounded to milliseconds in some runs.
    if abs(duration - (full["end"] - full["start"])) > .0011:
        raise ValueError("Candidate/source duration mismatch")
    events = candidate["events"]
    ids = [e["event_id"] for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate integrated event_id")
    counts, modalities = collections.Counter(), collections.Counter()
    intervals = collections.defaultdict(list)
    valid, invalid, uncategorized, non_smile = 0, 0, 0, 0
    for event in events:
        categories = set(event.get("categories") or [])
        counts.update(categories)
        uncategorized += not categories
        non_smile += bool(categories - {"smile"})
        modality = event.get("modality", "unknown")
        modalities[modality] += 1
        start, end = event.get("start"), event.get("end")
        if not (finite(start) and finite(end) and 0 <= start < end <= duration + 1e-6):
            invalid += 1
            continue
        valid += 1
        span = (start, min(end, duration))
        intervals["all"].append(span)
        intervals[modality].append(span)
    face = (full.get("face_geometry") or {}).get("listener") or {}
    crop = (source.get("video_transform") or {}).get("crop") or {}
    proposal = source.get("crop_proposal") or {}
    row = dict(
        sample_id=candidate["sample_id"], pair=full.get("pair"), group_id=full.get("group_id"),
        split=full["split"], admission_pool=source.get("admission_pool"), listener_id=full["listener_id"],
        source_start_s=source["source_start_s"], source_end_s=source["source_end_s"], duration_s=duration,
        event_count=len(events), visual_event_count=modalities["visual"], audio_event_count=modalities["audio"],
        other_modality_event_count=len(events)-modalities["visual"]-modalities["audio"],
        valid_timed_event_count=valid, invalid_timed_event_count=invalid,
        events_per_10s=len(events)*10/duration, timed_events_per_10s=valid*10/duration,
        event_coverage_s=union_seconds(intervals["all"]),
        visual_coverage_s=union_seconds(intervals["visual"]), audio_coverage_s=union_seconds(intervals["audio"]),
        event_coverage_fraction=union_seconds(intervals["all"])/duration,
        category_count=len(counts), uncategorized_event_count=uncategorized, non_smile_event_count=non_smile,
        category_counts_json=json.dumps(dict(sorted(counts.items())), ensure_ascii=False, separators=(",", ":")),
        **{f"{cat}_count": counts[cat] for cat in CATEGORIES},
        response_group_count=len(candidate.get("response_groups", [])),
        original_observation_count=candidate.get("original_observation_count", 0),
        duplicate_observations_removed=sum(e.get("duplicate_observations_removed", 0) for e in events),
        raw_issue_count=candidate.get("raw_issue_count", 0), semantic_issue_count=candidate.get("semantic_issue_count", 0),
        unresolved_observation_count=len(candidate.get("unresolved_observations", [])),
        raw_completion_status=candidate.get("raw_completion_status"),
        face_status=face.get("status", "missing"), face_scope="whole_clip_valid_frames",
        face_valid_fraction=face.get("valid_fraction"), face_valid_frames=face.get("valid_frames"),
        face_expected_frames=face.get("expected_frames"),
        native_width=face.get("width"), native_height=face.get("height"),
        crop_x=crop.get("x"), crop_y=crop.get("y"), crop_width=crop.get("width"), crop_height=crop.get("height"),
        crop_proposal_size=proposal.get("size"),
        source_manifest=source["source_manifest"], source_line_1based=source["source_line_1based"],
        source_line_sha256=source["source_line_sha256"],
    )
    for metric in ("short_side", "width", "height", "area_fraction"):
        key = f"face_{metric}" + ("" if metric == "area_fraction" else "_px")
        for q in ("p10", "p50"):
            value = (face.get(key) or {}).get(q)
            row[f"face_{metric}_{q}"] = value if finite(value) and value > 0 and face.get("status") == "measured" else None
    for q in ("p10", "p50"):
        area, width, height = row[f"face_area_fraction_{q}"], row["native_width"], row["native_height"]
        row[f"face_area_px2_{q}"] = area*width*height if area is not None and width and height else None
    row["label_issue_flag"] = int(bool(row["raw_issue_count"] or row["semantic_issue_count"] or invalid))
    return row


_READER = None


def init_worker(feature_run):
    global _READER
    _READER = SourceReader(feature_run)


def process_shard(task):
    fusion_run, output, entry = task
    shard = entry["shard"]
    path = Path(fusion_run) / entry["labels"]
    target = Path(output) / "shards" / f"{shard}.jsonl"
    receipt = target.with_suffix(".done.json")
    expected = entry["files"][entry["labels"]]["sha256"]
    if receipt.exists():
        done = read(receipt)
        if done["source_sha256"] == expected and sha(target) == done["sha256"]:
            return done
        raise ValueError(f"Invalid resume shard: {shard}")
    h, count, offset, event_count = hashlib.sha256(), 0, 0, 0
    temp = target.with_suffix(".partial")
    with path.open("rb") as f, temp.open("w") as out:
        for number, raw in enumerate(f, 1):
            h.update(raw)
            candidate = loads(raw)
            full = _READER.get(candidate["source"])
            row = sample_stats(candidate, full)
            row.update(label_file=str(path), label_line_1based=number, label_offset=offset,
                       label_bytes=len(raw), label_sha256=hashlib.sha256(raw).hexdigest())
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
            event_count += row["event_count"]
            offset += len(raw)
    if h.hexdigest() != expected or count != entry["counts"]["samples"] or event_count != entry["counts"]["events"]:
        raise ValueError(f"Fusion shard audit mismatch: {shard}")
    _READER.check_files()
    os.replace(temp, target)
    done = dict(shard=shard, samples=count, events=event_count, sha256=sha(target), source_sha256=expected)
    atomic_json(receipt, done)
    return done


def column_type(key):
    if key in {"sample_id", "pair", "group_id", "split", "admission_pool", "listener_id", "category_counts_json",
               "raw_completion_status", "face_status", "face_scope", "source_manifest", "source_line_sha256",
               "label_file", "label_sha256"}:
        return "TEXT"
    if key.endswith("_count") or key.endswith("_frames") or key.endswith("_1based") or key in {
        "original_observation_count", "duplicate_observations_removed", "label_issue_flag", "label_offset", "label_bytes"}:
        return "INTEGER"
    return "REAL"


def assemble(output, index):
    db_path = output / "stats.sqlite.partial"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    columns, total = None, 0
    try:
        with (output / "samples.csv.partial").open("w", newline="") as csvfile:
            for entry in index["shards"]:
                with (output / "shards" / f"{entry['shard']}.jsonl").open() as stream:
                    batch = [loads(line) for line in stream]
                if not batch:
                    continue
                if columns is None:
                    columns = list(batch[0])
                    conn.execute("CREATE TABLE samples (" + ",".join(
                        f'"{k}" {column_type(k)}' + (" PRIMARY KEY" if k == "sample_id" else "") for k in columns) + ")")
                    writer = csv.DictWriter(csvfile, fieldnames=columns)
                    writer.writeheader()
                conn.executemany("INSERT INTO samples VALUES (" + ",".join("?" for _ in columns) + ")",
                                 [tuple(row[k] for k in columns) for row in batch])
                writer.writerows(batch)
                total += len(batch)
                conn.commit()
        if total != index["samples"]:
            raise ValueError(f"Total mismatch: {total} vs {index['samples']}")
        conn.execute("CREATE INDEX sample_events ON samples(event_count DESC,face_short_side_p10 DESC,sample_id)")
        conn.execute("CREATE INDEX sample_face ON samples(face_short_side_p10,face_short_side_p50)")
        conn.execute("CREATE INDEX sample_pair ON samples(pair)")
        conn.commit()
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Statistics SQLite integrity failure")
    finally:
        conn.close()
    os.replace(db_path, output / "stats.sqlite")
    os.replace(output / "samples.csv.partial", output / "samples.csv")


def summarize(output):
    import numpy as np
    columns = ["duration_s", "event_count", "events_per_10s", "face_short_side_p10", "face_short_side_p50",
               "face_valid_fraction", "label_issue_flag", "event_coverage_fraction", "visual_event_count", "audio_event_count"]
    with sqlite3.connect(f"file:{output}/stats.sqlite?mode=ro", uri=True) as conn:
        data = np.array(conn.execute("SELECT " + ",".join(columns) + " FROM samples").fetchall(), dtype=float)
        categories = {c: conn.execute(f"SELECT SUM({c}_count) FROM samples").fetchone()[0] for c in CATEGORIES}
        totals = dict(conn.execute("SELECT split,COUNT(*) FROM samples GROUP BY split"))
        invalid = conn.execute("SELECT SUM(invalid_timed_event_count),SUM(raw_issue_count>0),SUM(semantic_issue_count>0) FROM samples").fetchone()
    col = {k: data[:, i] for i, k in enumerate(columns)}
    quantiles = {}
    for k in columns:
        values = col[k][np.isfinite(col[k])]
        quantiles[k] = {f"p{q:02}": float(np.percentile(values, q)) for q in (0, 10, 25, 50, 75, 90, 95, 99, 100)} if len(values) else {}
    summary = dict(schema=VERSION, samples=len(data), clip_hours=float(col["duration_s"].sum()/3600),
                   events=int(col["event_count"].sum()), visual_events=int(col["visual_event_count"].sum()),
                   audio_events=int(col["audio_event_count"].sum()), zero_candidate_samples=int((col["event_count"] == 0).sum()),
                   missing_face_p10=int((~np.isfinite(col["face_short_side_p10"])).sum()),
                   samples_with_label_issues=int(col["label_issue_flag"].sum()), invalid_timed_events=invalid[0],
                   samples_with_raw_issues=invalid[1], samples_with_semantic_issues=invalid[2],
                   splits=totals, per_category=categories, quantiles=quantiles,
                   notes=["Events are integrated model candidates, not verified physical occurrence counts.",
                          "Face size is native facial-landmark extent over valid frames of the whole clip, not crop size or event-local size.",
                          "p10 is the 10th percentile; missing face data is null, never replaced by crop dimensions.",
                          "Multi-category events count once in total; category totals can exceed event total.",
                          "Clip hours sum durations; overlapping source intervals are not deduplicated.",
                          "No candidate does not imply a verified negative; no training admission is granted."])
    atomic_json(output / "summary.json", summary)
    grid = []
    for metric in ("p10", "p50"):
        for clean in (False, True):
            for size in (64, 96, 128, 160, 192, 224, 256, 320, 384):
                base = (col[f"face_short_side_{metric}"] >= size) & (col["face_valid_fraction"] >= .9)
                if clean:
                    base &= col["label_issue_flag"] == 0
                for minimum in (0, 1, 2, 3, 5, 8, 10):
                    mask = base & (col["event_count"] >= minimum)
                    n = int(mask.sum())
                    grid.append(dict(face_metric=metric, min_face_short_side_px=size, min_face_valid_fraction=.9,
                                     exclude_label_issues=clean, min_events=minimum, samples=n,
                                     retained_fraction=n/len(data), clip_hours=float(col["duration_s"][mask].sum()/3600),
                                     events=int(col["event_count"][mask].sum()),
                                     mean_events=float(col["event_count"][mask].mean()) if n else None,
                                     mean_events_per_10s=float(col["events_per_10s"][mask].mean()) if n else None))
    with (output / "threshold_grid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0])); writer.writeheader(); writer.writerows(grid)
    return summary


def build(args):
    fusion, feature, output = (Path(x).resolve() for x in (args.fusion_run, args.feature_run, args.output))
    if args.workers < 1:
        raise ValueError("workers must be positive")
    index = read(fusion / "index.json")
    complete = read(fusion / "COMPLETE.json")
    feature_complete = read(feature / "COMPLETE.json")
    if complete["index_sha256"] != sha(fusion / "index.json") or complete["samples"] != index["samples"]:
        raise ValueError("Fusion completion identity mismatch")
    if feature_complete["samples"] != index["samples"] or feature_complete["index_sha256"] != sha(feature / "index.json"):
        raise ValueError("Feature completion identity mismatch")
    if Path(read(feature / "config.json")["fusion_run"]).resolve() != fusion:
        raise ValueError("Feature/fusion run mismatch")
    if sum(e["counts"]["samples"] for e in index["shards"]) != index["samples"] or not all(e["done"] for e in index["shards"]):
        raise ValueError("Fusion shard accounting mismatch")
    if len({e["shard"] for e in index["shards"]}) != len(index["shards"]):
        raise ValueError("Duplicate shard IDs")
    provenance = dict(schema=VERSION, fusion_run=str(fusion), feature_run=str(feature),
                      fusion_index_sha256=sha(fusion / "index.json"), feature_index_sha256=sha(feature / "index.json"),
                      source_index_sha256=sha(feature / "source_index/COMPLETE.json"), code_sha256=sha(__file__),
                      sample_scope="all_completed_fusion_samples", face_scope="whole_clip_valid_frames")
    if output.exists():
        if not args.resume or not (output / "provenance.json").exists() or read(output / "provenance.json") != provenance:
            raise ValueError("Output already exists; use --resume only with identical inputs and code")
        if (output / "COMPLETE.json").exists():
            for name, digest in read(output / "COMPLETE.json")["files"].items():
                if sha(output / name) != digest:
                    raise ValueError(f"Completed output changed: {name}")
            print("Already complete and output hashes verified", flush=True)
            return
    else:
        output.mkdir(parents=True)
        atomic_json(output / "provenance.json", provenance)
    (output / "shards").mkdir(exist_ok=True)
    reader = SourceReader(feature)
    reader.close()
    source_index = read(feature / "source_index/COMPLETE.json")
    if sha(feature / "source_index/offsets.sqlite") != source_index["db_sha256"]:
        raise ValueError("Source offsets database hash mismatch")
    started, done, count = time.time(), 0, 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(feature),)) as pool:
        futures = [pool.submit(process_shard, (str(fusion), str(output), e)) for e in index["shards"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1; count += result["samples"]
            if done % 16 == 0 or done == len(futures):
                status = dict(phase="statistics", shards_done=done, shards_total=len(futures), samples=count,
                              total=index["samples"], elapsed_s=round(time.time()-started, 2))
                atomic_json(output / "progress.json", status)
                print(json.dumps(status), flush=True)
    reader.check_files()
    atomic_json(output / "progress.json", dict(phase="assembling_csv_sqlite", samples=count, total=index["samples"]))
    assemble(output, index)
    summary = summarize(output)
    names = ["stats.sqlite", "samples.csv", "summary.json", "threshold_grid.csv", "provenance.json"]
    atomic_json(output / "COMPLETE.json", dict(schema=VERSION, samples=summary["samples"], events=summary["events"],
                                             elapsed_s=time.time()-started, files={name: sha(output/name) for name in names},
                                             training_admitted=False))
    atomic_json(output / "progress.json", dict(phase="complete", samples=count, total=index["samples"]))
    print(json.dumps(dict(samples=summary["samples"], events=summary["events"], output=str(output))), flush=True)


def open_stats(root):
    root = Path(root).resolve()
    if not (root / "COMPLETE.json").exists():
        raise ValueError("Statistics build is incomplete")
    return sqlite3.connect(f"file:{root}/stats.sqlite?mode=ro", uri=True)


def selection_query(args):
    thresholds = [args.min_events, args.min_density, args.min_face_p10, args.min_face_p50, args.min_face_valid]
    if any(not finite(x) or x < 0 for x in thresholds) or args.min_face_valid > 1:
        raise ValueError("Thresholds must be finite and nonnegative; face valid fraction must be <=1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    clauses = ["event_count >= ?", "events_per_10s >= ?"]
    values = [args.min_events, args.min_density]
    # Explicit nonzero face constraints exclude missing measurements. No fallback.
    for column, value in (("face_short_side_p10", args.min_face_p10), ("face_short_side_p50", args.min_face_p50),
                          ("face_valid_fraction", args.min_face_valid)):
        if value > 0:
            clauses.append(f"{column} >= ?"); values.append(value)
    if args.exclude_label_issues:
        clauses.append("label_issue_flag=0")
    if args.split:
        clauses.append("split=?"); values.append(args.split)
    order = {"events": "event_count", "density": "events_per_10s", "face": "face_short_side_p10"}[args.sort]
    query = "SELECT * FROM samples WHERE " + " AND ".join(clauses) + f" ORDER BY {order} DESC,face_short_side_p10 DESC,event_count DESC,sample_id ASC"
    if args.limit is not None:
        query += " LIMIT ?"; values.append(args.limit)
    return query, values


def select(args):
    query, params = selection_query(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Selection output already exists; choose a fresh directory")
    with open_stats(args.stats) as conn:
        cursor = conn.execute(query, params)
        fields = [col[0] for col in cursor.description]
        output.mkdir(parents=True)
        count, seconds, events = 0, 0.0, 0
        with (output / "samples.csv").open("w", newline="") as f, (output / "sample_ids.txt").open("w") as ids:
            writer = csv.writer(f); writer.writerow(fields)
            for row in cursor:
                writer.writerow(row)
                value = dict(zip(fields, row))
                ids.write(value["sample_id"] + "\n")
                count += 1; seconds += value["duration_s"]; events += value["event_count"]
    manifest = dict(schema=VERSION, stats=str(Path(args.stats).resolve()),
                    statistics_completion_sha256=sha(Path(args.stats)/"COMPLETE.json"),
                    criteria={k: v for k, v in vars(args).items() if k not in ("func", "command")},
                    sql=query, parameters=params, samples=count, clip_hours=seconds/3600, events=events,
                    training_admitted=False, feature_run=read(Path(args.stats)/"provenance.json")["feature_run"],
                    files={n: sha(output/n) for n in ("samples.csv", "sample_ids.txt")})
    atomic_json(output / "selection.json", manifest)
    print(json.dumps(dict(output=str(output), samples=count, clip_hours=seconds/3600, events=events)))


def inspect_sample(stats, sample_id):
    with open_stats(stats) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown sample_id")
        row = dict(row)
    with Path(row["label_file"]).open("rb") as f:
        f.seek(row["label_offset"]); raw = f.read(row["label_bytes"])
    if hashlib.sha256(raw).hexdigest() != row["label_sha256"]:
        raise ValueError("Referenced label changed")
    candidate = loads(raw)
    reader = SourceReader(read(Path(stats)/"provenance.json")["feature_run"])
    try:
        full = reader.get(candidate["source"])
    finally:
        reader.close()
    return dict(statistics=row, face_geometry=full.get("face_geometry", {}).get("listener"), integrated_label=candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="Join all integrated labels with native per-clip face statistics")
    p.add_argument("--fusion-run", required=True); p.add_argument("--feature-run", required=True)
    p.add_argument("--output", required=True); p.add_argument("--workers", type=int, default=16)
    p.add_argument("--resume", action="store_true"); p.set_defaults(func=build)
    p = sub.add_parser("select", help="Export sample IDs and statistics; leaves labels and caches unchanged")
    p.add_argument("--stats", required=True); p.add_argument("--output", required=True)
    p.add_argument("--min-events", type=int, default=0); p.add_argument("--min-density", type=float, default=0)
    p.add_argument("--min-face-p10", type=float, default=0); p.add_argument("--min-face-p50", type=float, default=0)
    p.add_argument("--min-face-valid", type=float, default=0)
    p.add_argument("--exclude-label-issues", action="store_true"); p.add_argument("--split")
    p.add_argument("--sort", choices=["events", "density", "face"], default="events")
    p.add_argument("--limit", type=int); p.set_defaults(func=select)
    p = sub.add_parser("show", help="Read original events and full face measurements for a sample")
    p.add_argument("--stats", required=True); p.add_argument("--sample-id", required=True)
    p.set_defaults(func=lambda a: print(json.dumps(inspect_sample(a.stats, a.sample_id), ensure_ascii=False, indent=2)))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
