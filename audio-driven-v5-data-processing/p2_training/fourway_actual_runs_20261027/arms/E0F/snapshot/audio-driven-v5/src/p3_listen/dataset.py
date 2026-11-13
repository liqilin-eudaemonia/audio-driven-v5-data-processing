"""Group split before windowing; labels live only in a separate review manifest."""
import itertools
import math
import random
import re
from collections import Counter
from pathlib import Path

from .common import digest, read, resolve, sha, write_once
from .prompts import contract


def identity(person):
    match = re.search(r"(V\d+)_(S\d+)_I\d+_(P\d+)$", person)
    if not match:
        raise ValueError(f"Unknown participant ID: {person}")
    version, session, participant = match.groups()
    return version + "_" + session, version + "_" + participant


def connected_groups(pairs):
    parent = list(range(len(pairs)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, pair in enumerate(pairs):
        for source in pair["sources"]:
            session, person = identity(source["id"])
            for key in ("session:" + session, "person:" + person):
                if key in owners:
                    parent[find(i)] = find(owners[key])
                owners[key] = i
    groups = {}
    for i, pair in enumerate(pairs):
        groups.setdefault(find(i), []).append(pair)
    result = []
    for group in groups.values():
        sessions, people = set(), set()
        counts = Counter()
        for pair in group:
            for src in pair["sources"]:
                s, p = identity(src["id"])
                sessions.add(s); people.add(p)
            counts.update(c["kind"] for c in pair["clips"])
        result.append(dict(group_id="+".join(sorted(sessions)), sessions=sorted(sessions),
                           participants=sorted(people), pairs=sorted(p["pair"] for p in group),
                           seconds=sum(c["duration"] for p in group for c in p["clips"]),
                           candidate_counts=dict(counts)))
    return sorted(result, key=lambda g: g["group_id"])


def split_groups(groups, seed, dev_count=2, test_count=2):
    if len(groups) <= dev_count + test_count or min(dev_count, test_count) < 1:
        raise ValueError("Not enough independent groups")
    # Exhaustive small-group allocation; objective uses data statistics only.
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    totals = Counter()
    for g in groups:
        totals.update(g["candidate_counts"])
    seconds = sum(g["seconds"] for g in groups)
    def score(indices):
        counts = Counter()
        for i in indices:
            counts.update(groups[i]["candidate_counts"])
        return ((sum(groups[i]["seconds"] for i in indices) / seconds - .15) ** 2
                + .2 * sum((counts[k] / n - .15) ** 2 for k, n in totals.items()))
    best = None
    for dev in itertools.combinations(order, dev_count):
        remaining = [i for i in order if i not in dev]
        for test in itertools.combinations(remaining, test_count):
            candidate = (score(dev) + score(test), dev, test)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return [dict(g, split="dev" if i in best[1] else "test" if i in best[2] else "train")
            for i, g in enumerate(groups)]


def windows(start, end, fps=25, max_frames=201, stride_seconds=4):
    if end <= start or fps <= 0 or max_frames < 9 or (max_frames - 1) % 8:
        raise ValueError("Invalid temporal geometry")
    available = math.floor((end - start) * fps + 1e-7)
    frames = min(max_frames, (available - 1) // 8 * 8 + 1)
    if frames < 9:
        return []
    duration = frames / fps
    stride = round(stride_seconds * fps)
    if stride < 1 or stride > max_frames:
        raise ValueError("Stride would leave gaps")
    offsets = list(range(0, available - frames + 1, stride))
    if offsets[-1] != available - frames:
        offsets.append(available - frames)
    return [dict(start=round(start + offset / fps, 9), end=round(start + offset / fps + duration, 9),
                 frames=frames, duration_s=duration) for offset in offsets]


def validate(manifest):
    seen, owners = set(), {}
    for g in manifest["groups"]:
        for key in ["s:" + s for s in g["sessions"]] + ["p:" + p for p in g["participants"]]:
            if key in owners and owners[key] != g["split"]:
                raise ValueError("Participant/session leakage")
            owners[key] = g["split"]
    pairs = {p: g["split"] for g in manifest["groups"] for p in g["pairs"]}
    for row in manifest["windows"]:
        if row["sample_id"] in seen or pairs[row["pair"]] != row["split"]:
            raise ValueError("Duplicate ID or split leakage")
        seen.add(row["sample_id"])
        if ((row["frames"] - 1) % 8 or row["frames"] > 201
                or row["start"] < row["clip_start"] - 1e-7 or row["end"] > row["clip_end"] + 1e-7
                or abs(row["end"] - row["start"] - row["frames"] / row["fps"]) > 1e-7):
            raise ValueError("Invalid window geometry")


def prepare(cfg):
    source = resolve(cfg["manifest"])
    original = read(source)
    if original["segmentation_revision"] not in (4, 5):
        raise ValueError("Expected speaker-span V4 or normalized subturn V5 data")
    if original["segmentation_revision"] == 5 and not original.get("normalization_complete"):
        raise ValueError("Audio normalization must complete before freezing training windows")
    d = cfg["data"]
    groups = connected_groups(original["pairs"])
    if cfg.get("frozen_splits"):
        frozen = {g["group_id"]:g for g in read(resolve(cfg["frozen_splits"]))["groups"]}
        if set(frozen) != {g["group_id"] for g in groups}:
            raise ValueError("Connected groups changed relative to frozen split")
        for g in groups:
            if g["participants"] != frozen[g["group_id"]]["participants"] or g["pairs"] != frozen[g["group_id"]]["pairs"]:
                raise ValueError("Frozen group membership changed")
            g["split"] = frozen[g["group_id"]]["split"]
    else:
        groups = split_groups(groups, cfg["seed"], d["dev_groups"], d["test_groups"])
    lookup = {p: g for g in groups for p in g["pairs"]}
    rows, reviews, short = [], [], []
    for pair in sorted(original["pairs"], key=lambda p: p["pair"]):
        group = lookup[pair["pair"]]
        for c in pair["clips"]:
            clip_id = pair["pair"] + "_" + Path(c["path"]).name
            if c["duration"] < d["min_seconds"]:
                short.append(dict(clip_id=clip_id, split=group["split"], start=c["start"], end=c["end"]))
                continue
            ws = windows(c["start"], c["end"], d["fps"], d["max_frames"], d["stride_seconds"])
            for i, w in enumerate(ws):
                sid = f"{clip_id}_w{i:03d}"
                row = dict(sample_id=sid, pair=pair["pair"], clip_id=clip_id, group_id=group["group_id"],
                           split=group["split"], **w, clip_start=c["start"], clip_end=c["end"],
                           fps=d["fps"], height=d["size"], width=d["size"],
                           speaker_id=c["speaker"], listener_id=c["listener"],
                           listener_crop=c["crop_boxes"]["listener"],
                           sources={role: pair["sources"][c[role + "_index"]]["source"] for role in ("speaker", "listener")})
                if c.get("normalized_audio"):
                    row["normalized_audio"] = {role:{k:c["normalized_audio"][role][k] for k in ("path","sha256","gain_db")}
                                               for role in ("speaker","listener")}
                rows.append(row)
                events = [dict(e, relative_start=max(e["start"], w["start"]) - w["start"],
                               relative_end=min(e["end"], w["end"]) - w["start"],
                               truncated=e["start"] < w["start"] or e["end"] > w["end"])
                          for e in c["listener_events"] if e["end"] > w["start"] and e["start"] < w["end"]]
                reviews.append(dict(sample_id=sid, split=group["split"], candidate_kind=c["kind"],
                                    events=events, unrepresented_tail_s=c["end"] - ws[-1]["end"]))
    # Deterministic shuffled order also avoids S0 consisting only of one long turn.
    random.Random(cfg["seed"]).shuffle(rows)
    manifest = dict(version=cfg["version"], source_sha256=sha(source), source=str(source),
                    official_source_split="test (repurposed for local training)", seed=cfg["seed"],
                    data_config=d, prompts=contract(), groups=groups, windows=rows, short_clips=short)
    validate(manifest)
    out = resolve(cfg["output"])
    write_once(out / "windows.json", manifest)
    write_once(out / "review_only.json", dict(note="Never a model input; automatic candidates, not verified labels", rows=reviews))
    selected = {}
    for split, count in (("train", 16), ("dev", 12)):
        pool = [r for r in rows if r["split"] == split]
        # Round-robin groups and candidate types, one window per clip first.
        buckets = {}
        kinds = {r["sample_id"]: r["candidate_kind"] for r in reviews}
        for row in pool:
            buckets.setdefault((row["group_id"], kinds[row["sample_id"]]), []).append(row)
        chosen, clips = [], set()
        while buckets and len(chosen) < count:
            for key in list(buckets):
                bucket = buckets[key]
                while bucket and bucket[0]["clip_id"] in clips:
                    bucket.pop(0)
                if bucket and len(chosen) < count:
                    row = bucket.pop(0); chosen.append(row["sample_id"]); clips.add(row["clip_id"])
                if not bucket:
                    del buckets[key]
        selected[split] = chosen
    write_once(out / "s0_selection.json", selected)
    summary = dict(groups=dict(Counter(g["split"] for g in groups)),
                   windows=dict(Counter(r["split"] for r in rows)), short_clips=len(short),
                   seconds={s: sum(g["seconds"] for g in groups if g["split"] == s) for s in ("train", "dev", "test")},
                   manifest_sha256=sha(out / "windows.json"))
    write_once(out / "data_summary.json", summary)
    print(summary, flush=True)
