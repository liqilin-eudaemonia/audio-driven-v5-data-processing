"""Small, CPU-only contracts. Inference never needs a target or label file."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]
VISUAL_TYPES = ("head_nod", "head_shake", "smile", "laugh", "frown", "surprise")
AUDIO_TYPES = ("listener_vocalization",)
EVENTS = VISUAL_TYPES + AUDIO_TYPES
TYPE_TO_MODALITY = {**dict.fromkeys(VISUAL_TYPES, "video"), **dict.fromkeys(AUDIO_TYPES, "audio")}
LABEL_SCHEMA = "listen-events-v2-visual6-audio1"


def class_subset(value, name="K_train"):
    if not isinstance(value, (list, tuple)) or list(value) != [k for k in EVENTS if k in value]:
        raise ValueError(f"{name} must be a unique subset in fixed taxonomy order")
    return tuple(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if digest(read(path)) != digest(value):
            raise ValueError(f"Existing identity differs: {path}")
        return
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        result = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    else:
        result = read(path)
    if not isinstance(result, list) or not result:
        raise ValueError("Expected a nonempty array or JSONL")
    ids = [safe_id(r["sample_id"]) for r in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    return result


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ValueError("Invalid sample/group ID")
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {name}")
    return value


def path_at(path, base=ROOT):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def observation(row, base=ROOT, verify=True):
    allowed = {"schema", "sample_id", "experiment_split", "official_split", "group_id",
               "grouping_ids", "speaker_audio", "listener_image", "requested_duration_s",
               "geometry", "media_hashes"}
    if set(row) - allowed or row.get("schema") != "p2-listen-observation-v1":
        raise ValueError("Observation schema/unknown fields (targets and plans are forbidden)")
    r = dict(row)
    safe_id(r["sample_id"]); safe_id(r["group_id"])
    if r["experiment_split"] not in ("train", "dev", "test") or r["official_split"] not in ("train", "dev", "test"):
        raise ValueError("Invalid split")
    if not isinstance(r.get("grouping_ids"), list) or not r["grouping_ids"] or not all(isinstance(x, str) and x for x in r["grouping_ids"]):
        raise ValueError("grouping_ids must include participant/session identities for split auditing")
    positive(r["requested_duration_s"], "duration")
    g = r["geometry"]
    if set(g) != {"fps", "height", "width"}:
        raise ValueError("Explicit fps/height/width required")
    positive(g["fps"], "fps")
    for key in ("height", "width"):
        if type(g[key]) is not int or g[key] <= 0 or g[key] % 32:
            raise ValueError("Video dimensions must be positive multiples of 32")
    if set(r.get("media_hashes", {})) != {"speaker_audio", "listener_image"}:
        raise ValueError("Input media hashes required")
    for key in ("speaker_audio", "listener_image"):
        r[key] = str(path_at(r[key], base))
        if verify and sha(r[key]) != r["media_hashes"][key]:
            raise ValueError(f"Input changed: {key}")
    return r


def input_identity(obs):
    # Routing/labels/target hashes must not affect an input cache.
    return {k: obs[k] for k in ("sample_id", "speaker_audio", "listener_image",
                                "requested_duration_s", "geometry", "media_hashes")}


def audit_splits(observations):
    seen = {}
    for r in observations:
        for key in ["group:" + r["group_id"], *["identity:" + x for x in r["grouping_ids"]]]:
            previous = seen.setdefault(key, r["experiment_split"])
            if previous != r["experiment_split"]:
                raise ValueError(f"Connected identity crosses splits: {key}")
    return {s: sum(r["experiment_split"] == s for r in observations) for s in ("train", "dev", "test")}


def target(row, obs, base=ROOT, verify=True):
    if set(row) != {"schema", "sample_id", "target_video", "target_audio", "duration_s", "media_hashes", "admitted"}:
        raise ValueError("Target schema fields differ")
    if row["schema"] != "p2-listen-target-v1" or row["sample_id"] != obs["sample_id"] or row["admitted"] is not True:
        raise ValueError("Unadmitted or mismatched target")
    if abs(positive(row["duration_s"], "target duration") - obs["requested_duration_s"]) > 1e-6:
        raise ValueError("Target must cover the exact observation window")
    r = dict(row)
    if set(r["media_hashes"]) != {"target_video", "target_audio"}:
        raise ValueError("Target hashes required")
    for key in ("target_video", "target_audio"):
        r[key] = str(path_at(r[key], base))
        if verify and sha(r[key]) != r["media_hashes"][key]:
            raise ValueError("Target media changed")
    return r
