"""Import caller-supplied, already cut media; never select production data implicitly."""
from pathlib import Path
from .contracts import (atomic_json, audit_splits, observation, target, rows,
                        path_at, digest, sha, read)
from .layout import geometry


def prepare(cfg, observations_path, targets_path):
    op, tp = Path(observations_path).resolve(), Path(targets_path).resolve()
    observations = [observation(x, op.parent) for x in rows(op)]
    raw = {x["sample_id"]: x for x in rows(tp)}
    if set(raw) != {x["sample_id"] for x in observations}:
        raise ValueError("Observation/target sample IDs must match exactly")
    targets = [target(raw[o["sample_id"]], o, tp.parent) for o in observations]
    counts = audit_splits(observations)
    for o, t in zip(observations, targets):
        g = geometry(o)
        d = cfg["data"]
        if (g["frames"] > d["max_frames"] or g["real_duration_s"] < d["min_seconds"] or
            g["fps"] != d["fps"] or g["height"] != d["size"] or g["width"] != d["size"]):
            raise ValueError(f"Sample outside pilot geometry: {o['sample_id']}; do not truncate")
        verify_media(o, t)
    root = path_at(cfg["output"]) / "manifests"
    atomic_json(root / "observations.json", observations, immutable=True)
    atomic_json(root / "targets.json", targets, immutable=True)
    report = dict(schema="p2-listen-admission-v1", counts=counts,
                  observations_sha256=digest(observations), targets_sha256=digest(targets),
                  source_files={str(op): sha(op), str(tp): sha(tp)},
                  grouping_verified=True, media_verified=True, training_started=False)
    atomic_json(root / "admission.json", report, immutable=True)
    return report


def verify_media(obs, target_row=None):
    import soundfile as sf
    from PIL import Image
    real = obs["requested_duration_s"]
    for path in [obs["speaker_audio"]] + ([] if target_row is None else [target_row["target_audio"]]):
        info = sf.info(path)
        if abs(info.frames / info.samplerate - real) > 1 / info.samplerate + 1e-6:
            raise ValueError(f"Audio duration does not cover exactly the window: {path}")
    with Image.open(obs["listener_image"]) as im:
        if im.size != (obs["geometry"]["width"],obs["geometry"]["height"]):
            raise ValueError("Reference must already match the declared crop/geometry")
        im.verify()
    if target_row is not None:
        import av
        with av.open(target_row["target_video"]) as video:
            stream = video.streams.video[0]
            g = obs["geometry"]
            if (stream.width, stream.height) != (g["width"], g["height"]) or abs(float(stream.average_rate) - g["fps"]) > 1e-4:
                raise ValueError("Pre-cut target video must match declared geometry/FPS")
            # Decode, rather than trust often-missing container frame counts.
            times = [float(f.time) for f in video.decode(video=0)]
        if not times or abs(times[0]) > 1e-4 or any(abs(t-i/g["fps"]) > .5/g["fps"] for i,t in enumerate(times)):
            raise ValueError("Target timestamps must be continuous and start at zero")
        if abs(len(times) / g["fps"] - real) > 1/g["fps"] + 1e-6:
            raise ValueError("Video duration differs from real audio support")


def load_observations(cfg, split=None, verify=True):
    values = read(path_at(cfg["output"]) / "manifests/observations.json")
    result = [observation(r, verify=verify) for r in values if split is None or r["experiment_split"] == split]
    audit_splits(values)
    return result


def load_targets(cfg, observations):
    raw = {r["sample_id"]: r for r in read(path_at(cfg["output"]) / "manifests/targets.json")}
    return {o["sample_id"]: target(raw[o["sample_id"]], o) for o in observations}


def import_labels(cfg, labels_path):
    from .annotations import validate
    observations = {o["sample_id"]: o for o in load_observations(cfg)}
    targets = load_targets(cfg, list(observations.values()))
    imported = []
    for label in rows(labels_path):
        sid = label["sample_id"]
        validate(label, observations[sid])
        if label["provenance"].get("media_hashes") != targets[sid]["media_hashes"]:
            raise ValueError("Labels must bind both target media hashes")
        if observations[sid]["experiment_split"] != "train" and label["provenance"].get("human_verified") is not True:
            raise ValueError("Dev/test reference labels require human verification")
        atomic_json(path_at(cfg["output"]) / "labels/canonical" / (sid + ".json"), label, immutable=True)
        imported.append(sid)
    return dict(imported=imported, input_sha256=sha(labels_path))
