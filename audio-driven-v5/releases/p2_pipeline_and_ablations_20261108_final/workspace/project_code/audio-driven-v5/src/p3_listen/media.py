"""Extract original archive MP4/WAV, never preview AAC or preview video."""
from functools import lru_cache
from pathlib import Path
import json
import subprocess
import tarfile

from .common import digest, read, resolve, sha, write
from .dataset import validate


@lru_cache(maxsize=64)
def members(archive):
    with tarfile.open(archive, "r:") as tar:
        return {m.name: (m.offset_data, m.size) for m in tar if m.isfile()}


def uri(source, extension):
    name = str(Path(source["member"]).with_suffix(extension))
    offset, size = members(source["archive"])[name]
    if extension == ".mp4" and (offset != source["offset"] or size != source["byte_size"]):
        raise ValueError("Original archive index changed")
    return f"subfile,,start,{offset},end,{offset + size},,:{source['archive']}"


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr[-4000:]}")
    return result.stdout


def selection(cfg, args):
    root = resolve(cfg["output"])
    manifest = read(root / "windows.json")
    validate(manifest)
    rows = [r for r in manifest["windows"] if r["split"] == args.split]
    if args.s0:
        ids = read(root / "s0_selection.json").get(args.split, [])
        if not ids:
            raise ValueError("S0 is train/dev only")
        lookup = {r["sample_id"]: r for r in rows}
        rows = [lookup[sid] for sid in ids]
    if args.limit < 0:
        raise ValueError("Negative limit")
    return manifest, rows[:args.limit] if args.limit else rows


def paths(root, row):
    folder = Path(root) / "media" / row["split"] / row["sample_id"]
    return {name: folder / filename for name, filename in dict(
        speaker_audio="speaker.wav", target_audio="listener.wav", target_video="listener.mp4",
        listener_image="reference.png", marker="complete.json").items()}


def verify_media(root, row):
    p = paths(root, row)
    marker = read(p["marker"])
    if marker["row_sha256"] != digest(row):
        raise ValueError("Export window contract changed")
    for key, value in marker["files"].items():
        if sha(p[key]) != value:
            raise ValueError(f"Export changed: {p[key]}")
    return p, marker


def export(root, row):
    p = paths(root, row)
    if p["marker"].exists():
        verify_media(root, row)
        return
    p["marker"].parent.mkdir(parents=True, exist_ok=True)
    common = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2"]
    # Accurate input seek decodes/discards to the common absolute start.
    for role, key in (("speaker", "speaker_audio"), ("listener", "target_audio")):
        tmp = p[key].with_name(p[key].stem + ".pending.wav")
        normalized = row.get("normalized_audio", {}).get(role)
        audio_source = normalized["path"] if normalized else uri(row["sources"][role], ".wav")
        if normalized and sha(audio_source) != normalized["sha256"]:
            raise ValueError("Normalized source audio changed")
        command(common + ["-ss", str(row["start"]), "-i", audio_source,
                          "-t", str(row["duration_s"]), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s24le", str(tmp)])
        tmp.replace(p[key])
    b = row["listener_crop"]
    vf = (f"setpts=PTS-STARTPTS,crop={b['size']}:{b['size']}:{b['x']}:{b['y']},"
          f"scale={row['width']}:{row['height']},fps={row['fps']},setsar=1")
    tmp = p["target_video"].with_name("listener.pending.mp4")
    command(common + ["-ss", str(row["start"]), "-i", uri(row["sources"]["listener"], ".mp4"),
                      "-vf", vf, "-frames:v", str(row["frames"]), "-an", "-c:v", "libx264", "-threads", "2",
                      "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    tmp.replace(p["target_video"])
    command(common + ["-i", str(p["target_video"]), "-frames:v", "1", str(p["listener_image"])])
    probe = json.loads(command(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(p["target_video"])]))
    v = probe["streams"][0]
    if (int(v["nb_frames"]), int(v["height"]), int(v["width"])) != (row["frames"], row["height"], row["width"]):
        raise ValueError("Extracted video geometry mismatch")
    import soundfile as sf
    for key in ("speaker_audio", "target_audio"):
        info = sf.info(str(p[key]))
        if info.frames != round(row["duration_s"] * info.samplerate):
            raise ValueError("Original audio too short or duration mismatch")
    write(p["marker"], dict(row_sha256=digest(row), files={k: sha(v) for k, v in p.items() if k != "marker"},
                           source="original archive MP4 and separate WAV", frames=row["frames"], duration_s=row["duration_s"]))


def run(cfg, args):
    manifest, rows = selection(cfg, args)
    root = resolve(cfg["output"])
    for i, row in enumerate(rows):
        if args.command == "export":
            export(root, row)
        else:
            verify_media(root, row)
        print(json.dumps(dict(operation=args.command, index=i + 1, total=len(rows), sample_id=row["sample_id"])), flush=True)
