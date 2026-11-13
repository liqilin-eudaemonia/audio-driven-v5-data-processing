"""Timestamped silent frames and isolated listener waveform, without VAD pruning."""
import base64
import hashlib
import io
import math
from pathlib import Path
import av
import numpy as np
import soundfile as sf
from PIL import Image
from p2_listen.contracts import positive, safe_id, sha


def validate_transform(value):
    if not isinstance(value, dict) or set(value) != {"crop", "source_size", "output_size", "resample"}:
        raise ValueError("Explicit crop/source_size/output_size/resample required")
    if value["output_size"] != [512, 512] or value["resample"] != "bicubic":
        raise ValueError("Crop annotation must use 512x512 bicubic resize")
    size, crop = value["source_size"], value["crop"]
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Invalid source dimensions")
    if not isinstance(crop, dict) or set(crop) != {"x", "y", "width", "height"}:
        raise ValueError("Resolved crop required; never fall back to whole frame")
    if any(type(v) is not int for v in crop.values()):
        raise ValueError("Crop must use integer source-pixel coordinates")
    x, y, w, h = (crop[k] for k in ("x", "y", "width", "height"))
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > size[0] or y + h > size[1]:
        raise ValueError("Crop outside source bounds; no silent clamping/padding")
    return value


def crop_resize(image, transform):
    validate_transform(transform)
    if list(image.size) != transform["source_size"]:
        raise ValueError("Decoded source size mismatch; cannot crop a pre-cropped video again")
    c = transform["crop"]
    return image.crop((c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"])).resize(
        (512, 512), resample=Image.Resampling.BICUBIC)


def prepare_row(raw, base, mode="split"):
    row = dict(raw)
    safe_id(row["sample_id"])
    positive(row["duration_s"], "duration_s")
    splits = ("train", "dev", "test", "unassigned") if mode == "visual" else ("train", "dev", "test")
    if row.get("experiment_split") not in splits:
        raise ValueError("Explicit experiment_split required")
    if (row.get("time_origin") != "clip_start_seconds" or row.get("video_role") != "listener"):
        raise ValueError("Require listener video on clip-start timeline")
    isolated = row.get("audio_role") == "listener" and row.get("audio_isolated") is True
    mixed_visual = mode == "visual" and row.get("audio_role") == "mixed" and row.get("audio_isolated") is False
    if not (isolated or mixed_visual):
        raise ValueError("Require pre-cut aligned listener video and isolated listener audio")
    if "video_transform" in row:
        validate_transform(row["video_transform"])
    hashes = {}
    for key in ("target_video", "target_audio"):
        path = Path(row[key])
        row[key] = str(path.resolve() if path.is_absolute() else (Path(base) / path).resolve())
        hashes[key] = sha(row[key])
    if "media_hashes" in row and row["media_hashes"] != hashes:
        raise ValueError("Input media hash mismatch")
    row["media_hashes"] = hashes
    return row


def frames(path, duration, cfg, transform=None):
    if transform is not None:
        validate_transform(transform)
    fps = positive(cfg["fps"], "sampling fps")
    limit, edge = cfg["max_frames"], cfg["max_edge"]
    if type(limit) is not int or limit < 1 or type(edge) is not int or edge < 32:
        raise ValueError("Invalid frame budget/edge")
    if math.ceil(duration * fps) > limit:
        raise ValueError("Frame budget exceeded; explicitly cut shorter clips, never silently reduce FPS")
    content, records = [], []
    next_grid, previous, last, end, decoded = 0, None, None, None, 0
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        if transform is not None and [stream.width, stream.height] != transform["source_size"]:
            raise ValueError("Decoded source size mismatch; cannot apply original crop to resized input")
        rate = float(stream.average_rate) if stream.average_rate else 0
        for frame in container.decode(video=0):
            decoded += 1
            if frame.pts is None:
                raise ValueError("Video frame lacks PTS")
            t = float(frame.pts * frame.time_base)
            if previous is None and abs(t) > 1e-4:
                raise ValueError("Video must start at zero; do not silently rebase")
            if previous is not None and t <= previous:
                raise ValueError("Non-monotonic video PTS")
            previous, last = t, t
            frame_duration = float(frame.duration * frame.time_base) if frame.duration else (1 / rate if rate else 0)
            end = t + frame_duration
            if t + 1e-8 < next_grid / fps or t >= duration:
                continue
            if len(records) >= limit:
                raise ValueError("Decoded frame budget exceeded")
            im = frame.to_image()
            if transform is not None:
                im = crop_resize(im, transform)
            else:
                im.thumbnail((edge, edge))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            jpeg = buf.getvalue()
            content.extend([{"type": "text", "text": f"listener frame at {t:.6f} seconds:"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}])
            records.append(dict(time_s=t, clip_frame=decoded - 1, jpeg_sha256=hashlib.sha256(jpeg).hexdigest(), size=list(im.size)))
            next_grid = math.floor((t + 1e-8) * fps) + 1
    if not records or last is None or end is None or abs(end - duration) > max(1e-4, 1 / rate if rate else 1e-4):
        raise ValueError("Video does not match declared duration")
    return content, dict(frames=records, requested_fps=fps, last_frame_s=last, decoded_end_s=end,
                         decoded_frames=decoded, video_transform=transform)


def waveform(path, duration, cfg):
    info = sf.info(path)
    if abs(info.frames / info.samplerate - duration) > 1 / info.samplerate + 1e-6:
        raise ValueError("Audio duration differs from clip")
    if info.channels != 1:
        raise ValueError("Expected isolated mono audio; never downmix speakers automatically")
    if duration > cfg["max_audio_seconds"]:
        raise ValueError("Audio exceeds configured request budget")
    samples, rate = sf.read(path, dtype="float32")
    if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) > 1:
        raise ValueError("Invalid/non-normalized waveform")
    # Lossless PCM16 transport, original rate, no ASR/VAD/normalization or resampling.
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()
    return [{"type": "text", "text": "Isolated LISTENER audio; sample zero is clip time 0 seconds."},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}], dict(
                samplerate=rate, samples=len(samples), wav_sha256=hashlib.sha256(wav).hexdigest(), transport="PCM_16")
