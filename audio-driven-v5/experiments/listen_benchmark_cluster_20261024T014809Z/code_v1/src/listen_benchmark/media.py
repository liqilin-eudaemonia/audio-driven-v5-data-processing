"""Prepare exact listener evidence and explicit model-native representations."""
import base64
import io
import math

import numpy as np
from PIL import Image
import soundfile as sf
from scipy.signal import resample_poly

from listen_labeling.pipeline import branch_inputs


def prepare(row, branch, config):
    content, metadata = branch_inputs(row, branch, config)
    pictures, audio, rate = [], None, None
    for part in content:
        if part["type"] == "image_url":
            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (512, 512):
                    raise ValueError("Benchmark requires explicit listener crop to 512x512")
                pictures.append(image.convert("RGB").copy())
        elif part["type"] == "input_audio":
            raw = base64.b64decode(part["input_audio"]["data"])
            audio, rate = sf.read(io.BytesIO(raw), dtype="float32")
    # All audio backends receive the same full waveform. No VAD or normalization.
    if audio is not None:
        if rate != 16000:
            factor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
        metadata["audio"]["encoder_samples"] = len(audio)
        metadata["audio"]["encoder_rate"] = 16000
        metadata["audio"]["resampling"] = "scipy.resample_poly" if rate != 16000 else "none"
    return {"content": content, "images": pictures, "audio": audio, "meta": metadata,
            "duration_s": row["duration_s"]}


def timestamped_images(media):
    result = []
    for image, record in zip(media["images"], media["meta"]["video"]["frames"]):
        result.extend([{"type": "text", "text": f"listener frame at {record['time_s']:.6f} seconds:"},
                       {"type": "image", "image": image}])
    return result


def uniform_video(media, fps):
    """Native Qwen video positions use a uniform grid; record the actual PTS error."""
    records = media["meta"]["video"]["frames"]
    # A valid clip can end between the last decoded frame and the next sampling
    # grid (e.g. 25 FPS, duration 1.28s, last PTS 1.24s, next grid 1.25s).
    # Do not invent a final frame. Record the native-grid tail discrepancy.
    count = math.floor(media["meta"]["video"]["last_frame_s"] * fps + 1e-8) + 1
    if len(records) != count or len(media["images"]) != count:
        raise ValueError("Native video needs one source frame per grid cell; no silent duplication")
    times = [r["time_s"] for r in records]
    offsets = [t - i / fps for i, t in enumerate(times)]
    if any(offset < -1e-6 or offset >= 1 / fps + 1e-6 for offset in offsets):
        raise ValueError("Irregular sampled timeline cannot be represented at configured FPS")
    return np.stack([np.asarray(image) for image in media["images"]]), {
        "representation": "qwen_native_video_audio", "fps": fps,
        "frame_pts_s": times, "grid_offset_s": offsets,
        "max_grid_offset_s": max(offsets), "use_audio_in_video": True,
        "native_grid_end_s": count / fps, "audio_duration_s": media["duration_s"],
        "native_grid_tail_gap_s": max(0., media["duration_s"] - count / fps),
        "native_grid_tail_padding_s": max(0., count / fps - media["duration_s"]),
        "audio_origin": "independent_listener_wav", "audio_padding": False}


def minicpm_native(media, concat_images):
    """Official high-refresh ordering: first frame, 1s audio, remaining-frame montage.

    Uses the official concat_images function, but takes already cropped frames and
    the independently verified WAV instead of extracting the video's embedded AAC.
    """
    records = media["meta"]["video"]["frames"]
    contents, segments = [], []
    waveform = media["audio"]
    for second in range(math.ceil(media["duration_s"])):
        indices = [i for i, record in enumerate(records) if second <= record["time_s"] < second + 1]
        if not indices:
            raise ValueError("No listener frame in native audio segment")
        start, end = second * 16000, min((second + 1) * 16000, len(waveform))
        if end <= start:
            raise ValueError("Native segment has no audio")
        contents.extend([f"listener second {second}; first frame PTS {records[indices[0]]['time_s']:.6f}s:",
                         media["images"][indices[0]], waveform[start:end]])
        stacked_size = None
        if len(indices) > 1:
            stacked = concat_images([media["images"][i] for i in indices[1:]])
            contents.extend(["Remaining frames, chronological montage; PTS seconds: " +
                             ", ".join(f"{records[i]['time_s']:.6f}" for i in indices[1:]), stacked])
            stacked_size = list(stacked.size)
        segments.append({"audio_samples": [start, end], "frame_indices": indices,
                         "stacked_image_size": stacked_size})
    return contents, {"representation": "minicpm_native_high_refresh", "segments": segments,
                      "audio_rate": 16000, "tail_padded": False,
                      "montage_changes_visual_encoding": True}
