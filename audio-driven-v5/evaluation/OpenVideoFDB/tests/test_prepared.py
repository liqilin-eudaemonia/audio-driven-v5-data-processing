"""Offline listener media can be placed into a scoreable run directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ovfdb.harness.prepared import bind
from ovfdb.schema import Bucket, DynamicType, Sample, write_manifest


def _sample(clip_id: str, bucket: Bucket, audio: Path) -> Sample:
    return Sample(
        clip_id=clip_id,
        conversation_id=clip_id,
        bucket=bucket,
        dynamic_type=DynamicType.PAUSE_HANDLING,
        dynamic_start_s=1.0,
        dynamic_end_s=2.0,
        duration_s=3.0,
        user_audio_path=str(audio),
        user_video_path=str(audio),
        source="nvidia",
    )


def test_bind_writes_16k_wav_and_keeps_video_path(tmp_path: Path):
    user = tmp_path / "user.wav"
    sf.write(user, np.zeros(1600, dtype=np.int16), 16000)
    generated = tmp_path / "listener.wav"
    sf.write(generated, np.ones(3200, dtype=np.int16), 8000)
    video = tmp_path / "listener.mp4"
    video.write_bytes(b"not-a-real-video")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([
        _sample("clip-a", Bucket.PERCEPTION, user),
        _sample("clip-b", Bucket.PERCEPTION, user),
    ], manifest)
    gen = tmp_path / "map.jsonl"
    gen.write_text(json.dumps({
        "clip_id": "clip-a", "audio": str(generated), "video": str(video), "model": "unit",
    }) + "\n", encoding="utf-8")
    out = tmp_path / "runs"
    summary = bind(manifest, gen, out)
    assert summary["bound"] == 1
    assert summary["missing"] == 1
    run = json.loads((out / "clip-a" / "run.json").read_text(encoding="utf-8"))
    assert run["provider"] == "prepared"
    assert run["model"] == "unit"
    assert run["agent_video_path"] == str(video.resolve())
    audio, sr = sf.read(run["agent_audio_path"])
    assert sr == 16000
    assert len(audio) == 6400
    again = bind(manifest, gen, out)
    assert again["skipped"] == 1
    assert again["bound"] == 0
