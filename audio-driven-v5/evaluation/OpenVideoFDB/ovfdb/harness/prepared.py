"""Bind already-generated listener media into an OpenVideoFDB run directory.

The harness providers stream a live agent. Our listener model writes a finished
wav or mp4 per clip. `bind` turns that map into the `agent.wav` + `run.json`
layout `ovfdb score` already reads. Audio is stored as 16 kHz mono PCM so the
scorer's ASR sees clip time 0 at the first sample.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ovfdb.schema import read_manifest, run_dir_names

PROVIDER = "prepared"


def load_generation_map(path: str | Path) -> dict[tuple[str, str | None], dict]:
    """JSONL rows keyed by (clip_id, bucket). A row without bucket matches every bucket."""
    out: dict[tuple[str, str | None], dict] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = str(row.get("clip_id") or row.get("sample_id") or "").strip()
            if not clip_id:
                raise ValueError(f"generation map row has no clip_id: {row}")
            bucket = row.get("bucket")
            bucket = str(bucket).strip() if bucket else None
            out[(clip_id, bucket)] = row
    return out


def _write_wav(src: Path, dst: Path) -> None:
    from ovfdb.harness.streamer import load_pcm16k
    import numpy as np
    import soundfile as sf

    pcm = load_pcm16k(src)
    samples = np.frombuffer(pcm, dtype=np.int16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), samples, 16000, subtype="PCM_16")


def _audio_from_video(video: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(video),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0 or not dst.exists():
        raise RuntimeError(f"could not read audio from {video}: {p.stderr[-800:]}")


def bind(manifest: str | Path, generation_map: str | Path, out: str | Path, mode: str = "av") -> dict:
    """Write one run directory per manifest row that the map covers. Existing run.json is kept."""
    if mode not in ("av", "a2a"):
        raise ValueError(f"mode must be av or a2a, got {mode!r}")
    samples = read_manifest(manifest)
    names = run_dir_names(samples)
    mapping = load_generation_map(generation_map)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.jsonl").write_text(Path(manifest).read_text(encoding="utf-8"), encoding="utf-8")
    bound = skipped = missing = 0
    missing_ids: list[str] = []
    for sample in samples:
        clip_dir = out_dir / names[(sample.clip_id, sample.bucket.value)]
        if (clip_dir / "run.json").exists():
            skipped += 1
            continue
        row = mapping.get((sample.clip_id, sample.bucket.value)) or mapping.get((sample.clip_id, None))
        if row is None:
            missing += 1
            missing_ids.append(f"{sample.clip_id} {sample.bucket.value}")
            continue
        audio = row.get("audio") or row.get("generated_audio")
        video = row.get("video") or row.get("generated")
        audio_path = Path(audio) if audio else None
        video_path = Path(video) if video else None
        if audio_path is None and video_path is None:
            raise ValueError(f"{sample.clip_id}: generation map row has neither audio nor video")
        if audio_path is not None and not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if video_path is not None and not video_path.exists():
            raise FileNotFoundError(video_path)
        wav_out = clip_dir / "agent.wav"
        if audio_path is not None:
            _write_wav(audio_path, wav_out)
        else:
            _audio_from_video(video_path, wav_out)
        payload = {
            "clip_id": sample.clip_id,
            "provider": PROVIDER,
            "mode": mode,
            "agent_audio_path": str(wav_out.resolve()),
            "agent_video_path": str(video_path.resolve()) if video_path else None,
            "model": str(row.get("model") or "listener"),
            "started_at": "",
            "drift_ms": 0.0,
            "events_path": None,
            "errors": [],
            "meta": {
                "bound": True,
                "source_audio": str(audio_path) if audio_path else None,
                "source_video": str(video_path) if video_path else None,
                "bucket": sample.bucket.value,
            },
        }
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        bound += 1
    return {"out": str(out_dir), "bound": bound, "skipped": skipped, "missing": missing,
            "missing_ids": missing_ids}
