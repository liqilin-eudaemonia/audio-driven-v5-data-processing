"""Typed causal inputs and deterministic GT-to-text conversion."""
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


@dataclass(frozen=True)
class Observation:
    sample_id: str
    speaker_audio: str
    listener_image: str
    speaker_start_s: float
    speaker_end_s: float
    output_duration_s: float

    def __post_init__(self):
        if not (0 <= self.speaker_start_s < self.speaker_end_s <= self.output_duration_s <= 30):
            raise ValueError("Invalid observation bounds (v1 supports <=30 seconds)")
        if not self.sample_id:
            raise ValueError("Missing sample ID")


@dataclass(frozen=True)
class TextPlan:
    reply: str
    delivery: str = ""
    behavior: str = ""
    vocal_events: str = ""

    def serialize(self):
        result = f"Reply: {self.reply or '[no main spoken reply]'}\nDelivery: {self.delivery}\nBehavior: {self.behavior}"
        if self.vocal_events:
            result += f"\nVocal: {self.vocal_events}"
        return result

    def render_prompt(self):
        pieces = ["A continuous shot of the listener shown in the reference image."]
        if self.behavior:
            pieces.append(self.behavior)
        pieces.append(f"The listener says exactly: {json.dumps(self.reply, ensure_ascii=False)}."
                      if self.reply else "The listener has no main spoken reply.")
        if self.delivery:
            pieces.append(f"Vocal delivery: {self.delivery}.")
        if self.vocal_events:
            pieces.append(f"Additional listener vocalizations: {self.vocal_events}.")
        return " ".join(pieces)

    @classmethod
    def parse(cls, text):
        matches = list(re.finditer(r"(?m)^(Reply|Delivery|Behavior|Vocal):[ \t]*", text.strip()))
        if not matches or matches[0].start() != 0:
            raise ValueError("Plan must begin with a named field")
        fields = {}
        for i, match in enumerate(matches):
            name = match.group(1)
            if name in fields:
                raise ValueError(f"Duplicate plan field: {name}")
            fields[name] = text.strip()[match.end():matches[i + 1].start() if i + 1 < len(matches) else None].strip()
        if not {"Reply", "Delivery", "Behavior"}.issubset(fields):
            raise ValueError("Incomplete plan; no silent GT fallback is allowed")
        reply = fields["Reply"]
        return cls("" if reply == "[no main spoken reply]" else reply,
                   fields["Delivery"], fields["Behavior"], fields.get("Vocal", ""))


def gt_text_plan(payload, sample_id):
    plan = payload["avatar_plan"]
    if plan["sample_id"] != sample_id:
        raise ValueError("GT sample ID mismatch")
    response, sem = plan["response"], plan.get("semantics", {})
    reply = response.get("text", "") if response["present"] else ""
    if response["present"] and not reply.strip():
        raise ValueError("Present reply has no transcript")
    human = lambda s: str(s).lower().replace("_", " ")
    delivery = [f"{human(sem[k])} {k.replace('_', ' ')}" for k in ("tone", "primary_emotion") if sem.get(k)]
    delivery += [f"{human(v)} {k}" for k, v in (response.get("prosody") or {}).items() if v]
    vocals = []
    for event in plan.get("vocal_track", []):
        if event.get("state") not in {None, "SILENCE", "REPLY_SPEECH"}:
            # Preserve order/content without replacing GT time with a fixed start rule.
            content = event.get("content", event.get("text", ""))
            vocals.append(human(event.get("subtype", event["state"]))
                          + (f" {json.dumps(content, ensure_ascii=False)}" if content else ""))
    return TextPlan(reply, "; ".join(delivery), plan.get("visual_action_description", ""), "; ".join(vocals))


def read_audio(path, end_s=None, start_s=0):
    with sf.SoundFile(path) as f:
        rate = f.samplerate
        if end_s is not None and round(end_s * rate) > len(f) + 2:
            raise ValueError("Observed end exceeds available audio")
        frames = -1 if end_s is None else round(end_s * rate)
        audio = f.read(frames=frames, dtype="float32", always_2d=True).mean(-1)
    if rate != 16000:
        from scipy.signal import resample_poly
        common = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // common, rate // common).astype(np.float32)
    audio[:round(start_s * 16000)] = 0
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Invalid waveform")
    return audio


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_training_examples(root, count, budget_s=8):
    """Small deterministic engineering subset; no media is truncated to fit."""
    root = Path(root)
    output = []
    for part in (0, 1):
        manifest = root / f"train_gt_qualified_20260920/train_part{part}_manifest.json"
        rows = json.loads(manifest.read_text())
        for row in rows:
            duration = float(row["duration_s"])
            end = float(row["speaker_speech_duration_s"])
            if not (2 <= duration <= budget_s and 0 < end < duration):
                continue
            sid = row["clip_id"]
            gt_path = root / f"omni_gt_train_part{part}_30fps" / sid / "avatar_plan.validated.json"
            if not gt_path.is_file():
                continue
            gt = json.loads(gt_path.read_text())
            if not gt.get("annotation_quality", {}).get("usable_for_ltx_training", False):
                continue
            text = gt_text_plan(gt, sid)
            files = row["files"]
            if not all(Path(files[k]).is_file() for k in ("speaker_audio", "listener_audio", "target_video", "first_frame_jpg")):
                continue
            obs = Observation(sid, files["speaker_audio"], files["first_frame_jpg"], 0, end, budget_s)
            output.append(dict(observation=asdict(obs), target_video=files["target_video"],
                               target_audio=files["listener_audio"], duration_s=duration,
                               text_plan=asdict(text), gt_path=str(gt_path), split="train"))
            if len(output) == count:
                return output
    raise ValueError(f"Only found {len(output)} usable samples")
