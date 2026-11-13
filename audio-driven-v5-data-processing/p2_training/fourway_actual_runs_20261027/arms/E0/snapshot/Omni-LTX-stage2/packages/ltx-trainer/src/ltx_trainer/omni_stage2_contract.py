"""Pinned Route-A v3 preprocessing, independent of shared extraction scripts.

Source: Qwen3-Omni/audio_caption/extract_causal_plan_features.py
Snapshot SHA256: 3495ae225eb8371c7cf11d2ea29b94c84f3058aabf1fbdc1e14375a732e09320
Only pure contract/compiler helpers are copied; no workers or model execution.
"""

from __future__ import annotations

import math
import re
import wave
from pathlib import Path
from typing import Any

import torch

PLAN_FIELD_NAMES = ("semantics", "response", "events")


def prepare_causal_speaker_audio(
    source_wav: str,
    cache_wav: Path,
    speaker_end_ms: int,
    tail_ms: int,
) -> str:
    """将 Speaker stem 规范为说话结束后固定尾音长度，不足部分补静音。"""
    cache_wav.parent.mkdir(parents=True, exist_ok=True)

    tmp_wav = cache_wav.with_suffix(".tmp.wav")
    with wave.open(source_wav, "rb") as src:
        params = src.getparams()
        keep_frames = math.ceil((speaker_end_ms + tail_ms) / 1000.0 * params.framerate)
        available_frames = min(params.nframes, keep_frames)
        frames = src.readframes(available_frames)
        if available_frames < keep_frames:
            bytes_per_frame = params.sampwidth * params.nchannels
            frames += b"\x00" * ((keep_frames - available_frames) * bytes_per_frame)

    with wave.open(str(tmp_wav), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    tmp_wav.replace(cache_wav)
    return str(cache_wav)


class PlanCompiler:
    """负责将 Avatar Plan JSON 编译为硬时序控制张量与自然语言 Prompt。"""

    FLOOR_ENUM_MAP = {"LISTEN": 0, "TAKE_TURN": 1, "REPLY": 2, "HOLD": 3}
    VISUAL_EVENT_MAP = {
        "NOD": 0,
        "SMILE_BURST": 1,
        "INHALE_PREP": 2,
        "BLINK_BURST": 3,
        "WAVE_HAND": 4,
        "HANDS_CLASPED": 5,
        "HAND_GESTURE": 6,
        "OPEN_PALMS": 7,
        "TOUCH_FACE": 8,
        "SHRUG": 9,
    }

    @classmethod
    def parse_hard_controls(cls, avatar_plan: dict[str, Any], duration_ms: int, fps: float = 25.0) -> torch.Tensor:
        duration_s = max(0.1, duration_ms / 1000.0)
        num_frames = math.ceil(duration_s * fps)
        # 16通道: [0..3: Floor, 4: Audio Gate, 5: Lip Gate, 6..15: 10类关键动作事件]
        controls = torch.zeros((num_frames, 16), dtype=torch.float32)

        # 默认起始填充 LISTEN
        controls[:, 0] = 1.0

        # 1. Floor Track
        floor_track = avatar_plan.get("floor_track", [])
        if isinstance(floor_track, list):
            for seg in floor_track:
                st_ms = seg.get("start", 0)
                ed_ms = seg.get("end", duration_ms)
                state = seg.get("state", "LISTEN")
                f_st = max(0, min(num_frames - 1, int(math.floor(st_ms / 1000.0 * fps))))
                f_ed = max(f_st + 1, min(num_frames, int(math.ceil(ed_ms / 1000.0 * fps))))
                idx = cls.FLOOR_ENUM_MAP.get(state, 0)
                controls[f_st:f_ed, 0:4] = 0.0
                controls[f_st:f_ed, idx] = 1.0

        # 2. Vocal Track (Audio Gate & Lip Gate)
        vocal_track = avatar_plan.get("vocal_track", [])
        if isinstance(vocal_track, list):
            for seg in vocal_track:
                st_ms = seg.get("start", 0)
                ed_ms = seg.get("end", duration_ms)
                state = seg.get("state", "SILENCE")
                articulation = seg.get("articulation", "NONE")
                f_st = max(0, min(num_frames - 1, int(math.floor(st_ms / 1000.0 * fps))))
                f_ed = max(f_st + 1, min(num_frames, int(math.ceil(ed_ms / 1000.0 * fps))))

                if state in ("BACKCHANNEL", "PARALINGUISTIC", "REPLY_SPEECH"):
                    controls[f_st:f_ed, 4] = 1.0  # Audio Gate 开
                if state != "SILENCE" and articulation != "NONE":
                    controls[f_st:f_ed, 5] = 1.0  # Lip Gate 开

        # 3. Visual Events (包含微动作与手势/身体动作)
        events = avatar_plan.get("visual_events", [])
        if isinstance(events, list):
            for ev in events:
                st_ms = ev.get("start", 0)
                ed_ms = ev.get("end", duration_ms)
                ev_type = ev.get("type", "")
                if ev_type in cls.VISUAL_EVENT_MAP:
                    ch_idx = 6 + cls.VISUAL_EVENT_MAP[ev_type]
                    f_st = max(0, min(num_frames - 1, int(math.floor(st_ms / 1000.0 * fps))))
                    f_ed = max(f_st + 1, min(num_frames, int(math.ceil(ed_ms / 1000.0 * fps))))
                    controls[f_st:f_ed, ch_idx] = 1.0

        return controls

    @classmethod
    def render_ltx_prompts(cls, avatar_plan: dict[str, Any]) -> tuple[str, str]:
        """将结构化 Plan 渲染为 LTX Tower 使用的半结构化时序语言。"""
        if not isinstance(avatar_plan, dict):
            avatar_plan = {}

        sem = avatar_plan.get("semantics") if isinstance(avatar_plan.get("semantics"), dict) else {}
        resp = avatar_plan.get("response") if isinstance(avatar_plan.get("response"), dict) else {}
        prosody = resp.get("prosody") if isinstance(resp.get("prosody"), dict) else {}

        act = sem.get("dialogue_act") or "ACKNOWLEDGE"
        tone = sem.get("tone") or "GENTLE"
        emotion = sem.get("primary_emotion") or "NEUTRAL"
        has_reply = bool(resp.get("present", False))
        reply_text = str(resp.get("text") or "").strip()
        pace = prosody.get("pace") or "MEDIUM"
        energy = prosody.get("energy") or "LOW"
        pitch = prosody.get("pitch") or "MEDIUM"
        volume = prosody.get("volume") or "SOFT"

        def safe_int(val: Any, default: int = 0) -> int:
            try:
                if val is None:
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default

        timebase = avatar_plan.get("timebase") if isinstance(avatar_plan.get("timebase"), dict) else {}
        timing_anchors = (
            avatar_plan.get("timing_anchors") if isinstance(avatar_plan.get("timing_anchors"), dict) else {}
        )

        duration_ms = safe_int(timebase.get("duration"), 0)
        speaker_end_ms = safe_int(timing_anchors.get("speaker_end_ms"), 0)
        floor_track = avatar_plan.get("floor_track") if isinstance(avatar_plan.get("floor_track"), list) else []
        vocal_track = avatar_plan.get("vocal_track") if isinstance(avatar_plan.get("vocal_track"), list) else []
        visual_track = (
            avatar_plan.get("visual_state_track") if isinstance(avatar_plan.get("visual_state_track"), list) else []
        )
        events = avatar_plan.get("visual_events") if isinstance(avatar_plan.get("visual_events"), list) else []

        reply_segments = [
            seg
            for seg in vocal_track
            if isinstance(seg, dict)
            and seg.get("state") == "REPLY_SPEECH"
            and safe_int(seg.get("end")) > safe_int(seg.get("start"))
        ]
        if not reply_segments:
            reply_segments = [
                seg
                for seg in floor_track
                if isinstance(seg, dict)
                and seg.get("state") == "REPLY"
                and safe_int(seg.get("end")) > safe_int(seg.get("start"))
            ]
        reply_start_ms = min(
            (safe_int(seg.get("start"), speaker_end_ms) for seg in reply_segments),
            default=speaker_end_ms,
        )
        reply_end_ms = max(
            (safe_int(seg.get("end"), reply_start_ms) for seg in reply_segments),
            default=reply_start_ms,
        )
        if duration_ms <= 0:
            all_ends = [
                safe_int(seg.get("end"), 0)
                for seg in [*floor_track, *vocal_track, *visual_track, *events]
                if isinstance(seg, dict)
            ]
            duration_ms = max(all_ends, default=max(speaker_end_ms, reply_end_ms, 1))

        def clean(value: Any, default: str = "") -> str:
            text = str(value if value is not None else default).strip().lower().replace("_", " ")
            return " ".join(text.split())

        def overlaps(seg: Any, start_ms: int, end_ms: int) -> bool:
            if not isinstance(seg, dict):
                return False
            st = safe_int(seg.get("start"), 0)
            ed = safe_int(seg.get("end"), 0)
            return ed > start_ms and st < end_ms

        def ordered_unique(values: list[str]) -> list[str]:
            return list(dict.fromkeys(v for v in values if v))

        def join_phrases(values: list[str]) -> str:
            values = ordered_unique(values)
            if not values:
                return ""
            if len(values) == 1:
                return values[0]
            return ", ".join(values[:-1]) + f", and {values[-1]}"

        vocal_phrase_map = {
            "BACKCHANNEL": "a brief backchannel",
            "PARALINGUISTIC": "a subtle paralinguistic vocalization",
            "REPLY_SPEECH": "the spoken reply",
        }
        subtype_phrase_map = {
            "INHALE": "a short audible inhale",
            "EXHALE": "a soft exhale",
            "LAUGH": "a brief natural laugh",
            "CHUCKLE": "a quiet chuckle",
            "SIGH": "a soft sigh",
            "HMM": "a soft “hmm”",
            "MM_HMM": "a soft “mm-hmm”",
        }

        def describe_vocal_interval(
            start_ms: int,
            end_ms: int,
            silence_fallback: str,
            lead_verb: str = "The listener produces",
        ) -> str:
            if end_ms <= start_ms:
                return silence_fallback
            phrases: list[str] = []
            for seg in vocal_track:
                if not overlaps(seg, start_ms, end_ms) or seg.get("state") == "SILENCE":
                    continue
                subtype = str(seg.get("subtype", "")).upper()
                phrase = (
                    subtype_phrase_map.get(subtype)
                    or vocal_phrase_map.get(str(seg.get("state", "")).upper())
                    or f"a {clean(seg.get('state'))} sound"
                )
                phrases.append(phrase)
            rendered = join_phrases(phrases)
            return f"{lead_verb} {rendered}" if rendered else silence_fallback

        listen_audio = describe_vocal_interval(
            0,
            speaker_end_ms,
            "The listener remains silent while the speaker is talking",
        )
        transition_audio = describe_vocal_interval(
            speaker_end_ms,
            reply_start_ms,
            "A short natural pause separates listening from the response",
            lead_verb="The transition features",
        )
        end_audio = describe_vocal_interval(
            reply_end_ms,
            duration_ms,
            "The final word is followed by clean listener-only silence",
            lead_verb="The response concludes with",
        )

        if has_reply and reply_text:
            spoken_text = reply_text if reply_text[-1] in ".?!" else reply_text + "."
            reply_audio = (
                f"The listener delivers a response whose dialogue act is {clean(act)}, tone is {clean(tone)}, "
                f"and affect is {clean(emotion)}, speaking at a {clean(pace)} pace with {clean(energy)} energy, "
                f"{clean(pitch)} pitch, and {clean(volume)} volume, saying “{spoken_text}”"
            )
        else:
            reply_audio = "The listener does not produce a spoken reply"

        def dot(s: str) -> str:
            text = s.rstrip()
            if text.endswith((".", "?", "!")):
                return text
            if len(text) >= 2 and text[-1] in ("”", '"') and text[-2] in ".?!":
                return text
            return text + "."

        audio_prompt = (
            f"LISTEN PHASE: {dot(listen_audio)} "
            f"TURN TRANSITION: {dot(transition_audio)} "
            f"REPLY PHASE: {dot(reply_audio)} "
            f"END PHASE: {dot(end_audio)} "
            f"AUDIO STYLE: Clean isolated listener audio with natural breathing, clear articulation, "
            f"and no speaker-voice leakage."
        )

        expression_map = {
            "ATTENTIVE": "maintains an attentive expression",
            "NEUTRAL": "maintains a neutral expression",
            "HAPPY": "shows a warm happy expression",
            "AMUSED": "shows an amused expression",
            "THOUGHTFUL": "shows a thoughtful expression",
            "CURIOUS": "shows a curious expression",
            "EMPATHETIC": "shows an empathetic expression",
            "CONFUSED": "shows a visibly uncertain expression",
            "SURPRISED": "shows a visibly surprised expression",
            "ENGAGED": "shows an engaged expression",
            "SMILE": "shows a warm smile",
            "DISCOMFORT": "shows a visibly uncomfortable expression",
            "CONFIDENT": "shows a confident expression",
            "DISAPPROVING": "shows a disapproving expression",
        }
        gaze_map = {
            "TOWARD_SPEAKER": "looks toward the speaker",
            "LOOK_AT_SPEAKER": "looks toward the speaker",
            "FORWARD": "looks forward",
            "DIRECT": "looks forward",
            "LOOK_DOWN": "looks downward",
            "DOWN": "looks downward",
            "LOOK_UP": "looks upward",
            "UP": "looks upward",
            "LOOK_LEFT": "looks to the left",
            "LEFT": "looks to the left",
            "LOOK_RIGHT": "looks to the right",
            "RIGHT": "looks to the right",
            "GAZE_AWAY": "briefly looks away",
            "AWAY": "briefly looks away",
        }
        head_map = {
            "NOD": "nods naturally",
            "SHAKE_HEAD": "shakes the head gently",
            "TILT_HEAD": "tilts the head slightly",
            "STILL": "keeps the head mostly still",
            "STABLE": "keeps the head stable",
            "NATURAL": "uses subtle natural head motion",
            "SPEAKING_NATURAL": "uses natural head motion while speaking",
        }
        posture_map = {
            "UPRIGHT": "keeps an upright posture",
            "RELAXED": "maintains a relaxed posture",
            "LEAN_FORWARD": "leans slightly forward",
            "LEANING_FORWARD": "leans slightly forward",
            "LEAN_BACK": "leans slightly back",
            "SHIFTING": "makes a natural posture shift",
        }
        hand_map = {
            "RESTING": "keeps the hands resting",
            "HANDS_CLASPED": "keeps both hands clasped",
            "CLASPED": "keeps both hands clasped",
            "OPEN_PALMS": "gestures with open palms",
            "HAND_GESTURE": "uses conversational hand gestures",
            "GESTURING": "uses conversational hand gestures",
            "POINTING": "points to emphasize the response",
            "TOUCH_FACE": "briefly touches the face or chin",
            "WAVE_HAND": "waves one hand",
            "WAVING": "waves one hand",
            "HANDS_ON_HIPS": "rests both hands on the hips",
            "ARMS_FOLDED": "keeps both arms folded",
        }
        event_phrase_map = {
            "NOD": "nods gently",
            "SHAKE_HEAD": "shakes the head",
            "TILT_HEAD": "tilts the head",
            "BLINK_BURST": "blinks naturally",
            "SMILE_BURST": "shows a brief warm smile",
            "BROW_RAISE": "raises the eyebrows",
            "BROW_FURROW": "briefly furrows the brow",
            "INHALE_PREP": "takes a visible preparatory breath",
            "WAVE_HAND": "waves one hand",
            "HANDS_CLASPED": "clasps both hands",
            "HAND_GESTURE": "uses conversational hand gestures",
            "OPEN_PALMS": "gestures with open palms",
            "TOUCH_FACE": "briefly touches the face or chin",
            "SHRUG": "makes a subtle shrug",
            "POINTING": "points to emphasize the response",
            "LEAN_FORWARD": "leans slightly forward",
            "LEAN_BACK": "leans slightly back",
            "POSTURE_SHIFT": "shifts posture naturally",
        }

        def describe_visual_interval(start_ms: int, end_ms: int, fallback: str) -> str:
            if end_ms <= start_ms:
                return fallback
            active_states = [seg for seg in visual_track if isinstance(seg, dict) and overlaps(seg, start_ms, end_ms)]
            phrases: list[str] = []
            for field, mapping, unknown_template in (
                ("expression", expression_map, "shows a {} expression"),
                ("gaze", gaze_map, "directs the gaze {}"),
                ("head_motion", head_map, "uses {} head motion"),
                ("body_posture", posture_map, "maintains a {} posture"),
                ("hand_action", hand_map, "uses {} hand movement"),
            ):
                values = ordered_unique(str(seg.get(field, "")).upper() for seg in active_states)
                for value in values[:2]:
                    if value and value not in ("NONE", "UNKNOWN", "NULL"):
                        phrases.append(mapping.get(value, unknown_template.format(clean(value))))

            event_counts: dict[str, int] = {}
            for ev in events:
                if isinstance(ev, dict) and overlaps(ev, start_ms, end_ms):
                    ev_type = str(ev.get("type", "")).upper()
                    if ev_type:
                        event_counts[ev_type] = event_counts.get(ev_type, 0) + 1
            for ev_type, count in event_counts.items():
                phrase = event_phrase_map.get(ev_type)
                if not phrase:
                    continue
                # 语义去重：避免 "keeps both hands clasped" 与 "clasps both hands" 冗余重叠
                norm_phrase = phrase.replace("both hands", "").replace("the head", "").replace("one hand", "").strip()
                if count == 1 and any(norm_phrase in existing for existing in phrases):
                    continue
                phrases.append(f"{phrase} several times" if count > 1 else phrase)

            rendered = join_phrases(phrases)
            return f"The listener {rendered}" if rendered else fallback

        listen_visual = describe_visual_interval(
            0,
            speaker_end_ms,
            "The listener watches attentively with restrained natural movement",
        )
        transition_visual = describe_visual_interval(
            speaker_end_ms,
            reply_start_ms,
            "The listener pauses naturally and prepares to take the turn",
        )
        reply_visual = describe_visual_interval(
            reply_start_ms,
            reply_end_ms,
            "The listener delivers the response with natural facial and upper-body motion",
        )
        end_visual = describe_visual_interval(
            reply_end_ms,
            duration_ms,
            "The listener settles back into a relaxed attentive posture",
        )

        vis_desc = str(avatar_plan.get("visual_action_description") or "").strip()
        action_sequence = (
            f"OBSERVED ACTION SEQUENCE: {vis_desc.rstrip(' .')}."
            if vis_desc
            else "OBSERVED ACTION SEQUENCE: The visible motion follows the ordered phase descriptions."
        )
        video_prompt = (
            "SHOT: A static portrait shot frames one listener and preserves the initial identity and composition. "
            f"LISTEN PHASE: {dot(listen_visual)} "
            f"TURN TRANSITION: {dot(transition_visual)} "
            f"REPLY PHASE: {dot(reply_visual)} "
            f"END PHASE: {dot(end_visual)} "
            f"{action_sequence} "
            "MOTION STYLE: Natural, smooth, understated conversational motion with stable identity, "
            "realistic facial behavior, and lip movement only during listener vocalization."
        )

        return audio_prompt, video_prompt


def _field_char_spans(raw_text: str) -> dict[str, tuple[int, int]]:
    """Locate semantic/reply/timing fields in the serialized Plan without ratios."""
    patterns = {
        "semantics": r'"semantics"\s*:',
        "response": r'"response"\s*:',
        "events": r'"vocal_track"\s*:',
        "_events_end": r'"turn_constraints"\s*:',
    }
    positions: dict[str, int] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, raw_text)
        if match is not None:
            positions[name] = match.start()
    required = {"semantics", "response", "events"}
    if not required.issubset(positions):
        raise ValueError(f"Avatar Plan is missing field anchors: {sorted(required - positions.keys())}")
    events_end = positions.get("_events_end", len(raw_text))
    spans = {
        "semantics": (positions["semantics"], positions["response"]),
        "response": (positions["response"], positions["events"]),
        "events": (positions["events"], events_end),
    }
    if any(end <= start for start, end in spans.values()):
        raise ValueError("Avatar Plan fields are not in the canonical order")
    return spans


def build_plan_field_mask(
    raw_text: str,
    tokenizer: Any,
    num_tokens: int,
) -> torch.Tensor:
    """Build exact boolean masks consumed by the trainable field-aware Resampler."""
    char_spans = _field_char_spans(raw_text)
    try:
        encoded = tokenizer(
            raw_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
    except (KeyError, TypeError, ValueError, NotImplementedError):
        offsets = None

    mask = torch.zeros((len(PLAN_FIELD_NAMES), num_tokens), dtype=torch.bool)
    if offsets is not None:
        for field_index, name in enumerate(PLAN_FIELD_NAMES):
            char_start, char_end = char_spans[name]
            for token_index, (token_start, token_end) in enumerate(offsets[:num_tokens]):
                if token_end > char_start and token_start < char_end:
                    mask[field_index, token_index] = True
    else:
        for field_index, name in enumerate(PLAN_FIELD_NAMES):
            char_start, char_end = char_spans[name]
            token_start = len(tokenizer(raw_text[:char_start], add_special_tokens=False)["input_ids"])
            token_end = len(tokenizer(raw_text[:char_end], add_special_tokens=False)["input_ids"])
            mask[field_index, token_start : min(token_end, num_tokens)] = True
    if not bool(mask.any(dim=1).all()):
        raise ValueError("At least one Avatar Plan field produced an empty token mask")
    return mask


CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT = """You are an expert conversational AI and avatar planner.
Given the provided causal inputs, your goal is to generate a comprehensive, structured Avatar Plan in valid JSON format.
Strict rules:
1. Only observe the causal input (speaker audio and listener initial frame).
2. Never invent ungrounded future knowledge.
3. Output strictly valid JSON."""


def derive_causal_timing(sample: dict[str, Any]) -> tuple[int, int]:
    """Return target duration and observable Speaker end; never infer from Listener data."""
    sid = sample.get("clip_id") or sample.get("sample_id") or "<unknown>"
    try:
        duration_s = float(sample["turn_duration_s"])
        speaker_duration_s = float(sample["speaker_speech_duration_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{sid}: turn_duration_s and speaker_speech_duration_s must both be explicit numbers") from exc

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"{sid}: invalid turn_duration_s={duration_s!r}")
    if not math.isfinite(speaker_duration_s) or speaker_duration_s <= 0 or speaker_duration_s > duration_s:
        raise ValueError(
            f"{sid}: invalid speaker_speech_duration_s={speaker_duration_s!r} for turn_duration_s={duration_s!r}"
        )

    duration_ms = int(round(duration_s * 1000))
    speaker_end_ms = int(round(speaker_duration_s * 1000))
    if not 0 < speaker_end_ms <= duration_ms:
        raise ValueError(f"{sid}: rounded speaker_end_ms={speaker_end_ms} is outside target duration_ms={duration_ms}")
    return duration_ms, speaker_end_ms
