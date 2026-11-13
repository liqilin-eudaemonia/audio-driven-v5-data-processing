#!/usr/bin/env python3
"""Prepare and run the vocal-ASR + nod evidence Omni natural-language captioning pilot on 172.29.189.119."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from p2_listen.contracts import atomic_json, digest, sha
from compare_listener_official_six import load_metadata, official

PROMPT_VERSION = "native-av-first-plus-nod-vocal-asr-v4"
PROMPT_STYLE = "first_plus_nod_and_vocal_asr_en_v4"

# Closed whitelist of pure phonetic vocalizations / non-speech sounds in non-English scripts
PHONETIC_INTERJECTION_WHITELIST = {
    # Chinese affirmative / nasal vocalizations
    "嗯": ("mhm", "affirmative vocalization"),
    "唔": ("mhm", "affirmative vocalization"),
    "姆": ("mm", "affirmative vocalization"),
    "哼": ("hm", "nasal vocalization"),
    # Chinese laughing syllables
    "哈": ("ha", "laughing syllable"),
    "呵": ("heh", "laughing syllable"),
    "嘿": ("heh", "laughing syllable"),
    "嘻": ("hee", "laughing syllable"),
    # Chinese short interjections / exclamations / hesitations
    "哎": ("eh", "short interjection"),
    "诶": ("eh", "short interjection"),
    "哦": ("oh", "short interjection"),
    "噢": ("oh", "short interjection"),
    "啊": ("ah", "short interjection"),
    "呀": ("ah", "short interjection"),
    "哇": ("wow", "short exclamation"),
    "呃": ("uh", "hesitation vocalization"),
    "耶": ("yeah", "short affirmative response"),
    # Chinese non-speech mouth / throat sounds
    "咳": (None, "cough or throat clearing"),
    "啧": ("tsk", "clicking / tsk sound"),
    # Cross-lingual nasal hums (Hindi / Japanese)
    "हम्म": ("mhm", "affirmative vocalization"),
    "हमम": ("mhm", "affirmative vocalization"),
    "हम्": ("mhm", "affirmative vocalization"),
    "हम": ("hm", "affirmative vocalization"),
    "うん": ("mhm", "affirmative vocalization"),
}

_PUNCT_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "–": "-",
        "…": "...",
    }
)


def normalize_asr_cue_for_english_prompt(
    asr_text: str | None, official_text: str | None, default_obs: str
) -> tuple[str | None, str | None, str]:
    """Three-tier complete normalization for ASR cues fed to the English Omni prompt:
    Tier 0 (Pure ASCII English): keep as-is.
    Tier 1 (Official English word available): when our ASR is non-ASCII, defer to official_text.
    Tier 2 (Closed phonetic interjection whitelist): map pure interjections (嗯/哈/呵/嘿/哼/哇/हम्म...) to exact English phonetic syllables.
    Tier 3 (Fallback for lexical non-English ASR hallucinations like 对/是/我/não/é): drop the unreliable text (asr_text=None) and keep the verified timing interval as an audible vocalization cue so Qwen3-Omni-30B listens directly to the waveform.
    """
    if not asr_text:
        return None, None, default_obs
    raw_cleaned = asr_text.strip()
    ascii_norm = unicodedata.normalize("NFKC", raw_cleaned).translate(_PUNCT_TRANSLATION).strip()
    if not ascii_norm:
        return None, None, default_obs

    # Tier 0: Pure ASCII English
    if all(ord(c) < 128 for c in ascii_norm):
        return ascii_norm, raw_cleaned, default_obs

    # Strip punctuation/spaces to look up phonetic interjection token
    token_key = re.sub(r"[\s\W_]+", "", ascii_norm, flags=re.UNICODE)

    # Tier 1: Official English word exists -> prefer official English word, while keeping phonetic observation if in whitelist
    if official_text and official_text.strip():
        _, wh_obs = PHONETIC_INTERJECTION_WHITELIST.get(token_key, (None, default_obs))
        return official_text.strip(), raw_cleaned, wh_obs

    # Tier 2: Closed phonetic interjection whitelist
    if token_key in PHONETIC_INTERJECTION_WHITELIST:
        en_phonetic, en_obs = PHONETIC_INTERJECTION_WHITELIST[token_key]
        return en_phonetic, raw_cleaned, en_obs

    # Tier 3: Non-ASCII lexical hallucination (e.g., '对', '是', '好', '我', 'não', 'é') without official word ->
    # Drop text cue and keep timing window so Omni listens directly to the purified audio
    return None, raw_cleaned, "audible spoken response or vocalization"


INSTRUCTION = """Watch and listen to this {duration:.3f}-second clip of a listener. Write one coherent English paragraph describing the listener's observable visual behavior and audible vocal responses across the whole clip.
Describe how actions unfold in order, which expressions or postures persist, which actions happen together, and which actions genuinely repeat. Use natural temporal language such as initially, then, while, and toward the end when useful. Describe a sustained state once rather than turning it into repeated actions. Give a repetition count only when clearly supported.
Describe directly visible movements of the face, eyes, head, hands, and body, including brief movements when visible. Use action names such as nodding only when the motion supports them. Describe audible laughter or other nonverbal vocalizations, and quote clearly intelligible spoken responses exactly as heard. Do not guess unclear words. Distinguish visible mouth movement from an audible vocal response, and mention periods without vocalization when relevant.
Use the supplied video and synchronized listener audio, with the supplementary evidence below. Do not infer thoughts, intentions, feelings, agreement, communicative meaning, or causes; avoid interpretive claims such as indicating, suggesting, or as if. Do not invent actions or sounds to fill a quiet interval. Keep a clear subject and write concrete, connected sentences without redundant descriptions or unrelated appearance details.
Output only the natural-language paragraph. Do not output JSON, a list, individual event intervals, or numeric timestamps.

Supplementary nod intervals: id is an identifier; start/end are pair-relative seconds; observation names the movement. Naturally include supported nodding, without treating intervals as counts or an empty list as proof of absence. Do not describe this metadata.
Supplementary vocal intervals: id is an identifier; start/end are pair-relative seconds within the unmuted official listener audio gate; asr_text/official_text/observation provide candidate spoken words or vocalizations detected in that window. Listen carefully to the supplied audio at each interval and naturally integrate audible spoken responses (quoting intelligible words in English) or nonverbal vocalizations (quoting phonetic syllables such as "heh", "ha", "mhm", "uh-huh", "hmm", or "oh" when provided) in temporal order with concurrent visual actions. Do not mention detector metadata, interval IDs, or numeric timestamps.
listener_audio_is_silent=true means the supplied waveform is entirely zero: do not claim audible speech or vocalizations. Otherwise describe only what you hear."""


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_vocal_cues(dur: float, full_silence: bool, aligned_events: list, official_with_text: dict) -> list[dict]:
    if full_silence:
        return []
    raw_items = []
    used_off_word_ids = set()

    for e in aligned_events:
        if e.get("method") == "excluded":
            continue
        ivs = e.get("intervals") or (
            [e["source_interval"]] if "source_interval" in e else [{"start": e["start"], "end": e["end"]}]
        )
        if not ivs:
            continue
        a = max(0.0, min(dur, float(ivs[0]["start"])))
        b = max(0.0, min(dur, float(ivs[-1]["end"])))
        if b <= a:
            continue
        ba = e.get("boundary_alignment", e)
        off_words = ba.get("official_evidence") or ba.get("overlapping_official_words") or []
        for w in off_words:
            if w.get("id"):
                used_off_word_ids.add(w["id"])
        overlapping_full = [
            w for w in official_with_text.get("words", [])
            if w.get("in_clip") and max(a, float(w["start"])) < min(b, float(w["end"])) - 1e-3
        ]
        for w in overlapping_full:
            if w.get("id"):
                used_off_word_ids.add(w["id"])
        off_txt_list = [w.get("text", "").strip() for w in (off_words or overlapping_full) if w.get("text")]
        dedup_off = []
        for t in off_txt_list:
            if not dedup_off or dedup_off[-1] != t:
                dedup_off.append(t)
        off_txt = " ".join(dedup_off).strip() or None
        default_obs = e.get("description") if not e.get("text") else "spoken response"
        en_asr, raw_asr, obs = normalize_asr_cue_for_english_prompt(e.get("text"), off_txt, default_obs)
        raw_items.append(
            {
                "start": round(a, 3),
                "end": round(b, 3),
                "asr_text": en_asr,
                "raw_asr_text": raw_asr,
                "official_text": off_txt,
                "observation": obs,
            }
        )

    for w in official_with_text.get("words", []):
        if not w.get("in_clip") or w.get("id") in used_off_word_ids:
            continue
        a = max(0.0, min(dur, float(w["start"])))
        b = max(0.0, min(dur, float(w["end"])))
        if b <= a:
            continue
        if any(max(a, x["start"]) < min(b, x["end"]) - 1e-3 and (x.get("official_text") or x.get("asr_text")) for x in raw_items):
            continue
        raw_items.append(
            {
                "start": round(a, 3),
                "end": round(b, 3),
                "asr_text": None,
                "raw_asr_text": None,
                "official_text": w.get("text"),
                "observation": "spoken response",
            }
        )

    for v in official_with_text.get("vad", []):
        if not v.get("in_clip"):
            continue
        a = max(0.0, min(dur, float(v["start"])))
        b = max(0.0, min(dur, float(v["end"])))
        if b <= a:
            continue
        if any(max(a, x["start"]) < min(b, x["end"]) - 1e-3 for x in raw_items):
            continue
        raw_items.append(
            {
                "start": round(a, 3),
                "end": round(b, 3),
                "asr_text": None,
                "raw_asr_text": None,
                "official_text": None,
                "observation": "audible vocalization / voice activity",
            }
        )

    raw_items.sort(key=lambda x: (x["start"], x["end"]))
    cues = []
    for idx, item in enumerate(raw_items, 1):
        cue = {"id": f"vocal-{idx:04d}", "start": item["start"], "end": item["end"]}
        if item["asr_text"]:
            cue["asr_text"] = item["asr_text"]
        if item.get("raw_asr_text") and item["raw_asr_text"] != item["asr_text"]:
            cue["raw_asr_text"] = item["raw_asr_text"]
        if item["official_text"]:
            cue["official_text"] = item["official_text"]
        cue["observation"] = item["observation"]
        cues.append(cue)
    return cues


def prompt_vocal_cues(vocal_cues: list[dict]) -> list[dict]:
    """Strip internal raw_asr_text before serializing into the English prompt."""
    out = []
    for c in vocal_cues:
        row = {"id": c["id"], "start": c["start"], "end": c["end"]}
        if c.get("asr_text"):
            row["asr_text"] = c["asr_text"]
        if c.get("official_text"):
            row["official_text"] = c["official_text"]
        row["observation"] = c["observation"]
        out.append(row)
    return out


def render_prompt(duration: float, full_silence: bool, nod_events: list[dict], vocal_cues: list[dict]) -> str:
    nod_cues = [
        {"id": e.get("id") or e.get("event_id"), "start": round(float(e["start"]), 3), "end": round(float(e["end"]), 3), "observation": "head nodding"}
        for e in nod_events
    ]
    clean_vocals = prompt_vocal_cues(vocal_cues)
    header = INSTRUCTION.format(duration=duration)
    return (
        header
        + "\n\nlistener_audio_is_silent: "
        + json.dumps(bool(full_silence))
        + "\n\nSUPPLEMENTARY NODDING INTERVALS:\n"
        + json.dumps(nod_cues, ensure_ascii=False)
        + "\n\nSUPPLEMENTARY VOCAL RESPONSE INTERVALS:\n"
        + json.dumps(clean_vocals, ensure_ascii=False)
    )


def prepare(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    old_rec_dir = ROOT / "experiments/p2_full_audio_postprocess_crop384_20261026/records"
    pilot50_dir = ROOT / "experiments/listen_native_caption_first_evidence50_20261030"
    pilot50_manifest = read_json(pilot50_dir / "manifest.json")

    samples = []

    # Part A: 10 audible clips from the 50-clip pilot
    for row in pilot50_manifest:
        if row["audio_preflight"]["full_silence"]:
            continue
        sid = row["sample_id"]
        dur = float(row["duration_s"])
        orig_meta = read_json(Path(row["source_reference"]["metadata_file"]["path"]))
        raw_meta, _ = load_metadata(orig_meta["sources"]["listener"])
        off_full = official(raw_meta, "listener", row["source_reference"]["source_start_s"], dur)

        rec_path = old_rec_dir / f"{sid}.json"
        if rec_path.exists():
            orec = read_json(rec_path)
            aligned_events = orec["audio_events"]
        else:
            aligned_events = row["audio_preflight"]["provenance"]["audit"]["audio_event_alignment"]

        vocal_cues = build_vocal_cues(dur, False, aligned_events, off_full)
        nod_events = row["nod_supplement"]["events"]
        prompt_text = render_prompt(dur, False, nod_events, vocal_cues)
        (run_dir / "prompts" / f"{sid}.txt").write_text(prompt_text + "\n", encoding="utf-8")

        prev_res = read_json(pilot50_dir / "results" / f"{sid}.json")
        item = {
            "sample_id": sid,
            "group": "pilot50_audible",
            "duration_s": dur,
            "mode": "pilot50",
            "pilot50_row": row,
            "nod_events": nod_events,
            "vocal_cues": vocal_cues,
            "prompt": prompt_text,
            "prior_caption": prev_res["caption"],
            "prior_source": str(pilot50_dir / "results" / f"{sid}.json"),
        }
        samples.append(item)

    # Part B: 14 large-face J clips where purified audio has clear ASR/official words but old J Omni caption missed vocalization
    db_old = sqlite3.connect(ROOT / "experiments/p2_full_audio_postprocess_crop384_20261026/scope.sqlite")
    old_map = {}
    for sid, obs_s, target_s in db_old.execute("SELECT sid, obs, target FROM samples"):
        old_map[sid] = (json.loads(obs_s), json.loads(target_s))

    db_j = sqlite3.connect(ROOT / "experiments/listen_j_final_quality_selection_20261101/selection.sqlite")
    db_r = sqlite3.connect(ROOT / "experiments/listen_richness_full_j_20261031/tables/sample_stats.sqlite")
    audio_sids = set(r[0] for r in db_r.execute('SELECT DISTINCT sid FROM behaviors WHERE type LIKE "audio:%" AND counted=1'))

    existing_sids = {s["sample_id"] for s in samples}
    j_selected = 0
    for sid, pair, split, dur, fp10, fpath, offset in db_j.execute(
        'SELECT sid, pair, split, duration, face_p10, file, "offset" FROM samples WHERE face_p10>=224 ORDER BY face_p10 DESC'
    ):
        if sid in existing_sids or sid in audio_sids or sid not in old_map:
            continue
        rec_path = old_rec_dir / f"{sid}.json"
        if not rec_path.exists():
            continue
        orec = read_json(rec_path)
        asr_events = [e for e in orec.get("audio_events", []) if e.get("text") or e.get("description")]
        off_words = [w for w in orec["official_listener"].get("words", []) if w.get("in_clip") and w.get("text")]
        if not asr_events and not off_words:
            continue
        with open(ROOT / "experiments/listen_richness_full_j_20261031" / fpath, "rb") as f:
            f.seek(offset)
            stat_rec = json.loads(f.readline())
        if stat_rec["audio"]["waveform_full_silence"]:
            continue

        h2 = hashlib.sha256(sid.encode()).hexdigest()[:2]
        j_cap_path = ROOT / f"experiments/listen_j_omni_20261030/caption/results/{h2}/{sid}.json"
        j_rec_path = ROOT / f"experiments/listen_j_omni_20261030/audio_recipes/results/{h2}/{sid}.json"
        if not j_cap_path.exists() or not j_rec_path.exists():
            continue
        j_cap = read_json(j_cap_path)
        nod_events = j_cap["input_evidence"]["nod_supplement"]["events"]
        vocal_cues = build_vocal_cues(float(dur), False, orec["audio_events"], orec["official_listener"])
        if not vocal_cues:
            continue

        prompt_text = render_prompt(float(dur), False, nod_events, vocal_cues)
        (run_dir / "prompts" / f"{sid}.txt").write_text(prompt_text + "\n", encoding="utf-8")

        obs_obj, target_obj = old_map[sid]
        item = {
            "sample_id": sid,
            "group": "j_face224_missed_vocal",
            "face_p10": float(fp10),
            "duration_s": float(dur),
            "mode": "j_omni",
            "j_caption_file": str(j_cap_path),
            "j_recipe_file": str(j_rec_path),
            "j_source": j_cap.get("effective_source") or j_cap["source"],
            "preview_media": {
                "video": target_obj["target_video"],
                "listener_wav": target_obj["target_audio"],
                "speaker_wav": obs_obj["speaker_audio"],
            },
            "nod_events": nod_events,
            "vocal_cues": vocal_cues,
            "prompt": prompt_text,
            "prior_caption": j_cap["caption"],
            "prior_source": str(j_cap_path),
        }
        samples.append(item)
        j_selected += 1
        if j_selected >= 14:
            break

    atomic_json(run_dir / "manifest.json", samples, immutable=False)
    protocol = {
        "schema": "listen-native-caption-nod-vocal-asr-pilot-v2",
        "prompt_version": PROMPT_VERSION,
        "prompt_style": PROMPT_STYLE,
        "instruction": INSTRUCTION,
        "single_han_gloss_map": PHONETIC_INTERJECTION_WHITELIST,
        "model": "/private/ckpts/Qwen3-Omni/Instruct",
        "total_samples": len(samples),
        "pilot50_audible_samples": sum(s["group"] == "pilot50_audible" for s in samples),
        "j_face224_missed_vocal_samples": sum(s["group"] == "j_face224_missed_vocal" for s in samples),
        "decoding_attempts": [
            {"temperature": 0, "max_tokens": 4096, "repetition_penalty": 1.05},
            {"temperature": 0, "max_tokens": 4096, "repetition_penalty": 1.12},
        ],
    }
    atomic_json(run_dir / "protocol.json", protocol, immutable=False)
    (run_dir / "prompt.txt").write_text(INSTRUCTION + "\n", encoding="utf-8")
    print(f"Prepared {len(samples)} samples in {run_dir}")


def run_inference(run_dir: Path, only_changed_prompts: bool = True) -> None:
    import numpy as np
    from scipy.signal import resample_poly
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from listen_benchmark.media import prepare as prepare_media50, uniform_video
    from listen_benchmark.full_raw import prepare as prepare_media_j
    from listen_audio_postprocess.recipes import read_listener_audio
    from listen_benchmark.j_omni import read_audio as read_j_audio
    from listen_benchmark.native_caption_quality import generation_failures

    samples = read_json(run_dir / "manifest.json")
    protocol = read_json(run_dir / "protocol.json")
    model_path = protocol["model"]

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM", "0.28")),
        max_model_len=16384,
        max_num_seqs=1,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=False,
        limit_mm_per_prompt={"image": 128, "audio": 1, "video": 1},
        skip_mm_profiling=True,
        enforce_eager=True,
    )

    media50_cfg = {"fps": 8, "max_frames": 128, "max_edge": 768, "max_audio_seconds": 16}

    for idx, item in enumerate(samples, 1):
        sid = item["sample_id"]
        dest = run_dir / "results" / f"{sid}.json"
        if dest.exists() and only_changed_prompts:
            existing = read_json(dest)
            if (
                existing.get("prompt_version") == PROMPT_VERSION
                and existing.get("prompt") == item["prompt"]
                and existing.get("vocal_cues") == item["vocal_cues"]
                and not any(ord(c) > 127 for c in existing.get("caption", ""))
            ):
                print(f"[{idx}/{len(samples)}] Skipping up-to-date v3 result for {sid}", flush=True)
                continue

        t0 = time.monotonic()
        if item["mode"] == "pilot50":
            row = item["pilot50_row"]
            recipe = read_json(Path(row["listener_audio_recipe"]))
            media = prepare_media50(row, "visual", media50_cfg)
            wave, sr, audio_identity = read_listener_audio(recipe)
        else:
            j_src = item["j_source"]
            recipe = read_json(Path(item["j_recipe_file"]))
            media = prepare_media_j(j_src, "visual", fast_crop=True, roi_crop=True)
            wave, sr, audio_identity = read_j_audio(recipe)

        audio = wave[:, 0]
        if sr != 16000:
            g = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

        frames, evidence = uniform_video(media, 8)
        second_per_grid_ts = [float(processor.image_processor.temporal_patch_size) / 8.0]
        prompt_text = item["prompt"]
        rendered = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video", "video": "listener://decoded"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        tag = f"vocal-pilot-v2-{sid}-{time.time_ns()}"
        req = {
            "prompt": rendered,
            "multi_modal_data": {"video": [frames], "audio": [audio]},
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
                "fps": 8,
                "do_sample_frames": False,
                "second_per_grid_ts": second_per_grid_ts,
            },
            "multi_modal_uuids": {"audio": [tag], "video": [tag]},
        }

        attempts = []
        for att_idx, dec in enumerate(protocol["decoding_attempts"], 1):
            params = SamplingParams(
                **dec,
                structured_outputs=StructuredOutputsParams(regex=r"[A-Za-z][ -~]*"),
            )
            llm.reset_mm_cache()
            att_start = time.monotonic()
            out = llm.generate([req], params, use_tqdm=False)[0]
            gen = out.outputs[0]
            raw_text = gen.text
            flags = generation_failures(raw_text, gen.finish_reason)
            attempts.append(
                {
                    "attempt": att_idx,
                    "caption": raw_text,
                    "finish_reason": gen.finish_reason,
                    "completion_tokens": len(gen.token_ids),
                    "seconds": round(time.monotonic() - att_start, 3),
                    "decoding": dec,
                    "failures": list(flags),
                }
            )
            if not flags:
                break

        result = {
            "sample_id": sid,
            "prompt_version": PROMPT_VERSION,
            "group": item["group"],
            "duration_s": item["duration_s"],
            "nod_events": item["nod_events"],
            "vocal_cues": item["vocal_cues"],
            "prompt": prompt_text,
            "rendered_prompt": rendered,
            "prior_caption": item["prior_caption"],
            "caption": raw_text,
            "finish_reason": gen.finish_reason,
            "completion_tokens": len(gen.token_ids),
            "seconds": round(time.monotonic() - t0, 3),
            "attempts": attempts,
            "audio_identity": audio_identity,
        }
        atomic_json(dest, result, immutable=False)
        print(
            f"[{idx}/{len(samples)}] {sid} ({item['group']}) -> {len(gen.token_ids)} tokens in {result['seconds']:.2f}s:\n"
            f"  Vocal cues: {json.dumps(item['vocal_cues'], ensure_ascii=False)}\n"
            f"  OLD: {item['prior_caption']}\n"
            f"  NEW: {raw_text}\n",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "run"])
    p.add_argument(
        "--run",
        type=Path,
        default=ROOT / "experiments/listen_native_caption_nod_vocal24_20261103",
    )
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.run.resolve())
    else:
        run_inference(args.run.resolve())


if __name__ == "__main__":
    main()
