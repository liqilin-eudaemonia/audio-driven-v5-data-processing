#!/usr/bin/env python3
"""Unit tests for the v4 3-tier ASR vocal cue normalization and prompt rendering."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_listen_native_caption_vocal_pilot import (
    INSTRUCTION,
    PHONETIC_INTERJECTION_WHITELIST,
    PROMPT_STYLE,
    PROMPT_VERSION,
    build_vocal_cues,
    normalize_asr_cue_for_english_prompt,
    prompt_vocal_cues,
    render_prompt,
)


class VocalAsrCueTests(unittest.TestCase):
    def test_tier0_pure_ascii_english_preserved(self) -> None:
        en_asr, raw_asr, obs = normalize_asr_cue_for_english_prompt("Yeah, I know’s", None, "spoken response")
        self.assertEqual(en_asr, "Yeah, I know's")
        self.assertEqual(raw_asr, "Yeah, I know’s")
        self.assertEqual(obs, "spoken response")

    def test_tier1_official_english_word_overrides_non_ascii_asr(self) -> None:
        # Lexical hallucination '是' with official word 'Sure.'
        en_asr, raw_asr, obs = normalize_asr_cue_for_english_prompt("是", "Sure.", "spoken response")
        self.assertEqual(en_asr, "Sure.")
        self.assertEqual(raw_asr, "是")
        # Interjection '嗯' with official word 'Mm-mm.'
        en_asr2, raw_asr2, obs2 = normalize_asr_cue_for_english_prompt("嗯", "Mm-mm.", "spoken response")
        self.assertEqual(en_asr2, "Mm-mm.")
        self.assertEqual(raw_asr2, "嗯")
        self.assertEqual(obs2, "affirmative vocalization")

    def test_tier2_closed_phonetic_interjection_whitelist(self) -> None:
        cases = {
            "嗯": ("mhm", "affirmative vocalization"),
            "嘿": ("heh", "laughing syllable"),
            "呵": ("heh", "laughing syllable"),
            "哈": ("ha", "laughing syllable"),
            "哼": ("hm", "nasal vocalization"),
            "哇": ("wow", "short exclamation"),
            "呃": ("uh", "hesitation vocalization"),
            "हम्म": ("mhm", "affirmative vocalization"),
            "うん": ("mhm", "affirmative vocalization"),
            "咳": (None, "cough or throat clearing"),
        }
        for raw, (exp_en, exp_obs) in cases.items():
            en_asr, raw_asr, obs = normalize_asr_cue_for_english_prompt(raw, None, "spoken response")
            self.assertEqual(en_asr, exp_en, raw)
            self.assertEqual(raw_asr, raw, raw)
            self.assertEqual(obs, exp_obs, raw)
            if en_asr is not None:
                self.assertTrue(all(ord(c) < 128 for c in en_asr))

    def test_tier3_lexical_non_english_hallucination_falls_back_to_timing_only(self) -> None:
        for hallucination in ("对", "是", "好", "我", "你", "não", "então", "é", "हाँ"):
            en_asr, raw_asr, obs = normalize_asr_cue_for_english_prompt(hallucination, None, "spoken response")
            self.assertIsNone(en_asr, hallucination)
            self.assertEqual(raw_asr, hallucination)
            self.assertEqual(obs, "audible spoken response or vocalization")

    def test_prompt_render_contains_zero_non_ascii_and_preserves_body(self) -> None:
        cues = build_vocal_cues(
            dur=5.0,
            full_silence=False,
            aligned_events=[
                {"start": 0.5, "end": 0.9, "text": "嘿", "description": "嘿"},
                {"start": 2.0, "end": 2.4, "text": "对", "description": "对"},
            ],
            official_with_text={"words": [], "vad": []},
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["asr_text"], "heh")
        self.assertEqual(cues[0]["raw_asr_text"], "嘿")
        self.assertNotIn("asr_text", cues[1])
        self.assertEqual(cues[1]["raw_asr_text"], "对")
        self.assertEqual(cues[1]["observation"], "audible spoken response or vocalization")

        prompt = render_prompt(5.0, False, [{"id": "nod-0001", "start": 1.0, "end": 1.5}], cues)
        self.assertTrue(all(ord(c) < 128 for c in prompt))
        self.assertIn("quote clearly intelligible spoken responses exactly as heard.", prompt)
        self.assertIn("SUPPLEMENTARY VOCAL RESPONSE INTERVALS:", prompt)
        serialized_vocals = json.loads(prompt.split("SUPPLEMENTARY VOCAL RESPONSE INTERVALS:\n", 1)[1])
        for v in serialized_vocals:
            self.assertNotIn("raw_asr_text", v)


if __name__ == "__main__":
    unittest.main()
