import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from run_glare_reaction_omni_pilot import (
    build_prompt,
    check_prompt,
    compact_vocal_cues,
    cue_gaps,
    retry_note,
)


class TestJOmniReactionVocal(unittest.TestCase):
    def test_compact_vocal_cues_merges_overlaps_and_drops_pure_vad(self):
        raw = [
            {'id': 'vocal-0001', 'start': 2.92, 'end': 3.16, 'asr_text': 'Yeah', 'official_text': 'Yeah.', 'observation': 'spoken response'},
            {'id': 'vocal-0002', 'start': 3.00, 'end': 3.40, 'official_text': 'Yeah.', 'observation': 'Affirmative backchannel vocalization'},
            {'id': 'vocal-0003', 'start': 6.16, 'end': 6.83, 'observation': 'audible vocalization / voice activity'},
            {'id': 'vocal-0004', 'start': 8.13, 'end': 8.46, 'asr_text': 'Yeah', 'official_text': 'Yeah.', 'observation': 'spoken response'},
            {'id': 'vocal-0005', 'start': 8.52, 'end': 8.68, 'official_text': 'Yeah.', 'observation': 'spoken response'},
        ]
        compacted = compact_vocal_cues(raw)
        self.assertEqual(len(compacted), 2)
        self.assertAlmostEqual(compacted[0]['start'], 2.92)
        self.assertAlmostEqual(compacted[0]['end'], 3.40)
        self.assertAlmostEqual(compacted[1]['start'], 8.13)
        self.assertAlmostEqual(compacted[1]['end'], 8.68)

    def test_prompt_omits_absent_reaction_classes_and_keeps_nod_rule(self):
        nod = [{'id': 'nod-0001', 'start': 6.56, 'end': 8.12}]
        react = [{'id': 'reaction-0001', 'start': 7.56, 'end': 7.88, 'observation': 'brow lowering'}]
        text = build_prompt('sample_1931', 9.682, False, nod, [], react)
        check_prompt(
            {
                'short': '1931',
                'nod_events': nod,
                'listener_audio_is_silent': False,
                'vocal_cues': [],
                'reaction_cues': react,
            },
            text,
        )
        self.assertIn('brow lowering', text)
        self.assertNotIn('head shaking', text)
        self.assertNotIn('brows raised', text)
        self.assertIn('Keep every supplied nodding interval as nodding', text)

    def test_cue_gaps_and_retry_note(self):
        vocal = [
            {'id': 'vocal-0001', 'start': 1.05, 'end': 1.39, 'official_text': 'Yeah.', 'observation': 'spoken response'},
            {'id': 'vocal-0002', 'start': 9.15, 'end': 9.45, 'official_text': 'Yeah.', 'observation': 'spoken response'},
        ]
        nod = [{'id': 'nod-0001', 'start': 6.56, 'end': 8.12}]
        react = [{'id': 'reaction-0001', 'start': 7.56, 'end': 7.88, 'observation': 'brow lowering'}]
        gaps = cue_gaps(vocal, react, nod, 'The listener says "Yeah" once and smiles.')
        self.assertEqual(len(gaps['vocal']), 1)
        self.assertEqual(len(gaps['reaction']), 1)
        self.assertEqual(len(gaps['nod']), 1)
        note = retry_note(gaps)
        self.assertIn('Yeah', note)
        self.assertIn('nodding', note)
        self.assertIn('brow lowering', note)


if __name__ == '__main__':
    unittest.main()
