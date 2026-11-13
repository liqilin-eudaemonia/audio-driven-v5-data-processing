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

    def test_fast_yeah_is_one_burst_and_does_not_cover_a_second_window(self):
        one = [{'start': 1.0, 'end': 1.4, 'official_text': 'Yeah, yeah.', 'observation': 'spoken response'}]
        self.assertEqual(cue_gaps(one, [], [], 'She says yeah, yeah.')['vocal'], [])
        distant = [
            {'start': 1.0, 'end': 1.3, 'official_text': 'Yeah.', 'observation': 'spoken response'},
            {'start': 8.2, 'end': 8.5, 'official_text': 'Yeah.', 'observation': 'spoken response'},
        ]
        burst = cue_gaps(distant, [], [], 'She says yeah, yeah.')
        self.assertEqual(burst['vocal'][0]['needed'], 2)
        self.assertEqual(burst['vocal'][0]['written'], 1)
        separated = cue_gaps(distant, [], [], 'She says yeah at the start. Toward the end she says yeah again.')
        self.assertEqual(separated['vocal'], [])

    def test_silent_caption_claiming_speech_is_not_a_pass(self):
        gaps = cue_gaps([], [], [], 'She says "yeah" and laughs.', silent=True)
        self.assertTrue(gaps['silent_vocal'])
        self.assertIn('silent', retry_note(gaps).lower())

    def test_negated_head_shake_is_uncertain_and_not_retried(self):
        react = [{'start': 1.0, 'end': 1.8, 'observation': 'head shaking'}]
        gaps = cue_gaps([], react, [], 'She does not shake her head.')
        self.assertEqual(gaps['reaction'], [])
        self.assertEqual(gaps['uncertain'][0]['reason'], 'negated_mention')
        self.assertNotIn('side-to-side', retry_note(gaps))

    def test_distant_nods_are_not_passed_by_one_word_or_forced(self):
        nods = [{'start': 0.4, 'end': 0.9}, {'start': 7.2, 'end': 7.8}]
        gaps = cue_gaps([], [], nods, 'She nods her head.')
        self.assertEqual(gaps['nod'], [])
        self.assertEqual(gaps['uncertain'][0]['reason'], 'distant_intervals_not_separated')
        self.assertEqual(retry_note(gaps), '')
        single = cue_gaps([], [], [nods[0]], 'She nods.')
        self.assertEqual(single['nod'], [])
        self.assertNotIn('uncertain', single)

    def test_okay_abbreviation_matches_and_a_bare_letter_does_not(self):
        cue = [{'start': 1.0, 'end': 1.3, 'official_text': 'OK.'}]
        self.assertEqual(cue_gaps(cue, [], [], 'She says okay.')['vocal'], [])
        self.assertEqual(len(cue_gaps(cue, [], [], 'She says k.')['vocal']), 1)

    def test_silent_clauses_do_not_hide_a_later_vocal_claim(self):
        first = cue_gaps([], [], [], 'She remains silent at first, then laughs.', silent=True)
        second = cue_gaps([], [], [], 'She does not move her head but says "yeah".', silent=True)
        quiet = cue_gaps([], [], [], 'She remains silent.', silent=True)
        denied = cue_gaps([], [], [], 'She does not speak.', silent=True)
        self.assertTrue(first['silent_vocal'])
        self.assertTrue(second['silent_vocal'])
        self.assertNotIn('silent_vocal', quiet)
        self.assertNotIn('silent_vocal', denied)

    def test_thats_right_and_okay_abbreviation_match_either_way(self):
        thats = [{'start': 1.0, 'end': 1.5, 'official_text': "That's right."}]
        self.assertEqual(cue_gaps(thats, [], [], "That's right.")['vocal'], [])
        okay = [{'start': 1.0, 'end': 1.3, 'official_text': 'okay'}]
        self.assertEqual(cue_gaps(okay, [], [], 'She says OK.')['vocal'], [])

    def test_prompt_does_not_force_a_single_nod(self):
        text = build_prompt('sample_1931', 9.682, False, [{'id': 'nod-0001', 'start': 1.0, 'end': 2.0}], [], [])
        self.assertIn('consecutive nods', text)
        self.assertNotIn('one lowering', text)
        self.assertNotIn('one raising', text)


if __name__ == '__main__':
    unittest.main()
