import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_event_presence.hints import audio_hints, smile_trace, visual_hints
from p2_event_presence.prompts import prompt
from run_listen_native_caption_vocal_pilot import normalize_asr_cue_for_english_prompt


class EventPresenceHintTests(unittest.TestCase):
    def test_missing_smile_measurement_is_not_zero(self):
        rows = [dict(detected=True, blendshapes=dict(mouthSmileLeft=.2, mouthSmileRight=.6)),
                dict(detected=True, blendshapes=dict(mouthSmileLeft=.2)),
                dict(detected=False, blendshapes=dict(mouthSmileLeft=.2, mouthSmileRight=.6)),
                dict(detected=True, blendshapes=dict(mouthSmileLeft=0., mouthSmileRight=0.))]
        self.assertEqual(smile_trace(rows), [.4, None, None, 0.])

    def test_smile_peaks_reach_final_prompt_without_binary_classification(self):
        trace = [0.] * 25
        trace[7] = .9
        c = dict(fps=25, frame_coverage=1., framewise_face_detected=[True]*25,
                 smile_blendshape=dict(framewise_mean_left_right=trace))
        h = visual_hints(c, dict(start_s=0., end_s=1.))
        bins = h['smile_blendshape']['bins']
        self.assertEqual(bins[0][3], .9)
        self.assertGreater(bins[0][3], bins[0][2])
        self.assertIsNone(h['smile_blendshape']['binary_threshold'])
        text = prompt('visual', 1., h)
        self.assertIn(json.dumps(h), text)
        self.assertIn('not probabilities or binary labels', text)

    def test_window_has_only_local_cues_and_local_coverage(self):
        c = dict(fps=25, frame_coverage=.5, framewise_face_detected=[False]*25+[True]*25,
                 cues=[dict(type='nodding', start=.5, end=1.5),
                       dict(type='frowning', start=0., end=.8)],
                 smile_blendshape=dict(framewise_mean_left_right=[.9]*25+[.1]*25))
        h = visual_hints(c, dict(start_s=1., end_s=2.))
        self.assertEqual(h['frame_coverage'], 1.)
        self.assertEqual(h['cues'], [dict(type='nodding', start=0., end=.5)])
        self.assertTrue(all(b[3] == .1 for b in h['smile_blendshape']['bins']))
        self.assertEqual(h['smile_blendshape']['bins'][0][:2], [0., .5])

    def test_annotation_observations_and_brow_context_reach_prompt(self):
        c = dict(fps=25, frame_coverage=.9,
                 cues=[dict(type='frowning', observation='brow lowering', start=0., end=.5),
                       dict(type='surprised', observation='brows raised', start=.5, end=1.)])
        h = visual_hints(c, dict(start_s=0., end_s=1.))
        text = prompt('visual', 1., h)
        self.assertEqual([x['observation'] for x in h['cues']], ['brow lowering','brows raised'])
        self.assertIn('not an extra output category', text)
        from p2_event_presence.prompts import schema
        self.assertEqual(len(schema('visual')['properties']['classes']['required']), 4)
        self.assertNotIn('surprised', schema('visual')['properties']['classes']['required'])

    def test_short_tail_and_missing_measurements_remain_explicit(self):
        c = dict(fps=25, framewise_face_detected=[True]*26,
                 smile_blendshape=dict(framewise_mean_left_right=[None]*26))
        h = visual_hints(c, dict(start_s=0., end_s=1.02))
        self.assertEqual(h['frame_coverage'], 1.)
        self.assertEqual(h['smile_blendshape']['bins'][-1], [1., 1.02, None, None, 0, 1])

    def test_legacy_coverage_is_not_claimed_as_window_coverage(self):
        h = visual_hints(dict(frame_coverage=.8), dict(start_s=1., end_s=2.))
        self.assertEqual(h['coverage_scope'], 'clip_fallback')
        self.assertEqual(h['frame_coverage'], .8)

    def test_annotation_normalization_really_filters_prompt(self):
        raw = dict(transcript='对', nonlexical_syllables=['嗯', '咳'], uncertain=True)
        before = copy.deepcopy(raw)
        h = audio_hints(raw, normalize_asr_cue_for_english_prompt)
        self.assertEqual(h['normalized_candidate_texts'], ['mhm'])
        self.assertIn('cough or throat clearing', h['observations'])
        self.assertTrue(h['uncertain'])
        text = prompt('audio', 2., asr=h)
        for value in ('对', '嗯', '咳', 'raw_asr_text'):
            self.assertNotIn(value, text)
        self.assertIn('mhm', text)
        self.assertEqual(raw, before)

    def test_english_asr_survives_and_empty_asr_is_not_absence(self):
        h = audio_hints(dict(transcript='okay', nonlexical_syllables=['uh-huh'], uncertain=False),
                        normalize_asr_cue_for_english_prompt)
        self.assertEqual(h['normalized_candidate_texts'], ['okay', 'uh-huh'])
        empty = audio_hints(dict(transcript='', nonlexical_syllables=[], uncertain=True),
                            normalize_asr_cue_for_english_prompt)
        self.assertEqual(empty['normalized_candidate_texts'], [])
        self.assertTrue(empty['uncertain'])
        self.assertNotIn('absent', empty)


if __name__ == '__main__':
    unittest.main()
