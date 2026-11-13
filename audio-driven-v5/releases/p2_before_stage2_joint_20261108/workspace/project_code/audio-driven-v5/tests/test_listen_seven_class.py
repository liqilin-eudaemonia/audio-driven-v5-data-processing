import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from listen_benchmark.seven_class import TAXONOMY, map_sample
from build_listen_seven_class import export


def sample(specs):
    events, traces, sources = [], {}, []
    for i, spec in enumerate(specs):
        sig, text, raw = spec[:3]
        mod = spec[3] if len(spec) > 3 else 'visual'
        eid, cid = f'event-{i:04d}', f'candidate-{i:04d}'
        event = dict(event_id=eid, start=1 + i * .1, end=1.5 + i * .1,
                     modality=mod, behavior=text, review_flags=[])
        events.append(event)
        sources.append(dict(id=cid, start=event['start'], end=event['end'],
                            modality=mod, behavior=raw,
                            source='qwen3_vl' if mod == 'visual' else 'omni_audio'))
        traces[eid] = dict(signature=sig, source_candidate_ids=[cid],
                           boundary_method='observed_occurrence_interval', atoms=[])
    return (dict(sample_id='sample1', duration_s=10, time_origin='clip_start_seconds', events=events),
            dict(events=traces, source_candidates=sources, excluded=[]))


class SevenClassTests(unittest.TestCase):
    def test_explicit_seven_classes_keep_order_modality_and_timestamps(self):
        a, p = sample([
            ('nod', 'Nods.', 'Nods slightly.'),
            ('shake head', 'Shakes head.', 'Shakes her head left and right.'),
            ('smile', 'Smiles.', 'Smiles warmly.'),
            ('laugh', 'Laughs.', 'Laughs visibly.'),
            ('frown', 'Frowns.', 'Frowns.'),
            ('show surprise', 'Looks surprised.', 'Looks surprised with eyes wide and eyebrows raised.'),
            ('say okay', 'Says "okay".', 'Says "okay".', 'audio'),
        ])
        result = map_sample(a, p)
        self.assertEqual([e['type'] for e in result['events']], list(TAXONOMY))
        self.assertEqual([(e['start'], e['end']) for e in result['events']],
                         [(e['start'], e['end']) for e in a['events']])

    def test_generic_movement_cannot_become_nod_via_signature(self):
        a, p = sample([('nod', 'Moves head slightly.', 'Slight head movement, indicating attentiveness.')])
        r = map_sample(a, p)
        self.assertFalse(r['events'])
        self.assertEqual(r['decisions'][0]['status'], 'needs_review')
        self.assertIn('normalized_class_not_supported_by_original_description', r['decisions'][0]['reasons'])

    def test_generic_signature_can_use_explicit_atomic_description(self):
        a, p = sample([('head | direction=', 'Nods slightly.', 'Smiles and nods.')])
        self.assertEqual(map_sample(a, p)['events'][0]['type'], 'head_nod')

    def test_hedged_negated_and_repeated_cycles_require_review(self):
        for raw in ['Possibly nodding or adjusting posture.', 'Does not nod.',
                    'Nods repeatedly at multiple points.']:
            a, p = sample([('nod', 'Nods.', raw)])
            self.assertEqual(map_sample(a, p)['decisions'][0]['status'], 'needs_review')

    def test_side_turn_open_mouth_and_eye_widening_not_forced_into_classes(self):
        a, p = sample([('turn head', 'Turns head left.', 'Turns head left.'),
                       ('open mouth', 'Opens mouth.', 'Opens mouth.'),
                       ('widen eyes', 'Widens eyes.', 'Widens eyes.')])
        r = map_sample(a, p)
        self.assertFalse(r['events'])
        self.assertEqual({d['status'] for d in r['decisions']}, {'out_of_scope'})

    def test_normalized_description_multiple_cycles_retains_review_requirement(self):
        a, p = sample([('nod', 'Nods at multiple points.', 'Head nods at 0.5s, 1s and 1.5s.')])
        r = map_sample(a, p)
        self.assertFalse(r['events'])
        self.assertIn('multiple_cycles_without_individual_boundaries', r['decisions'][0]['reasons'])

    def test_surprise_requires_cues_in_original_not_invented_normalization(self):
        a, p = sample([('show surprise', 'Eyes widen and eyebrows raise.', 'Shows surprise.')])
        r = map_sample(a, p)
        self.assertIn('surprise_requires_original_multicue_evidence', r['decisions'][0]['reasons'])
        self.assertFalse(r['events'])

    def test_expression_end_transition_does_not_create_positive_state(self):
        a, p = sample([('smile | change=smile fades', 'Smile fades.', 'Smile fades to neutral.')])
        self.assertIn('state_transition_boundaries_require_review',
                      map_sample(a, p)['decisions'][0]['reasons'])

    def test_multi_label_overlap_and_audio_laughter_use_separate_modalities(self):
        a, p = sample([('nod', 'Nods.', 'Nods and smiles.'),
                       ('smile', 'Smiles.', 'Nods and smiles.'),
                       ('laugh', 'Laughs.', 'Laughs audibly.', 'audio')])
        r = map_sample(a, p)
        self.assertEqual([e['type'] for e in r['events']], ['head_nod', 'smile', 'listener_vocalization'])
        self.assertEqual(len(r['events']), 3)

    def test_breath_and_alignment_conflicts_stay_pending(self):
        a, p = sample([('vocalize', 'Emits a breathy sound.', 'Breathes softly.', 'audio'),
                       ('say okay', 'Says "okay".', 'Says "okay".', 'audio')])
        a['events'][1]['review_flags'] = ['multiple_word_anchors_overlap']
        r = map_sample(a, p)
        self.assertFalse(r['events'])
        self.assertEqual({d['status'] for d in r['decisions']}, {'needs_review'})

    def test_asr_repeated_words_keep_exact_anchors_and_original_text(self):
        a, p = sample([('say okay', 'Says "Okay".', 'Okay', 'audio')] * 2)
        for e, s, t in zip(a['events'], p['source_candidates'], p['events'].values()):
            e['text'] = s['behavior']
            s['source'] = 'asr_aligner'
            t['boundary_method'] = 'asr_word_anchor'
        before = copy.deepcopy((a, p))
        r = map_sample(a, p)
        self.assertEqual(len(r['events']), 2)
        self.assertEqual([(e['start'], e['end'], e['text']) for e in r['events']],
                         [(e['start'], e['end'], e['text']) for e in a['events']])
        self.assertTrue(all(e['event_unit'] == 'word' for e in r['events']))
        self.assertEqual((a, p), before)
        a['events'][0]['end'] += .01
        with self.assertRaisesRegex(ValueError, 'ASR anchor'): map_sample(a, p)

    def test_every_observation_preserved_without_negative_or_training_admission(self):
        a, p = sample([('smile', 'Smiles.', 'Smiles.'),
                       ('nod', 'Nods.', 'Might nod.'),
                       ('twirl pen', 'Twirls pen.', 'Twirls pen.')])
        p['excluded'] = [{'source_candidate_id': 'older-exclusion', 'reason': 'static posture'}]
        r = map_sample(a, p)
        self.assertEqual(len(r['decisions']), len(a['events']))
        self.assertEqual(r['upstream_exclusions'], p['excluded'])
        self.assertEqual(r['coverage'], {k: [] for k in TAXONOMY})
        self.assertEqual(r['unknown'], {k: [[0, 10]] for k in TAXONOMY})
        self.assertEqual(r['K_train'], [])
        self.assertFalse(r['training_admitted'])
        self.assertFalse(r['human_ground_truth'])

    def test_invalid_intervals_missing_sources_and_mismatched_modalities_fail(self):
        for mutate in [lambda a, p: a['events'][0].update(start=True),
                       lambda a, p: a['events'][0].update(end=float('nan')),
                       lambda a, p: p['source_candidates'].clear(),
                       lambda a, p: p['source_candidates'][0].update(modality='audio'),
                       lambda a, p: a['events'].append(copy.deepcopy(a['events'][0]))]:
            a, p = sample([('nod', 'Nods.', 'Nods.')])
            mutate(a, p)
            with self.assertRaises(ValueError): map_sample(a, p)

    def test_export_replay_and_input_immutability(self):
        a, p = sample([('smile', 'Smiles.', 'Smiles.')])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);run = root / 'open';run.mkdir()
            data = {'labels.jsonl': json.dumps(a) + '\n',
                    'provenance.json': json.dumps({'samples': {'sample1': p}}),
                    'input_review.json': json.dumps({'samples': [dict(id='sample1', duration=10,
                        video='../media/sample1/video.mp4', audio='../media/sample1/audio.wav')]})}
            for name, value in data.items(): (run / name).write_text(value)
            export(run, root / 'one');export(run, root / 'two')
            for name in ['labels.jsonl', 'labels.json', 'mapping_policy.json']:
                self.assertEqual((root/'one'/name).read_bytes(), (root/'two'/name).read_bytes())
            for name, value in data.items(): self.assertEqual((run/name).read_text(), value)
            with self.assertRaises(ValueError): export(run, root/'one')
            with self.assertRaises(ValueError): export(run, run/'nested')


if __name__ == '__main__':
    unittest.main()
