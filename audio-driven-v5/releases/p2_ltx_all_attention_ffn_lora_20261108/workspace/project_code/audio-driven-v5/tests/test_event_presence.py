import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p2_event_presence import SCHEMA
from p2_event_presence.contracts import collapse_events, labels
from p2_event_presence.scoring import score
from p2_event_presence.prompts import parse_label_response


def record(sid, gt=(), pred=(), method='a', draw='0'):
    g, p = labels('absent', 'explicit_negative'), labels('absent', 'explicit_negative')
    for side, present in [(g, gt), (p, pred)]:
        for mod in side:
            for c in present:
                if c in side[mod]: side[mod][c] = dict(state='present', evidence='observed')
    return dict(schema=SCHEMA, dataset='test', id=sid, method=method, draw_id=draw,
                reference_identity='reference-' + sid, protocol_identity='fixed-protocol',
                capabilities=dict(visual=True, audio=True), reference=g, generated=p)


def visual(rows, method='a'):
    return score(rows)['datasets']['test']['methods'][method]['draws'][0]['visual']


class EventPresenceTests(unittest.TestCase):
    def test_corpus_counts_not_mean_of_clip_f1(self):
        rows = [record('1', ['nodding'], ['nodding']), record('2', ['nodding'], []),
                record('3', [], ['nodding']), record('4', ['nodding'], ['nodding'])]
        result = visual(rows); c = result['per_class']['nodding']
        self.assertEqual((c['tp'], c['fp'], c['fn']), (2, 1, 1))
        self.assertAlmostEqual(result['full_dataset_metrics']['macro_f1'], 2/3)

    def test_timing_and_repetition_are_irrelevant(self):
        g = collapse_events([dict(class_='unused', **{'class': 'nodding'}, start=.1, end=.3)], 10)
        p = collapse_events([{'class': 'nodding', 'start': a, 'end': a+.1} for a in [5., 6., 8.]], 10)
        row = record('x'); row.update(reference=g, generated=p)
        self.assertEqual(visual([row])['full_dataset_metrics']['macro_f1'], 1.)

    def test_all_empty_is_not_perfect(self):
        result = visual([record('x')])
        self.assertIsNone(result['full_dataset_metrics']['macro_f1'])
        self.assertEqual(result['reference_all_empty_pairs'], 1)

    def test_unsupported_reference_class_false_positive_is_exposed(self):
        result = visual([record('x', ['nodding'], ['nodding', 'frowning'])])
        self.assertEqual(result['macro_classes'], ['nodding'])
        self.assertEqual(result['full_dataset_metrics']['macro_f1'], 1.)
        self.assertEqual(result['per_class']['frowning']['fp'], 1)
        self.assertEqual(result['per_class']['frowning']['false_positive_rate'], 1.)

    def test_no_prediction_of_supported_class_is_zero(self):
        result = visual([record('x', ['nodding'])])
        self.assertEqual(result['full_dataset_metrics']['macro_precision'], 0.)
        self.assertEqual(result['full_dataset_metrics']['macro_f1'], 0.)

    def test_unknown_and_failure_are_not_negatives(self):
        row = record('x', ['nodding']); row['generated']['visual']['nodding']['state'] = 'error'
        result = visual([row])
        self.assertIsNone(result['full_dataset_metrics'])
        self.assertEqual(result['per_class']['nodding']['fn'], 0)
        self.assertEqual(result['per_class']['nodding']['evaluated_pairs'], 0)
        self.assertIsNone(result['available_subset_metrics']['macro_f1'])

    def test_common_masks_are_per_class_and_fixed_classes_do_not_disappear(self):
        a = record('x', ['nodding'], ['nodding', 'smiling'])
        b = record('x', ['nodding'], ['nodding', 'smiling'], method='b')
        b['generated']['visual']['nodding']['state'] = 'unknown'
        d = score([a, b])['datasets']['test']
        self.assertEqual(d['common_evaluable_ids_by_draw_class']['0']['visual']['nodding'], [])
        self.assertEqual(d['common_evaluable_ids_by_draw_class']['0']['visual']['smiling'], ['x'])
        for m in ['a', 'b']:
            r = d['methods'][m]['common_subset_draws'][0]['visual']
            self.assertEqual(r['macro_classes'], ['nodding'])
            self.assertIsNone(r['available_subset_metrics']['macro_f1'])

    def test_draws_are_scored_separately_not_union_or_best(self):
        a = record('x', ['nodding'], ['nodding'], draw='0')
        b = record('x', ['nodding'], [], draw='1')
        d = score([a, b])['datasets']['test']['methods']['a']['full_dataset_draw_aggregate']['visual']['macro_f1']
        self.assertEqual(d['values'], [1., 0.]); self.assertEqual(d['mean'], .5)

    def test_audio_and_visual_are_separate(self):
        d = score([record('x', ['nodding', 'audible_laughter'], ['nodding'])])['datasets']['test']['methods']['a']['draws'][0]
        self.assertEqual(d['visual']['full_dataset_metrics']['macro_f1'], 1.)
        self.assertEqual(d['audio']['full_dataset_metrics']['macro_f1'], 0.)

    def test_visual_only_method_audio_is_na(self):
        row = record('x', ['audible_laughter']); row['capabilities']['audio'] = False
        d = score([row])['datasets']['test']['methods']['a']['draws'][0]
        self.assertEqual(d['audio']['status'], 'not_applicable')

    def test_reference_conflicts_and_duplicate_pairs_rejected(self):
        a = record('x', ['nodding']); b = record('x', [], method='b')
        with self.assertRaisesRegex(ValueError, 'Inconsistent reference'): score([a, b])
        with self.assertRaisesRegex(ValueError, 'Duplicate'): score([a, copy.deepcopy(a)])

    def test_missing_frozen_prediction_becomes_failure_not_dropped_sample(self):
        a = record('x', ['nodding'], ['nodding']); b = record('y', ['nodding'], ['nodding'])
        expected = [a, b, dict(b, method='b'), dict(a, method='b')]
        result = score([a, b, dict(a, method='b')], expected)
        self.assertEqual(len(result['missing_detection_records']), 1)
        d = result['datasets']['test']['methods']['b']['draws'][0]['visual']
        self.assertEqual(d['requested_pairs'], 2); self.assertIsNone(d['full_dataset_metrics'])

    def test_unknown_is_not_implicit_absent_and_event_bounds_are_checked(self):
        a = record('x'); del a['generated']['visual']['frowning']
        with self.assertRaises(ValueError): score([a])
        with self.assertRaises(ValueError): collapse_events([{'class': 'nodding', 'start': 9, 'end': 12}], 10)

    def test_model_omissions_and_invalid_confidence_are_not_labels(self):
        with self.assertRaises(ValueError):parse_label_response({'classes':{},'long_speech_observed':False},'audio')
        r={'classes':labels()['audio'],'long_speech_observed':False}
        self.assertEqual(parse_label_response(r,'audio')['classes']['audible_laughter']['state'],'unknown')
        r['classes']['audible_laughter']['state']=.9
        with self.assertRaises(ValueError):parse_label_response(r,'audio')

    def test_manual_disagreement_stays_unknown(self):
        import tempfile
        from p2_event_presence.reporting import human_labels
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'labels.csv'
            p.write_text('media_id,modality,class,state,evidence,rater_id\nx,visual,nodding,present,nod seen,a\nx,visual,nodding,absent,no nod,b\n')
            gold,audit=human_labels(p)
            self.assertEqual(gold['x']['visual']['nodding']['state'],'unknown')
            self.assertEqual(gold['x']['visual']['frowning']['state'],'unknown')
            self.assertEqual(audit['disagreement_units'],1)


if __name__ == '__main__': unittest.main()
