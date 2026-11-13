import copy
import unittest
from reference_events import align
from p2_event_presence.contracts import SCHEMA, labels
from p2_event_presence.scoring import score


def record(sid='a', method='baseline'):
    return dict(schema=SCHEMA, dataset='seamless', id=sid, method=method,
                draw_id='0', reference_identity='media-' + sid,
                protocol_identity='detector-v1', capabilities=dict(visual=True, audio=True),
                reference=labels('unknown'), generated=labels('absent'))


class ReferenceChecks(unittest.TestCase):
    def test_generated_stays_new_reference_stays_frozen(self):
        base = record()
        base['reference']['visual']['smiling']['state'] = 'present'
        _, registry = align([base], {}, ['a'], True)
        candidate = record(method='event020')
        candidate['reference'] = labels('absent')
        candidate['generated']['visual']['smiling']['state'] = 'present'
        fixed, _ = align([candidate], registry, ['a'])
        self.assertEqual(fixed[0]['reference'], base['reference'])
        self.assertEqual(fixed[0]['generated'], candidate['generated'])
        self.assertEqual(fixed[0]['reference']['audio']['brief_verbal_response']['state'], 'unknown')
        summary = score(fixed)
        self.assertIsNone(summary['datasets']['seamless']['methods']['event020']['full_dataset_draw_aggregate']['audio']['macro_f1']['mean'])

    def test_subset_expansion_never_relabels_existing(self):
        first = record()
        _, registry = align([first], {}, ['a'], True)
        changed = record(); changed['reference'] = labels('present')
        fixed, merged = align([changed, record('b')], registry, ['a', 'b'], True)
        self.assertEqual(merged['a']['reference'], first['reference'])
        self.assertEqual(len(merged), 2)
        self.assertEqual(fixed[0]['reference'], first['reference'])

    def test_missing_duplicate_and_outside_population_rejected(self):
        for records, ids in [([record()], ['a', 'b']), ([record(), record()], ['a']),
                             ([record('b')], ['a'])]:
            with self.assertRaises(ValueError): align(records, {}, ids, True)
        with self.assertRaises(ValueError): align([record()], {}, ['a'])

    def test_changed_media_and_protocol_rejected(self):
        _, registry = align([record()], {}, ['a'], True)
        for key in ('reference_identity', 'protocol_identity', 'dataset'):
            row = record(); row[key] = 'changed'
            with self.assertRaises(ValueError): align([row], registry, ['a'])

    def test_mixed_runs_rejected(self):
        with self.assertRaises(ValueError):
            align([record(), record('b', 'another')], {}, ['a', 'b'], True)


if __name__ == '__main__': unittest.main(verbosity=2)
