import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dual_seed_compare as dual
import dual_seed_queue as queue
import promotion_policy as policy
from test_promotion_policy import fixtures
from p2_listen.contracts import atomic_json,sha


class DualSeedChecks(unittest.TestCase):
    def compare(self,data):
        membership,_=fixtures()
        with patch.dict(policy.DEFAULT,bootstrap_draws=1000):
            return dual.compare(data,membership,copy.deepcopy(policy.DEFAULT))

    def test_average_seed_f1_not_pooled_confusion_counts(self):
        _,good=fixtures();_,bad=fixtures(False)
        report=self.compare(dict(baseline={137:good,238:bad}))
        self.assertEqual(report['mean_metrics']['baseline']['reaction'],.5)
        self.assertEqual(report['per_seed_metrics']['baseline']['137']['reaction'],1.)
        self.assertEqual(report['per_seed_metrics']['baseline']['238']['reaction'],0.)
        self.assertEqual(report['samples_per_seed'],24)
        self.assertEqual(report['independent_source_groups'],12)
        self.assertFalse(report['automatic_promotion'])

    def test_opposite_seed_directions_are_exposed(self):
        _,good=fixtures();_,bad=fixtures(False)
        report=self.compare(dict(baseline={137:good,238:bad},challenger={137:bad,238:good}))
        for row in report['comparisons']:
            self.assertEqual(row['mean_bounds']['reaction']['delta'],0.)
            self.assertFalse(row['reaction_improves_in_both_seeds'])
            self.assertFalse(row['robust_preference'])

    def test_missing_seed_or_different_reference_rejected(self):
        _,good=fixtures()
        with self.assertRaises(ValueError):self.compare(dict(baseline={137:good}))
        altered=copy.deepcopy(good);altered['events'][0]['reference_identity']='other source'
        with self.assertRaises(ValueError):self.compare(dict(baseline={137:good,238:altered}))

    def test_unknown_in_either_draw_is_not_absence(self):
        _,good=fixtures();unknown=copy.deepcopy(good)
        for r in unknown['events']:r['generated']['visual']['nodding']['state']='unknown'
        report=self.compare(dict(baseline={137:good,238:unknown}))
        self.assertEqual(report['common_coverage']['visual/nodding'],0.)
        self.assertTrue(report['issues'])

    def test_no_average_over_only_available_seed(self):
        self.assertEqual(dual.strict_mean([10.,20.]),15.)
        self.assertIsNone(dual.strict_mean([10.,None]))

    def test_second_draw_reuses_export_and_changes_only_seed_and_paths(self):
        with tempfile.TemporaryDirectory()as folder,patch.object(queue,'WORK',Path(folder)),\
             patch.object(queue,'validate_frozen_plan'),patch('stage_driver.verify_scoring'):
            source=queue.work_for('baseline',137);source.mkdir(parents=True)
            for directory in ['export','conditions','whisper','negative']:(source/directory).mkdir()
            for f in ['REQUEST.json','cluster.json','DATA_CONFIG.json']:atomic_json(source/f,{})
            atomic_json(source/'shared_items.json',[dict(id='a',key='seamless/a')])
            for i in range(4):atomic_json(source/'shards'/f'{i:02d}.json',[])
            p=dict(seed=137,total_samples=64,workers=4,arms=[dict(original_sha256='same checkpoint')],
                   shared_items_sha256=sha(source/'shared_items.json'),generation=dict(steps=30),
                   cluster=str(source/'cluster.json'),metrics_cluster=str(source/'cluster.json'),
                   shared_items_manifest=str(source/'shared_items.json'))
            atomic_json(source/'protocol.json',p)
            dest=queue.prepare_second('baseline');new=json.loads((dest/'protocol.json').read_text())
            self.assertEqual(new['seed'],238)
            self.assertEqual(new['generation'],p['generation'])
            self.assertEqual(new['arms'],p['arms'])
            self.assertEqual((dest/'export').resolve(),(source/'export').resolve())
            self.assertEqual(queue.prepare_second('baseline'),dest)
            new['seed']=999;atomic_json(dest/'protocol.json',new)
            with self.assertRaises(ValueError):queue.prepare_second('baseline')


if __name__=='__main__':unittest.main()
