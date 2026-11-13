import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rerun',ROOT/'scripts/p2_seamless_whisper_joint_seed238_20261108.py')
rerun=importlib.util.module_from_spec(spec);spec.loader.exec_module(rerun)


class SeedTests(unittest.TestCase):
    def test_sealed_membership_and_conditions(self):
        parents=[rerun.read(p/'protocol.json') for p in rerun.PARENTS.values()]
        for name in ['A','B']:
            p=rerun.read(rerun.WORK/name/'protocol.json')
            self.assertEqual(p['seed'],238);self.assertEqual(list(p['datasets']),['seamless'])
            self.assertEqual(p['total_outputs'],200)
            self.assertEqual(p['planner_adapter_identity'],parents[0]['planner_adapter_identity'])
            self.assertEqual(p['generation'],parents[0]['generation'])
            self.assertEqual(p['arms'],parents[0 if name=='A' else 1]['arms'])
            c=rerun.read(rerun.WORK/name/'cluster.json')
            self.assertEqual([(n['host'],n['gpus']) for n in c['nodes']],[('172.29.189.119',[0])])
            items=rerun.read(rerun.WORK/name/'shared_items.json')
            self.assertEqual(len(items),200)
            self.assertTrue(all(x['dataset']=='seamless' for x in items))
        self.assertEqual([p['seed'] for p in parents],[137,137])

    def test_previous_seed_is_rejected(self):
        p=dict(seed=238,arms=[dict(original_sha256='abc')],planner_adapter_identity={},generation={})
        m=dict(seed=238,step=6000,checkpoint_sha256='abc',planning_adapter={},generation={},
               listener_future_input=False,target_description_access=False)
        rerun.verify_metadata(m,p)
        m['seed']=137
        with self.assertRaises(AssertionError):rerun.verify_metadata(m,p)

    def test_device_report_cannot_select_other_gpu(self):
        d=rerun.gpu0('1, 0, 143771, 0\n0, 40000, 143771, 100')
        self.assertEqual(d['used'],40000);self.assertEqual(d['free'],103771)
        with self.assertRaises(ValueError):rerun.gpu0('7, 0, 143771, 0')


if __name__=='__main__':unittest.main()
