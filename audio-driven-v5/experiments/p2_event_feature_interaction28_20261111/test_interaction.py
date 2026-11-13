import copy
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extension as ext
ext.activate()
import torch
from p2_ltx_events.heads import objective
from p2_loss_search28.schedule import Schedule
from p2_loss_search28 import queue, config


class InteractionChecks(unittest.TestCase):
    def test_only_weights_name_output_change(self):
        base = ext.read(ext.ORIGINAL / 'runs/baseline/config.json')
        for name, (event, feature) in ext.CANDIDATES.items():
            value = ext.validate(ext.make(name))
            self.assertEqual(value['loss_weights'], dict(av=1., event=event, video_feature=feature, audio_feature=feature))
            value = copy.deepcopy(value)
            value['output'] = base['output']
            value['loss_search']['candidate'] = 'baseline'
            value['loss_weights'] = base['loss_weights']
            self.assertEqual(value, base)

    def test_coefficients_affect_gradients(self):
        for name, (event, feature) in ext.CANDIDATES.items():
            x = torch.tensor(2., requires_grad=True)
            values = dict(av=x, features=dict(video=2*x, audio=3*x), event_sums={'clip/example':4*x})
            loss, _, _ = objective(values, ext.make(name), {'clip/example':torch.tensor(1.)}, 1, 28)
            loss.backward()
            self.assertAlmostEqual(x.grad.item(), 1/28 + 4*event + 5*feature/28, places=5)

    def test_training_entrypoints_use_only_new_candidates(self):
        self.assertEqual(config.WORK, ext.WORK)
        self.assertEqual(queue.WORK, ext.WORK)
        self.assertEqual(list(queue.stages()), [(500, list(ext.CANDIDATES))])
        for name in ext.CANDIDATES:
            cluster = ext.validate_cluster(ext.make(name), ext.cluster_for(name))
            self.assertEqual(cluster['training_script'], str(ext.WORK / 'worker.py'))
            self.assertEqual(sum(len(n['gpus']) for n in cluster['nodes']), 28)
            self.assertEqual(cluster['nodes'][3]['gpus'], [0, 1, 2, 3])

    def test_original_source_kept_and_schedule_identical(self):
        expected = ext.read(ext.ORIGINAL / 'PREFLIGHT_ACCEPTED.json')['source']
        actual = ext.source_identity()
        self.assertTrue(all(actual[k] == v for k, v in expected.items()))
        schedules = [Schedule(ext.make(name)) for name in ext.CANDIDATES]
        self.assertEqual(schedules[0].pairs, schedules[1].pairs)
        original = ext.read(ext.ORIGINAL / 'schedule.json')
        self.assertEqual(schedules[0].pairs, [tuple(v) for v in original['pairs']])

    def test_wrong_weights_and_geometry_rejected(self):
        for name in ext.CANDIDATES:
            value = ext.make(name)
            value['loss_weights']['video_feature'] = .025
            with self.assertRaises(ValueError): ext.validate(value)
            cluster = ext.cluster_for(name)
            cluster['nodes'][3]['gpus'].append(7)
            with self.assertRaises(ValueError): ext.validate_cluster(ext.make(name), cluster)

    def test_fresh_worker_entry_import(self):
        code = '''import runpy,sys,pathlib
path=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(path.parent))
runpy.run_path(str(path),run_name='entry_import')
import p2_loss_search28.config as c
assert set(c.CANDIDATES)=={'event020_feature010','event100_feature010'}
c.validate_cluster(c.make('event020_feature010'),__import__('extension').cluster_for('event020_feature010'))
import queue
assert pathlib.Path(queue.__file__).parent!=path.parent
'''
        subprocess.run([sys.executable, '-c', code, str(ext.WORK / 'worker.py')], check=True, timeout=60)

    def test_rolling_dispatch_uses_extended_inference_entry(self):
        code = '''import sys,pathlib
sys.path.insert(0,sys.argv[1])
import rolling_video as r
assert r.native.SELF==pathlib.Path(sys.argv[1])/'evaluate28.py'
assert r.D.WORK==pathlib.Path(sys.argv[1])
assert r.E.CLI==r.E.WORK/'evaluate28.py'
import p2_loss_search28.config as c
assert c.WORK==pathlib.Path(sys.argv[1])
'''
        subprocess.run([sys.executable, '-c', code, str(ext.WORK)], check=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
