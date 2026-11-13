"""Protect idle-only acquisition and reuse of the sealed latest-SFT test inputs."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('current_av_eval', ROOT/'scripts/p2_current_av6000_sft2752_eval_20261107.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CurrentAVTests(unittest.TestCase):
    def test_idle_does_not_admit_a_small_active_job(self):
        self.assertFalse(module.gpu_is_idle('1100, 0', '12345, 1100'))
        self.assertFalse(module.gpu_is_idle('533, 0', '12345, 10'))
        self.assertTrue(module.gpu_is_idle('533, 0', ''))

    def test_idle_requires_both_headroom_and_idle_compute(self):
        self.assertFalse(module.gpu_is_idle('30000, 0', ''))
        self.assertFalse(module.gpu_is_idle('533, 99', ''))

    def test_reused_conditions_and_test_protocol_are_exact(self):
        p = module.read(module.WORK/'protocol.json')
        parent = module.read(module.CONDITIONS/'protocol.json')
        for name in ['datasets', 'seed', 'generation', 'planner_adapter_identity', 'shared_items_sha256']:
            self.assertEqual(p[name], parent[name])
        self.assertEqual(len(module.read(module.WORK/'REUSED_CONDITIONS.json')['plans']), 400)
        self.assertEqual(module.read(module.WORK/'cluster.json')['nodes'][0]['gpus'], [7])
        self.assertFalse((module.WORK/'whisper').exists())
        self.assertEqual(p['arms'][0]['step'], 6000)
        self.assertEqual(p['checkpoint_identity']['experiment']['loss_weights'],
                         dict(av=1., event=0., video_feature=0., audio_feature=0.))


if __name__ == '__main__':
    unittest.main()
