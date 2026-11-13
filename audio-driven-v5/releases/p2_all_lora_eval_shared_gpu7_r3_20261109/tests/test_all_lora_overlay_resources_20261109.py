import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('overlay',ROOT/'scripts/p2_all_lora_followups_gpu7_overlay_20261109.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class SharedResourceTests(unittest.TestCase):
 def test_shared_render_has_four_gib_margin(self):
  free=71226;cap=M.render_budget(free)
  self.assertLessEqual(cap*1024,free-4096)
  self.assertGreater(cap,64)
  self.assertEqual(M.render_budget(140*1024),96)
  with self.assertRaises(ValueError):M.render_budget(63*1024)
 def test_two_decoder_caps_fit_beside_observed_foreign_reservation(self):
  foreign=72538;total=143771
  self.assertLess(2*M.decode_budget(2)*1024+foreign,total-10*1024)
  self.assertEqual(M.decode_budget(1),40)
  self.assertEqual(M.decode_budget(4),32)
 def test_only_gpu7_for_every_stage(self):
  for s in ['render','decode','metric']:
   self.assertEqual(M.worker_environment(s,3)['CUDA_VISIBLE_DEVICES'],'7')
   for p in [0,1,2,4,7]:
    with self.assertRaises(AssertionError):M.worker_environment(s,p)
 def test_frozen_model_and_sampler_helper_unchanged(self):
  self.assertEqual(M.E.__file__,str(ROOT/'src/p2_all_lora_followups_eval_20261109.py'))
  for step in [5500,7000,7500]:
   M.E.configure(step);p=M.read(M.E.WORK/'protocol.json')
   self.assertEqual(p['execution_code'],M.E.execution_code())
   self.assertEqual(p['arms'][0]['step'],step)
   self.assertEqual(p['generation'],dict(steps=30,video_cfg=3,audio_cfg=7))
 def test_native_decode_environment_stays_isolated(self):
  for stage in ['render','decode']:self.assertEqual(M.worker_environment(stage,3)['PYTHONPATH'],'')
  self.assertEqual(M.worker_environment('metric',3)['PYTHONPATH'],'/private/liqi_workspace/.cache/paired_video_metrics/python')

if __name__=='__main__':unittest.main()
