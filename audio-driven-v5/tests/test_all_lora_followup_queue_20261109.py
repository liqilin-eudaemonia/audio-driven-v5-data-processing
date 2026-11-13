"""Source/step and GPU isolation checks for the queued checkpoint evaluations."""
import ast
import copy
import fcntl
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_all_lora_followups_eval_20261109 as E
import p2_all_lora_followup_queue_20261109 as Q
from p2_testsets_eval.protocol import read,save
spec=importlib.util.spec_from_file_location('followup',E.CLI);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class FollowupTests(unittest.TestCase):
 def test_only_gpu7_in_every_stage(self):
  for stage in ['render','decode','metric']:
   self.assertEqual(entry.worker_environment(stage,3)['CUDA_VISIBLE_DEVICES'],'7')
   for i in [0,1,2,4,5,6,7,24,27]:
    with self.assertRaises(AssertionError):entry.worker_environment(stage,i)
  self.assertEqual(entry.worker_environment('decode',3)['PYTHONPATH'],'')
 def test_checkpoint_and_source_identity(self):
  for step,count in [(5500,282),(7000,295),(7500,295)]:
   E.configure(step);self.assertEqual(read(E.TRAIN/f'step_{step:06d}.complete.json')['step'],step)
   self.assertEqual(len(E.expected_source()),count)
   raw=read(E.TRAIN/'run.json')['experiment'];before=copy.deepcopy(raw);normalized=E.model_config(raw)
   self.assertEqual(raw,before);self.assertEqual(normalized['training']['global_batch'],31)
   self.assertEqual(raw['training']['global_batch'],31 if step==5500 else 28)
 def test_changed_training_plan_rejected(self):
  E.configure(7000);c=read(E.TRAIN/'run.json')['experiment'];c=copy.deepcopy(c);c['training']['global_batch']=29
  with self.assertRaises(AssertionError):E.model_config(c)
  c=read(E.TRAIN/'run.json')['experiment'];c=copy.deepcopy(c);c['batch28_continuation']['prefix_pairs']+=1
  with self.assertRaises(AssertionError):E.model_config(c)
 def test_unapproved_steps_rejected(self):
  for step in [5000,6000,6500,7501]:
   with self.assertRaises(ValueError):E.configure(step)
 def test_sampler_math_unchanged(self):
  def fn(path):return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='render')
  native=fn(ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py')
  class Step(ast.NodeTransformer):
   def visit_Constant(self,n):return ast.copy_location(ast.Name(id='STEP',ctx=ast.Load()),n) if n.value==6000 else n
  self.assertEqual(ast.dump(Step().visit(native)),ast.dump(fn(Path(E.__file__))))
 def test_queue_requires_finished_scores_and_released_controller(self):
  with tempfile.TemporaryDirectory() as tmp:
   w=Path(tmp);save(w/'status.json',dict(stage='progressive_evaluation',latents=200))
   self.assertFalse(Q.completed(w,5000))
   save(w/'status.json',dict(stage='complete'))
   save(w/'protocol.json',dict(seed=137,arms=[dict(step=5000,name='a')],datasets=dict(seamless={})))
   save(w/'FINAL_AUDIT.json',dict(success=True,step=5000,outputs=200));save(w/'HANDOFF_COMPLETE.json',dict(success=True,outputs=200))
   result=dict(requested_pairs=200,audio=dict(pairs_with_errors=0),coverage={k:dict(pairs=200) for k in ['psnr','ssim','lpips','fid','fvd']})
   save(w/'RESULTS.json',{'a/seamless':result});self.assertTrue(Q.completed(w,5000))
   with (w/'controller.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertFalse(Q.completed(w,5000))
   (w/'STOP').touch();self.assertFalse(Q.completed(w,5000));(w/'STOP').unlink()
   with self.assertRaises(AssertionError):Q.completed(w,5500)
   result['coverage']['fvd']['pairs']=199;save(w/'RESULTS.json',{'a/seamless':result})
   with self.assertRaises(AssertionError):Q.completed(w,5000)

if __name__=='__main__':unittest.main()
