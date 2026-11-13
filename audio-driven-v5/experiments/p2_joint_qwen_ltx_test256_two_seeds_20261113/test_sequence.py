"""CPU checks: sequencing, seed isolation, exact condition reuse and mean math."""
import importlib.util,json,math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import aggregate,followup
class SequenceTests(unittest.TestCase):
 def test_gate_requires_all_outputs_all_scores_and_no_workers(self):
  with tempfile.TemporaryDirectory() as t:
   w=Path(t)
   def write(name,x):(w/name).write_text(json.dumps(x))
   self.assertFalse(followup.ready(w))
   write('COMPLETE.json',dict(success=True,outputs=1024,combinations=4));write('RESULTS.json',dict(completed_combinations=4));s=dict(generated=1024,latent=1024,active_gpus=1,active=[{'pid':1}]);write('status.json',s);self.assertFalse(followup.ready(w))
   s.update(active_gpus=0,active=[]);write('status.json',s);self.assertTrue(followup.ready(w))
   write('RESULTS.json',dict(completed_combinations=3));self.assertFalse(followup.ready(w))
 def test_means_and_missing_values(self):
  s=aggregate.statistics2({'137':10.,'238':14.});self.assertEqual(s['mean'],12.);self.assertAlmostEqual(s['sample_std'],math.sqrt(8));self.assertEqual(s['absolute_seed_difference'],4.)
  s=aggregate.statistics2({'137':None,'238':14.});self.assertIsNone(s['mean']);self.assertEqual(s['available_seeds'],1)
 def test_seed238_scope_and_no_generated_reuse(self):
  first,second=aggregate.SOURCES[137],aggregate.SOURCES[238]
  self.assertEqual(aggregate.read(first/'shared_items.json'),aggregate.read(second/'shared_items.json'))
  self.assertEqual(aggregate.read(second/'PLAN.json')['seed'],238)
  for combo in aggregate.read(second/'PLAN.json')['combinations']:
   new=Path(combo['work']);old=Path(combo['first_seed_work']);a,b=aggregate.read(old/'protocol.json'),aggregate.read(new/'protocol.json')
   self.assertEqual(a['seed'],137);self.assertEqual(b['seed'],238)
   for k in ['arms','generation','planner_adapter_identity','datasets','sampling']:self.assertEqual(a[k],b[k])
   for k in ['conditions','whisper','negative','shards']:self.assertEqual((old/k).resolve(),(new/k).resolve())
   self.assertFalse((new/'generated').exists());self.assertFalse((new/'latents').exists());self.assertFalse((new/'metric_shards').exists())
   refs=[x['key'] for i in range(10) for x in aggregate.read(new/'shards'/f'{i:02d}.json')];self.assertEqual(len(refs),256);self.assertEqual(len(set(refs)),256)
 def test_native_sampling_uses_protocol_seed(self):
  source=(aggregate.ROOT/'src/p2_all_lora_followups_eval_20261109.py').read_text().split('def render(work,items,rank,p):',1)[1]
  self.assertEqual(source.count("torch.manual_seed(p['seed'])"),2)
  new=(aggregate.SOURCES[238]/'run.py').read_text();publish=new.split('def publish(key,slot):',1)[1].split('def snapshot():',1)[0]
  self.assertIn("m['seed']==p['seed']==238",publish);self.assertNotIn("m['seed']==137",publish)
if __name__=='__main__':unittest.main()
