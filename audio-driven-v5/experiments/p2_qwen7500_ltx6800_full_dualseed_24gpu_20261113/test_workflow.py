"""CPU checks for seed isolation, real manifests, scheduling and resume order."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common as c
import run

class WorkflowTests(unittest.TestCase):
 def test_real_manifest_scope_and_reuse(self):
  c.verify();plan=c.read(c.BASE/'PLAN.json');self.assertEqual(plan['total_outputs'],14420);self.assertEqual(plan['new_outputs'],13908)
  for dataset,count in c.COUNTS.items():
   first=c.read(c.work(dataset,137)/'shared_items.json');second=c.read(c.work(dataset,238)/'shared_items.json')
   self.assertEqual(first,second);self.assertEqual(len({x['key'] for x in first}),count)
   for seed in c.SEEDS:
    w=c.work(dataset,seed);p=c.read(w/'protocol.json');keys=[x['key'] for i in range(p['workers']) for x in c.read(w/'shards'/f'{i:02d}.json')]
    self.assertEqual(len(keys),count);self.assertEqual(set(keys),{x['key'] for x in first});self.assertEqual(p['seed'],seed)
    self.assertEqual((w/'conditions').resolve(),c.conditions(dataset).joinpath('conditions').resolve())
  self.assertEqual(c.read(c.work('responsenet',137)/'protocol.json')['datasets']['responsenet']['face_padding'],64)
  self.assertEqual(c.read(c.BASE/'PLAN.json')['datasets']['responsenet']['fps_counts'],{'23.98':23,'24.0':49,'25.0':14,'29.97':429,'30.0':71})

 def test_reused_seed_mismatch_rejected(self):
  w=c.work('seamless',137);p=c.read(w/'protocol.json');x=c.read(w/'shards/00.json')[0];f=w/'generated'/c.ARM/x['key'];m=c.read(f/'metadata.json');h=c.sha(w/'conditions'/x['key']/'qwen/plan.json')
  self.assertTrue(c.valid_media(m,p,x,h))
  for key,value in [('seed',238),('step',6500),('planning_sha256','wrong'),('listener_future_input',True),('planning_adapter',None)]:
   bad=copy.deepcopy(m);bad[key]=value;self.assertFalse(c.valid_media(bad,p,x,h),key)
  self.assertNotEqual(f.resolve(),(c.work('seamless',238)/'generated'/c.ARM/x['key']).resolve())

 def test_source_changes_rejected(self):
  x=c.read(c.dataset_work('seamless')/'shared_items.json')[0];y=copy.deepcopy(x);self.assertTrue(c.same_source(x,y))
  y['reference']['gt_reference']['row_sha256']='different';self.assertFalse(c.same_source(x,y))

 def test_24_slots_use_22_render_and_2_decode_then_expand_decode(self):
  active=[dict(stage='render') for _ in range(22)]
  self.assertIsNone(c.choose_inference(active,100,0,140))
  self.assertEqual(c.choose_inference(active,100,20,140),'decode');active.append(dict(stage='decode'))
  self.assertEqual(c.choose_inference(active,100,20,140),'decode');active.append(dict(stage='decode'))
  self.assertIsNone(c.choose_inference(active,100,20,140))
  self.assertEqual(c.choose_inference([dict(stage='decode')]*2,0,20,140),'decode')
  self.assertIsNone(c.choose_inference([],10,0,80));self.assertEqual(c.choose_inference([],10,1,80),'decode')

 def test_exact_order_no_scoring_and_resume_skip(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);calls=[]
   work=lambda d,s:root/d/f'seed{s}';conditions=lambda d:root/d/'conditioning'
   for d,s in c.ORDER:work(d,s).mkdir(parents=True)
   c.save(work('seamless',137)/'INFERENCE_COMPLETE.json',dict(success=True))
   c.save(conditions('seamless')/'CACHE_COMPLETE.json',dict(success=True))
   def stage(d,s,phase):
    calls.append((d,s,phase))
    if phase=='cache':c.save(conditions(d)/'CACHE_COMPLETE.json',dict(success=True))
    return True
   with patch.multiple(run,BASE=root,work=work,conditions=conditions,verify=lambda:None,activate=lambda:None,stopped=lambda *a:False,stage_loop=stage),patch.object(run.signal,'signal'),patch.object(run.subprocess,'Popen',side_effect=AssertionError('Must not launch GPU in CPU test')):
    run.run_queue('infer')
   self.assertEqual(calls,[('seamless',238,'infer'),('responsenet',137,'cache'),('responsenet',137,'infer'),('responsenet',238,'infer')])
   self.assertTrue((root/'INFERENCE_COMPLETE.json').exists());self.assertFalse((root/'SCORING_COMPLETE.json').exists())

 def test_failure_does_not_start_next_dataset(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);calls=[]
   def stage(d,s,phase):calls.append((d,s,phase));raise RuntimeError('simulated worker failure')
   with patch.multiple(run,BASE=root,work=lambda d,s:root/d/f'seed{s}',conditions=lambda d:root/d/'conditioning',verify=lambda:None,activate=lambda:None,stopped=lambda *a:False,stage_loop=stage),patch.object(run.signal,'signal'):
    with self.assertRaisesRegex(RuntimeError,'simulated'):run.run_queue('infer')
   self.assertEqual(calls,[('seamless',137,'cache')]);self.assertFalse((root/'INFERENCE_COMPLETE.json').exists())

 def test_inference_commit_distinct_from_latent_and_partial_export(self):
  with tempfile.TemporaryDirectory() as tmp:
   w=Path(tmp);item=dict(key='seamless/example');index=c.CompletionIndex(w,[item])
   latent=w/'latents'/c.ARM/item['key'];latent.mkdir(parents=True);(latent/'payload.pt').write_bytes(b'latent')
   c.save(latent/'complete.json',dict(files={'payload.pt':dict(bytes=6)}));self.assertEqual(index.refresh(),(0,1));self.assertEqual(len(index.done),0)
   gen=w/'generated'/c.ARM/item['key'];gen.mkdir(parents=True);(gen/'generated.mp4').write_bytes(b'video')
   self.assertEqual(index.refresh(),(0,1));c.save(gen/'complete.json',dict(files={'generated.mp4':dict(bytes=5)}));self.assertEqual(index.refresh(),(0,0));self.assertEqual(len(index.done),1)

if __name__=='__main__':unittest.main()
