"""CPU-only admission, allocation and native concurrent claim/recovery tests."""
import ast,collections,copy,importlib.util,json,multiprocessing as mp,os,socket,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('cluster29',Path(__file__).with_name('cluster29.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def consume(work,protocol,crash=False):
 from p2_testsets_eval.shared_items import claim_items
 from p2_testsets_eval.runtime import complete
 for item in claim_items(Path(work),[],protocol,{'name':'a'},'render'):
  if crash:os._exit(9)
  folder=Path(work)/'latents/a'/item['key'];folder.mkdir(parents=True,exist_ok=True)
  # O_EXCL proves no two consumers compute the same completed item.
  with (folder/'payload').open('x') as f:f.write(item['key'])
  time.sleep(.01);complete(folder,[folder/'payload'])

class Checks(unittest.TestCase):
 def test_mapping_and_dispatch(self):
  c=m.cfg();mapping=m.slots(c);self.assertEqual(len(mapping),29)
  self.assertEqual([s['gpu'] for s in mapping if s['node']=='node11'],list(range(5)))
  bad=copy.deepcopy(c);bad['nodes'][-1]['gpus'].append(7)
  with self.assertRaises(AssertionError):m.slots(bad)
  for rank in [0,8,16,28]:
   for stage in ['render','decode','metric']:
    cmd=m.command(stage,rank,m.STEPS[0]);self.assertIn('CUDA_VISIBLE_DEVICES='+str(mapping[rank]['gpu']),cmd)
 def test_admission(self):
  n={'name':'node11','gpus':[4]};r={'gpus':{'4':{'uuid':'x','used':90000}},'apps':[{'uuid':'x','state':'S','owned':True,'pid':1}]}
  self.assertFalse(m.blockers(n,r,True));self.assertTrue(m.blockers(n,r,False))
  r['apps'].append({'uuid':'x','state':'S','owned':False,'pid':2})
  self.assertTrue(m.blockers(n,r,True))
  r['apps']=[];self.assertTrue(m.blockers(n,r,True))
 def test_balanced_initial_allocation(self):
  state={s:{'remaining':64,'cost':1000} for s in m.STEPS};active={}
  for rank in range(29):
   step=m.choose_render(rank,state,active);self.assertIsNotNone(step);active[rank]={'stage':'render','step':step}
  counts=collections.Counter(j['step'] for j in active.values())
  self.assertEqual(set(counts),set(m.STEPS));self.assertLessEqual(max(counts.values()),2)
 def test_handoff_marker_ownership(self):
  with tempfile.TemporaryDirectory() as td,patch.object(m,'WORK',Path(td)),patch.object(m.base,'child',lambda step:Path(td)/str(step)):
   m.handoff_stops(True)
   self.assertTrue(all((Path(td)/str(s)/'STOP').exists() for s in m.STEPS))
   m.handoff_stops();self.assertFalse((Path(td)/'STOP').exists())
   foreign=Path(td)/str(m.STEPS[-1])/'STOP';m.save(foreign,{'owner':'other'})
   with self.assertRaises(AssertionError):m.handoff_stops(True)
   self.assertFalse((Path(td)/'STOP').exists());m.handoff_stops();self.assertTrue(foreign.exists())
 def test_native_concurrent_claim_and_crash_resume(self):
  from p2_testsets_eval.runtime import complete,ready
  with tempfile.TemporaryDirectory() as td:
   w=Path(td);items=[{'key':f's/{i}','geometry':{'native_frames':100-i}} for i in range(24)]
   m.save(w/'items.json',items);p={'shared_item_queue':True,'shared_items_manifest':str(w/'items.json'),'shared_items_sha256':m.sha(w/'items.json'),'total_samples':24}
   pre=w/'latents/a'/items[0]['key'];pre.mkdir(parents=True);(pre/'payload').write_text('preexisting');complete(pre,[pre/'payload'])
   ctx=mp.get_context('fork');failed=ctx.Process(target=consume,args=(td,p,True));failed.start();failed.join(15);self.assertEqual(failed.exitcode,9)
   jobs=[ctx.Process(target=consume,args=(td,p)) for _ in range(4)]
   for j in jobs:j.start()
   for j in jobs:j.join(30);self.assertEqual(j.exitcode,0)
   self.assertTrue(all(ready(w/'latents/a'/x['key']) for x in items));self.assertEqual((pre/'payload').read_text(),'preexisting')
   consume(td,p) # resume does no work; exclusive writes would fail otherwise
 def test_frozen_inputs_and_card_receipts(self):
  self.assertEqual(m.verify()['total_gpus'],29)
  self.assertFalse((m.WORK/'STOP').exists())
  tree=ast.parse(m.SELF.read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
  expr=next(n.value for n in worker.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets))
  for slot in m.slots(m.cfg()):
   receipt=eval(compile(ast.Expression(expr),str(m.SELF),'eval'),dict(time=time,os=os,socket=socket,s=slot,stage='render',step=5600))
   self.assertEqual(receipt['host'],slot['host']);self.assertEqual(receipt['hostname'],socket.gethostname())

if __name__=='__main__':unittest.main(verbosity=2)
