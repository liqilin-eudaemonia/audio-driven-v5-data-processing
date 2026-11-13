import ast,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dispatch_v2',HERE/'manage_p2_full_features.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class DispatchTests(unittest.TestCase):
 def test_only_dispatch_and_root_changed(self):
  old=(HERE/'original_manage_p2_full_features.py').read_text();new=(HERE/'manage_p2_full_features.py').read_text()
  new=new.replace(f"ROOT=Path({str(m.ROOT)!r});sys.path.insert(0,str(ROOT/'src'))","ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))").replace("if len(pending)>=cfg.get('controller_pending_jobs',64):break",'if len(pending)>=8:break')
  new=new.replace("                    if role=='monitor' and (run/'monitor/FINISHED.json').exists():continue\n",'')
  self.assertEqual(new,old)
 def test_pool_stays8_and_pending64_no_duplicate(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp)/'run';fusion=Path(tmp)/'fusion';(r/'locks').mkdir(parents=True);(fusion/'state/fusion').mkdir(parents=True)
   for d in ['state/published','state/media','state/features','state/final','watchdogs','source_index','nodes','alerts']:(r/d).mkdir(parents=True,exist_ok=True)
   ids=[f'j{i:03d}' for i in range(100)]
   m.atomic(r/'config.json',dict(fusion_run=str(fusion),expected_samples=100,nodes={},poll_seconds=0,controller_pending_jobs=64));m.atomic(r/'source_index/COMPLETE.json',{});m.atomic(r/'monitor/FINISHED.json',dict(status='disabled_by_user'));m.atomic(r/'state/published/p.json',dict(jobs=ids,samples=100))
   for j in ids:m.atomic(r/f'state/features/{j}.json',dict(samples=1,worker='test'))
   calls=[];pool_sizes=[];ticks=[]
   class Future:
    def done(self):return False
   class Pool:
    def __init__(self,max_workers):pool_sizes.append(max_workers)
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def submit(self,fn,*args):calls.append((fn.__name__,args));return Future()
   def sleep(_):
    ticks.append(1)
    if len(ticks)==2:(r/'STOP').touch()
   with patch.object(m.concurrent.futures,'ProcessPoolExecutor',Pool),patch.object(m,'ensure_nodes'),patch.object(m,'alive',return_value=False),patch.object(m,'launch',side_effect=AssertionError('Disabled monitor must not restart')),patch.object(m.time,'sleep',sleep),patch.object(m,'ROOT',Path(tmp)):
    m.controller(r)
   self.assertEqual(pool_sizes,[8]);self.assertEqual(len(calls),64);self.assertEqual(len({c[1][1] for c in calls}),64);self.assertTrue(all(c[0]=='audit_job' for c in calls));self.assertEqual(m.read(r/'status.json')['counts']['verified'],0)
 def test_audit_rejects_corruption_and_preserves_checks(self):
  from p2_listen.contracts import digest
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);(r/'locks').mkdir();sid='sample';row=dict(sample_id=sid)
   for part in ['input','teacher','vae','events']:
    folder=r/f'cache/{part}/{sid}';folder.mkdir(parents=True);(folder/'payload.pt').write_bytes(b'original-payload');contract=dict(input=dict(sample_id=sid),part=part);m.atomic(folder/'complete.json',dict(contract=contract,identity=digest(contract),payload_sha256=m.file_sha(folder/'payload.pt')))
   m.atomic(r/'manifests/shards/good.json',[row]);result=m.audit_job(r,'good');self.assertTrue(result['done']);self.assertEqual(result['samples'],1)
   (r/f'cache/teacher/{sid}/payload.pt').write_bytes(b'corrupted');m.atomic(r/'manifests/shards/bad.json',[row])
   with self.assertRaisesRegex(ValueError,'cache audit failed'):m.audit_job(r,'bad')
   self.assertFalse((r/'state/final/bad.json').exists())
if __name__=='__main__':unittest.main()
