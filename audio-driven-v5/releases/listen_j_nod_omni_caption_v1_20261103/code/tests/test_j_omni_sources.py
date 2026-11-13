import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import write,JobQueue,read
from listen_benchmark.j_omni_sources import resolve_nod_receipt,journal_for,validate_item
from listen_benchmark.j_omni import producer_step
class SourceOwnershipTests(unittest.TestCase):
 def plan(self,w,nodes,owners=None):
  p=w/'plan.json';write(p,dict(schema='j-omni-source-ownership-v1',nodes=nodes,delegated_nod_owners=owners or {}))
  return dict(source_ownership=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),nodes={},backlog_sources=100)
 def test_delegated_owner_and_missing_real_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);a=w/'a';b=w/'b';write(a/'nod/completed/j.json',dict(job_id='j',clips=0,status='delegated_branch_not_measurement',owner_work=str(b)))
   cfg=self.plan(w,{}, {'j':str(b)});self.assertIsNone(resolve_nod_receipt(cfg,a,dict(job_id='j',count=3)))
   write(b/'nod/completed/j.json',dict(job_id='j',clips=3,source_result_sha256='hash'))
   self.assertEqual(resolve_nod_receipt(cfg,a,dict(job_id='j',count=3))[0],b)
   with self.assertRaises(ValueError):resolve_nod_receipt({},a,dict(job_id='j',count=3))
 def test_filtered_import_requires_complete_real_queue(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);(w/'ready.jsonl').write_text('abcdef');p=w/'filtered.jsonl';p.write_text('xyz')
   node=dict(old_work=str(w),imported_completed_journal=str(p),journal_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
   q=dict(offset=6,inflight={},failed={});self.assertEqual(journal_for(node,q),(p,3))
   q['inflight']={'unfinished':{}}
   with self.assertRaises(ValueError):journal_for(node,q)
 def test_whitelist(self):
  validate_item(dict(allowed_jobs={'a':2}),dict(job_id='a',count=2))
  for item in [dict(job_id='b',count=2),dict(job_id='a',count=3)]:
   with self.assertRaises(ValueError):validate_item(dict(allowed_jobs={'a':2}),item)
 def test_producer_imports_all_sources_without_gpu_nodes(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);old=w/'old';job=dict(job_id='j',source_result_sha256='hash',rows=[])
   write(old/'jobs/j.json',job);write(old/'nod/completed/j.json',dict(job_id='j',clips=2,source_result_sha256='hash'))
   (old/'ready.jsonl').write_text(json.dumps(dict(job_id='j',count=2))+'\n')
   write(old/'queues/nod/state.json',dict(offset=(old/'ready.jsonl').stat().st_size,inflight={},failed={}))
   cfg=self.plan(w,{'history':dict(old_work=str(old),scope_sources=1,allowed_jobs={'j':2})});write(w/'config.json',cfg)
   state=producer_step(w);self.assertTrue(state['all_sources_published']);self.assertEqual(state['published_clips'],2)
   self.assertEqual(read(w/'job_origins/j.json')['old_work'],str(old))
   self.assertTrue((w/'producer_complete.json').exists())
if __name__=='__main__':unittest.main()
