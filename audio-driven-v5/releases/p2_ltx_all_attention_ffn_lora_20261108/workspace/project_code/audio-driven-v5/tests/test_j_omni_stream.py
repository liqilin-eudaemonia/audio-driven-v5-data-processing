import sys,json,tempfile,threading,time,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import JobQueue,write,read,result_path,acquire
from listen_benchmark.j_omni_stream import CrossSourceStream
from listen_benchmark.j_omni_queued import ready_groups

class StreamTests(unittest.TestCase):
 def fixture(self,w,jobs):
  for jid,ids in jobs:
   rows=[dict(sample_id=s,source=dict(sample_id=s,pair=jid,duration_s=2 if jid=='a' else 1)) for s in ids]
   write(w/'jobs'/f'{jid}.json',dict(job_id=jid,rows=rows,source_result_sha256=jid))
   write(w/'job_origins'/f'{jid}.json',dict(old_work='owner-'+jid,source_result_sha256=jid))
  (w/'ready.jsonl').write_text(''.join(json.dumps(dict(job_id=j))+'\n' for j,_ in jobs))
  write(w/'producer_complete.json',{})
  return JobQueue(w,'caption')
 def stream(self,w,q,pool,stop):
  state=dict(visual_excluded_clips=0,visual_review_clips=0)
  return CrossSourceStream(w,q,'test',dict(pipeline_id='p'),SimpleNamespace(pool=pool),'resources',dict(prefetch=2),None,state,stop,2)
 def terminal(self,w,row):write(result_path(w,'caption',row['sample_id']),dict(source=row['source'],pipeline_id='p',generation_checks_passed=True))
 def test_cross_source_ready_work_does_not_wait_for_slow_source_and_routes_owner(self):
  with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(2) as pool:
   w=Path(d);q=self.fixture(w,[('a',['a1','a2']),('b',['b1'])]);release=threading.Event();owners=[]
   def prepare(row,owner,*_):
    owners.append((row['sample_id'],owner))
    if row['source']['pair']=='a':release.wait(5)
    return row['sample_id']
   with patch('listen_benchmark.j_omni.verified_job',side_effect=read),patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=prepare):
    s=self.stream(w,q,pool,threading.Event())
    try:
     first=next(s);self.assertEqual(first[0]['sample_id'],'b1');self.assertEqual(first[1].result(),'b1');self.terminal(w,first[0]);release.set()
     rest=[]
     for row,f,_ in s:self.assertEqual(f.result(),row['sample_id']);rest.append(row['sample_id']);self.terminal(w,row)
     self.assertEqual(set(rest),{'a1','a2'});self.assertEqual(read(q.path)['completed'],2)
     self.assertEqual(set(owners),{('a1','owner-a'),('a2','owner-a'),('b1','owner-b')})
    finally:release.set();s.close()
 def test_cancel_drains_cpu_before_releasing_source_lock(self):
  with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(2) as pool,ThreadPoolExecutor(1) as closer:
   w=Path(d);q=self.fixture(w,[('a',['a1']),('b',['b1'])]);release=threading.Event();started=threading.Event()
   def prepare(row,*_):
    if row['sample_id']=='a1':started.set();release.wait(5)
    return row['sample_id']
   with patch('listen_benchmark.j_omni.verified_job',side_effect=read),patch('listen_benchmark.j_omni_stream.prepare_call',side_effect=prepare):
    s=self.stream(w,q,pool,threading.Event());next(s);self.assertTrue(started.wait(1))
    ending=closer.submit(s.close);time.sleep(.05)
    self.assertFalse(ending.done());self.assertIsNone(acquire(q.folder/'claims/a.lock',False))
    release.set();ending.result(timeout=2);h=acquire(q.folder/'claims/a.lock',False);self.assertIsNotNone(h);h.close()
 def test_failure_charges_only_the_affected_source(self):
  with tempfile.TemporaryDirectory() as d,ThreadPoolExecutor(2) as pool:
   w=Path(d);q=self.fixture(w,[('a',['a1']),('b',['b1'])])
   with patch('listen_benchmark.j_omni.verified_job',side_effect=read),patch('listen_benchmark.j_omni_stream.prepare_call',return_value={}):
    s=self.stream(w,q,pool,threading.Event());next(s);s.record_failure('b1',ValueError('bad b'));s.close()
   self.assertEqual(read(q.path)['attempts'],{'b':1});self.assertEqual(read(q.path)['completed'],0)
 def test_ready_mailbox_close_cancels_infinite_cross_source_iterator(self):
  class Source:
   def __init__(self):self.cancel=threading.Event();self.first=True
   def __iter__(self):return self
   def __next__(self):
    if self.first:self.first=False;return 1
    self.cancel.wait(2);raise StopIteration
   def cancel_pending(self):self.cancel.set()
  source=Source();g=ready_groups(source,4,threading.Event());self.assertEqual(next(g),[1]);t=time.monotonic();g.close();self.assertLess(time.monotonic()-t,.5)
if __name__=='__main__':unittest.main()
