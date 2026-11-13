import sys,time,threading,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.j_omni_prefetch import prefetch_stream

class PrefetchTests(unittest.TestCase):
 def test_ready_first_keeps_identity_and_does_not_wait_for_long_head(self):
  release=threading.Event();slow_started=threading.Event();stop=threading.Event()
  def task(i):
   if i==0:slow_started.set();release.wait(5)
   return 'value-'+str(i)
  with ThreadPoolExecutor(max_workers=2) as pool:
   stream=prefetch_stream([0,1,2],lambda i:pool.submit(task,i),2,stop,True)
   try:
    i,f,_=next(stream);self.assertTrue(slow_started.is_set());self.assertEqual(i,1);self.assertEqual(f.result(),'value-1')
    release.set();rest=[(i,f.result()) for i,f,_ in stream];self.assertEqual(sorted(rest),[(0,'value-0'),(2,'value-2')])
   finally:release.set();stream.close()
 def test_ordered_baseline_preserves_input_order(self):
  with ThreadPoolExecutor(max_workers=3) as pool:
   stream=prefetch_stream(range(7),lambda i:pool.submit(lambda x:x*x,i),2,threading.Event(),False)
   self.assertEqual([(i,f.result()) for i,f,_ in stream],[(i,i*i) for i in range(7)])
 def test_close_drains_source_before_caller_releases_lock(self):
  release=threading.Event();started=threading.Event();ended=threading.Event()
  def work(i):
   if i==1:started.set();release.wait(5);ended.set()
   return i
  with ThreadPoolExecutor(max_workers=2) as pool:
   stream=prefetch_stream([0,1],lambda i:pool.submit(work,i),2,threading.Event(),False)
   _,f,_=next(stream);f.result();self.assertTrue(started.wait(1))
   closer=threading.Thread(target=stream.close);closer.start();time.sleep(.03);self.assertTrue(closer.is_alive());release.set();closer.join(2);self.assertFalse(closer.is_alive());self.assertTrue(ended.is_set())
 def test_exceptions_keep_their_own_sample_identity(self):
  def fn(i):
   if i==2:raise ValueError('sample 2')
   return i
  with ThreadPoolExecutor(max_workers=2) as pool:
   stream=prefetch_stream(range(4),lambda i:pool.submit(fn,i),2,threading.Event(),True);ok=[];bad=[]
   for i,f,_ in stream:
    try:ok.append((i,f.result()))
    except ValueError as e:bad.append((i,str(e)))
   self.assertEqual(sorted(ok),[(0,0),(1,1),(3,3)]);self.assertEqual(bad,[(2,'sample 2')])
 def test_close_also_drains_yielded_but_unconsumed_future(self):
  release=threading.Event();started=threading.Event()
  def fn(i):started.set();release.wait(5);return i
  with ThreadPoolExecutor(max_workers=1) as pool:
   stream=prefetch_stream([0],lambda i:pool.submit(fn,i),1,threading.Event(),False)
   next(stream);self.assertTrue(started.wait(1));closer=threading.Thread(target=stream.close);closer.start();time.sleep(.03)
   self.assertTrue(closer.is_alive());release.set();closer.join(2);self.assertFalse(closer.is_alive())
if __name__=='__main__':unittest.main()
