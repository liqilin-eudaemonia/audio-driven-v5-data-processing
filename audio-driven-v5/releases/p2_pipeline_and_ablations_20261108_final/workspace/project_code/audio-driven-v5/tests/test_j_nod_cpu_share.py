import importlib.util,json,socket,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import listen_j_nod_cpu_share as share
from listen_benchmark.j_annotation import JobQueue,write
class CPUSharingTests(unittest.TestCase):
 def config(self,w):
  write(w/'config.json',{'nodes':{'test':{'hostname':socket.gethostname(),'resource_work':str(w/'resource')}}})
  write(w/'nod_cpu_share/config.json',{'source_work':str(w/'source'),'python':sys.executable,'nodes':{'test':{'workers':2,'max_workers':4,'slot_base':10000}}})
 def test_resource_only_redirect(self):
  import listen_j_annotation_nod as old
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);self.config(w)
   def run():
    self.assertEqual(sys.argv[-4:],['--work',str(w/'source'),'--slot','10001'])
    old.annotate('row','face','nod','params','model','intervals',w/'source',fast_crop=True)
   saved=old.annotate
   try:
    with patch.object(old,'annotate') as annotate,patch.object(old,'main',side_effect=run),patch.object(sys,'argv',[]):
     share.borrowed_worker(w,'test',10001)
     annotate.assert_called_once_with('row','face','nod','params','model','intervals',w/'resource',fast_crop=True)
   finally:old.annotate=saved
 def test_reject_original_slots(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);self.config(w)
   with self.assertRaises(ValueError):share.borrowed_worker(w,'test',1)
 def test_reject_wrong_host(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);self.config(w)
   with patch.object(socket,'gethostname',return_value='wrong'):
    with self.assertRaises(ValueError):share.settings(w,'test')
 def test_claim_recovery_and_single_completion(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);(w/'ready.jsonl').write_text(json.dumps({'job_id':'source','count':2})+'\n')
   a=JobQueue(w,'nod');b=JobQueue(w,'nod');jid,handle=a.claim('original')
   self.assertIsNone(b.claim('borrowed'))
   handle.close();jid,newhandle=b.claim('borrowed')
   b.finish(jid,receipt={'clips':2});newhandle.close()
   self.assertIsNone(a.claim('original'))
   self.assertEqual(json.loads(a.path.read_text())['completed_clips'],2)
if __name__=='__main__':unittest.main()
