from pathlib import Path
import sys,tempfile,copy,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import adapter as A
from p2_listen.contracts import read,atomic_json as save
class Checks(unittest.TestCase):
 def test_prepare_preserves_population_and_conditions(self):
  template=A.TRAIN/'evaluations/event020_feature010_500_seed137';old=read(template/'protocol.json')
  receipt=copy.deepcopy(read(template/'export/RECEIPT.json'));receipt['step']=1661
  cp=A.TRAIN/'runs/event020_feature010/step_001661.pt'
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);save(w/'REQUEST.json',dict(checkpoint=str(cp),samples=64))
   with patch.object(A,'export',return_value=receipt):p=A.prepare(w)
   for k in ['generation','sampling','planner_adapter_identity','shared_items_sha256','datasets']:self.assertEqual(p[k],old[k])
   self.assertEqual(p['arms'][0]['step'],1661);self.assertEqual(p['workers'],28)
   self.assertEqual(read(w/'shared_items.json'),read(template/'shared_items.json'))
   self.assertEqual((w/'conditions').resolve(),(template/'conditions').resolve())
 def test_reject_unapproved_checkpoint(self):
  with self.assertRaises(ValueError):A.select('/tmp/fake/step_001500.pt')
 def test_own_dispatch_modules(self):
  sys.path.insert(0,str(A.WORK))
  import rolling_driver as r
  self.assertEqual(Path(r.D.__file__).parent,A.WORK)
  self.assertEqual(Path(r.original.__file__).parent,A.WORK)
  import rolling_video as v
  self.assertEqual(Path(v.native.SELF).parent,A.WORK)
if __name__=='__main__':unittest.main()
