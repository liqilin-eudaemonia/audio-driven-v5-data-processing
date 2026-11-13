import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import watch as M
import recover as C
class Safety(unittest.TestCase):
 def test_owned_exact_arm(self):
  argv=['python',str(M.W/'runner.py'),'train','--arm','no_feature']
  self.assertTrue(M.owned({'argv':argv},'no_feature'))
  self.assertFalse(M.owned({'argv':argv},'no_whisper'))
  self.assertFalse(M.owned({'argv':['python','other.py','train','--arm','no_feature']},'no_feature'))
 def test_partial_observation_never_restarts(self):
  with patch.object(M.os,'kill')as k:
   with self.assertRaises(AssertionError):M.recover({},'no_feature',{'observed':False})
   k.assert_not_called()
 def test_stop_preserved(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'STOP').touch()
   with patch.object(M,'W',p),patch.object(M.os,'kill')as k:
    with self.assertRaisesRegex(RuntimeError,'STOP'):M.recover({},'no_feature',{'observed':True})
    k.assert_not_called()
 def test_latest_skips_incomplete(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'step_002500.pt').touch();(p/'step_002000.pt').touch();(p/'step_002000.complete.json').write_text('{}')
   self.assertEqual(C.latest(p),p/'step_002000.pt')
 def test_existing_run_never_reinitialized(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'run.json').write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'no complete checkpoint'):C.latest(p)
 def test_fresh_arm_allowed(self):
  with tempfile.TemporaryDirectory()as d:self.assertIsNone(C.latest(Path(d)))
 def test_pid_reuse_rejected(self):
  with patch.object(M,'record',return_value=dict(pid=42,start='new',cmd='same')):
   self.assertFalse(M.alive(dict(pid=42,start='old',cmd='same')))
 def test_remote_probe_compiles(self):
  compile(M.PROBE.replace('SCRIPT',repr(str(M.W/'runner.py'))),'<remote>','exec')
if __name__=='__main__':unittest.main()
