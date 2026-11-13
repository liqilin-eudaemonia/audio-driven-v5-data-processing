import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('handoff',ROOT/'scripts/p2_all_lora_overlay_handoff_20261109.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class HandoffTests(unittest.TestCase):
 def setup_files(self,p,complete=False):
  work=p/'handoff';run=p/'run';watch=p/'watch';chain=p/'chain'
  for d in [work,run,watch,chain]:d.mkdir()
  M.save(work/'PREPARED_NOT_ACTIVATED.json',dict(code={}))
  items=[dict(key='seamless/'+str(i).zfill(3),geometry=dict(native_frames=57))for i in range(200)]
  M.save(run/'shared_items.json',items);M.save(run/'protocol.json',dict(arms=[dict(name='a')]))
  M.save(watch/'CONFIG.json',dict(end_at=M.time.time()+100))
  if complete:
   for i in items:M.save(run/'latents/a'/i['key']/'complete.json',{})
  return work,run,watch,chain
 def test_no_controller_stop_before_render200(self):
  with tempfile.TemporaryDirectory()as tmp:
   w,r,watch,chain=self.setup_files(Path(tmp));(w/'STOP').touch()
   with patch.multiple(M,WORK=w,RUN=r,WATCH=watch,CHAIN=chain),patch.object(M,'stop_owned_controller')as stop:
    M.main();stop.assert_not_called()
 def test_released_foreign_reservation_keeps_original_policy(self):
  with tempfile.TemporaryDirectory()as tmp:
   w,r,watch,chain=self.setup_files(Path(tmp),True)
   with patch.multiple(M,WORK=w,RUN=r,WATCH=watch,CHAIN=chain),patch.object(M.M,'remote',return_value=dict(apps=[])),patch.object(M,'stop_owned_controller')as stop:
    M.main();stop.assert_not_called();self.assertTrue((w/'NOT_NEEDED.json').exists())
 def test_user_stop_after_render_is_respected(self):
  with tempfile.TemporaryDirectory()as tmp:
   w,r,watch,chain=self.setup_files(Path(tmp),True);(r/'STOP').touch();M.save(chain/'status.json',dict(step=5500,stage='evaluation'))
   with patch.multiple(M,WORK=w,RUN=r,WATCH=watch,CHAIN=chain),patch.object(M.M,'remote',return_value=dict(apps=[dict(owned=False,used_mib='72538')])),patch.object(M,'stop_owned_controller')as stop:
    M.main();stop.assert_not_called()

if __name__=='__main__':unittest.main()
