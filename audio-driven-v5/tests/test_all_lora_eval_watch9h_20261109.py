import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('watch',ROOT/'scripts/p2_all_lora_eval_watch9h_20261109.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)

class WatchTests(unittest.TestCase):
 def test_finished_render_cannot_trigger_slow_decode_restart(self):
  active=dict(status=dict(latents=200,decoded=86,scored_shards=0),recent_render_samples=[dict(key=str(i),frames=57,seconds=50)for i in range(20)])
  baseline={str(i):dict(frames=57,seconds=20)for i in range(20)}
  self.assertEqual(W.phase(active['status']),'decode')
  self.assertEqual(W.matched_slowdown(active,baseline),(False,[]))
 def test_short_and_long_shapes_are_compared_to_same_samples(self):
  active=dict(status=dict(latents=100,decoded=0,scored_shards=0),recent_render_samples=[dict(key=str(i),frames=57,seconds=31)for i in range(20)])
  baseline={str(i):dict(frames=57,seconds=31)for i in range(20)}
  self.assertFalse(W.matched_slowdown(active,baseline)[0])
  for x in active['recent_render_samples']:x['seconds']=50
  self.assertTrue(W.matched_slowdown(active,baseline)[0])
 def test_slowdown_signal_rechecks_phase_before_any_termination(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);fake=dict(work=p,script=p/'own.py',args=[],lock='controller.lock',runs=[p])
   with patch.object(W,'WORK',p),patch.dict(W.MODES,{'test':fake}),patch.object(W,'remote',return_value=dict(apps=[])),patch.object(W,'controller',return_value=dict(pid=123,live=True)),patch.object(W,'run_state',return_value=dict(complete=False,status=dict(latents=200,decoded=86))),patch.object(W.os,'kill')as kill:
    self.assertEqual(W.recover('test','confirmed_slowdown',W.time.time()+100,{})['skipped'],'render_already_finished')
    kill.assert_not_called()
 def test_slowdown_needs_two_sustained_windows(self):
  self.assertFalse(W.slowdown([.32]*20,.31))
  self.assertFalse(W.slowdown([.32]*10+[.7]*10,.31))
  self.assertFalse(W.slowdown([.7]*10,.31))
  self.assertTrue(W.slowdown([.5]*20,.31))
 def test_dead_pid_is_not_same_as_reused_foreign_pid(self):
  self.assertFalse(W.process_identity(99999999,Path('/missing/own.py'))['live'])
  import os
  self.assertFalse(W.process_identity(os.getpid(),Path('/some_other_controller.py'))['live'])
 def test_explicit_stop_prevents_any_probe_or_restart(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);(p/'STOP').touch()
   fake=dict(work=p,script=p/'x.py',args=[],lock='controller.lock',runs=[p])
   with patch.object(W,'WORK',p),patch.dict(W.MODES,{'test':fake}),patch.object(W,'remote',side_effect=AssertionError('must not probe')):
    r=W.recover('test','controller_exited',W.time.time()+100,{})
    self.assertEqual(r['skipped'],'deadline_or_user_stop')
 def test_observation_failure_never_launches_replacement(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);fake=dict(work=p,script=p/'own.py',args=[],lock='controller.lock',runs=[p])
   with patch.object(W,'WORK',p),patch.dict(W.MODES,{'test':fake}),patch.object(W,'remote',side_effect=ConnectionError('SSH timed out')),patch.object(W.subprocess,'Popen')as launch:
    with self.assertRaises(ConnectionError):W.recover('test','controller_exited',W.time.time()+100,{})
    launch.assert_not_called()
 def test_live_controller_not_restarted_for_observation_or_stale_pid_claim(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);fake=dict(work=p,script=p/'own.py',args=[],lock='controller.lock',runs=[p])
   with patch.object(W,'WORK',p),patch.dict(W.MODES,{'test':fake}),patch.object(W,'remote',return_value=dict(apps=[])),patch.object(W,'controller',return_value=dict(pid=123,live=True)),patch.object(W.os,'kill')as kill,patch.object(W.subprocess,'Popen')as launch:
    r=W.recover('test','controller_exited',W.time.time()+100,{})
    self.assertEqual(r['skipped'],'controller_still_live');kill.assert_not_called();launch.assert_not_called()
 def test_confirm_stall_cancels_when_progress_changes(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);(p/'status.json').write_text('{}')
   with patch.object(W,'remote',return_value=dict(gpu={},apps=[])),patch.object(W,'read',side_effect=[dict(latents=10),dict(latents=11)]),patch.object(W.time,'sleep'):
    okay,proof=W.confirm_stall(dict(work=str(p)));self.assertFalse(okay)
 def test_recovery_never_past_window(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=Path(tmp);fake=dict(work=p,script=p/'own.py',args=[],lock='controller.lock',runs=[p])
   with patch.object(W,'WORK',p),patch.dict(W.MODES,{'test':fake}),patch.object(W,'remote',side_effect=AssertionError('no late probe')):
    self.assertEqual(W.recover('test','controller_exited',W.time.time()-1,{})['skipped'],'deadline_or_user_stop')

if __name__=='__main__':unittest.main()
