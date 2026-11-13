import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_gpu4_gpu7_watch9h_20261112 as W
import p2_lora6000_seamless_full_gpu4_20261112 as E
import p2_seamless_gpu7_assist_20261112 as A
from p2_testsets_eval import runtime as rt

class SchedulingChecks(unittest.TestCase):
 def test_shared_render_claim_is_exclusive_and_released_after_crash(self):
  with tempfile.TemporaryDirectory() as d:
   work=Path(d);items=[dict(key='seamless/'+k,id=k) for k in ['a','b']]
   (work/'shared_items.json').write_text(json.dumps(items));(work/'ready_queue.jsonl').write_text(''.join(json.dumps(dict(key=x['key']))+'\n' for x in items))
   for x in items:
    for p in [work/'conditions'/x['key']/'qwen',work/'conditions'/x['key']/'text',work/'whisper'/x['key']]:p.mkdir(parents=True);(p/'complete.json').touch()
   with patch.object(E,'WORK',work),patch.object(rt,'ready',lambda p:(p/'complete.json').exists()):
    g1=E.ready_claims(work,[],{}, {},'render');g2=E.ready_claims(work,[],{}, {},'render')
    self.assertEqual(next(g1)['id'],'a');self.assertEqual(next(g2)['id'],'b')
    g1.close();g3=E.ready_claims(work,[],{}, {},'render');self.assertEqual(next(g3)['id'],'a');g3.close();g2.close()
 def test_gpu7_waits_for_both_prioritized_weights(self):
  with tempfile.TemporaryDirectory() as d,patch.object(A,'RN',Path(d)):
   w=Path(d);self.assertFalse(A.eligible());(w/'COMPLETE.json').write_text('{"success":true}')
   for s in [6000,6500]:
    p=w/f'step_{s:06d}/FINAL_AUDIT.json';p.parent.mkdir();p.write_text('{"success":true}')
    if s==6000:self.assertFalse(A.eligible())
   self.assertTrue(A.eligible())
 def test_no_recovery_outside_window(self):
  with patch.object(W,'controller',side_effect=AssertionError('no probe')),patch.object(W.subprocess,'Popen') as launch:
   W.recovery('gpu4',dict(end_at=time.time()-1),{});launch.assert_not_called()
 def test_foreign_pid_reuse_not_owned(self):
  self.assertFalse(W.identity(os.getpid(),['this-is-not-an-owned-worker.py'])['live'])
 def test_never_signal_a_reused_pid(self):
  with patch.object(W,'identity',return_value=dict(live=True,start_ticks='other')),patch.object(W.os,'kill') as kill:
   self.assertFalse(W.stop_owned(dict(pid=123,cmd='owned',start_ticks='original')));kill.assert_not_called()
 def test_busy_gpu7_never_borrows(self):
  with tempfile.TemporaryDirectory() as d,patch.object(W,'SEAM',Path(d)),patch.object(W.assist,'eligible',return_value=True),patch.object(W,'children',return_value=[]),patch.object(W.subprocess,'Popen') as launch:
   W.enable_gpu7(dict(end_at=time.time()+100),{7:dict(processes=[dict(pid=999)],free_mib=140000)},{});launch.assert_not_called()
 def test_slowdown_requires_twenty_shape_matched_observations(self):
  state=dict(baseline_seconds_per_frame=dict(task=1),baseline_shapes=dict(task=305))
  data=dict(task=dict(recent=[dict(seconds_per_frame=2,frames=81)]*20))
  self.assertFalse(W.slowdowns(data,state));data['task']['recent']=[dict(seconds_per_frame=2,frames=305)]*20;self.assertTrue(W.slowdowns(data,state))

 def test_three_severe_shape_matched_commits_alert_without_waiting_twenty(self):
  state=dict(baseline_seconds_per_frame=dict(task=.35),baseline_shapes=dict(task=297))
  data=dict(task=dict(recent=[dict(seconds_per_frame=1.8,frames=289)]*3))
  self.assertEqual(W.slowdowns(data,state)[0]['type'],'severe_2x_three_commits')
 def test_moderate_slowdown_still_requires_twenty(self):
  state=dict(baseline_seconds_per_frame=dict(task=1),baseline_shapes=dict(task=305))
  data=dict(task=dict(recent=[dict(seconds_per_frame=1.6,frames=305)]*19))
  self.assertFalse(W.slowdowns(data,state))
  data['task']['recent'].append(dict(seconds_per_frame=1.6,frames=305));self.assertTrue(W.slowdowns(data,state))

if __name__=='__main__':unittest.main()
