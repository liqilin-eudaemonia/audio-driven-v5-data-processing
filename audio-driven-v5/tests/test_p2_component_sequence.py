"""The queue may hand off only after6000 save, process exit and verification."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest
from p2_listen.contracts import ROOT,atomic_json,read

def module():
 path=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence.py'
 spec=importlib.util.spec_from_file_location('component_sequence_test',path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_boundary_never_accepts_late_or_early_step():
 m=module();assert not m.boundary(5999);assert m.boundary(6000)
 with pytest.raises(RuntimeError):m.boundary(6001)

def test_each_launch_is_fresh_and_stops_exactly6000():
 m=module()
 for arm in m.ARMS:
  cmd=m.arm_command(arm,Path('/tmp/sequence-test'))
  assert cmd[cmd.index('--stop-after')+1]=='6000'and '--resume'not in cmd
  assert cmd[cmd.index('--config')+1]==str(m.HERE/arm/'config.json')

@pytest.mark.parametrize('change',[lambda v:v.update(step=5999),lambda v:v['optimizer']['state'][0].update(step=5999),lambda v:v.update(rank_rng=[{}]),lambda v:v.update(identity={})])
def test_reject_incomplete_or_mismatched_state(change):
 m=module();identity={'world':31};v=dict(step=6000,identity=identity.copy(),rank_rng=[{}]*31,optimizer={'state':{0:{'step':6000}}})
 m.verified_state(v,identity);change(v)
 with pytest.raises(ValueError):m.verified_state(v,identity)

def test_parent_handoff_order_and_failed_first_arm_blocks_next(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m,'HERE',tmp_path);parent=tmp_path/'parent';parent.mkdir();monkeypatch.setattr(m,'PARENT',parent)
 atomic_json(parent/'status.json',{'step':6000});atomic_json(parent/'CONTROLLER.json',{'pid':123,'start_ticks':'1'})
 order=[]
 monkeypatch.setattr(m,'validate_ready',lambda:None)
 monkeypatch.setattr(m,'await_exit',lambda *a:order.append('exit'))
 def verify(*args):
  assert read(parent/'STOP')['target_step']==6000;order.append('verify');return {'success':True}
 monkeypatch.setattr(m,'verify_checkpoint',verify)
 monkeypatch.setattr(m,'wait_idle',lambda *args:order.append('idle'))
 monkeypatch.setattr(m.subprocess,'Popen',Mock(side_effect=RuntimeError('simulated first launch failure')))
 (tmp_path/'no_whisper').mkdir()
 with pytest.raises(RuntimeError,match='simulated'):m.run()
 assert order==['exit','verify','idle']
 assert m.subprocess.Popen.call_count==1
 assert not (tmp_path/'no_feature/SEQUENCE_LAUNCH_COMMAND.json').exists()

def test_user_queue_stop_blocks_action(tmp_path,monkeypatch):
 m=module();monkeypatch.setattr(m,'HERE',tmp_path);(tmp_path/'QUEUE_STOP').touch()
 with pytest.raises(RuntimeError,match='queue stop'):m.cancelled()

def v2module():
 path=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence_v2.py'
 spec=importlib.util.spec_from_file_location('component_sequence_v2_test',path)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_v2_event_first_and_correct_training_entry():
 m=v2module();assert m.ARMS==('av_event','no_whisper','no_feature','no_event')
 for arm in m.ARMS:
  cmd=m.arm_command(arm,Path('/tmp/queue-test'))
  expected='p2_current500h_av_event_only.py'if arm=='av_event'else'p2_current500h_component_ablations.py'
  assert Path(cmd[1]).name==expected and '--resume'not in cmd
  assert cmd[cmd.index('--stop-after')+1]=='6000'

@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_v2_full_queue_requires_save_exit_verify_before_every_next_launch(tmp_path,monkeypatch,version):
 import os
 if version=='v2':m=v2module()
 else:
  path=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence_v3.py'
  spec=importlib.util.spec_from_file_location('component_sequence_v3_test',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  assert m.ARMS==('av_event',)
 monkeypatch.setattr(m,'HERE',tmp_path)
 parent=tmp_path/'parent';parent.mkdir();monkeypatch.setattr(m,'PARENT',parent)
 atomic_json(parent/'status.json',{'step':6000});atomic_json(parent/'CONTROLLER.json',{})
 for arm in m.ARMS:(tmp_path/arm).mkdir()
 calls=[]
 monkeypatch.setattr(m,'validate_ready',lambda:None)
 monkeypatch.setattr(m,'await_exit',lambda folder,*args:calls.append(('exit',folder.name)))
 monkeypatch.setattr(m,'verify_checkpoint',lambda folder,kind:calls.append(('verify',folder.name))or{'success':True})
 monkeypatch.setattr(m,'wait_idle',lambda folder:calls.append(('idle',folder.name)))
 def launch(cmd,**kwargs):
  arm=Path(cmd[cmd.index('--config')+1]).parent.name;calls.append(('launch',arm))
  return Mock(pid=os.getpid(),poll=lambda:0,returncode=0)
 monkeypatch.setattr(m.subprocess,'Popen',launch)
 m.run()
 expected=[('exit','parent'),('verify','parent')]
 for arm in m.ARMS:expected.extend([('idle',arm),('launch',arm),('exit',arm),('verify',arm)])
 assert calls==expected and read(tmp_path/'SEQUENCE_FINISHED.json')['success']
