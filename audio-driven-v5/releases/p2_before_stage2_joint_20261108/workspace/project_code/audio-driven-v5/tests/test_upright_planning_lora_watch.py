"""Fault tests for the user-authorized single-GPU training supervisor."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
spec=importlib.util.spec_from_file_location('upright_watch',ROOT/'scripts/watch_upright_planning_lora.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

@pytest.fixture
def state(tmp_path,monkeypatch):
 monkeypatch.setattr(w,'WORK',tmp_path/'lora');monkeypatch.setattr(w,'FILTER',tmp_path/'filter');monkeypatch.setattr(w,'MON',tmp_path/'monitor')
 w.WORK.mkdir();w.FILTER.mkdir();w.MON.mkdir()
 return dict(at=2000.,training_complete=False,filter_complete=True,gpu=dict(ok=True,memory_mib=10000,utilization=0),training_process=None,smoke_process=None,queue_process=None,training_record={},step=1,metrics_age_s=None)

def test_reused_pid_or_changed_command_never_owned(monkeypatch):
 record=dict(pid=42,start_ticks='5',command=['python',str(w.WORK/'vendor/train_entry.py')])
 info=dict(pid=42,state='S',pgid=42,start_ticks='6',args=record['command'],alive=True)
 monkeypatch.setattr(w,'process',lambda pid:info)
 assert w.owned(record) is None
 info['start_ticks']='5';info['args']=['python','unrelated.py'];assert w.owned(record) is None
 info['args']=record['command'];assert w.owned(record)==info


def test_failed_gpu_observation_cannot_restart_or_stop(state,monkeypatch):
 state['gpu']={'ok':False}
 monkeypatch.setattr(w,'spawn',lambda *args,**kwargs:pytest.fail('must not launch'))
 monkeypatch.setattr(w,'stop_verified_group',lambda *args:pytest.fail('must not stop'))
 assert w.repair(state,[],{},[])['action']=='defer_observation_unavailable'


def test_live_filter_child_prevents_duplicate_filter(state,monkeypatch):
 state.update(filter_complete=False,filter_process=None,filter_worker={'pid':7})
 monkeypatch.setattr(w,'spawn',lambda *args,**kwargs:pytest.fail('must not launch'))
 assert w.repair(state,[],{},[]) is None


def test_brief_zero_utilization_does_not_kill_advancing_training(state,monkeypatch):
 state.update(training_process={'pid':7},training_record={'started_at':1},metrics_age_s=30)
 monkeypatch.setattr(w,'stop_verified_group',lambda *args:pytest.fail('must not stop'))
 history=[dict(at=state['at']-i*30,gpu=dict(ok=True,utilization=0)) for i in reversed(range(6))]
 assert w.repair(state,history,{},[]) is None


def test_active_gpu_never_treated_as_idle_stall(state,monkeypatch):
 state.update(training_process={'pid':7},training_record={'started_at':1},metrics_age_s=900)
 monkeypatch.setattr(w,'stop_verified_group',lambda *args:pytest.fail('must not stop'))
 history=[dict(at=state['at']-i*30,gpu=dict(ok=True,utilization=100)) for i in reversed(range(6))]
 assert w.repair(state,history,{},[]) is None


def test_multiple_confirmed_idle_samples_and_stale_steps_required(state,monkeypatch):
 state.update(training_process={'pid':7},training_record={'started_at':1},metrics_age_s=900)
 calls=[];monkeypatch.setattr(w,'stop_verified_group',lambda r:calls.append(r))
 too_few=[dict(at=state['at']-i*30,gpu=dict(ok=True,utilization=0)) for i in reversed(range(3))]
 assert w.repair(state,too_few,{},[]) is None and not calls
 confirmed=[dict(at=state['at']-i*30,gpu=dict(ok=True,utilization=0)) for i in reversed(range(6))]
 assert w.repair(state,confirmed,{},[])['action']=='confirmed_owned_training_stall_stopped_for_checkpoint_resume';assert len(calls)==1


def test_adopted_child_prevents_duplicate_queue(state,monkeypatch):
 state['gpu']['memory_mib']=4
 monkeypatch.setattr(w,'other_live_inputs',lambda:[{'pid':8}]);monkeypatch.setattr(w,'spawn',lambda *args,**kwargs:pytest.fail('must not launch'))
 assert w.repair(state,[],{},[])['action']=='defer_live_owned_child'


def test_incomplete_or_wrong_step_checkpoint_falls_back(state):
 out=w.WORK/'training';out.mkdir()
 for step in (500,1000,1500):
  p=out/f'checkpoint-{step}';p.mkdir()
  for name in w.CHECKPOINT_FILES:(p/name).write_bytes(b'tensor placeholder')
  (p/'adapter_config.json').write_text(json.dumps(dict(r=32,lora_alpha=64)))
  (p/'trainer_state.json').write_text(json.dumps(dict(global_step=step,log_history=[dict(loss=.1,grad_norm=.5)])))
 (out/'checkpoint-1500/optimizer.pt').unlink()
 (out/'checkpoint-1000/trainer_state.json').write_text(json.dumps(dict(global_step=999)))
 r=w.latest_checkpoint();assert r['valid'] and r['step']==500 and len(r['rejected_newer'])==2


def test_nonfinite_checkpoint_not_resumed(state):
 out=w.WORK/'training/checkpoint-500';out.mkdir(parents=True)
 for name in w.CHECKPOINT_FILES:(out/name).write_bytes(b'placeholder')
 (out/'adapter_config.json').write_text(json.dumps(dict(r=32,lora_alpha=64)))
 (out/'trainer_state.json').write_text(json.dumps(dict(global_step=500,log_history=[dict(loss=float('nan'))])))
 assert not w.latest_checkpoint()['valid']


def test_hourly_estimate_includes_input_wait(state,monkeypatch):
 state.update(step=20,loss_health=dict(finite=True,observed=True),recent_mean_seconds=10.,recent_mean_wall_seconds=30.)
 (w.MON/'reports').mkdir();(w.WORK/'DATA_READY.json').write_text(json.dumps(dict(total_optimizer_steps=100)))
 monkeypatch.setattr(w,'frozen_ok',lambda:True);monkeypatch.setattr(w,'latest_checkpoint',lambda cache:dict(valid=False))
 report=w.full_check(state,1,2000,{},{});assert report['rough_remaining_hours']==pytest.approx(80*30/3600)


def test_worker_warnings_do_not_hide_latest_loss(state):
 logs=w.WORK/'logs';logs.mkdir();(logs/'training.log').write_text("{'loss': 1.2, 'grad_norm': 0.7}\n"+'WARNING: '+('worker output '*50+'\n')*200)
 result=w.losses();assert result['observed'] and result['finite'] and result['latest']==[1.2,.7]
