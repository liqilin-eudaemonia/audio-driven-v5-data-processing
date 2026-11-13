import copy,json,signal
from pathlib import Path
from unittest.mock import patch
import pytest
import common as C
import guardian as G

@pytest.mark.parametrize('flags,ratio,seconds,steps,grace,want',[
 ([],1.,0,0,False,None),(['stalled'],None,0,0,False,'stalled'),
 (['stalled'],None,0,0,True,None),([],1.8,60,50,False,None),
 ([],1.8,200,19,False,None),([],1.8,200,20,False,'sustained_slowdown'),
 (['nonfinite'],1,0,0,False,'numerical_failure')])
def test_recovery_trigger_bounds(flags,ratio,seconds,steps,grace,want):
 assert G.trigger(dict(flags=flags),ratio,seconds,steps,grace)==want

def test_checkpoint_save_skips_speed_restart_but_preserves_failure_recovery():
 assert G.trigger(dict(flags=[],saving=True),2.,999,200)is None
 assert G.trigger(dict(flags=['stalled'],saving=True),2.,999,200)=='stalled'
 assert G.trigger(dict(flags=['nonfinite'],saving=True),2.,999,200)=='numerical_failure'

def snapshot():
 cluster=C.read(C.OUT/'cluster.json');return dict(healthy31=False,nodes=[dict(node=n['name'],cards=[dict(gpu=i,used_mib=1,util=0)for i in n['gpus']],workers=[],foreign=[],errors=[])for n in cluster['nodes']])

def test_ready_allows_known_vllm_and_rejects_other_tasks():
 s=snapshot();assert C.ready(s)
 n=s['nodes'][-1];n['foreign']=[dict(pid=1,allowed_background=True)]
 for c in n['cards']:c['used_mib']=28000
 assert C.ready(s);n['foreign'][0]['allowed_background']=False;assert not C.ready(s)

def test_missing_node_or_query_failure_never_authorizes_cleanup():
 s=snapshot();s['nodes'].pop();assert not C.observed(s)and not C.ready(s)
 s=snapshot();s['nodes'][0]['error']='ssh timeout';assert not C.observed(s)
 assert not G.confirm('confirmed_dead',s,s,dict(flags=[]),None)

def test_active_owned_worker_is_not_ready():
 s=snapshot();s['nodes'][0]['workers']=[dict(pid=1)];assert not C.ready(s)

def test_contention_requires_two_live_samples_of_authorized_shared_compute(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path)
 s=snapshot();s['healthy31']=True;n=next(x for x in s['nodes']if x['node']=='node11')
 n['workers']=[dict(gpu=1),dict(gpu=2)];n['foreign']=[dict(pid=123,allowed_background=True),dict(pid=124,allowed_background=True)]
 state=dict(flags=[],metrics_age_seconds=1)
 rows='1 123 C 30 5\n2 124 C 40 5\n1 123 C 25 5\n2 124 C 35 5\n'
 from types import SimpleNamespace
 with patch.object(G.subprocess,'run',return_value=SimpleNamespace(stdout=rows)):
  assert G.shared_compute(s,state)['active_shared_gpus']==[1,2]
  assert G.shared_compute(s,dict(flags=['stalled'],metrics_age_seconds=200))is None
  assert G.shared_compute(s,dict(flags=['nonfinite'],metrics_age_seconds=1))is None
  assert G.shared_compute(s,dict(flags=[],metrics_age_seconds=61))is None
  n['foreign'][0]['allowed_background']=False;assert G.shared_compute(s,state)is None
 with patch.object(G.subprocess,'run',return_value=SimpleNamespace(stdout='1 123 C 30 5\n2 124 C 40 5\n')):
  assert G.shared_compute(s,state)is None

def test_contention_grace_expires_without_refreshing_measurement(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path)
 C.atomic_json(tmp_path/'SHARED_COMPUTE.json',dict(at=100,step=50,active_shared_gpus=[1,2]))
 s=snapshot();s['healthy31']=True;state=dict(flags=[],metrics_age_seconds=1,step=60)
 with patch.object(G.time,'time',return_value=200):
  assert G.shared_compute(s,state)['at']==100
  assert G.shared_compute(s,dict(state,step=82))is None
  assert G.shared_compute(s,dict(state,flags=['stalled']))is None
 with patch.object(G.time,'time',return_value=281):assert G.shared_compute(s,state)is None

def test_stop_or_deadline_priority_and_finish_transaction():
 with patch.object(C,'stop_intent',return_value=False),patch.object(G,'read',return_value=dict(deadline=0)):
  assert G.cancelled()and not G.cancelled(finishing=True)
 with patch.object(C,'stop_intent',return_value=True):assert G.cancelled(finishing=True)

@pytest.mark.parametrize('change',[None,'pid_reuse','wrong_config','wrong_rank'])
def test_only_exact_owned_process_can_be_signalled(tmp_path,change):
 pid=123;p=tmp_path/str(pid);p.mkdir();script='/owned/train.py';cfg='/owned/config.json'
 argv=[script,'train','--config',cfg if change!='wrong_config'else'/foreign/config.json']
 (p/'cmdline').write_bytes(chr(0).join(argv).encode())
 stat=['S']+['0']*18+['999'if change=='pid_reuse'else'1234']
 (p/'stat').write_text('123 (python) '+' '.join(stat))
 env=dict(RANK='2'if change=='wrong_rank'else'1',WORLD_SIZE='31',CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7')
 (p/'environ').write_bytes(chr(0).join(k+'='+v for k,v in env.items()).encode())
 payload=dict(workers=[dict(pid=pid,rank=1,start_ticks='1234')],script=script,config=cfg,mask=env['CUDA_VISIBLE_DEVICES'])
 code=C.KILL.replace('PAYLOAD',repr(json.dumps(payload))).replace("Path('/proc',",'Path('+repr(str(tmp_path))+',')
 with patch('os.kill')as kill:
  if change:
   with pytest.raises(RuntimeError):exec(code,{})
   kill.assert_not_called()
  else:exec(code,{});kill.assert_called_once_with(pid,signal.SIGTERM)
