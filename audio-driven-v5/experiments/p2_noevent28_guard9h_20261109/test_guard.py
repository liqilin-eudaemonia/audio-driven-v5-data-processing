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
 cluster=C.read(C.OUT/'cluster.json');return dict(healthy28=False,nodes=[dict(node=n['name'],cards=[dict(gpu=i,used_mib=1,util=0)for i in n['gpus']],workers=[],foreign=[],errors=[])for n in cluster['nodes']])

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

def test_context_presence_is_distinct_from_measured_active_compute(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path)
 s=snapshot();s['healthy28']=True;n=next(x for x in s['nodes']if x['node']=='node11')
 n['workers']=[dict(gpu=1),dict(gpu=2)];n['foreign']=[dict(pid=123,allowed_background=True),dict(pid=124,allowed_background=True)]
 state=dict(flags=[],metrics_age_seconds=1)
 rows='1 123 C 30 5\n2 124 C 40 5\n1 123 C 25 5\n2 124 C 35 5\n'
 from types import SimpleNamespace
 with patch.object(G.subprocess,'run',return_value=SimpleNamespace(stdout=rows)):
  assert G.shared_compute(s,state)['active_shared_gpus']==[1,2]
  assert G.shared_compute(s,dict(flags=['stalled'],metrics_age_seconds=200))is None
  assert G.shared_compute(s,dict(flags=['nonfinite'],metrics_age_seconds=1))is None
  assert G.shared_compute(s,dict(flags=[],metrics_age_seconds=61))is None
  n['foreign'][0]['allowed_background']=False;assert G.shared_compute(s,state)['evidence_kind']=='authorized_gpu_context_presence'
 with patch.object(G.subprocess,'run',return_value=SimpleNamespace(stdout='1 123 C 30 5\n2 124 C 40 5\n')):
  assert G.shared_compute(s,state)['evidence_kind']=='authorized_gpu_context_presence'

def test_sampling_timeout_never_proves_absence_of_context_contention(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path)
 s=snapshot();s['healthy28']=True;n=next(x for x in s['nodes']if x['node']=='node11')
 n['workers']=[dict(gpu=1)];n['foreign']=[dict(pid=123,allowed_background=True)]
 state=dict(flags=[],metrics_age_seconds=1,step=10)
 with patch.object(G.subprocess,'run',side_effect=G.subprocess.TimeoutExpired('pmon',20)):
  result=G.shared_compute(s,state)
  assert result['evidence_kind']=='authorized_gpu_context_presence'and result['context_pids']==[123]
  assert 'pmon_error'in result
  assert G.shared_compute(s,dict(state,flags=['stalled']))is None
  s['healthy28']=False;assert G.shared_compute(s,state)is None

def test_contention_grace_expires_without_refreshing_measurement(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path)
 C.atomic_json(tmp_path/'SHARED_COMPUTE.json',dict(at=100,step=50,active_shared_gpus=[1,2]))
 s=snapshot();s['healthy28']=True;state=dict(flags=[],metrics_age_seconds=1,step=60)
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
 env=dict(RANK='2'if change=='wrong_rank'else'1',WORLD_SIZE='28',CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7')
 (p/'environ').write_bytes(chr(0).join(k+'='+v for k,v in env.items()).encode())
 payload=dict(workers=[dict(pid=pid,rank=1,start_ticks='1234')],script=script,config=cfg,mask=env['CUDA_VISIBLE_DEVICES'])
 code=C.KILL.replace('PAYLOAD',repr(json.dumps(payload))).replace("Path('/proc',",'Path('+repr(str(tmp_path))+',')
 with patch('os.kill')as kill:
  if change:
   with pytest.raises(RuntimeError):exec(code,{})
   kill.assert_not_called()
  else:exec(code,{});kill.assert_called_once_with(pid,signal.SIGTERM)

def test_reserved_gpu_foreign_task_is_ignored(monkeypatch):
 s=C.read(C.WORK/'INITIAL_CLUSTER.json');raw=[]
 for n in s['nodes']:
  g=copy.deepcopy(n['cards']);foreign=[];errors=[]
  if n['node']=='node11':
   g.append(dict(gpu=4,uuid='reserved_gpu',used_mib=60000,free_mib=50000,util=100));foreign=[dict(pid=991,uuid='reserved_gpu',allowed_vllm=False)];errors=[dict(foreign_pid=991)]
  raw.append(dict(node=n['node'],gpus=g,workers=n['workers'],foreign=foreign,errors=errors))
 monkeypatch.setattr(C.C,'probe',lambda arm:dict(nodes=raw))
 result=C.probe();assert result['healthy28']and all(not n['foreign']for n in result['nodes'])

@pytest.mark.parametrize('alive,age,saving,complete,want',[(False,31,True,False,'confirmed_dead'),(False,31,False,False,'confirmed_dead'),(False,10,True,False,None),(True,900,True,False,None),(False,500,False,True,None)])
def test_launcher_exit_during_save(alive,age,saving,complete,want):
 assert G.owner_exit_reason(dict(metrics_age_seconds=age,saving=saving,complete=complete),alive)==want

def test_recovery_keeps_small_verified_vllm_on_node4():
 s=snapshot();n=next(n for n in s['nodes']if n['node']=='node4');n['foreign']=[dict(pid=123,allowed_background=True)]
 for card in n['cards']:card['used_mib']=21000;card['util']=100
 assert C.ready(s)
 n['foreign'][0]['allowed_background']=False;assert not C.ready(s)
 n['foreign'][0]['allowed_background']=True;n['cards'][0]['used_mib']=35000;assert not C.ready(s)

def test_node4_competition_is_detected_without_node11_vllm(tmp_path,monkeypatch):
 monkeypatch.setattr(G,'WORK',tmp_path);s=snapshot();s['healthy28']=True;n=next(n for n in s['nodes']if n['node']=='node4')
 n['workers']=[dict(gpu=0),dict(gpu=1)];n['foreign']=[dict(pid=123,allowed_background=True),dict(pid=124,allowed_background=True)]
 from types import SimpleNamespace
 with patch.object(G.subprocess,'run',return_value=SimpleNamespace(stdout='0 123 C 30 5\n1 124 C 40 5\n0 123 C 25 5\n1 124 C 35 5\n')):
  result=G.shared_compute(s,dict(flags=[],metrics_age_seconds=1,step=2000));assert result['node']=='node4'and result['active_shared_gpus']==[0,1]

def test_failed_recovery_alerts_before_retry_limit(tmp_path):
 from agent_errors import pending_error
 (tmp_path/'RECOVERY_ERROR.json').write_text(json.dumps(dict(at=2,error='GPU startup failed')))
 assert pending_error(tmp_path)['file']=='RECOVERY_ERROR.json'
 (tmp_path/'RECOVERY_RESULT.json').write_text(json.dumps(dict(at=3,restarted=True)))
 assert pending_error(tmp_path)is None

def test_success_receipt_does_not_hide_newer_failure(tmp_path):
 from agent_errors import pending_error
 (tmp_path/'RECOVERY_RESULT.json').write_text(json.dumps(dict(at=1,restarted=True)))
 (tmp_path/'RECOVERY_ERROR.json').write_text(json.dumps(dict(at=2,error='new error')))
 assert pending_error(tmp_path)['file']=='RECOVERY_ERROR.json'

@pytest.mark.parametrize('execute',[False,True])
def test_pause_plan_never_signals_training_or_reserved_gpu(execute):
 import pause_competitors as P
 payload=dict(gpus=[0,1,2,3],execute=execute,records=[dict(pid=100,start_ticks='777',uuid='selected')])
 outputs=['0, selected\n7, reserved\n','11, selected\n100, selected\n300, reserved\n']
 def contents(path,*a,**kw):
  if path.name=='comm':return 'VLLM::EngineCore'if path.parent.name in ['100','300']else'python'
  if path.name=='stat':return '100 (engine) S '+'0 '*18+'777 0'
  raise AssertionError(str(path))
 with patch.object(P.subprocess,'check_output',side_effect=outputs),patch.object(Path,'read_text',contents),patch.object(__import__('os'),'kill')as kill:
  exec(P.REMOTE.replace('PAYLOAD',repr(json.dumps(payload))),{})
  if execute:kill.assert_called_once_with(100,signal.SIGSTOP)
  else:kill.assert_not_called()

def test_pause_rejects_reused_pid():
 import pause_competitors as P
 payload=dict(gpus=[0],execute=True,records=[dict(pid=100,start_ticks='OLD',uuid='selected')])
 def contents(path,*a,**kw):return 'VLLM::EngineCore'if path.name=='comm'else'100 (engine) S '+'0 '*18+'NEW 0'
 with patch.object(P.subprocess,'check_output',side_effect=['0, selected\n','100, selected\n']),patch.object(Path,'read_text',contents),patch.object(__import__('os'),'kill')as kill:
  with pytest.raises(RuntimeError):exec(P.REMOTE.replace('PAYLOAD',repr(json.dumps(payload))),{})
  kill.assert_not_called()

def test_pause_execution_requires_explicit_user_approval():
 import pause_competitors as P
 with patch.object(P.C,'read',return_value=dict(user_approved=False)),patch.object(P.subprocess,'run')as run:
  with pytest.raises(AssertionError):P.main(True)
  run.assert_not_called()

def test_pause_rejects_one_process_spanning_reserved_gpu():
 import pause_competitors as P
 payload=dict(gpus=[0],execute=True,records=[dict(pid=100,start_ticks='777',uuid='selected')])
 def contents(path,*a,**kw):return 'VLLM::EngineCore'if path.name=='comm'else'100 (engine) S '+'0 '*18+'777 0'
 with patch.object(P.subprocess,'check_output',side_effect=['0, selected\n7, reserved\n','100, selected\n100, reserved\n']),patch.object(Path,'read_text',contents),patch.object(__import__('os'),'kill')as kill:
  with pytest.raises(RuntimeError,match='reserved GPU'):exec(P.REMOTE.replace('PAYLOAD',repr(json.dumps(payload))),{})
  kill.assert_not_called()
