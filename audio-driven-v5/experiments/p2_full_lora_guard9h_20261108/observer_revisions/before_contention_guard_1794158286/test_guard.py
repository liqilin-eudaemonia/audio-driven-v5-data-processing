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
