import copy,os,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest
import policy,remote

@pytest.mark.parametrize('chain,complete,protected,owner',[
 ([dict(cwd='/private/liqi_workspace/project_code',task_paths=[])],True,False,'owned'),
 ([dict(cwd='/tmp',task_paths=['/private/liqi_workspace/project_code/train.py'])],True,False,'owned'),
 ([dict(cwd='/tmp',task_paths=[]),dict(cwd='/private/liqi_workspace',task_paths=[])],True,False,'owned'),
 ([dict(cwd='/other',task_paths=[])],True,False,'foreign'),
 ([dict(cwd='/other',task_paths=[])],False,False,'unknown'),
 ([dict(cwd='/other',error='PermissionError')],True,False,'unknown'),
 ([],False,True,'owned'),
 ([dict(cwd='/other',comm='nvidia-cuda-mps-server')],True,False,'infrastructure'),
])
def test_ownership_evidence(chain,complete,protected,owner):
 assert policy.classify(chain,'/private/liqi_workspace',complete,protected)[0]==owner

def test_workspace_python_and_model_are_not_task_ownership(tmp_path):
 v=tmp_path/'workspace/.venv/bin';v.mkdir(parents=True);exe=v/'python';exe.touch();model=tmp_path/'workspace/model.safetensors';model.touch()
 assert policy.task_paths([str(exe),'--model',str(model)],'/tmp')==[]
 assert not policy.inside(str(tmp_path/'workspace_other'),str(tmp_path/'workspace'))

def start_external(tmp_path):
 workspace=tmp_path/'workspace';workspace.mkdir(exist_ok=True)
 script=tmp_path/'outside.py';script.write_text('import time\ntime.sleep(60)\n')
 p=subprocess.Popen([sys.executable,str(script)],cwd=tmp_path,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 return p,dict(gpus=[0],workspace_root=str(workspace),protected=[])

def nvml(pid,reserved=False):
 def output(cmd,**kw):
  if '--query-gpu=index,uuid'in cmd:return '0, selected\n7, reserved\n'
  return f'{pid}, selected\n'+(f'{pid}, reserved\n'if reserved else'')
 return output

def test_real_foreign_process_terminates(tmp_path):
 p,x=start_external(tmp_path)
 try:
  with patch.object(remote.subprocess,'check_output',side_effect=nvml(p.pid)):
   scan=remote.scan(x);assert scan['records'][0]['owner']=='foreign'
   result=remote.act(dict(x,expected=scan['records']))
   assert result['results'][0]['sigterm'];assert p.wait(timeout=5)==-15
 finally:
  if p.poll()is None:p.kill();p.wait()

@pytest.mark.parametrize('case',['protected','reserved','pid_reuse','now_owned'])
def test_real_process_is_preserved_when_not_confirmed_foreign(tmp_path,case):
 p,x=start_external(tmp_path)
 try:
  with patch.object(remote.subprocess,'check_output',side_effect=nvml(p.pid,case=='reserved')):
   before=remote.scan(x);record=copy.deepcopy(before['records'][0])
   if case=='protected':x['protected']=[dict(pid=p.pid,start_ticks=record['start_ticks'])]
   elif case=='pid_reuse':record['start_ticks']='wrong-start'
   elif case=='now_owned':x['workspace_root']=str(tmp_path)
   if case=='reserved':assert record['owner']=='unknown'
   else:record['owner']='foreign'
   r=remote.act(dict(x,expected=[record]));assert p.poll()is None
   assert not any(v.get('sigterm')for v in r['results'])
 finally:
  p.terminate();p.wait(timeout=5)

@pytest.mark.parametrize('move_into_workspace',[False,True])
def test_force_kill_rechecks_attribution(tmp_path,move_into_workspace):
 workspace=tmp_path/'workspace';workspace.mkdir();flag=tmp_path/'ready';script=tmp_path/'outside.py'
 handler=f'lambda *a: os.chdir({str(workspace)!r})'if move_into_workspace else'signal.SIG_IGN'
 script.write_text(f'import os,signal,time\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,{handler})\nPath({str(flag)!r}).touch()\ntime.sleep(60)\n')
 p=subprocess.Popen([sys.executable,str(script)],cwd=tmp_path,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 x=dict(gpus=[0],workspace_root=str(workspace),protected=[])
 try:
  deadline=time.time()+5
  while not flag.exists():assert time.time()<deadline;time.sleep(.02)
  with patch.object(remote.subprocess,'check_output',side_effect=nvml(p.pid)):
   record=remote.scan(x)['records'];assert record[0]['owner']=='foreign';result=remote.act(dict(x,expected=record))
   if move_into_workspace:assert p.poll()is None and result['results'][0]['force_skipped']=='ownership_changed'
   else:assert p.wait(timeout=5)==-9 and result['results'][0]['sigkill']
 finally:
  if p.poll()is None:p.kill();p.wait()
