from pathlib import Path
import sys,os,time,json,statistics,subprocess,fcntl,traceback,copy,threading,concurrent.futures
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';WORK=OUT/'network8hca_clean'
SINGLE='=mlx5_0:1';MULTI='='+','.join(f'mlx5_{i}:1'for i in range(8))

def emit(stage,**kw):
 v=dict(at=time.time(),pid=os.getpid(),stage=stage,**kw);atomic_json(WORK/'STATE.json',v);print(json.dumps(v),flush=True)

def workers_exited():
 import importlib.util
 p=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence_v2.py';s=importlib.util.spec_from_file_location('helpers',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
 return h.workers_exited(OUT)

EIGHT_CROSS='0'

def wait_idle(cluster,label):
 from p2_whisper_event_feature_train31 import selected_resources
 good=0
 for attempt in range(60):
  if (WORK/'CANCEL').exists():raise RuntimeError('User cancelled network test')
  snapshot=selected_resources(cluster);atomic_json(WORK/(label+'_resources.json'),snapshot)
  good=good+1 if snapshot['selected_all_idle']else 0
  if good>=2:return
  time.sleep(5)
 raise RuntimeError('Selected31 did not become idle; preserve other tasks')

def unexpected_apps(cluster):
 from listen_cluster import transport
 code="""import subprocess,json,sys
from pathlib import Path
selected=json.loads(sys.argv[1]);root=sys.argv[2]
m=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
ids={u.strip():int(i)for i,u in (l.split(',')for l in m.splitlines())}
a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);bad=[];owned=[]
for line in a.splitlines():
 u,p=[s.strip()for s in line.split(',')];i=ids.get(u)
 if i not in selected:continue
 path=Path('/proc')/p
 try:cmd=(path/'cmdline').read_bytes().decode().replace(chr(0),' ')
 except FileNotFoundError:continue
 if root in cmd:owned.append(dict(gpu=i,pid=int(p)))
 elif cmd:bad.append(dict(gpu=i,pid=int(p)))
print(json.dumps(dict(unexpected=bad,owned=owned)))
"""
 def one(n):
  p=subprocess.run(transport(cluster,n,[cluster['python'],'-c',code,json.dumps(n['gpus']),str(WORK)]),capture_output=True,text=True,timeout=20,check=True)
  v=json.loads(p.stdout);v['node']=n['name'];return v
 with concurrent.futures.ThreadPoolExecutor(4)as pool:reports=list(pool.map(one,cluster['nodes']))
 return dict(at=time.time(),nodes=reports,clean=all(not n['unexpected']for n in reports))

def stage(cluster,name,script,hca,checkpoint,step,smoke=False):
 from listen_cluster import launch
 wait_idle(cluster,name+'_pre')
 work=WORK/name;work.mkdir(exist_ok=True);commands=[];cl=copy.deepcopy(cluster);cl['network']['NCCL_IB_HCA']=hca
 cl['network']['NCCL_CROSS_NIC']=EIGHT_CROSS if hca==MULTI else '0'
 cl['network']['TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC']='0'
 if smoke:cl['network'].update(NCCL_DEBUG='INFO',TORCH_NCCL_TRACE_BUFFER_SIZE='1000')
 for i,n in enumerate(cl['nodes']):
  commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],str(WORK/script),'node','--config',str(OUT/'config.json'),'--work',str(work),'--checkpoint',str(checkpoint),'--node-index',str(i),'--start-step',str(step+1),'--steps','16'])
 atomic_json(work/'COMMANDS.json',commands);emit('measuring',case=name,hca=hca,cross_nic=cl['network']['NCCL_CROSS_NIC'])
 done=threading.Event();cancel=threading.Event();errors=[]
 def monitor():
  while not done.wait(5):
   try:
    snapshot=unexpected_apps(cl);atomic_json(work/'RESOURCE_OWNERSHIP.json',snapshot)
    with(work/'resource_ownership.jsonl').open('a')as f:f.write(json.dumps(snapshot)+'\n')
    if not snapshot['clean']:errors.append(snapshot);cancel.set();return
   except Exception as exc:atomic_json(work/'RESOURCE_CHECK_ERROR.json',dict(error=repr(exc)))
 watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
 try:launch(cl,commands,work/'nodes',cancel_event=cancel)
 finally:done.set();watcher.join(timeout=25)
 if errors:raise RuntimeError('Foreign GPU process contaminated trial')
 result=read(work/('SMOKE_ACCEPTED.json'if smoke else'RESULT.json'))
 if not result['success']:raise RuntimeError('Stage failed '+name)
 return result

def main():
 global EIGHT_CROSS
 with(WORK/'driver.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  from p2_ltx_all_lora.config import load,source_identity
  from p2_ltx_all_lora.schedule import Schedule
  c=load(OUT/'config.json');schedule=Schedule(c);accepted=read(OUT/'CODE_ACCEPTED.json')
  if source_identity()!=accepted['source']:raise ValueError('Production source changed')
  cluster=read(OUT/'cluster.json');assert all(n['gpus']==(list(range(1,8))if n['name']=='node11'else list(range(8)))for n in cluster['nodes'])
  if cluster['network']['NCCL_IB_HCA']!=SINGLE:raise ValueError('Unexpected current HCA config')
  atomic_json(WORK/'PRODUCTION_CLUSTER_BEFORE.json',cluster,immutable=True)
  if(OUT/'STOP').exists():raise RuntimeError('Preserve existing STOP')
  atomic_json(OUT/'STOP',dict(owner='single-vs-eight-hca-trial',at=time.time()));emit('using_complete_checkpoint871')
  while not workers_exited():time.sleep(5)
  step=read(OUT/'status.json')['step'];checkpoint=OUT/f'step_{step:06d}.pt';marker=read(checkpoint.with_suffix('.complete.json'));assert marker['step']==step
  atomic_json(WORK/'PARENT.json',dict(step=step,checkpoint=str(checkpoint),marker=marker,source=source_identity(),config_sha256=sha(OUT/'config.json')))
  selected=SINGLE;decision=None
  try:
   from p2_whisper_event_feature_train31 import selected_resources
   snapshot=selected_resources(cluster);atomic_json(WORK/'RESOURCE_PRECHECK.json',snapshot)
   wait_idle(cluster,'initial_selected31')
   stage(cluster,'smoke1','smoke31.py',SINGLE,checkpoint,step,True)
   smoke_ok=False;smoke_failures=[]
   for cross in ['0','1']:
    EIGHT_CROSS=cross
    try:
     stage(cluster,'smoke8_cross'+cross,'smoke31.py',MULTI,checkpoint,step,True);smoke_ok=True;break
    except Exception:
     smoke_failures.append(dict(cross_nic=cross,error=traceback.format_exc()));atomic_json(WORK/'SMOKE8_FAILURES.json',smoke_failures)
   if not smoke_ok:raise RuntimeError('Eight HCA smoke failed for cross_nic0 and1 with clean-resource monitoring')
   measurements=[]
   for name,hca in [('A1_single',SINGLE),('B1_eight',MULTI),('B2_eight',MULTI),('A2_single',SINGLE)]:
    result=stage(cluster,name,'benchmark31.py',hca,checkpoint,step);row=result['results'][0];assert row['safe']
    measurements.append(dict(case=name,hca=hca,result=row));atomic_json(WORK/'PARTIAL.json',measurements)
   reference=measurements[0]['result'];keys=['gradient_sha256','model_sha256','optimizer_sha256','losses','diagnostics']
   equivalent=all(all(m['result'][k]==reference[k]for k in keys)for m in measurements)
   A=[m['result']['mean_seconds']for m in measurements if m['hca']==SINGLE];B=[m['result']['mean_seconds']for m in measurements if m['hca']==MULTI]
   reduction=1-statistics.mean(B)/statistics.mean(A)
   adopt=equivalent and reduction>=.03 and max(B)<min(A)
   decision=dict(success=True,at=time.time(),equivalent=equivalent,single_seconds=A,eight_seconds=B,step_time_reduction=reduction,adopt_eight=adopt,eight_cross_nic=EIGHT_CROSS,criterion='all loss/gradient/model/Adam/diagnostics equal; >=3% faster; both B runs faster than both A runs',world=31,matched_steps=16,matched_samples=496)
   atomic_json(WORK/'DECISION.json',decision)
   if adopt:selected=MULTI
   emit('comparison_complete',**{'decision':decision})
  except Exception:
   atomic_json(WORK/'ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));emit('test_failed_restore_single')
  finally:
   if read(OUT/'STOP').get('owner')!='single-vs-eight-hca-trial':raise RuntimeError('STOP owner changed; preserve newer user intent')
   if selected==MULTI:
    updated=copy.deepcopy(cluster);updated['network']['NCCL_IB_HCA']=MULTI;updated['network']['NCCL_CROSS_NIC']=EIGHT_CROSS;updated['training_script']=str(WORK/'train8hca.py');atomic_json(OUT/'cluster.json',updated)
    atomic_json(WORK/'NETWORK_RECEIPT.json',dict(at=time.time(),wrapper_sha256=sha(WORK/'train8hca.py'),cluster_sha256=sha(OUT/'cluster.json'),config_sha256=sha(OUT/'config.json'),parent_cluster_sha256=sha(WORK/'PRODUCTION_CLUSTER_BEFORE.json'),decision_sha256=sha(WORK/'DECISION.json'),hca=MULTI,world=31,reserved_gpu='node11:0'))
    entry=WORK/'train8hca.py'
   else:entry=ROOT/'scripts/p2_ltx_all_lora_node11shift_20261108.py'
   (OUT/'STOP').unlink();wait_idle(cluster,'production_restore');emit('resuming_production',hca=selected,from_step=step)
   command=[sys.executable,str(entry),'launch','--resume',str(checkpoint),'--logs',str(WORK/'resume_nodes')]
   with(WORK/'resume.log').open('w')as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   atomic_json(WORK/'RESUME_CONTROLLER.json',dict(pid=p.pid,command=command,at=time.time()))
   while p.poll()is None:
    latest=read(OUT/'status.json')['step']
    if latest>=step+8:
     atomic_json(WORK/'RESUMED_ACCEPTED.json',dict(success=True,at=time.time(),hca=selected,from_step=step,step=latest,world=31));emit('production_resumed',step=latest,hca=selected);return
    time.sleep(5)
   raise RuntimeError('Production resume failed')

if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(WORK/'DRIVER_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
