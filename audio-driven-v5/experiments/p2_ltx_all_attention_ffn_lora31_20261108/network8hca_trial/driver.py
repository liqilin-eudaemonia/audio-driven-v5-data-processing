from pathlib import Path
import sys,os,time,json,statistics,subprocess,fcntl,traceback,copy
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';WORK=OUT/'network8hca_trial'
SINGLE='=mlx5_0:1';MULTI='='+','.join(f'mlx5_{i}:1'for i in range(8))

def emit(stage,**kw):
 v=dict(at=time.time(),pid=os.getpid(),stage=stage,**kw);atomic_json(WORK/'STATE.json',v);print(json.dumps(v),flush=True)

def workers_exited():
 import importlib.util
 p=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence_v2.py';s=importlib.util.spec_from_file_location('helpers',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
 return h.workers_exited(OUT)

def stage(cluster,name,script,hca,checkpoint,step,smoke=False):
 from listen_cluster import launch
 work=WORK/name;work.mkdir(exist_ok=True);commands=[];cl=copy.deepcopy(cluster);cl['network']['NCCL_IB_HCA']=hca
 if smoke:cl['network']['NCCL_DEBUG']='INFO'
 for i,n in enumerate(cl['nodes']):
  commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],str(WORK/script),'node','--config',str(OUT/'config.json'),'--work',str(work),'--checkpoint',str(checkpoint),'--node-index',str(i),'--start-step',str(step+1),'--steps','16'])
 atomic_json(work/'COMMANDS.json',commands);emit('measuring',case=name,hca=hca)
 launch(cl,commands,work/'nodes')
 result=read(work/('SMOKE_ACCEPTED.json'if smoke else'RESULT.json'))
 if not result['success']:raise RuntimeError('Stage failed '+name)
 return result

def main():
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
  atomic_json(OUT/'STOP',dict(owner='single-vs-eight-hca-trial',at=time.time()));emit('saving_production')
  while not workers_exited():time.sleep(5)
  step=read(OUT/'status.json')['step'];checkpoint=OUT/f'step_{step:06d}.pt';marker=read(checkpoint.with_suffix('.complete.json'));assert marker['step']==step
  atomic_json(WORK/'PARENT.json',dict(step=step,checkpoint=str(checkpoint),marker=marker,source=source_identity(),config_sha256=sha(OUT/'config.json')))
  selected=SINGLE;decision=None
  try:
   from p2_whisper_event_feature_train31 import selected_resources
   snapshot=selected_resources(cluster);atomic_json(WORK/'RESOURCE_PRECHECK.json',snapshot)
   if not snapshot['selected_all_idle']:raise RuntimeError('Selected GPUs not idle')
   stage(cluster,'smoke8','smoke31.py',MULTI,checkpoint,step,True)
   stage(cluster,'smoke1','smoke31.py',SINGLE,checkpoint,step,True)
   measurements=[]
   for name,hca in [('A1_single',SINGLE),('B1_eight',MULTI),('B2_eight',MULTI),('A2_single',SINGLE)]:
    result=stage(cluster,name,'benchmark31.py',hca,checkpoint,step);row=result['results'][0];assert row['safe']
    measurements.append(dict(case=name,hca=hca,result=row));atomic_json(WORK/'PARTIAL.json',measurements)
   reference=measurements[0]['result'];keys=['gradient_sha256','model_sha256','optimizer_sha256','losses','diagnostics']
   equivalent=all(all(m['result'][k]==reference[k]for k in keys)for m in measurements)
   A=[m['result']['mean_seconds']for m in measurements if m['hca']==SINGLE];B=[m['result']['mean_seconds']for m in measurements if m['hca']==MULTI]
   reduction=1-statistics.mean(B)/statistics.mean(A)
   adopt=equivalent and reduction>=.03 and max(B)<min(A)
   decision=dict(success=True,at=time.time(),equivalent=equivalent,single_seconds=A,eight_seconds=B,step_time_reduction=reduction,adopt_eight=adopt,criterion='all loss/gradient/model/Adam/diagnostics equal; >=3% faster; both B runs faster than both A runs',world=31,matched_steps=16,matched_samples=496)
   atomic_json(WORK/'DECISION.json',decision)
   if adopt:selected=MULTI
   emit('comparison_complete',**{'decision':decision})
  except Exception:
   atomic_json(WORK/'ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));emit('test_failed_restore_single')
  finally:
   if read(OUT/'STOP').get('owner')!='single-vs-eight-hca-trial':raise RuntimeError('STOP owner changed; preserve newer user intent')
   if selected==MULTI:
    updated=copy.deepcopy(cluster);updated['network']['NCCL_IB_HCA']=MULTI;updated['training_script']=str(WORK/'train8hca.py');atomic_json(OUT/'cluster.json',updated)
    atomic_json(WORK/'NETWORK_RECEIPT.json',dict(at=time.time(),wrapper_sha256=sha(WORK/'train8hca.py'),cluster_sha256=sha(OUT/'cluster.json'),config_sha256=sha(OUT/'config.json'),parent_cluster_sha256=sha(WORK/'PRODUCTION_CLUSTER_BEFORE.json'),decision_sha256=sha(WORK/'DECISION.json'),hca=MULTI,world=31,reserved_gpu='node11:0'))
    entry=WORK/'train8hca.py'
   else:entry=ROOT/'scripts/p2_ltx_all_lora_node11shift_20261108.py'
   (OUT/'STOP').unlink();emit('resuming_production',hca=selected,from_step=step)
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
