from pathlib import Path
import os,sys,time,json,subprocess,fcntl,traceback
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';WORK=OUT/'recompute_trial_retry'

def emit(stage,**kw):
 v=dict(stage=stage,at=time.time(),pid=os.getpid(),**kw);atomic_json(WORK/'STATE.json',v);print(json.dumps(v),flush=True)

def worker_exit():
 import importlib.util
 p=ROOT/'experiments/p2_current_component_ablations31_20261108/sequence_v2.py';s=importlib.util.spec_from_file_location('helpers',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 return m.workers_exited(OUT)

def main():
 with (WORK/'driver.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'STOP').exists():raise RuntimeError('Preserve existing STOP')
  atomic_json(OUT/'STOP',dict(owner='recompute-trial-save-and-resume',at=time.time()))
  emit('saving_production')
  while not worker_exit():time.sleep(5)
  step=read(OUT/'status.json')['step'];checkpoint=OUT/f'step_{step:06d}.pt'
  marker=read(checkpoint.with_suffix('.complete.json'));assert marker['step']==step
  atomic_json(WORK/'PARENT.json',dict(step=step,path=str(checkpoint),marker=marker,config_sha256=sha(OUT/'config.json')))
  emit('production_saved',step=step)
  try:
   cluster=read(OUT/'cluster.json');commands=[]
   for i,n in enumerate(cluster['nodes']):
    commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cluster['network'].items()],cluster['python'],str(WORK/'benchmark31.py'),'node','--config',str(OUT/'config.json'),'--work',str(WORK),'--checkpoint',str(checkpoint),'--node-index',str(i),'--start-step',str(step+1),'--steps','6'])
   atomic_json(WORK/'COMMANDS.json',commands);emit('benchmark31',step=step)
   from listen_cluster import launch
   launch(cluster,commands,WORK/'nodes')
   result=read(WORK/'RESULT.json');emit('benchmark_complete',selected=result['selected']['name'],relative_speedup=result['relative_speedup'])
  except Exception:
   atomic_json(WORK/'ERROR.json',dict(at=time.time(),error=traceback.format_exc()));emit('benchmark_failed_restore_all')
  finally:
   stop=read(OUT/'STOP')
   if stop.get('owner')!='recompute-trial-save-and-resume':raise RuntimeError('STOP owner changed; preserve new user intent')
   (OUT/'STOP').unlink()
   emit('restoring_production_all',step=step)
   command=[sys.executable,str(ROOT/'scripts/p2_ltx_all_lora_node11shift_20261108.py'),'launch','--resume',str(checkpoint),'--logs',str(WORK/'resume_nodes')]
   with(WORK/'resume.log').open('w')as f:
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   atomic_json(WORK/'RESUME_CONTROLLER.json',dict(pid=p.pid,command=command,step=step,at=time.time()))
   while p.poll()is None:
    latest=read(OUT/'status.json')['step']
    if latest>step:
     atomic_json(WORK/'RESUMED_ACCEPTED.json',dict(success=True,source_step=step,step=latest,policy='all',at=time.time()));emit('production_resumed',step=latest);return
    time.sleep(5)
   raise RuntimeError('Production resume failed')

if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(WORK/'DRIVER_ERROR.json',dict(error=traceback.format_exc(),at=time.time()));raise
