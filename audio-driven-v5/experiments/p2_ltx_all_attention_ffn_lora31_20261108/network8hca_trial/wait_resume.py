from pathlib import Path
import sys,time,json,subprocess,fcntl,os,traceback
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_whisper_event_feature_train31 import selected_resources
OUT=ROOT/'experiments/p2_ltx_all_attention_ffn_lora31_20261108';WORK=OUT/'network8hca_trial'
def state(stage,**kw):atomic_json(WORK/'RESTORE_STATE.json',dict(at=time.time(),pid=os.getpid(),stage=stage,**kw))
try:
 with(WORK/'restore.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  parent=read(WORK/'PARENT.json');step=parent['step'];cluster=read(OUT/'cluster.json');assert cluster['network']['NCCL_IB_HCA']=='=mlx5_0:1'
  consecutive=0
  while consecutive<2:
   if(WORK/'RESTORE_CANCEL').exists()or(OUT/'STOP').exists():raise RuntimeError('Preserve new stop intent')
   snapshot=selected_resources(cluster);atomic_json(WORK/'RESTORE_RESOURCES.json',snapshot)
   consecutive=consecutive+1 if snapshot['selected_all_idle']else 0
   state('waiting_selected31_idle',from_step=step,consecutive=consecutive)
   if consecutive<2:time.sleep(10)
  command=[sys.executable,str(ROOT/'scripts/p2_ltx_all_lora_node11shift_20261108.py'),'launch','--resume',parent['checkpoint'],'--logs',str(WORK/'resume_after_idle_nodes')]
  with(WORK/'resume_after_idle.log').open('w')as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  atomic_json(WORK/'RESUME_AFTER_IDLE_CONTROLLER.json',dict(pid=p.pid,command=command,at=time.time()));state('restoring_single_hca',from_step=step)
  while p.poll()is None:
   latest=read(OUT/'status.json')['step']
   if latest>=step+8:
    atomic_json(WORK/'RESUMED_ACCEPTED.json',dict(success=True,at=time.time(),from_step=step,step=latest,hca='=mlx5_0:1',world=31));state('production_resumed',step=latest);break
   time.sleep(5)
  else:raise RuntimeError('Restore launch failed; preserve error log')
except Exception:
 atomic_json(WORK/'RESTORE_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
