"""Retarget a safely stopped run using the unchanged native28 worker."""
from pathlib import Path
import sys,fcntl,time,traceback,os
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extension import activate,WORK,ORIGINAL,source_identity,validate,validate_cluster
activate()
from p2_listen.contracts import read,sha,atomic_json as save
from p2_loss_search28.checkpoint import checked,validate_stage
from p2_loss_search28.resources import idle
from listen_cluster import launch

def run():
 out=WORK/'runs/event020_feature010'
 with (ORIGINAL/'queue.lock').open('a') as allocation,(WORK/'queue.lock').open('a') as lock:
  fcntl.flock(allocation,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_EX)
  c=validate(read(out/'config.json'));cl=validate_cluster(c,read(out/'cluster.json'));expected=source_identity()
  assert expected==read(WORK/'PREFLIGHT_ACCEPTED.json')['source']
  if (WORK/'STOP').exists():return
  if (out/'STAGE_1661.json').exists():return
  cp=sorted(out.glob('step_*.complete.json'))[-1].with_suffix('').with_suffix('.pt');v=checked(cp)
  validate_stage(v,c,v['step'],expected);assert 500<=v['step']<=1661
  save(WORK/'RETARGET1661_CHECKPOINT_ACCEPTED.json',dict(success=True,step=v['step'],checkpoint=str(cp),sha256=sha(cp),rank_rng=len(v['rank_rng']),at=time.time()))
  del v
  marker=out/'STOP'
  if marker.exists():
   assert marker.read_text()=='User requested retarget to1661; safe complete save before strict resume\n'
   marker.rename(WORK/'RETARGET1661_STOP_ARCHIVED')
  while not idle(cl)['idle']:
   if (WORK/'STOP').exists() or (out/'STOP').exists():return
   time.sleep(10)
  if read(cp.with_suffix('.complete.json'))['step']<1661:
   commands=[]
   for i,n in enumerate(cl['nodes']):
    commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v)for k,v in cl['network'].items()],cl['python'],'-u',cl['training_script'],'node','--index',str(i),'--config',str(out/'config.json'),'--stop-after','1661','--resume',str(cp)])
   save(WORK/'STATE.json',dict(stage='training',candidate='event020_feature010',target=1661,resume=str(cp),at=time.time()))
   launch(cl,commands,out/f'nodes_1661_{time.time_ns()}')
  cp=out/'step_001661.pt';v=checked(cp);validate_stage(v,c,1661,expected)
  save(out/'STAGE_1661.json',dict(success=True,step=1661,checkpoint=str(cp),sha256=read(cp.with_suffix('.complete.json'))['sha256'],at=time.time()))
  save(WORK/'STATE.json',dict(stage='awaiting_evaluation',completed_target=1661,at=time.time()))
if __name__=='__main__':
 try:run()
 except Exception as exc:save(WORK/'RESUME1661_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()));raise
