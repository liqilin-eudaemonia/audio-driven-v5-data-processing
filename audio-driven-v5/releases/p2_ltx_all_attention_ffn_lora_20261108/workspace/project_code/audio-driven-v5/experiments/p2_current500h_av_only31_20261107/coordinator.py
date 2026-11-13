"""User-authorized transition at parent7000 complete checkpoint to fresh AV-only31."""
from pathlib import Path
import os,sys,json,time,threading,subprocess,concurrent.futures,traceback,statistics,math,fcntl
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');OUT=Path(__file__).resolve().parent
PARENT=ROOT/'experiments/p2_ltx_clip_train24_reserve_node11_20261107'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from listen_cluster import transport,launch


def state(stage,**kw):
 value=dict(stage=stage,at=time.time(),controller_pid=os.getpid(),**kw);atomic_json(OUT/'QUEUE_STATE.json',value);print(json.dumps(value),flush=True)


def selected_resources(cluster):
 def one(node):
  for attempt in range(3):
   try:
    r=subprocess.run(transport(cluster,node,['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=25)
    if r.returncode:raise RuntimeError(r.stderr[-500:])
    cards=[]
    for line in r.stdout.splitlines():
     i,m,u=line.split(',')
     if int(i)in node['gpus']:cards.append(dict(index=int(i),free_mib=float(m),utilization=float(u)))
    return dict(node=node['name'],gpus=cards,idle=len(cards)==len(node['gpus'])and all(g['free_mib']>=108*1024 and g['utilization']<10 for g in cards))
   except Exception as e:error=repr(e)
  return dict(node=node['name'],error=error,idle=False)
 with concurrent.futures.ThreadPoolExecutor(4)as pool:return list(pool.map(one,cluster['nodes']))


def main():
 with (OUT/'queue.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state('armed_waiting_parent7000',parent=str(PARENT),target_step=7000,reserved_gpu='node11:7')
  while True:
   if (OUT/'QUEUE_STOP').exists():state('cancelled_before_parent_stop');return
   step=read(PARENT/'status.json')['step']
   if step>=7000:
    if not (PARENT/'STOP').exists():atomic_json(PARENT/'STOP',dict(owner='user7000-to-av-only31',at=time.time(),target_step=7000))
    state('waiting_parent_complete_save_and_exit',observed_step=step);break
   time.sleep(.5)
  ctrl=read(PARENT/'CONTROLLER.json')
  while True:
   step=read(PARENT/'status.json')['step'];marker=PARENT/f'step_{step:06d}.complete.json';proc=Path('/proc',str(ctrl['pid']))
   alive=False
   if proc.exists():
    st=(proc/'stat').read_text().split();alive=st[2]!='Z'and st[21]==ctrl['start_ticks']
   if marker.exists()and not alive:break
   time.sleep(2)
  # Usually exactly7000; if a late user request races past the boundary,
  # preserve that later complete state rather than discarding any update.
  target=PARENT/f'step_{step:06d}.pt'
  if sha(target)!=read(marker)['sha256']:raise ValueError('Parent complete checkpoint checksum mismatch')
  atomic_json(OUT/'PARENT_SAVED.json',dict(success=True,step=step,checkpoint=str(target),sha256=read(marker)['sha256'],at=time.time()))
  state('waiting_prelaunch_acceptance',parent_saved_step=step)
  while not (OUT/'PRELAUNCH_ACCEPTED.json').exists():
   if (OUT/'QUEUE_STOP').exists():state('cancelled_after_parent_save');return
   time.sleep(2)
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  from p2_current_av_only.config import load,source_identity
  import p2_current500h_av_only as runtime
  c=load(OUT/'config.json');cluster=read(OUT/'cluster.json');assert read(OUT/'preflight.json')['source']==source_identity()
  consecutive=0
  while consecutive<2:
   if (OUT/'QUEUE_STOP').exists()or (OUT/'STOP').exists():state('cancelled_waiting_resources');return
   observations=selected_resources(cluster);atomic_json(OUT/'RESOURCES.json',dict(at=time.time(),nodes=observations))
   ready=all(n['idle']for n in observations);consecutive=consecutive+1 if ready else 0
   state('waiting_selected31_idle',parent_saved_step=step,consecutive_idle=consecutive,busy_nodes=[n['node']for n in observations if not n['idle']])
   if consecutive<2:time.sleep(10)
  assert read(OUT/'preflight.json')['source']==source_identity()
  cmds=runtime.commands(cluster,OUT/'config.json');atomic_json(OUT/'LAUNCH_COMMANDS.json',cmds)
  done=threading.Event();started=time.time()
  def monitor():
   while not done.wait(15):
    try:
     f=OUT/'metrics.jsonl'
     if not f.exists():continue
     with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-150000));lines=stream.read().splitlines()
     rows=[]
     for line in lines:
      try:rows.append(json.loads(line))
      except ValueError:pass
     if not rows:continue
     row=rows[-1]
     for k in ('av','av_video','av_audio','weighted_total'):assert math.isfinite(row[k])
     assert abs(row['weighted_total']-row['av'])<1e-9
     if row['step']>=3:
      for k in ('connector','text_lora'):assert math.isfinite(row['grad_norm'][k])and row['grad_norm'][k]>0
     perf=dict(at=time.time(),step=row['step'],world=31,mean_last30_seconds=statistics.mean(r['seconds']for r in rows[-30:]),
      metrics_age_seconds=time.time()-f.stat().st_mtime,peak_reserved_gib=max(r['peak_reserved_gib']for r in rows[-30:]),loss=row['weighted_total'])
     atomic_json(OUT/'PERFORMANCE_STATUS.json',perf)
     if row['step']>=20 and not (OUT/'RUNNING_VERIFIED.json').exists():
      atomic_json(OUT/'RUNNING_VERIFIED.json',dict(success=True,**perf,pure_av=True,optimizer_groups=['connector','text_lora'],reserved_gpu='node11:7'))
      state('training31_verified',**{k:v for k,v in perf.items()if k!='at'})
    except Exception:
     atomic_json(OUT/'MONITOR_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()))
  watcher=threading.Thread(target=monitor,daemon=True);watcher.start();state('starting_fresh_av_only31',parent_saved_step=step,budget=c['training']['steps'],global_batch=31)
  try:launch(cluster,cmds,OUT/'launch')
  finally:done.set();watcher.join(timeout=5)
  state('training_finished',status=read(OUT/'status.json'))


if __name__=='__main__':
 try:main()
 except Exception:
  atomic_json(OUT/'QUEUE_ERROR.json',dict(at=time.time(),traceback=traceback.format_exc()));state('failed');raise
