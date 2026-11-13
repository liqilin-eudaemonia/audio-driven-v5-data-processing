"""One-shot matched benchmark with guaranteed original-queue restoration."""
import os,sys,json,time,threading,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_seamless200_whisper_eval import idle_snapshot
from listen_cluster import launch
from queue_p2_whisper_ablation31 import resume_point
W=ROOT/'experiments/p2_whisper_execution_trial_20261106';Q=ROOT/'experiments/p2_whisper_single_aux31_20261106';O=ROOT/'experiments/p2_whisper_event_20261106'

def live(record):
 try:
  s=Path('/proc',str(record['pid']),'stat').read_text().split();return s[21]==record['start_ticks'] and s[2]!='Z'
 except FileNotFoundError:return False

def state(stage,**kw):
 v=dict(stage=stage,at=time.time(),**kw);atomic_json(W/'state.json',v);print(json.dumps(v),flush=True)

def run():
 W.mkdir(exist_ok=True)
 with (W/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (W/'RESULT.json').exists():raise ValueError('Do not rerun completed benchmark')
  cluster=read(O/'cluster.json');original=read(Q/'controller.json');token=json.dumps(dict(owner='execution_trial',pid=os.getpid(),at=time.time()))
  if (O/'STOP').exists() or (Q/'STOP').exists():raise ValueError('Existing user STOP')
  state('waiting_step1000')
  while read(O/'status.json')['step']<1000:
   if not live(original):raise RuntimeError('Original controller ended before pause')
   time.sleep(1)
  with (O/'STOP').open('x') as f:f.write(token)
  state('waiting_complete_pause')
  try:
   deadline=time.time()+900
   while live(original):
    if time.time()>deadline:raise TimeoutError('Graceful pause did not finish; do not terminate healthy workers')
    time.sleep(2)
   while not idle_snapshot(cluster)['all_idle']:time.sleep(5)
   checkpoint=resume_point(O);step=int(checkpoint.stem.split('_')[-1]);atomic_json(W/'PAUSE_RECEIPT.json',dict(checkpoint=str(checkpoint),step=step,original_controller=original))
   script=ROOT/'scripts/p2_whisper_ablation_benchmark.py';commands=[]
   for i,n in enumerate(cluster['nodes']):
    commands.append(['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,n['gpus'])),'PYTHONUNBUFFERED=1',*[k+'='+str(v) for k,v in cluster['network'].items()],cluster['python'],str(script),'node','--config',str(O/'config.json'),'--work',str(W),'--checkpoint',str(checkpoint),'--node-index',str(i),'--start-step',str(step+1),'--steps','8'])
   state('benchmark31',checkpoint=str(checkpoint));cancel=threading.Event();done=threading.Event();started=time.time()
   def monitor():
    while not done.wait(15):
     p=W/'progress.json';stamp=p.stat().st_mtime if p.exists() else started
     if time.time()-stamp>900:atomic_json(W/'TIMEOUT.json',dict(at=time.time()));cancel.set();return
   t=threading.Thread(target=monitor,daemon=True);t.start()
   try:launch(cluster,commands,W/'launch',cancel_event=cancel)
   finally:done.set();t.join(20)
   result=read(W/'RESULT.json');state('benchmark_complete',selected=result['selected']['name'])
  finally:
   # Never erase a STOP written or replaced by the user.
   if live(original):state('original_controller_still_live_no_restart')
   elif (O/'STOP').exists() and (O/'STOP').read_text()==token and not (Q/'STOP').exists():
    while not idle_snapshot(cluster)['all_idle']:time.sleep(5)
    (O/'STOP').unlink()
    with (Q/'controller.log').open('ab',buffering=0) as log:
     p=subprocess.Popen([cluster['python'],'-u',str(ROOT/'scripts/queue_p2_whisper_ablation31.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
    atomic_json(W/'ORIGINAL_QUEUE_RESTARTED.json',dict(pid=p.pid,at=time.time()));state('original_queue_restarted',pid=p.pid)

if __name__=='__main__':
 try:run()
 except Exception as exc:atomic_json(W/'ERROR.json',dict(error=repr(exc),at=time.time()));raise
