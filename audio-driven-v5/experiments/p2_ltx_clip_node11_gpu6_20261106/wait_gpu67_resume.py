"""Wait for physical GPUs6 AND7 to be idle, then resume unchanged run on GPU7."""
from pathlib import Path
import sys,json,time,os,fcntl,subprocess,datetime,runpy,traceback
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import transport,launch
OUT=Path(__file__).resolve().parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def idle(probe):
    cards=probe['cards']
    return set(cards)=={'6','7'} and all(not card['processes'] and card['memory_used_mib']<256 and card['utilization']==0 for card in cards.values())

PROBE="""import subprocess,json

def query(fields,kind):
 r=subprocess.run(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
 if r.returncode:raise RuntimeError(r.stderr)
 return [list(map(str.strip,line.split(','))) for line in r.stdout.splitlines() if line.strip()]
cards={}
for index,uuid,mem,util in query('index,uuid,memory.used,utilization.gpu','gpu'):
 if index in ('6','7'):cards[index]={'uuid':uuid,'memory_used_mib':int(mem),'utilization':int(util),'processes':[]}
for uuid,pid,name in query('gpu_uuid,pid,process_name','compute-apps'):
 for card in cards.values():
  if card['uuid']==uuid:card['processes'].append({'pid':pid,'name':name})
print(json.dumps({'cards':cards}))
"""

def probe(cluster):
 r=subprocess.run(transport(cluster,cluster['nodes'][0],[cluster['python'],'-c',PROBE]),capture_output=True,text=True,timeout=45)
 if r.returncode:raise RuntimeError(r.stderr[-2000:])
 return json.loads(r.stdout)

def resume(cluster):
 from p2_whisper_av.bootstrap import activate_ltx
 activate_ltx()
 import torch
 from p2_ltx_events.config import source_identity
 from p2_ltx_events.data import data_identity
 from p2_ltx_events.training import check_preflight
 torch.set_num_threads(2)
 c=read(OUT/'config.json');step=43;checkpoint=OUT/f'step_{step:06d}.pt'
 marker=read(checkpoint.with_suffix('.complete.json'));assert marker['step']==step
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert saved['step']==step and saved['identity']['world']==1 and saved['identity']['experiment']==c
 assert saved['optimizer']['state'] and len(saved['rank_rng'])==1
 check_preflight(c,read(OUT/'preflight.json'),source_identity(),data_identity(c))
 assert saved['identity']['source']==source_identity()
 cli=runpy.run_path(str(ROOT/'scripts/p2_ltx_events.py'),run_name='auto_gpu7_commands')
 commands=cli['commands'](cluster,OUT/'config.json',resume=checkpoint)
 assert len(commands)==1 and 'CUDA_VISIBLE_DEVICES=7' in commands[0]
 atomic_json(OUT/'COMMANDS_GPU7.json',commands)
 # Check once more after CPU identity validation, immediately before allocating.
 if not idle(probe(cluster)):return False
 if (OUT/'CANCEL_AUTO_RESUME').exists():return False
 (OUT/'STOP').unlink(missing_ok=True)
 state=dict(state='launching',time=now(),controller_pid=os.getpid(),gpu=7,resume_step=43,next_step=44,save_every=50,
            trigger='GPUs6_and7_both_idle_for_two_polls',checkpoint_sha256=marker['sha256'])
 atomic_json(OUT/'AUTO_RESUME_STATUS.json',state);atomic_json(OUT/'GPU7_MIGRATION.json',state)
 print(json.dumps(state),flush=True)
 logs=OUT/('launch_gpu7_auto_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 launch(cluster,commands,logs)
 state.update(state='finished' if read(OUT/'status.json')['complete'] else 'stopped',time=now(),training_status=read(OUT/'status.json'))
 atomic_json(OUT/'AUTO_RESUME_STATUS.json',state)
 return True

def main():
 with (OUT/'AUTO_RESUME.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cluster=read(OUT/'cluster_gpu7.json');assert cluster['nodes'][0]['gpus']==[7]
  stable=0
  while not (OUT/'CANCEL_AUTO_RESUME').exists():
   try:
    observation=probe(cluster);stable=stable+1 if idle(observation) else 0
    state=dict(state='waiting_idle',time=now(),controller_pid=os.getpid(),required_gpus=[6,7],training_gpu=7,
               resume_step=43,stable_idle_checks=stable,required_idle_checks=2,poll_seconds=30,observation=observation)
    atomic_json(OUT/'AUTO_RESUME_STATUS.json',state);print(json.dumps(state),flush=True)
    if stable>=2:
     try:
      if resume(cluster):return
      stable=0
     except Exception as exc:
      (OUT/'STOP').touch()
      atomic_json(OUT/'AUTO_RESUME_STATUS.json',dict(state='blocked_validation_or_launch',time=now(),error=str(exc),controller_pid=os.getpid()))
      print(traceback.format_exc(),flush=True)
      return
   except Exception as exc:
    stable=0
    atomic_json(OUT/'AUTO_RESUME_STATUS.json',dict(state='waiting_probe_retry',time=now(),error=str(exc),controller_pid=os.getpid()))
   time.sleep(30)
  atomic_json(OUT/'AUTO_RESUME_STATUS.json',dict(state='cancelled',time=now(),controller_pid=os.getpid()))

if __name__=='__main__':main()
