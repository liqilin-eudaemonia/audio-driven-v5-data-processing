from pathlib import Path
import os,sys,json,time,subprocess,fcntl,argparse,hashlib
R=Path(__file__).resolve().parents[1]
from pipeline import atomic

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();lock=(R/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'validation.json').read_text())['approved_pilot_pairs_exactly_reproduced']==137
 stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());snapshot=R/'code_snapshots'/stamp;snapshot.mkdir(parents=True)
 import shutil
 for p in (R/'code').glob('*.py'):shutil.copy2(p,snapshot/p.name)
 atomic(snapshot/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')})
 children=[];logs=[];rounds=1 if args.smoke else 3
 for retry_round in range(rounds):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);slots=[]
  for row in raw.strip().splitlines():
   gpu,free=map(int,row.split(','));count=2 if free>=45000 else 1 if free>=22000 else 0
   slots.extend([str(gpu)]*count)
  if args.smoke:slots=['6'] if '6' in slots else slots[:1]
  if not slots:raise RuntimeError('Insufficient free GPU memory; no other process was stopped')
  children=[]
  for i,gpu in enumerate(slots):
   f=(R/'logs'/f'worker_r{retry_round}_{i}_gpu{gpu}.log').open('a');logs.append(f);cmd=[str(R.parent/'av_baseline_pairs_20261024/venv/bin/python'),str(R/'code/pipeline.py')]
   if args.smoke:cmd+=['--only=-1wCGAxqT_4']
   children.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=f,stderr=subprocess.STDOUT))
  atomic(R/'runtime.json',{'stage':'running','smoke':args.smoke,'round':retry_round,'started_at':time.time(),'supervisor_pid':os.getpid(),'workers':[p.pid for p in children],'gpu_slots':slots,'code_snapshot':str(snapshot)})
  while any(p.poll() is None for p in children):
   subprocess.run([sys.executable,str(R/'code/report.py')],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
   time.sleep(30)
  atomic(R/'round_exit.json',{'round':retry_round,'workers':[{'pid':p.pid,'exit':p.returncode} for p in children]})
 subprocess.run([sys.executable,str(R/'code/report.py'),'--final'],check=True)
 states=[json.loads(p.read_text()) for p in (R/'state').glob('*.json')];allcomplete=len(states)==692 and all(s['stage']=='complete' for s in states)
 atomic(R/'runtime.json',{'stage':'complete' if allcomplete else 'smoke_finished' if args.smoke else 'finished_with_unresolved','completed':sum(s['stage']=='complete' for s in states),'sources':692,'finished_at':time.time(),'supervisor_pid':os.getpid()})
 subprocess.run([sys.executable,str(R/'code/export_samples.py')],check=True)
 subprocess.run([sys.executable,str(R/'code/report.py'),'--final'],check=True)
if __name__=='__main__':main()
