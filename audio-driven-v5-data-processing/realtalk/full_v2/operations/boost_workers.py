"""Add exact unchanged pipeline workers; preserve cache signatures and in-flight work."""
from pathlib import Path
import os,sys,time,json,subprocess,fcntl,collections,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
from pipeline import atomic
PYTHON=str(R.parent/'av_baseline_pairs_20261024/venv/bin/python')
def alive(pid):
 try:return b'full_v2_20261025/code/' in Path(f'/proc/{pid}/cmdline').read_bytes()
 except OSError:return False
def readstates():
 return [json.loads(p.read_text()) for p in (R/'state').glob('*.json')]
def main():
 lock=(R/'operations/boost.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.time();children=[];logs=[];history=[];last_report=0;last_target=0
 while True:
  states=readstates();done=sum(s['stage']=='complete' for s in states);pending=692-sum(s['stage']=='complete' or (s['stage']=='error' and s.get('attempt',0)>=3) for s in states)
  runtime=json.loads((R/'runtime.json').read_text());original={int(p):str(g) for p,g in zip(runtime.get('workers',[]),runtime.get('gpu_slots',[])) if alive(p)}
  own={p.pid:g for p,g in children if p.poll() is None};workers={**original,**own};counts=collections.Counter(workers.values())
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=30)
  gpus={};uuids={}
  for line in raw.strip().splitlines():
   idx,uid,total,free,util=[x.strip() for x in line.split(',')];gpus[idx]={'total':int(total),'free':int(free),'util':int(util),'external':0};uuids[uid]=idx
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],text=True,timeout=30)
  for line in apps.strip().splitlines():
   pid,mem,uid=[x.strip() for x in line.split(',')]
   if int(pid) not in workers and mem.isdigit():gpus[uuids[uid]]['external']+=int(mem)
  capacities={g:max(0,min(8,(v['total']-v['external']-16000)//13000)) for g,v in gpus.items()}
  # First add to 32, then use memory-budgeted capacity after warm-up. Existing jobs are never killed.
  target=min(sum(capacities.values()),32 if time.time()-started<90 else 64)
  cpu_pressure=float(Path('/proc/pressure/cpu').read_text().split('avg10=')[1].split()[0]);load=os.getloadavg()[0]
  if cpu_pressure>30 or load>240:target=min(target,len(workers))
  while len(workers)<target and len(workers)<pending:
   candidates=[g for g in gpus if counts[g]<capacities[g] and gpus[g]['free']>=18000]
   if not candidates:break
   g=min(candidates,key=lambda g:(counts[g]/max(capacities[g],1),-gpus[g]['free']));i=len(children);f=(R/'logs'/f'boost_{i:03d}_gpu{g}.log').open('a');logs.append(f)
   p=subprocess.Popen([PYTHON,str(R/'code/pipeline.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);children.append((p,g));workers[p.pid]=g;counts[g]+=1
  event={'time':time.time(),'elapsed_seconds':time.time()-started,'completed_sources':done,'stages':dict(collections.Counter(s['stage'] for s in states)),'active_workers':len(workers),'workers_per_gpu':dict(counts),'memory_budget_capacities':capacities,'gpus':gpus,'cpu_load1':load,'cpu_pressure_avg10':cpu_pressure,'pids':list(workers),'controller_pid':os.getpid(),'original_supervisor_pid':runtime.get('supervisor_pid'),'target':target,'pipeline_code_unchanged':True}
  atomic(R/'operations/boost_status.json',event)
  with (R/'operations/boost_metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  if target!=last_target:print('TARGET',target,'ACTIVE',len(workers),'GPU_COUNTS',dict(counts),flush=True);last_target=target
  if time.time()-last_report>120:subprocess.run([sys.executable,str(R/'code/report.py')],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT);last_report=time.time()
  if not pending and not workers and not alive(runtime.get('supervisor_pid',0)):break
  time.sleep(30)
 subprocess.run([sys.executable,str(R/'code/report.py'),'--final'],check=True)
 subprocess.run([PYTHON,str(R/'code/export_samples.py')],check=True)
 states=readstates();atomic(R/'runtime.json',{'stage':'complete' if len(states)==692 and all(s['stage']=='complete' for s in states) else 'finished_with_unresolved','completed':sum(s['stage']=='complete' for s in states),'sources':692,'finished_at':time.time(),'boost_controller_pid':os.getpid()})
if __name__=='__main__':main()
