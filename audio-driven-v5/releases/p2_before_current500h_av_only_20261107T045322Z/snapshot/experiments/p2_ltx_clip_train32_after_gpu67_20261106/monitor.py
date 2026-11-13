"""Read-only persistent 32-GPU utilization and throughput monitor."""
from pathlib import Path
import sys,os,json,time,datetime,subprocess,concurrent.futures,statistics,fcntl
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from listen_cluster import transport
OUT=Path(__file__).resolve().parent

def numeric(v):
 try:return float(v.strip())
 except ValueError:return None

def sample_node(cluster,node):
 try:
  command=['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits']
  r=subprocess.run(transport(cluster,node,command),capture_output=True,text=True,timeout=25)
  if r.returncode:return dict(node=node['name'],error=r.stderr[-600:])
  cards=[]
  for line in r.stdout.splitlines():
   fields=line.split(',')
   if len(fields)!=5:continue
   i,m,u,p,c=fields;cards.append(dict(index=int(i),memory_used_mib=numeric(m),utilization=numeric(u),power_w=numeric(p),sm_clock_mhz=numeric(c)))
  return dict(node=node['name'],gpus=cards)
 except Exception as exc:return dict(node=node['name'],error=str(exc))

def recent_metrics():
 p=OUT/'metrics.jsonl'
 if not p.exists():return []
 with p.open('rb') as f:
  size=p.stat().st_size;f.seek(max(0,size-1024*1024));raw=f.read().splitlines()
 if size>1024*1024:raw=raw[1:]
 rows=[]
 for line in raw:
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows[-50:]

def main():
 with (OUT/'monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cluster=read(OUT/'cluster.json');baseline=None;idle_streak={}
  while not (OUT/'MONITOR_STOP').exists():
   began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
   with concurrent.futures.ThreadPoolExecutor(4) as pool:nodes=list(pool.map(lambda n:sample_node(cluster,n),cluster['nodes']))
   rows=recent_metrics();migration=read(OUT/'MIGRATION.json') if (OUT/'MIGRATION.json').exists() else None
   queue=read(OUT/'QUEUE_STATE.json');step=rows[-1]['step'] if rows else None
   record=dict(time=utc,queue_state=queue['state'],step=step,nodes=nodes)
   alerts=[]
   if rows:
    warm=[r for r in rows if migration is None or r['step']>migration['step']+3][-20:]
    if warm:
     mean=statistics.mean(r['seconds'] for r in warm);wait=statistics.mean(r['input_wait_seconds'] for r in warm)
     record.update(window_steps=len(warm),mean_step_seconds=mean,median_step_seconds=statistics.median(r['seconds'] for r in warm),
       mean_input_wait_seconds=wait,input_wait_fraction=wait/max(mean,1e-8),compute_samples_per_second=32/max(mean,1e-8),
       rank0_peak_reserved_gib=max(r['peak_reserved_gib'] for r in warm),loss=rows[-1]['weighted_total'])
     if baseline is None:baseline=(time.monotonic(),step)
     elapsed=time.monotonic()-baseline[0]
     if elapsed>30:record['wall_samples_per_second']=32*(step-baseline[1])/elapsed
     if len(warm)>=10 and mean>15:alerts.append('mean_step_over_15s')
     if len(warm)>=10 and wait/max(mean,1e-8)>.2:alerts.append('input_wait_over_20_percent')
    utils=[];by_node={};low=[]
    for n in nodes:
     values=[g['utilization'] for g in n.get('gpus',[]) if g['utilization'] is not None]
     if values:by_node[n['node']]=statistics.mean(values);utils+=values
     if len(n.get('gpus',[]))!=8:alerts.append('telemetry_missing_'+n['node'])
     for g in n.get('gpus',[]):
      key=n['node']+':'+str(g['index']);idle_streak[key]=idle_streak.get(key,0)+1 if g['utilization'] is not None and g['utilization']<10 else 0
      if idle_streak[key]>=3:low.append(key)
    record.update(gpu_utilization_mean=statistics.mean(utils) if utils else None,gpu_utilization_by_node=by_node,low_utilization_streak_cards=low)
    if low and step is not None and migration and step>migration['step']+8:alerts.append('sustained_low_gpu_utilization')
    age=time.time()-(OUT/'metrics.jsonl').stat().st_mtime;record['seconds_since_progress']=age
    if age>180:alerts.append('no_step_progress_over_180s')
   record['alerts']=alerts
   atomic_json(OUT/'PERFORMANCE_STATUS.json',record)
   with (OUT/'performance.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   atomic_json(OUT/'PERFORMANCE_ALERTS.json',dict(time=utc,step=step,alerts=alerts))
   if queue['state'] in ('failed','finished'):break
   time.sleep(max(1,15-(time.monotonic()-began)))

if __name__=='__main__':main()
