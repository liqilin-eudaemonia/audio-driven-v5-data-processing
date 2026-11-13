#!/usr/bin/env python3
"""Persist 20-minute reports and bounded recovery of explicitly registered own services."""
import argparse,datetime,json,os,signal,subprocess,time,urllib.request
from pathlib import Path

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def read(path,default=None):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--hours',type=float,default=8);a=p.parse_args()
 root=Path(a.root).resolve();watch=root/'watchdog';watch.mkdir(exist_ok=True)
 import fcntl
 lock=(watch/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.time();end=start+a.hours*3600;last_report=0;restarts={};first_counts=None
 write(watch/'process.json',dict(pid=os.getpid(),started_at=start,deadline=end,check_seconds=60,report_seconds=1200))
 while time.time()<end:
  now=time.time();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();alerts=[];services=[]
  config=read(root/'watch_services.json',dict(services=[]))
  for spec in config['services']:
   record=read(Path(spec['record']))
   if not record:continue
   pid=record['pid'];proc=Path('/proc')/str(pid)/'cmdline';alive=proc.exists() and bool(proc.read_bytes())
   health=alive
   if alive and spec.get('url'):
    try:
     with urllib.request.urlopen(spec['url'],timeout=5) as response:health=response.status==200
    except Exception:health=False
   state=dict(name=spec['name'],pid=pid,alive=alive,healthy=health);services.append(state)
   if not alive and spec.get('autorestart',False) and now<record.get('stop_at',record.get('deadline',float('inf'))) and not (root/'maintenance').exists():
    recent=[t for t in restarts.get(spec['name'],[]) if now-t<3600]
    if len(recent)<3:
     log=open(root/(spec['name']+'.recovery.log'),'a')
     child=subprocess.Popen(record['command'],cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,**spec.get('env',{})))
     record['previous_pid']=pid;record['pid']=child.pid;write(Path(spec['record']),record)
     restarts[spec['name']]=recent+[now];state['restarted_pid']=child.pid
    else:alerts.append(dict(service=spec['name'],reason='Three restarts in one hour; needs diagnosis'))
   elif alive and not health:alerts.append(dict(service=spec['name'],reason='Process alive but health request failed; not killing a possibly busy engine'))
  worker_alert=read(root/'production'/'worker_alert.json')
  if worker_alert:alerts.append(dict(service='production-worker',details=worker_alert))
  counts=dict(processed=0,schema_pass=0,failed=0,supervision_supported=0);shards=[]
  for path in (root/'production'/'state').glob('*.json'):
   state=read(path)
   if not state:continue
   shards.append(state)
   heartbeat=state.get('heartbeat')
   if heartbeat and not state.get('done'):
    age=now-datetime.datetime.fromisoformat(heartbeat).timestamp()
    if age>1200:alerts.append(dict(service='production-shard',shard=state['shard'],reason='Heartbeat older than 20 minutes',age_seconds=age))
   for key in counts:counts[key]+=state.get(key,0)
  index=read(root/'queue'/'index.json') or read(root/'queue'/'progress.json',{})
  comparison=read(root/'comparison50'/'summary.json',{})
  try:
   r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
   gpu=r.stdout.strip();gpu_error=r.stderr.strip() if r.returncode else None
  except Exception as exc:gpu='';gpu_error=str(exc)
  snapshot=dict(utc=utc,elapsed_seconds=now-start,remaining_seconds=max(0,end-now),counts=counts,index_counts=index.get('counts'),comparison_completed=comparison.get('completed'),comparison_total=comparison.get('total_requests'),services=services,alerts=alerts,gpu=gpu,gpu_error=gpu_error,active_shards=len(shards),training_admitted=False)
  if first_counts is None:first_counts=(now,counts['schema_pass'])
  elapsed=now-first_counts[0];rate=(counts['schema_pass']-first_counts[1])/elapsed if elapsed>0 else 0
  snapshot['schema_pass_per_hour_since_watch_start']=rate*3600
  eligible=(index.get('counts') or {}).get('eligible',0);snapshot['remaining_candidates_lower_bound']=max(0,eligible-counts['processed'])
  snapshot['eta_hours_at_observed_rate']=snapshot['remaining_candidates_lower_bound']/rate/3600 if rate>0 else None
  write(watch/'latest.json',snapshot)
  if alerts:write(watch/'alerts.json',dict(utc=utc,alerts=alerts))
  if now-last_report>=1200:
   stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');write(watch/'reports'/(stamp+'.json'),snapshot);last_report=now
   print(json.dumps(snapshot,ensure_ascii=False),flush=True)
  time.sleep(60)
 write(watch/'deadline_reached.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),note='Eight-hour reporting window reached. This watcher does not delete outputs or stop other jobs.'))
if __name__=='__main__':main()
