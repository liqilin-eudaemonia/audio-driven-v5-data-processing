"""Observe supervised staged rollout, audit each new worker, then measure all eight."""
from pathlib import Path
import json,time,subprocess,os
W=Path(__file__).resolve().parent

def read(name):return json.loads((W/name).read_text())
def save(name,data):
 p=W/name;t=p.with_suffix('.observe-tmp');t.write_text(json.dumps(data,indent=2));os.replace(t,p)
def live(record):
 try:
  f=Path(f'/proc/{record["pid"]}/stat').read_text().rsplit(')',1)[1].split()
  return f[0]!='Z' and f[19]==str(record['start_ticks'])
 except (FileNotFoundError,KeyError):return False

def main():
 started=time.time();captures={};audited=set();targets=[f'gpu-{i}' for i in range(1,7)]
 save('monitor_snapshots/process_all_rollout_observer.json',{'pid':os.getpid(),'started_at':started})
 while time.time()-started<5400:
  s=read('supervisor.json');statuses={}
  if not live(s) or not live(read('watchdog.json')):raise RuntimeError('A supervision guard is not live')
  for name in targets:
   c=s['children'][name];h=read(f'workers/{name}.json')
   statuses[name]={'pid':c['pid'],'live':live(c),'revision':h.get('revision'),'completed':h.get('completed_clips',0),'stage':h.get('stage')}
   if h.get('pid')!=c['pid'] or h.get('revision')!=11 or not live(c):continue
   if h['execution'].get('decode_backend')!='process' or not h['execution'].get('rolling_inference'):
    raise RuntimeError(f'Unexpected execution on {name}')
   if name not in captures:
    q=read('queues/vlm/state.json');p=W/f'monitor_snapshots/process_all_{name}_before.json'
    record={'at':time.time(),'snapshot':{'vlm_queue_range':{'offset':q['offset'],'inflight':list(q['inflight'])}},'worker':c,'initial_completed':h.get('completed_clips',0)}
    save(p.relative_to(W),record);captures[name]=record
   previous=captures[name]
   if c['pid']!=previous['worker']['pid'] or c['start_ticks']!=previous['worker']['start_ticks']:
    raise RuntimeError(f'{name} restarted during rollout validation')
   if h.get('failed_clips',0):raise RuntimeError(f'{name} reported input failures')
   if name not in audited and h.get('completed_clips',0)>=previous['initial_completed']+150:
    subprocess.run(['python3',str(W/'audit_rolling_canary.py'),'--worker',name,'--decode-backend','process','--count','100',
       '--tag',f'process_all_{name}_first100_audit','--queue-snapshot',str(W/f'monitor_snapshots/process_all_{name}_before.json')],check=True)
    audited.add(name)
  save('monitor_snapshots/process_all_rollout_status.json',{'at':time.time(),'started_at':started,'audited':sorted(audited),'statuses':statuses,'rolling':s.get('rolling_restart'),'warming':s.get('warming_gpus'),'complete':False})
  print(json.dumps({'at':time.time(),'audited':sorted(audited),'new_workers':sorted(captures),'counts':{k:v['completed'] for k,v in statuses.items()},'rolling':s.get('rolling_restart',{}).get('name'),'warming':s.get('warming_gpus')}),flush=True)
  if len(audited)==len(targets) and not s.get('rolling_restart') and not s.get('warming_gpus'):
   subprocess.run(['python3',str(W/'measure_runtime_window.py'),'--tag','process_all_card_stable_window','--seconds','600','--revision','11'],check=True)
   report=read('monitor_snapshots/process_all_card_stable_window.json')
   if not report['summary']['valid_for_throughput_comparison']:raise RuntimeError('All-card window was not stable')
   save('monitor_snapshots/process_all_rollout_complete.json',{'at':time.time(),'audited':sorted(audited),'measurement':'process_all_card_stable_window.json'})
   return
  time.sleep(30)
 raise TimeoutError('Rollout observation expired; production is not stopped by this observer')

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save('monitor_snapshots/process_all_rollout_observer_error.json',{'at':time.time(),'error':repr(exc)})
  raise
