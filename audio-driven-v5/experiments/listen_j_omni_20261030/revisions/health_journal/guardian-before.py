#!/usr/bin/env python3
"""Bounded ten-hour health window; restores only this pipeline's owned processes."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,acquire
KEY='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
REMOTE=r'''
import os,sys,time,signal,socket,subprocess
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_benchmark.j_annotation import read,write
from listen_j_annotation_supervisor import alive,identity,spawn
from listen_j_omni import environment,complete
work=Path(WORK);nw=work/'nodes'/NODE;cfg=read(work/'config.json');assert socket.gethostname()==cfg['nodes'][NODE]['hostname']
statepath=nw/'window_guard_state.json';previous=read(statepath) if statepath.exists() else {'suspects':{},'pending':{}}
actions=[];now=time.time();watch=read(nw/'watchdog.json') if (nw/'watchdog.json').exists() else None
stop=(nw/'STOP_SUPERVISION').exists() or complete(work)
if not stop and not alive(watch):
 child=spawn(nw,'window-watchdog-recovery',['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/listen_j_omni.py','watchdog','--work',WORK,'--node',NODE],environment());actions.append(dict(action='restore_watchdog',child=child))
s=read(nw/'supervisor.json') if (nw/'supervisor.json').exists() else {}
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
dev={int(x.split(',')[0]):list(map(float,x.split(',')[1:])) for x in raw.strip().splitlines()}
workers=[]
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 key=str(gpu);child=s.get('children',{}).get('gpu-'+key);hp=work/'workers'/f'{NODE}-gpu-{gpu}.json';h=read(hp) if hp.exists() else {}
 live=alive(child);matching=bool(child and h.get('pid')==child.get('pid'))
 progress=h.get('last_progress_at',h.get('started_at',now));age=now-progress
 token=[child.get('pid'),child.get('start_ticks'),progress,h.get('completed_clips'),h.get('held_clips'),h.get('benchmark_phase'),h.get('benchmark_clips')] if child else None
 suspect=previous['suspects'].get(key,{})
 suspect_count=suspect.get('count',0)+1 if suspect.get('token')==token else 1
 stalled=(live and matching and age>900 and h.get('stage') not in ('loading_model','waiting','stopped','complete') and dev[gpu][1]<=5)
 previous['suspects'][key]=dict(token=token,count=suspect_count if stalled else 0)
 pending=previous['pending'].get(key)
 if pending and now-pending['at']>120:
  if matching and (progress>pending['progress'] or token!=pending.get('progress_token',token)):
   previous['pending'].pop(key,None)
  else:
   killed=[]
   for rec in reversed(pending['owned_tree']):
    if alive(rec):os.kill(rec['pid'],signal.SIGKILL);killed.append(rec['pid'])
   if killed:actions.append(dict(action='kill_confirmed_stalled_owned_tree',gpu=gpu,pids=killed,evidence=pending))
   previous['pending'].pop(key,None)
 if not stop and stalled and suspect_count>=3 and not pending:
  cmd=Path(f"/proc/{child['pid']}/cmdline").read_bytes().decode().replace('\x00',' ')
  assert 'listen_j_omni.py worker' in cmd and WORK in cmd
  # Snapshot exact descendants and start ticks before signaling; PID reuse is
  # checked again before any escalation. Other jobs are never selected by GPU.
  pairs={}
  for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
   p,pp=map(int,line.split());pairs[p]=pp
  owned=[child['pid']]
  for _ in range(12):
   extra=[p for p,pp in pairs.items() if pp in owned and p not in owned]
   if not extra:break
   owned+=extra
  tree=[dict(pid=p,start_ticks=identity(p)) for p in owned]
  record=dict(at=now,progress=progress,progress_token=token,owned_tree=tree,worker=h,gpu=dev[gpu])
  previous['pending'][key]=record;os.kill(child['pid'],signal.SIGTERM)
  actions.append(dict(action='terminate_confirmed_stalled_worker',gpu=gpu,evidence=record))
 workers.append(dict(gpu=gpu,pid=child.get('pid') if child else None,alive=live,matching_heartbeat=matching,stage=h.get('stage'),completed=h.get('completed_clips',0),progress_age_s=age,heartbeat_age_s=now-h.get('heartbeat',now),memory_mib=dev[gpu][0],gpu_util=dev[gpu][1],execution_revision=h.get('execution_revision'),last_error=(h.get('last_error') or '')[-350:]))
previous['at']=now;write(statepath,previous)
cpu=dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
print(__import__('json').dumps(dict(node=NODE,at=now,hostname=socket.gethostname(),watchdog_alive=alive(watch),supervisor_alive=alive(s),supervisor_error=s.get('error'),blocked_gpus=s.get('blocked_gpus',{}),workers=workers,cpu_usage_usec=int(cpu['usage_usec']),actions=actions,stop_requested=stop)))
'''
def probe(work,node,host):
 code=f'ROOT={str(ROOT)!r}\nWORK={str(work)!r}\nNODE={node!r}\n'+REMOTE
 command=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@'+host,'python3 -']
 for attempt in range(3):
  try:
   r=subprocess.run(command,input=code,text=True,capture_output=True,timeout=40)
   if r.returncode:raise RuntimeError(r.stderr[-1500:] or r.stdout[-1500:])
   return json.loads(r.stdout.strip().splitlines()[-1])
  except (subprocess.TimeoutExpired,RuntimeError):
   if attempt==2:raise
   time.sleep(2*(attempt+1))
def main(work,start,end):
 folder=work/'monitoring_10h';folder.mkdir(parents=True,exist_ok=True);owner=acquire(folder/'leader.lock',False)
 if owner is None:return
 plan=folder/'plan.json'
 if plan.exists():p=read(plan);assert p['start']==start and p['end']==end
 else:write(plan,dict(start=start,end=end,report_interval_s=1800,health_poll_s=30,scope='three nodes, 24 Omni workers; restore owned watchdogs and confirmed stalled workers only'))
 from importlib.util import spec_from_file_location,module_from_spec
 spec=spec_from_file_location('supervisor',ROOT/'scripts/listen_j_annotation_supervisor.py');m=module_from_spec(spec);spec.loader.exec_module(m)
 state=dict(pid=os.getpid(),start_ticks=m.identity(os.getpid()),started_at=time.time(),start=start,end=end,completed_reports=0)
 last=read(folder/'state.json') if (folder/'state.json').exists() else {};reported=set(last.get('reported_slots',[]));cfg=read(work/'config.json')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while True:
   t=time.time();nodes={};fs={pool.submit(probe,work,n,c['host']):n for n,c in cfg['nodes'].items()}
   for f,n in fs.items():
    try:nodes[n]=f.result()
    except Exception as e:nodes[n]=dict(node=n,at=time.time(),observation_error=repr(e))
   q=read(work/'queues/caption/state.json');report=dict(at=time.time(),nodes=nodes,resolved_clips=q['completed_clips'],failed_sources=len(q['failed']))
   write(folder/'latest.json',report)
   slot=min(20,max(0,int((t-start)//1800)))
   if slot not in reported or t>=end:
    write(folder/f'report-{slot:02d}.json',report);reported.add(slot)
   state.update(heartbeat=time.time(),reported_slots=sorted(reported),completed_reports=len(reported),last_resolved_clips=q['completed_clips'],stage='complete' if t>=end else 'monitoring')
   write(folder/'state.json',state)
   if t>=end:write(folder/'complete.json',dict(at=time.time(),start=start,end=end,report_slots=sorted(reported)));break
   time.sleep(min(30,max(0,end-time.time())))
 owner.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True);a=p.parse_args();main(a.work.resolve(),a.start,a.end)
