#!/usr/bin/env python3
"""Own only this run's producers/workers; finish extraction, merge and selection."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_j_upright_av500h_v3 as task
import listen_richness_cluster_control as cluster
from listen_benchmark.j_annotation import acquire,write
W=task.WORK
PY='/private/software/conda/envs/vllm/bin/python'
VENDOR=str(W/'vendor')

def environment():
 return dict(os.environ,PYTHONPATH=VENDOR,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',TOKENIZERS_PARALLELISM='false')

def local_child(role):
 p=W/(role+'-child.json');old=task.read(p) if p.exists() else None
 if old:
  stat=Path('/proc')/str(old['pid'])/'stat'
  if stat.exists():
   parts=stat.read_text().rsplit(')',1)[1].split()
   if parts[0] not in ('Z','X') and int(parts[19])==old['start_ticks']:return old
 cmd=[PY,str(ROOT/'scripts/run_j_upright_av500h_v3.py'),role]
 with (W/'logs'/f'{role}.log').open('a') as f:child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=environment())
 ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]);record=dict(pid=child.pid,start_ticks=ticks,cmd=cmd,started_at=time.time());write(p,record);return record

def launch_producers():
 return cluster.all_nodes(W,cluster.REMOTE_COMMON+'''
nodework=W/'nodes'/NODE
p=nodework/'manifest-producer.json';old=read(p) if p.exists() else None
if alive(old):child=old
else:
 env=environment();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(W/'vendor'),TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='8')
 cmd=['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/run_j_upright_av500h_v3.py','prepare','--node',NODE]
 child=spawn(cmd,nodework/'logs/manifest-producer.log',env);save(p,child)
print(json.dumps(child))
''')

def launch_workers():
 # Existing launch preserves GPU UUID/memory checks and checks owned PID ticks.
 return cluster.launch(W,'production')

def wait_subprocess(command,stage):
 write(W/'RUN_STATUS.json',dict(stage=stage,at=time.time(),pid=os.getpid()))
 log=W/'logs'/f'{stage}.log'
 with log.open('a') as f:p=subprocess.Popen([PY,str(ROOT/'scripts/run_j_upright_av500h_v3.py'),command],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment())
 write(W/(stage+'-child.json'),dict(pid=p.pid,started_at=time.time(),cmd=command,log=str(log)))
 while p.poll() is None:
  write(W/'RUN_STATUS.json',dict(stage=stage,at=time.time(),pid=os.getpid(),child_pid=p.pid));time.sleep(10)
 if p.returncode:raise RuntimeError(stage+' failed: '+str(log))

def main():
 lock=acquire(W/'controller.lock',False)
 if lock is None:raise RuntimeError('Controller already active')
 ensure=task.ensure;cfg=task.read(W/'config.json')
 ensure(task.read(W/'CPU_VALIDATION.json')['passed'],'CPU validation failed')
 write(W/'RUN_STATUS.json',dict(stage='launching',at=time.time(),pid=os.getpid()))
 local_child('coordinate')
 write(W/'producer_launch.json',launch_producers())
 cluster.start_resources(W)
 launched=launch_workers();write(W/'worker_launch.json',launched)
 ensure(sum(len(v) for v in launched.values())==32,'Not all 32 workers launched; inspect occupancy')
 started=time.time();history=[];last_launch=time.time();restarts=0
 try:
  while True:
   ss=cluster.workers(W);qpath=W/'queues/extract/state.json';q=task.read(qpath) if qpath.exists() else {'completed_clips':0,'inflight':{},'failed':{}}
   completed=q.get('completed_clips',0);now=time.time();history.append((now,completed));history=[x for x in history if now-x[0]<360]
   rate=(completed-history[0][1])/(now-history[0][0]) if len(history)>1 and now>history[0][0] else 0
   prep=task.read(W/'preparation.json') if (W/'preparation.json').exists() else {}
   stage_counts={k:sum(s['stage']==k for s in ss.values()) for k in sorted({s['stage'] for s in ss.values()})}
   state=dict(stage='preparing_and_extracting',at=now,pid=os.getpid(),elapsed_s=now-started,completed=completed,expected=task.N,rate_window_clips_per_s=rate,remaining_seconds_at_recent_rate=(task.N-completed)/rate if rate>0 else None,worker_stages=stage_counts,workers_seen=len(ss),inflight_chunks=len(q['inflight']),failed_chunks=q['failed'],preparation=prep)
   write(W/'RUN_STATUS.json',state)
   if q['failed']:raise RuntimeError('Extraction failed queue is nonempty')
   pq=W/'prepare/queues/prepare/state.json'
   if pq.exists() and task.read(pq)['failed']:raise RuntimeError('Preparation failed queue is nonempty')
   if completed==task.N and (W/'producer_complete.json').exists() and not q['inflight']:
    if all(s['stage']=='complete' for s in ss.values()) and len(ss)==32:break
   if now-last_launch>90:
    if any(s['stage']=='error' for s in ss.values()):
     restarts+=1;ensure(restarts<=3,'Workers repeatedly failed; retained outputs are resumable')
     write(W/'worker_launch.json',launch_workers())
    if not (W/'producer_complete.json').exists():
     local_child('coordinate');write(W/'producer_launch.json',launch_producers())
    last_launch=now
   time.sleep(10)
  wait_subprocess('merge','merging_statistics')
  ensure(task.read(W/'complete.json')['status']=='complete','Statistics have unresolved inputs')
  wait_subprocess('select','selecting_500h')
  summary=task.read(W/'selection/SUMMARY.json')
  (W/'STOP_RESOURCES').write_text('task complete')
  write(W/'TASK_COMPLETE.json',dict(status='complete',samples=summary['samples'],hours=summary['hours'],selection_summary_sha256=task.sha(W/'selection/SUMMARY.json'),statistics_summary_sha256=task.sha(W/'tables/summary.json'),manifest_sha256=task.EXPECTED_SHA,at=time.time()))
  write(W/'RUN_STATUS.json',dict(stage='complete',at=time.time(),samples=summary['samples'],hours=summary['hours'],statistics_samples=task.N))
 except BaseException as exc:
  write(W/'CONTROLLER_ERROR.json',dict(error=repr(exc),at=time.time(),pid=os.getpid()));raise
 finally:lock.close()
if __name__=='__main__':main()
