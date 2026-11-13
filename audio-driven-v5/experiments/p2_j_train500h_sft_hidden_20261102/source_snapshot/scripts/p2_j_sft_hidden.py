#!/usr/bin/env python3
"""Resume-safe four-node guided-H extraction after audited teacher completion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,read,sha


def environment(gpu=None):
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    return env


def run_node(a):
    from p2_j_sft_hidden import Reader,require_ready
    require_ready(Reader(a.work))
    folder=Path(a.work)/'nodes'/a.node;folder.mkdir(parents=True,exist_ok=True)
    children={};logs=[];attempts={s:0 for s in range(8*a.per_gpu)}
    def launch(slot):
        wid=a.offset+slot;gpu=slot//a.per_gpu
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--work',a.work,
                 '--worker-id',str(wid),'--producers',str(a.producers),'--prefetch',str(a.prefetch)]
        log=open(folder/f'worker{wid}.log','a');logs.append(log)
        child=subprocess.Popen(command,env=environment(gpu),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[slot]=child
        atomic_json(folder/f'worker{wid}.pid.json',dict(pid=child.pid,gpu=gpu,command=command,time=time.time()))
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGHUP,stop)
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for slot in attempts:launch(slot)
            while True:
                for slot,child in list(children.items()):
                    if child.poll() not in (None,0) and attempts[slot]<2:
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        attempts[slot]+=1;launch(slot)
                states={a.offset+s:p.poll() for s,p in children.items()}
                atomic_json(folder/'status.json',dict(time=time.time(),node=a.node,pid=os.getpid(),states=states,
                            per_gpu=a.per_gpu,producers=a.producers,prefetch=a.prefetch))
                gpu=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=15)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=gpu.stdout))+'\n')
                if all(x is not None for x in states.values()):break
                time.sleep(15)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
            if any(x!=0 for x in states.values()):raise RuntimeError('Extraction worker failed; inspect node logs')
        finally:
            for child in children.values():
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            for log in logs:log.close()


def remote(cluster,node,args,timeout=45):
    from p2_planning_cluster import ssh
    r=subprocess.run(ssh(node,cluster)+[shlex.join(args)],capture_output=True,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(node['name']+': '+r.stderr[-3000:])
    return r.stdout


def resource_probe(cluster,node):
    """Fail closed on query errors or any foreign compute process."""
    try:
        gpu=remote(cluster,node,['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
        processes=remote(cluster,node,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
        rows=[[float(x.strip()) for x in line.split(',')] for line in gpu.strip().splitlines()]
        available=(len(rows)==8 and {int(r[0]) for r in rows}==set(range(8))
                   and all(r[1]>=100*1024 and r[2]<=10 for r in rows) and not processes.strip())
        return dict(node=node['name'],available=available,gpu=rows,compute_processes=processes.strip())
    except Exception:return dict(node=node['name'],available=False,error=traceback.format_exc())


def launch_remote(cluster,node,work,offset,a):
    """Idempotent launch; node.lock protects against double workers on resume."""
    code='''import os,json,subprocess,sys,time
from pathlib import Path
root,work,node,offset,per_gpu,producers,prefetch,script=sys.argv[1:]
folder=Path(work)/'nodes'/node;folder.mkdir(parents=True,exist_ok=True)
launch=folder/'launch.json'
if launch.exists():
 old=json.loads(launch.read_text());pid=old['pid']
 try:
  cmd=Path('/proc')/str(pid)/'cmdline'
  text=cmd.read_bytes().replace(b'\\0',b' ').decode()
  if script in text and work in text and ' node ' in text:
   print(json.dumps(dict(already_running=True,pid=pid)));sys.exit(0)
 except FileNotFoundError:pass
cmd=[sys.executable,script,'node','--work',work,'--node',node,'--offset',offset,'--per-gpu',per_gpu,'--producers',producers,'--prefetch',prefetch]
env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (folder/'supervisor.log').open('a') as log:
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=p.pid,command=cmd,time=time.time());tmp=launch.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(launch)
print(json.dumps(row))'''
    return json.loads(remote(cluster,node,[cluster['python'],'-c',code,str(ROOT),str(work),node['name'],str(offset),
            str(a.per_gpu),str(a.producers),str(a.prefetch),str(Path(__file__).resolve())]))


def chain(a):
    from p2_j_sft_hidden import Reader,predecessor_ready,require_ready,status,audit
    work=Path(a.work);reader=Reader(work);cluster=read(a.cluster)
    proof=read(work/'CPU_CHECK.json')
    if not proof['success'] or proof['protocol_sha256']!=reader.protocol_sha:
        raise ValueError('Matching CPU input proof is required before scheduling')
    if len(cluster['nodes'])!=4 or any(n['gpus']!=list(range(8)) for n in cluster['nodes']):
        raise ValueError('Expected the authorized four nodes and all 32 cards')
    execution=dict(cluster=cluster,per_gpu=a.per_gpu,producers=a.producers,prefetch=a.prefetch,
                   native_batch=1,writer_threads=2,job_size=128,protocol_sha256=reader.protocol_sha)
    atomic_json(work/'execution.json',execution,immutable=True)
    def state(stage,**kw):atomic_json(work/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (work/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (work/'COMPLETE.json').exists():state('complete',result=read(work/'COMPLETE.json'));return
            while not predecessor_ready(reader.protocol):
                teacher=Path(reader.protocol['teacher'])/'cluster_status.json'
                p=read(teacher) if teacher.exists() else {}
                state('waiting_teacher_audit',teacher_samples=p.get('samples'),teacher_total=p.get('total'),teacher_failures=p.get('failures'))
                time.sleep(30)
            atomic_json(work/'predecessor_verified.json',dict(complete_sha256=sha(Path(reader.protocol['teacher'])/'COMPLETE.json'),time=time.time()))
            # A resumed production chain must not wait for its own GPU workers.
            if not (work/'production_started.json').exists():
                stable=0
                while stable<2:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        rows=list(pool.map(lambda n:resource_probe(cluster,n),cluster['nodes']))
                    stable=stable+1 if all(r['available'] for r in rows) else 0
                    state('waiting_resource_release',resource_checks=rows,consecutive_clear=stable)
                    if stable<2:time.sleep(30)
                if not (work/'CANARY.json').exists():
                    state('gpu_canary')
                    cmd=[sys.executable,str(Path(__file__).resolve()),'canary','--work',str(work)]
                    with (work/'canary.log').open('a') as log:
                        child=subprocess.Popen(cmd,env=environment(0),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        atomic_json(work/'canary_launch.json',dict(pid=child.pid,command=cmd,time=time.time()))
                        while child.poll() is None:
                            state('gpu_canary',canary_pid=child.pid);time.sleep(15)
                        if child.returncode:raise RuntimeError('GPU canary failed; full extraction remains blocked; see canary.log')
                require_ready(reader)
                atomic_json(work/'production_started.json',dict(time=time.time(),protocol_sha256=reader.protocol_sha),immutable=True)
            require_ready(reader)
            offsets={'node8':0,'node17':100,'node12':200,'node4':300}
            with ThreadPoolExecutor(max_workers=4) as pool:
                launches=list(pool.map(lambda n:launch_remote(cluster,n,work,offsets[n['name']],a),cluster['nodes']))
            state('extracting',launches=launches)
            idle_since=time.monotonic();last=-1
            while True:
                s=status(work);atomic_json(work/'cluster_status.json',s)
                with (work/'cluster_history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in s.items() if k!='workers'})+'\n')
                state('extracting',samples=s['samples'],total=s['total'],failures=s['failures'])
                if s['failures']:raise RuntimeError('Extraction has failed blocks; retained for diagnosis, not skipped silently')
                if s['samples']==s['total']:
                    state('auditing');result=audit(work);state('complete',result=result);return
                if s['samples']!=last:last=s['samples'];idle_since=time.monotonic()
                if time.monotonic()-idle_since>1800:raise RuntimeError('No completed block for 30 minutes; inspect worker/node logs')
                time.sleep(30)
        except BaseException:
            state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','cpu-check','canary','worker','node','chain','status','audit'])
    p.add_argument('--work',required=True);p.add_argument('--source');p.add_argument('--teacher');p.add_argument('--adapter');p.add_argument('--canary-ids')
    p.add_argument('--cluster',default=str(ROOT/'configs/p2_planning_cluster_filtered_20261102.json'))
    p.add_argument('--node',default='node12');p.add_argument('--offset',type=int,default=0);p.add_argument('--worker-id',type=int,default=0)
    p.add_argument('--per-gpu',type=int,default=2);p.add_argument('--producers',type=int,default=8);p.add_argument('--prefetch',type=int,default=8)
    a=p.parse_args()
    from p2_j_sft_hidden import prepare_work,cpu_check,canary,worker,status,audit
    if a.command=='prepare':
        for name in ('source','teacher','adapter','canary_ids'):
            if not getattr(a,name):p.error('--'+name.replace('_','-')+' is required for prepare')
        result=prepare_work(a.source,a.teacher,a.adapter,a.work,read(a.canary_ids));print(json.dumps(dict(samples=result['samples'],work=a.work)))
    elif a.command=='canary':print(json.dumps(canary(a.work)))
    elif a.command=='cpu-check':
        r=cpu_check(a.work);print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
    elif a.command=='worker':worker(a.work,a.worker_id,a.producers,a.prefetch)
    elif a.command=='node':run_node(a)
    elif a.command=='chain':chain(a)
    elif a.command=='audit':print(json.dumps(audit(a.work)))
    else:
        s=status(a.work);s.pop('workers',None);print(json.dumps(s))


if __name__=='__main__':main()
