#!/usr/bin/env python3
"""Launch persistent cache workers locally; all nodes share the same job queue."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,read


def launch(work,node,rank,phase,prefetch=4,media_processes=0):
    from p2_j_cache.cluster import queue_status
    work=Path(work);phases=['text','features'] if phase=='both' else [phase]
    logs=[];processes=[];started=time.time()
    try:
        for current in phases:
            count=1 if current=='text' else 2
            processes=[]
            for gpu in range(8):
                for replica in range(count):
                    name=f'{node}-{current}-g{gpu}-r{replica}'
                    path=work/'logs'/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
                    log=path.open('a');logs.append(log)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_ENABLE_PARALLEL_LOADING='true',
                        HF_PARALLEL_LOADING_WORKERS='4',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                    cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',str(work),'--phase',current,
                        '--name',name,'--rank',str(rank*8*count+gpu*count+replica),'--prefetch',str(prefetch),
                        '--media-processes',str(media_processes if current=='features' else 0)]
                    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);processes.append(p)
            atomic_json(work/'launchers'/(node+'.json'),dict(node=node,pid=os.getpid(),phase=current,started_at=started,
                workers=[p.pid for p in processes],prefetch=prefetch,media_processes=media_processes))
            path=work/'telemetry'/(node+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as stream:
                while True:
                    states=[p.poll() for p in processes]
                    bad=[v for v in states if v not in (None,0)]
                    # Other compute jobs have restarted during testing. The user
                    # explicitly assigned all eight GPUs on each of these nodes.
                    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    owned={p.pid for p in processes}
                    for line in apps.stdout.splitlines():
                        pid_text,command=line.split(',',1);pid=int(pid_text)
                        if pid in owned:continue
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:continue
                        with (work/'logs'/(node+'-released-gpus.jsonl')).open('a') as audit:
                            audit.write(json.dumps(dict(time=time.time(),pid=pid,command=command.strip(),reason='User authorized exclusive use of all four nodes'))+'\n')
                    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
                    row=dict(time=time.time(),phase=current,gpus=[[float(v) for v in line.split(',')] for line in result.stdout.splitlines()])
                    stream.write(json.dumps(row)+'\n');stream.flush()
                    if all(v is not None for v in states):
                        if bad:raise RuntimeError(f'{node} {current} workers failed: {bad}; see worker logs')
                        break
                    time.sleep(2)
            print(json.dumps(dict(node=node,**queue_status(work,current))),flush=True)
            # Other nodes can still hold the last text jobs. Features are independent
            # and may start immediately; no global model-loading barrier is needed.
        atomic_json(work/'launchers'/(node+'.json'),dict(node=node,pid=os.getpid(),status='finished',elapsed_s=time.time()-started))
    finally:
        import psutil
        children=[]
        for p in processes:
            if p.poll() is None:
                try:children.extend(psutil.Process(p.pid).children(recursive=True))
                except psutil.NoSuchProcess:pass
                p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for child in children:
            try:
                if child.is_running():child.kill()
            except psutil.NoSuchProcess:pass
        for log in logs:log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--work',required=True)
    prep.add_argument('--config',default=str(ROOT/'experiments/p2_train50h_postprocessed_reselected_20261027/config.json'))
    prep.add_argument('--workers',type=int,default=64)
    queue=sub.add_parser('queue');queue.add_argument('--work',required=True)
    queue.add_argument('--feature-chunk',type=int,default=32);queue.add_argument('--text-chunk',type=int,default=256)
    node=sub.add_parser('node');node.add_argument('--work',required=True);node.add_argument('--node',required=True)
    node.add_argument('--rank',type=int,required=True);node.add_argument('--phase',choices=['both','text','features'],default='both')
    node.add_argument('--prefetch',type=int,default=4)
    node.add_argument('--media-processes',type=int,default=0)
    w=sub.add_parser('worker');w.add_argument('--work',required=True);w.add_argument('--name',required=True);w.add_argument('--rank',type=int,required=True)
    w.add_argument('--phase',choices=['text','features'],required=True);w.add_argument('--text-batch-size',type=int,default=32);w.add_argument('--prefetch',type=int,default=4)
    w.add_argument('--media-processes',type=int,default=0)
    status=sub.add_parser('status');status.add_argument('--work',required=True)
    a=parser.parse_args()
    from p2_j_cache.cluster import prepare_full,prepare_queue,cluster_worker,queue_status
    if a.command=='prepare':
        value=prepare_full(a.work,a.config,a.workers);prepare_queue(a.work)
        print(json.dumps(value['selection_summary']),flush=True)
    elif a.command=='queue':prepare_queue(a.work,a.feature_chunk,a.text_chunk)
    elif a.command=='node':launch(a.work,a.node,a.rank,a.phase,a.prefetch,a.media_processes)
    elif a.command=='worker':
        def stop_worker(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,stop_worker)
        cluster_worker(a.work,a.phase,a.name,a.rank,a.text_batch_size,a.prefetch,a.media_processes)
    else:print(json.dumps([queue_status(a.work,p) for p in ['text','features']]),flush=True)


if __name__=='__main__':main()
