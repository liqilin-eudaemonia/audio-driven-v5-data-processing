#!/usr/bin/env python3
"""Shared-lock queue: matching old parents, 24 GPUs per arm, guarded retries."""
from pathlib import Path
import argparse,fcntl,json,os,socket,sys,time,threading,subprocess
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json,sha
from listen_cluster import launch,distributed_commands,transport

ARMS=('E0','E1','E0F','E1F')

def latest(out):
    from p2_matrix16.checkpoint import checked
    for marker in sorted(out.glob('step_*.complete.json'),reverse=True):
        path=marker.with_suffix('').with_suffix('.pt')
        if not path.exists():continue
        try:
            value=checked(path)
            if value['step']!=read(marker)['step']:raise ValueError('Checkpoint step mismatch')
            return path,value['step']
        except Exception as e:print('Rejected checkpoint',path,str(e),flush=True)
    return None,0

def resources(cluster):
    def one(node):
        cmd=['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        r=subprocess.run(transport(cluster,node,cmd),capture_output=True,text=True,timeout=30)
        if r.returncode:raise RuntimeError(r.stderr[-500:])
        rows=[[int(v.strip()) for v in line.split(',')] for line in r.stdout.splitlines()]
        if len(rows)!=8:raise RuntimeError('Expected eight GPUs per node')
        return dict(node=node['host'],gpus=rows)
    with ThreadPoolExecutor(max_workers=3) as pool:return list(pool.map(one,cluster['nodes']))

def run(root):
    from p2_legacy_qwen_lora import backend
    cluster=read(root/'cluster.json');queue=root/'queue';manifest=read(queue/'frozen.json')
    def state(**kw):atomic_json(queue/'state.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),**kw))
    def verify():
        for path,want in manifest.items():
            if sha(path)!=want:raise RuntimeError('Frozen source/config changed: '+path)
    for arm in ARMS:
        config=root/'configs'/f'{arm}.json';c=read(config);out=Path(c['output'])
        verify()
        checkpoint,step=latest(out)
        if checkpoint and step==1268:
            state(arm=arm,status='arm_complete',step=step);continue
        for retry in range(12):
            if (queue/'STOP').exists():state(arm=arm,status='stopped');return
            verify();checkpoint,step=latest(out)
            if checkpoint and step==1268:break
            while True:
                try:
                    inventory=resources(cluster)
                    minimum=cluster.get('minimum_free_gib',67)
                    enough=all(g[1]>=minimum*1024 for n in inventory for g in n['gpus'])
                    if enough:break
                    state(arm=arm,status='waiting_for_memory',inventory=inventory,minimum_free_gib=minimum)
                except Exception as e:state(arm=arm,status='retrying_resource_check',error=str(e))
                if (queue/'STOP').exists():return
                time.sleep(20)
            if not checkpoint and (out/'run.json').exists():
                # A failed initial model registration before the atomic step-zero save.
                # Prior launch has fully exited; preserve metadata instead of overwriting.
                (out/'run.json').rename(out/f'run.failed_start.{time.time_ns()}.json')
            tag=f'{arm}-{time.time_ns()}'
            logs=queue/'attempts'/tag
            cc=dict(cluster);cc['master_port']=29991+retry
            extra=['train','--config',str(config)]
            if checkpoint:extra+=['--resume',str(checkpoint)]
            # Exercise an actual 24-rank restart after the first two production steps.
            # It preserves optimizer/RNG/data order and consumes no extra samples.
            if arm=='E0' and step<2 and not (queue/'resume_verified.json').exists():extra+=['--stop-after','2']
            commands=distributed_commands(cc,Path(cc.get('training_script',str(ROOT/'scripts/p2_legacy_qwen_lora.py'))),extra)
            cancel=threading.Event();result=[]
            def execute():
                try:launch(cc,commands,logs,cancel_event=cancel);result.append(None)
                except Exception as e:result.append(str(e))
            thread=threading.Thread(target=execute);thread.start();begin=time.time()
            state(arm=arm,status='running',resume=str(checkpoint) if checkpoint else None,step=step,attempt=tag)
            while thread.is_alive():
                thread.join(10)
                progress=read(out/'status.json') if (out/'status.json').exists() else {}
                metric=out/'metrics.jsonl';changed=max(begin,metric.stat().st_mtime if metric.exists() else begin)
                state(arm=arm,status='running',step=progress.get('step',step),attempt=tag,resume=str(checkpoint) if checkpoint else None)
                if time.time()-changed>1200 or (queue/'STOP').exists():cancel.set()
            if (queue/'STOP').exists():state(arm=arm,status='stopped');return
            if result==[None]:
                done,last=latest(out)
                if last==1268:break
                if arm=='E0' and last==2:
                    atomic_json(queue/'resume_verified.json',dict(checkpoint=str(done),phase='checkpoint_saved_restart_pending'))
                    continue
            state(arm=arm,status='retrying',error=result,attempt=tag)
            time.sleep(15)
        else:raise RuntimeError('Twelve attempts failed for '+arm)
    state(status='complete',arms=list(ARMS))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    q=root/'queue';q.mkdir(exist_ok=True)
    # Three independent node daemons, one shared filesystem lock. If the active
    # controller dies its guard pipes close, and a standby takes over and resumes.
    with (q/'controller.lock').open('a') as lock:
        while True:
            if (q/'STOP').exists():return
            if (q/'state.json').exists() and read(q/'state.json').get('status')=='complete':return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:run(root);return
            except Exception as e:
                atomic_json(q/'state.json',dict(status='blocked',error=repr(e),at=time.time(),host=socket.gethostname()))
                print(repr(e),flush=True);return
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':main()
