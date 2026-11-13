#!/usr/bin/env python3
"""Wait for successful E1 and released GPUs, then launch and monitor matched E0."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from control import OUT,ROOT,TRAIN,PYTHON,E1,E0,HOSTS,HOSTNAMES,KEY,BUDGET,read,write,now,verify_release,latest_step,completed_gate,verify_checkpoint,verify_pair


def remote(rank,mode):
    cmd=[str(PYTHON),str(OUT/'run_node.py'),'--node-rank',str(rank),mode]
    if rank==0:
        result=subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True,timeout=60)
    else:
        result=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10',
            'root@'+HOSTS[rank],shlex.join(cmd)],check=True,capture_output=True,text=True,timeout=60)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def parallel(mode):
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(remote,i,mode) for i in range(3)]
        for i,future in enumerate(futures):
            try:results.append(dict(node_rank=i,result=future.result(),ok=True))
            except Exception as error:results.append(dict(node_rank=i,error=str(error),ok=False))
    return results


def node_states(directory):
    return [read(directory/f'node{i}.json') if (directory/f'node{i}.json').exists() else {} for i in range(3)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--detach',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if socket.gethostname()!=HOSTNAMES[0]:raise RuntimeError('Coordinator must run on node0')
    verify_release()
    if a.check:
        results=parallel('--probe')
        print(json.dumps(dict(source_verified=True,nodes=results,e1_step=latest_step(E1)),indent=2))
        if not all(r['ok'] for r in results):raise SystemExit(1)
        return
    if a.detach:
        # Test singleton lock before forking; child holds it throughout queue/training.
        with (OUT/'queue.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'queue.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(coordinator_pid=child.pid,log=str(OUT/'queue.log'))));return
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(coordinator_pid=os.getpid(),created_at=now(),arm='E0',predecessor=str(E1),output=str(E0),budget=BUDGET)
        launched=False
        def publish(stage,**extra):
            state.update(state=stage,updated_at=now(),**extra);write(OUT/'queue_status.json',state)
        try:
            if (E0/'run.json').exists() or any(n.get('state') in ('running','starting') for n in node_states(OUT)):
                raise RuntimeError('E0 already exists/active; refusing duplicate launch')
            while True:
                if (OUT/'STOP').exists():publish('cancelled_before_launch');return
                nodes=node_states(TRAIN);status=read(E1/'status.json') if (E1/'status.json').exists() else None
                if completed_gate(status,nodes):break
                publish('waiting_e1',predecessor_step=latest_step(E1),predecessor_nodes=[n.get('state') for n in nodes])
                time.sleep(20)
            verify_release();checkpoint=verify_checkpoint(E1)
            write(OUT/'predecessor_verified.json',dict(checked_at=now(),status=status,checkpoint=checkpoint,nodes=nodes))
            while True:
                if (OUT/'STOP').exists():publish('cancelled_before_launch');return
                probes=parallel('--probe')
                if all(p['ok'] and p['result']['idle'] for p in probes):break
                publish('waiting_resources',predecessor_step=BUDGET,resource_checks=probes);time.sleep(20)
            verify_release()
            publish('launching',predecessor_step=BUDGET,resource_checks=probes)
            launched=True;acks=parallel('--detach');write(OUT/'launch_results.json',acks)
            if not all(a['ok'] for a in acks):raise RuntimeError('One or more E0 nodes failed to launch')
            pair_checked=False
            while True:
                nodes=node_states(OUT)
                if (OUT/'STOP').exists():
                    # Native step-boundary stop preserves a resumable E0 checkpoint.
                    E0.mkdir(parents=True,exist_ok=True);(E0/'STOP').touch()
                    publish('stopping_e0',nodes=nodes)
                if any(n.get('state')=='failed' for n in nodes):raise RuntimeError('E0 node failed; stopping only owned E0 workers')
                if (E0/'run.json').exists() and not pair_checked:
                    write(OUT/'pair_validation.json',verify_pair());pair_checked=True
                status=read(E0/'status.json') if (E0/'status.json').exists() else None
                if all(n.get('state')=='finished' and n.get('returncode')==0 for n in nodes):
                    if status and status.get('stopped'):
                        publish('stopped_e0',step=status['step'],nodes=nodes);return
                    if not completed_gate(status,nodes):raise RuntimeError('Invalid final E0 status')
                    if not pair_checked:raise RuntimeError('Missing matched E0 identity')
                    checkpoint=verify_checkpoint(E0)
                    publish('complete',step=BUDGET,nodes=nodes,checkpoint=checkpoint);return
                publish('training' if pair_checked else 'training_precheck',step=latest_step(E0),nodes=nodes,pair_verified=pair_checked)
                time.sleep(20)
        except Exception as error:
            cleanup=parallel('--stop-owned') if launched else []
            publish('failed',error_type=type(error).__name__,error=str(error),cleanup=cleanup)
            raise


if __name__=='__main__':main()
