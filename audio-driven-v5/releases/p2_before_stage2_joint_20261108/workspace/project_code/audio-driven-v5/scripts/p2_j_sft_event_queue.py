"""Wait for current AV+Feature, then run frozen-SFT-H AV+0.05 Event on 32 GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_listen.contracts import read, sha, atomic_json
from p2_matrix16.config import load
from p2_j_s1_queue import commands, verify_metrics
from listen_cluster import launch, transport


def predecessor_finished(folder):
    """Counters alone must never authorize using occupied GPUs."""
    folder = Path(folder)
    paths = [folder/name for name in ('status.json','controller_state.json','logs/completed.json')]
    if not all(p.exists() for p in paths):return False
    progress, controller, exits = map(read, paths)
    budget = read(folder/'run.json')['experiment']['training']['steps']
    return (progress.get('complete') is True and progress.get('step') == budget
            and controller.get('state') == 'complete' and controller.get('step') == budget
            and exits.get('returncodes') == [0,0,0,0])


def verify_initial(rows):
    result = verify_metrics(rows,3)
    for r in rows:
        if r['feature_video'] != 0 or r['feature_audio'] != 0:
            raise ValueError('Feature branch must stay disabled')
        if not math.isclose(r['weighted_total'],r['av']+.05*r['event'],rel_tol=1e-8,abs_tol=1e-9):
            raise ValueError('AV+0.05 Event objective differs')
        for group in ('connector','events','text_lora'):
            if not math.isfinite(r['grad_norm'][group]):raise ValueError('Nonfinite gradient')
    return dict(result,objective_verified=True,feature_disabled=True)


def resources(cluster):
    def check(node):
        code = """import json,subprocess
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(dict(gpus=[[int(v.strip()) for v in line.split(',')] for line in g.splitlines()],compute_pids=apps)))
"""
        p=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code]),capture_output=True,text=True,timeout=30)
        if p.returncode:return dict(node=node['name'],free=False,error=p.stderr[-1500:])
        r=json.loads(p.stdout.strip().splitlines()[-1]); g=r['gpus']
        return dict(node=node['name'],free=(len(g)==8 and [x[0] for x in g]==list(range(8))
            and all(x[1]>=135*1024 and x[2]<1024 for x in g) and not r['compute_pids']),**r)
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(check,cluster['nodes']))


def run(config, cluster, predecessor_index):
    import p2_j_sft_event_backend as backend
    c=load(config,formal=True);backend.base_config(c)
    out=Path(c['output']);q=out/'queue';q.mkdir(exist_ok=True)
    last_report=0
    def status(state,**kw):
        nonlocal last_report
        value=dict(state=state,time=time.time(),config=str(config),world=32,budget=c['training']['steps'],**kw)
        atomic_json(q/'state.json',value)
        if time.time()-last_report>60 or state not in ('waiting_predecessor','waiting_resource_release'):
            print(json.dumps(value),flush=True);last_report=time.time()
    def render(**kw):
        cmds=commands(cluster,config,**kw)
        for cmd in cmds:
            cmd[cmd.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts/p2_j_sft_event_train.py')
        return cmds
    def execute(cmds, logdir):
        cancel=threading.Event();done=threading.Event();began=time.time()
        old_mtime=(out/'metrics.jsonl').stat().st_mtime if (out/'metrics.jsonl').exists() else 0
        def watch():
            while not done.wait(15):
                f=out/'metrics.jsonl';mtime=f.stat().st_mtime if f.exists() else 0
                started=mtime>old_mtime
                age=time.time()-(mtime if started else began)
                if age>(180 if started else 600):
                    atomic_json(q/'STALL.json',dict(time=time.time(),logdir=str(logdir),age_s=age,
                        action='Cancel only owned workers; preserve complete checkpoints'))
                    cancel.set();return
        monitor=threading.Thread(target=watch,daemon=True);monitor.start()
        try:launch(cluster,cmds,logdir,cancel_event=cancel)
        finally:done.set();monitor.join(timeout=2)
    with (q/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (out/'run.json').exists():raise ValueError('Existing new run; explicit recovery required')
            while True:
                if (out/'STOP').exists():status('stopped');return
                prior=Path(read(predecessor_index)['output'])
                if predecessor_finished(prior):break
                progress=read(prior/'status.json')
                status('waiting_predecessor',predecessor=str(prior),predecessor_step=progress['step'],
                       predecessor_budget=progress['budget'])
                time.sleep(15)
            # Require the real final checkpoint receipt and checksum, not only status.
            previous=read(prior/'run.json');step=previous['experiment']['training']['steps']
            path=prior/f'step_{step:06d}.pt';receipt=read(path.with_suffix('.complete.json'))
            status('verifying_predecessor_checkpoint',checkpoint=str(path))
            if receipt['step']!=step or sha(path)!=receipt['sha256']:
                raise ValueError('Predecessor final checkpoint incomplete/corrupt')
            while True:
                if (out/'STOP').exists():status('stopped');return
                checks=resources(cluster);atomic_json(q/'resources.json',dict(time=time.time(),nodes=checks))
                if all(n['free'] for n in checks):break
                status('waiting_resource_release',nodes=checks);time.sleep(15)
            pre=read(out/'preflight.json')
            if pre['source']!=backend.source_identity() or pre['data_identity']!=backend.data_identity(c):
                raise ValueError('New experiment source/data changed after preflight')
            from p2_j_duration_execution import Schedule
            Schedule(c)
            initial=render(stop_after=3);checkpoint=out/'step_000003.pt';continuation=render(resume=checkpoint)
            atomic_json(out/'launch_commands.json',dict(executed=True,cluster=cluster,initial=initial,continuation=continuation))
            status('initial_32gpu_training',validation_steps=3)
            execute(initial,q/'initial3')
            import torch
            from p2_matrix16.checkpoint import checked
            torch.set_num_threads(2)
            saved=checked(checkpoint)
            if saved['step']!=3 or saved['identity']['world']!=32 or saved['identity']['experiment']!=c:
                raise ValueError('Initial checkpoint identity mismatch')
            del saved
            rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            result=verify_initial(rows)
            atomic_json(out/'GPU_VALIDATION.json',dict(success=True,checkpoint=str(checkpoint),**result))
            ready=read(out/'READY.json');ready.update(training_started=True,gpu_training_validated=True,queued=True)
            atomic_json(out/'READY.json',ready)
            if (out/'STOP').exists():status('stopped_after_validation');return
            status('training',resume_step=3,**result)
            execute(continuation,q/'continue_epoch')
            final=read(out/'status.json')
            if not final['complete'] or final['step']!=c['training']['steps']:
                raise ValueError('Training ended before budget')
            status('complete',step=final['step'])
        except Exception as exc:
            status('failed',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--predecessor-index',type=Path,required=True)
    p.add_argument('--cluster',type=Path,required=True);a=p.parse_args()
    run(a.config.resolve(),read(a.cluster),a.predecessor_index.resolve())
