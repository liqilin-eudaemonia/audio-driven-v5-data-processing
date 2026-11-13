"""Base-H AV+Event+Feature: preflight, wait for 32 free GPUs, verify then train."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json,sha
from p2_matrix16.config import load
from listen_cluster import launch,transport
from concurrent.futures import ThreadPoolExecutor
import subprocess


def commands(cluster,config,**kw):
    result=[]
    for rank,node in enumerate(cluster['nodes']):
        cmd=['env','CUDA_VISIBLE_DEVICES='+','.join(map(str,node['gpus'])),'PYTHONUNBUFFERED=1',
             'TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cluster['network'].items()],
             'TORCH_ELASTIC_WORKER_IDENTICAL=0',
             cluster['python'],'-m','torch.distributed.run','--max_restarts=0','--nnodes=4',
             '--nproc_per_node='+str(len(node['gpus'])),'--node_rank='+str(rank),
             '--master_addr='+cluster['nodes'][0]['host'],'--master_port='+str(cluster['master_port']),
             str(ROOT/'scripts/p2_j_train_observed.py'),'train','--config',str(config)]
        if kw.get('resume'):cmd+=['--resume',str(kw['resume'])]
        if kw.get('stop_after') is not None:cmd+=['--stop-after',str(kw['stop_after'])]
        result.append(cmd)
    return result


def resources(cluster):
    def check(node):
        try:
            p=subprocess.run(transport(cluster,node,['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=25)
            if p.returncode:return dict(node=node['name'],free=False,error=p.stderr)
            g={int(x.split(',')[0]):[int(v.strip()) for v in x.split(',')[1:]] for x in p.stdout.splitlines()}
            return dict(node=node['name'],free=all(i in g and g[i][0]>=135*1024 and g[i][1]<1024 for i in node['gpus']),selected=node['gpus'],gpu_memory=g)
        except Exception as exc:return dict(node=node['name'],free=False,error=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as ex:return list(ex.map(check,cluster['nodes']))


def verify_metrics(rows,n):
    if [r['step'] for r in rows]!=list(range(1,n+1)):raise ValueError('Initial updates incomplete')
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('av','event','feature_video','feature_audio','weighted_total')):
            raise ValueError('Nonfinite loss')
        expected=r['av']+.05*r['event']+.025*(r['feature_video']+r['feature_audio'])
        if not math.isclose(r['weighted_total'],expected,rel_tol=1e-8,abs_tol=1e-9):
            raise ValueError('Wrong combined objective')
    for group in ('connector','events','features','text_lora'):
        if not all(math.isfinite(r['grad_norm'][group]) for r in rows):raise ValueError('Nonfinite gradient')
        if not any(r['grad_norm'][group]>0 for r in rows):raise ValueError('Missing gradient: '+group)
    if not all(any(r[k]>0 for r in rows) for k in ('event','feature_video','feature_audio')):
        raise ValueError('Inactive auxiliary branch')
    return dict(success=True,verified_steps=n,objective_verified=True,
        gradient_groups=['connector','events','features','text_lora'],
        mean_step_seconds=sum(r['seconds'] for r in rows)/n,
        rank0_peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows))


def run(config,cluster):
    from p2_j_duration_execution import install_backend,install_sampler
    c=load(config,formal=True);out=Path(c['output']);q=out/'queue';q.mkdir(exist_ok=True)
    world=sum(len(n['gpus']) for n in cluster['nodes'])
    if world!=28 or c['training']['global_batch']!=28:raise ValueError('Approved run requires 28 ranks, batch28')
    if (c['loss'],c['stage'],c['qwen_hidden_source'],c['event_head_training'],c['qwen_lora']['enabled'])!=('av_event_feature',1,'base_pre_sft',True,False):
        raise ValueError('Wrong combined base-H experiment')
    if c['loss_weights']!=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025):raise ValueError('Wrong weights')
    def status(state,**kw):
        row=dict(state=state,time=time.time(),config=str(config),world=world,budget=c['training']['steps'],**kw)
        atomic_json(q/'state.json',row);print(json.dumps(row),flush=True)
    def render(**kw):
        return commands(cluster,config,**kw)
    def execute(cmds,path):
        cancel=threading.Event();done=threading.Event();begin=time.time()
        f=out/'metrics.jsonl';old=f.stat().st_mtime if f.exists() else 0
        def watch():
            while not done.wait(15):
                stamp=f.stat().st_mtime if f.exists() else 0;started=stamp>old
                age=time.time()-(stamp if started else begin)
                if age>(180 if started else 600):
                    atomic_json(q/'STALL.json',dict(time=time.time(),age_s=age,logs=str(path)));cancel.set();return
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        try:launch(cluster,cmds,path,cancel_event=cancel)
        finally:done.set();watcher.join(timeout=2)
    with (q/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            recovery=None
            if (out/'run.json').exists():
                if not (out/'network_recovery.json').exists() or (out/'metrics.jsonl').exists():
                    raise ValueError('Run exists; recover from a complete checkpoint explicitly')
                recovery=out/'step_000000.pt'
                if not recovery.with_suffix('.complete.json').exists():raise ValueError('No complete step0 recovery')
            backend=install_backend();install_sampler(c)
            status('preflight')
            previous=ROOT/'experiments/p2_j_guided_av_feature_base025_20261102/duration_memorysafe_save500/preflight.json'
            pre=read(previous)
            if pre['source']==backend.source_identity() and pre['data_identity']==backend.data_identity(c):
                pre['experiment_mode']=[c['text_condition'],c['loss'],c['stage']]
                pre['reused_full_preflight']=dict(path=str(previous),sha256=sha(previous),
                    reason='Exact same full285638 cache, Event labels, teacher and implementation; only loss switches/batch/output changed')
                atomic_json(out/'preflight.json',pre)
            else:pre=backend.preflight(c,processes=16)
            if pre['samples']!=285638:raise ValueError('Expected full 500h train')
            while True:
                if (out/'STOP').exists():status('stopped');return
                try:checks=resources(cluster)
                except Exception as exc:
                    status('waiting_resource_query_retry',error=repr(exc));time.sleep(15);continue
                atomic_json(q/'resources.json',dict(time=time.time(),nodes=checks))
                if all(n['free'] for n in checks):break
                status('waiting_selected_28_free_gpus',busy=[n for n in checks if not n['free']]);time.sleep(15)
            if backend.source_identity()!=pre['source'] or backend.data_identity(c)!=pre['data_identity']:
                raise ValueError('Source/data changed while waiting')
            n=16;checkpoint=out/f'step_{n:06d}.pt'
            initial=render(stop_after=n,**({'resume':recovery} if recovery else {}));continuation=render(resume=checkpoint)
            atomic_json(out/'launch_commands.json',dict(initial=initial,continuation=continuation,cluster=cluster))
            status('initial_28gpu_training',validation_steps=n)
            execute(initial,q/'initial16')
            import torch
            from p2_matrix16.checkpoint import checked
            torch.set_num_threads(2);saved=checked(checkpoint)
            if saved['step']!=n or saved['identity']['experiment']!=c or saved['identity']['world']!=world:
                raise ValueError('Initial checkpoint mismatch')
            del saved
            result=verify_metrics([json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()],n)
            atomic_json(out/'GPU_VALIDATION.json',dict(checkpoint=str(checkpoint),**result))
            if (out/'STOP').exists():status('stopped_after_validation');return
            status('training',resume_step=n,**result)
            execute(continuation,q/'continue_epoch')
            final=read(out/'status.json')
            if not final['complete'] or final['step']!=c['training']['steps']:raise ValueError('Incomplete training')
            status('complete',step=final['step'])
        except Exception as exc:status('failed',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cluster',type=Path,required=True);a=p.parse_args()
    run(a.config.resolve(),read(a.cluster))
