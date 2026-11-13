"""One guided AV+Event S1 epoch on four nodes, after cache audit succeeds."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json
from p2_matrix16.config import load
from listen_cluster import launch


def commands(cluster,config,*,resume=None,stop_after=None,probe_output=None):
    if len(cluster['nodes'])!=4 or any(n['gpus']!=list(range(8)) for n in cluster['nodes']):
        raise ValueError('Expected four nodes with eight GPUs each')
    result=[]
    for rank,node in enumerate(cluster['nodes']):
        command=['env','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','PYTHONUNBUFFERED=1',
            'TOKENIZERS_PARALLELISM=false',*[k+'='+str(v) for k,v in cluster['network'].items()],
            cluster['python'],'-m','torch.distributed.run','--max_restarts=0','--nnodes=4','--nproc_per_node=8',
            '--node_rank='+str(rank),'--master_addr='+cluster['nodes'][0]['host'],
            '--master_port='+str(cluster['master_port'])]
        if probe_output:
            command += [str(ROOT/'scripts/p2_j_train_probe.py'),'--output',str(probe_output)]
        else:
            command += [str(ROOT/'scripts/p2_clip_events.py'),'train','--config',str(config)]
            if resume:command += ['--resume',str(resume)]
            if stop_after is not None:command += ['--stop-after',str(stop_after)]
        result.append(command)
    return result


def verify_metrics(rows,steps):
    if [r['step'] for r in rows]!=list(range(1,steps+1)):
        raise ValueError('Initial training steps incomplete')
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('av','av_video','av_audio','event','weighted_total')):
            raise ValueError('Nonfinite training metrics')
    for group in ('connector','events','text_lora'):
        if not any(math.isfinite(r['grad_norm'][group]) and r['grad_norm'][group]>0 for r in rows):
            raise ValueError('Missing training gradient: '+group)
    return dict(steps=steps,mean_step_seconds=sum(r['seconds'] for r in rows)/len(rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),losses_finite=True,
                connector_event_text_lora_gradients_nonzero=True)


def run(config,cluster):
    c=load(config,formal=True);out=Path(c['output']);q=out/'queue';q.mkdir(parents=True,exist_ok=True)
    if (c['text_condition'],c['loss'],c['stage'])!=('guided','av_event',1) or c['training']['global_batch']!=32:
        raise ValueError('Queue is for guided AV+Event S1, global batch 32')
    cache=Path(c['j_cache']);stop=out/'STOP'
    def status(state,**extra):
        value=dict(state=state,time=time.time(),config=str(config),world=32,budget=c['training']['steps'],**extra)
        atomic_json(q/'state.json',value);print(json.dumps(value),flush=True)
    with (q/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                if stop.exists():status('stopped');return
                finished=(cache/'COMPLETE.json').exists()
                released=all((cache/'launchers'/(n['name']+'.json')).exists() and
                    read(cache/'launchers'/(n['name']+'.json')).get('status')=='finished' for n in cluster['nodes'])
                if finished and released:break
                progress=read(cache/'status.json')
                status('waiting_cache_audit',features=progress['phases']['features']['samples'],
                       cache_audit_complete=finished,cache_launchers_released=released)
                time.sleep(30)
            complete=read(cache/'COMPLETE.json');samples=read(Path(c['clip_labels'])/'manifest.json')['samples']
            if not complete['success'] or complete['samples']!=samples or c['training']['steps']!=math.ceil(samples/32):
                raise ValueError('Cache completeness or one-epoch budget differs')
            status('preflight')
            with (q/'preflight.log').open('w') as log:
                subprocess.run([cluster['python'],str(ROOT/'scripts/p2_clip_events.py'),'preflight','--config',str(config)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if stop.exists():status('stopped');return
            # These first 16 updates belong to the same formal run and budget.
            first_steps=16
            status('initial_32gpu_training',validation_steps=first_steps)
            launch(cluster,commands(cluster,config,stop_after=first_steps),q/'initial16')
            if stop.exists():status('stopped');return
            import torch
            from p2_matrix16.checkpoint import checked
            torch.set_num_threads(2)
            path=out/f'step_{first_steps:06d}.pt';saved=checked(path)
            if saved['step']!=first_steps or saved['identity']['world']!=32 or saved['identity']['experiment']!=c:
                raise ValueError('Initial checkpoint identity differs')
            rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            metrics=verify_metrics(rows,first_steps)
            atomic_json(q/'INITIAL_TRAINING_VERIFIED.json',dict(success=True,checkpoint=str(path),**metrics))
            del saved
            status('training',resume_step=first_steps,**metrics)
            launch(cluster,commands(cluster,config,resume=path),q/'continue_epoch')
            if stop.exists():status('stopped');return
            final=read(out/'status.json')
            if not final['complete'] or final['step']!=c['training']['steps']:
                raise ValueError('Training ended before the epoch budget')
            status('complete',step=final['step'])
        except Exception as exc:
            status('failed',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--cluster',required=True)
    parser.add_argument('--probe-output')
    args=parser.parse_args();cluster=read(args.cluster);config=Path(args.config).resolve()
    if args.probe_output:
        output=Path(args.probe_output).resolve()
        launch(cluster,commands(cluster,config,probe_output=output),output/'transport')
    else:run(config,cluster)
