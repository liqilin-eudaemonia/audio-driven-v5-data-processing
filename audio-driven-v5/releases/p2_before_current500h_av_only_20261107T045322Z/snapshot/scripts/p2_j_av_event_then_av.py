"""Resume AV+Event, then automatically launch the independent pure-AV control."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
from p2_j_duration_execution import install_backend,Schedule
from p2_j_s1_queue import commands
from listen_cluster import launch,transport

WORK=ROOT/'experiments/p2_j_av_event_then_av_20261101'
OLD=ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101'
NEW=ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_alternating_20261101'
AV_CONFIG=ROOT/'configs/p2_j_av_only_20261101.json'


def status(stage,**kw):
    row=dict(stage=stage,at=time.time(),**kw)
    atomic_json(WORK/'state.json',row);print(json.dumps(row),flush=True)


def prepare():
    import torch
    from p2_listen.cache import atomic_torch
    from p2_matrix16.checkpoint import checked
    from p2_matrix16.config import validate
    from p2_j_duration_benchmark import equal_tensors
    torch.set_num_threads(4)
    result=read(ROOT/'experiments/p2_j_av_only_benchmark_20261101/RESULT.json')
    if not (result['success'] and result['selected']=='alternating' and
            all(result['recomputation_bitwise_equal'].values()) and all(result['strict_resume_bitwise_equal'].values())):
        raise ValueError('32-GPU AV-only validation did not pass')
    if not (OLD/'STOP').exists() or read(OLD/'status.json')['step']!=4201:
        raise ValueError('Expected gracefully paused AV+Event at4201')
    old_identity=read(OLD/'run.json');c=copy.deepcopy(old_identity['experiment'])
    c['output']=str(NEW);c['training']['gradient_checkpointing']='alternating'
    validate(c,formal=True);backend=install_backend()
    if backend.source_identity()!=old_identity['source'] or backend.data_identity(c)!=old_identity['data_identity']:
        raise ValueError('AV+Event source/data changed')
    Schedule(c)
    original=OLD/'step_004201.pt';saved=checked(original)
    if saved['identity']!=old_identity or saved['step']!=4201:
        raise ValueError('Wrong resume checkpoint')
    identity=dict(old_identity,experiment=c)
    target=NEW/original.name
    if target.exists():raise FileExistsError('Prepared checkpoint already exists')
    atomic_torch(target,dict(saved,identity=identity))
    atomic_json(target.with_suffix('.complete.json'),dict(step=4201,sha256=sha(target)))
    restored=checked(target)
    unchanged={k:equal_tensors(saved[k],restored[k]) for k in ('model','optimizer','rank_rng')}
    if not all(unchanged.values()):raise ValueError('Migration changed training tensors')
    atomic_json(NEW/'run.json',identity,immutable=True)
    atomic_json(NEW/'preflight.json',read(OLD/'preflight.json'),immutable=True)
    atomic_json(NEW/'status.json',dict(step=4201,budget=c['training']['steps'],complete=False))
    atomic_json(NEW/'migration.json',dict(source=str(original),source_sha256=sha(original),step=4201,
        model_optimizer_rng_bitwise_unchanged=unchanged,
        change='gradient_checkpointing all -> alternating; output directory only',
        validation=str(ROOT/'experiments/p2_j_av_only_benchmark_20261101/RESULT.json')),immutable=True)
    config=WORK/'av_event.json';atomic_json(config,c,immutable=True)
    cluster=read(ROOT/'configs/p2_j_av_only_32gpu_20261101.json');cluster['master_port']=29841
    atomic_json(WORK/'cluster.json',cluster,immutable=True)
    av=read(AV_CONFIG)
    if (Path(av['output'])/'run.json').exists():raise ValueError('Pure AV run already started')
    if av['training']['steps']!=8927 or av['training']['save_every']!=200:
        raise ValueError('Unexpected AV budget/checkpoint interval')
    atomic_json(WORK/'job.json',dict(av_event_config=str(config),av_event_config_sha256=sha(config),
        av_event_checkpoint=str(target),av_event_checkpoint_sha256=sha(target),
        av_event_output=str(NEW),av_config=str(AV_CONFIG),av_config_sha256=sha(AV_CONFIG),
        av_output=av['output'],budget=8927,save_every=200,world=32),immutable=True)
    status('prepared',resume_step=4201,av_event_output=str(NEW),av_output=av['output'],unchanged=unchanged)


def run():
    from p2_matrix16.checkpoint import checked
    job=read(WORK/'job.json');cluster=read(WORK/'cluster.json')
    def cancelled():return (WORK/'STOP').exists()
    def available(node):
        s=subprocess.check_output(transport(cluster,node,['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits']),text=True,timeout=30)
        vals=[int(x) for x in s.splitlines() if x.strip()]
        return len(vals)==8 and min(vals)>=135*1024
    def wait_free():
        while not cancelled():
            with ThreadPoolExecutor(max_workers=4) as pool:
                if all(pool.map(available,cluster['nodes'])):return True
            time.sleep(3)
        return False
    def verify_config(key):
        if sha(job[key])!=job[key+'_sha256']:raise ValueError('Queued config changed: '+key)
    def verify_finished(output):
        final=read(Path(output)/'status.json')
        if not final['complete'] or final['step']!=job['budget']:return False
        p=Path(output)/f"step_{job['budget']:06d}.pt"
        saved=checked(p)
        if saved['step']!=job['budget']:raise ValueError('Wrong final checkpoint step')
        return True
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(WORK/'state.json')['stage']!='prepared':
            raise ValueError('Queue already started; inspect its recorded state')
        try:
            verify_config('av_event_config');verify_config('av_config')
            if sha(job['av_event_checkpoint'])!=job['av_event_checkpoint_sha256']:
                raise ValueError('Prepared checkpoint changed')
            if not wait_free():status('stopped_before_av_event');return
            if (NEW/'STOP').exists():status('stopped_before_av_event');return
            status('av_event_running',resume_step=4201,budget=job['budget'],output=str(NEW),next='av_only')
            cmds=commands(cluster,job['av_event_config'],resume=job['av_event_checkpoint'])
            for cmd in cmds:cmd[cmd.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts/p2_j_duration_train.py')
            launch(cluster,cmds,WORK/'transport'/f'av_event_{time.time_ns()}')
            if not verify_finished(NEW):status('av_event_stopped',output=str(NEW));return
            status('waiting_release_for_av_only',av_event_final_step=job['budget'])
            if not wait_free():status('stopped_before_av_only');return
            verify_config('av_config')
            av=Path(job['av_output'])
            if cancelled() or (av/'STOP').exists():status('stopped_before_av_only');return
            if (av/'run.json').exists():raise ValueError('Pure AV output already in use')
            # Recheck the prepared AV data/source contract before consuming GPUs.
            subprocess.run([cluster['python'],str(ROOT/'scripts/p2_j_av_only.py'),'preflight',
                            '--config',job['av_config']],check=True)
            status('av_only_running',start_step=0,budget=job['budget'],output=str(av))
            cc=copy.deepcopy(cluster);cc['master_port']=29842
            cmds=commands(cc,job['av_config'])
            for cmd in cmds:cmd[cmd.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts/p2_j_av_only.py')
            launch(cc,cmds,WORK/'transport'/f'av_only_{time.time_ns()}')
            if not verify_finished(av):status('av_only_stopped',output=str(av));return
            status('complete',av_event_final_step=job['budget'],av_only_final_step=job['budget'])
        except BaseException as exc:
            status('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run()
