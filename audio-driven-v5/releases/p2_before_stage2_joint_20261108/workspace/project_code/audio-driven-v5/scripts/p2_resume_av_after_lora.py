"""Resume the paused AV+Event run after planning epoch2; save every 200 steps."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p2_listen.contracts import read, atomic_json, sha

WORK = ROOT/'experiments/p2_j_av_event_resume_after_lora2_20261101'
OLD = ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_duration512_20261101'
NEW = ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_resume200_20261101'
LORA = ROOT/'experiments/p2_planning_sft_3b_32gpu_epoch2_20261101'


def prepare():
    import torch
    from p2_listen.cache import atomic_torch
    from p2_matrix16.checkpoint import checked
    from p2_matrix16.config import validate
    from p2_j_duration_benchmark import equal_tensors
    from p2_j_duration_execution import install_backend, Schedule
    torch.set_num_threads(4)
    old_identity = read(OLD/'run.json')
    old_step = read(OLD/'status.json')['step']
    if old_step != 2301 or not (OLD/'STOP').is_file():
        raise ValueError('Expected the explicitly paused S1 at step2301')
    config = copy.deepcopy(old_identity['experiment'])
    if (config['stage'],config['text_condition'],config['loss']) != (1,'guided','av_event'):
        raise ValueError('Wrong AV training experiment')
    config['output'] = str(NEW)
    config['training']['save_every'] = 200
    validate(config,formal=True)
    backend = install_backend(); source = backend.source_identity()
    changed = {k for k in source if source[k] != old_identity['source'].get(k)}
    allowed = {str(ROOT/'src/p2_matrix16'/n) for n in ('config.py','training.py')}
    if set(source) != set(old_identity['source']) or changed != allowed:
        raise ValueError('Unexpected training source changes')
    replacements = {
        'config.py':("t['save_every']!=100", "type(t['save_every']) is not int or t['save_every']<1"),
        'training.py':("if step%100==0:save(step)", "if step%t['save_every']==0:save(step)")}
    for name,(before,after) in replacements.items():
        backup=WORK/'previous_source'/name;current=ROOT/'src/p2_matrix16'/name
        if sha(backup)!=old_identity['source'][str(current)] or backup.read_text().replace(before,after)!=current.read_text():
            raise ValueError('Migration permits only the reviewed save-interval edit')
    if backend.data_identity(config) != old_identity['data_identity']:
        raise ValueError('Data changed since S1 paused')
    Schedule(config)  # Verify the original deterministic duration schedule.
    parent = OLD/f'step_{old_step:06d}.pt'
    saved = checked(parent)
    if saved['identity'] != old_identity or saved['step'] != old_step:
        raise ValueError('Checkpoint/run identity mismatch')
    identity = dict(old_identity,experiment=config,source=source)
    target = NEW/parent.name
    if target.exists():
        raise FileExistsError('Migration already exists; do not overwrite it')
    atomic_torch(target,dict(saved,identity=identity))
    atomic_json(target.with_suffix('.complete.json'),dict(step=old_step,sha256=sha(target)))
    restored = checked(target)
    unchanged = {k:equal_tensors(saved[k],restored[k]) for k in ('model','optimizer','rank_rng')}
    if not all(unchanged.values()):
        raise ValueError('Checkpoint migration changed training state')
    atomic_json(NEW/'run.json',identity,immutable=True)
    pre = read(OLD/'preflight.json')
    if pre['source'] != old_identity['source'] or pre['data_identity'] != identity['data_identity']:
        raise ValueError('Original preflight does not match the paused run')
    pre['source'] = source
    pre['reused_full_preflight'] = dict(path=str(OLD/'preflight.json'),sha256=sha(OLD/'preflight.json'),
        reason='Only checkpoint frequency and output directory changed; data/model/objective/sampler unchanged')
    atomic_json(NEW/'preflight.json',pre,immutable=True)
    atomic_json(NEW/'status.json',dict(step=old_step,budget=config['training']['steps'],complete=False))
    config_path = WORK/'av_resume200.json'
    atomic_json(config_path,config,immutable=True)
    cluster = read(ROOT/'experiments/p2_j_guided_av_event_s1_32gpu_20261101/cluster.json')
    cluster['master_port'] = 29821
    atomic_json(WORK/'cluster.json',cluster,immutable=True)
    atomic_json(NEW/'migration.json',dict(source=str(parent),source_sha256=sha(parent),
        step=old_step,unchanged=unchanged,old_save_every=100,new_save_every=200,
        changed_sources={k:dict(before=old_identity['source'][k],after=source[k]) for k in sorted(changed)}),immutable=True)
    job = dict(wait_for=str(LORA),expected_lora_step=2232,config=str(config_path),
        config_sha256=sha(config_path),checkpoint=str(target),checkpoint_sha256=sha(target),
        output=str(NEW),cluster=str(WORK/'cluster.json'),resume_step=old_step,save_every=200)
    atomic_json(WORK/'job.json',job,immutable=True)
    atomic_json(WORK/'state.json',dict(stage='prepared',time=time.time(),job=job))
    print(json.dumps(dict(prepared=True,unchanged=unchanged,next_periodic_checkpoint=2400)),flush=True)


def watch():
    from listen_cluster import launch, transport
    from p2_j_s1_queue import commands
    job = read(WORK/'job.json'); cluster = read(job['cluster']); upstream = Path(job['wait_for'])
    def status(stage, **fields):
        row = dict(stage=stage,time=time.time(),**fields)
        atomic_json(WORK/'state.json',row)
        print(json.dumps(row),flush=True)
    def free_memory(node):
        result = subprocess.run(transport(cluster,node,['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits']),capture_output=True,text=True,check=True,timeout=25)
        values = [float(s)/1024 for s in result.stdout.splitlines() if s.strip()]
        if len(values)!=8:raise ValueError('Expected eight GPU memory readings')
        return dict(node=node['name'],minimum_free_gib=min(values))
    with (WORK/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(WORK/'state.json')['stage'] not in ('prepared','waiting_lora','waiting_gpu_release'):
            raise ValueError('Queue already launched or failed; inspect before restarting')
        status('waiting_lora',upstream=str(upstream),expected_step=job['expected_lora_step'])
        try:
            while not (upstream/'COMPLETE.json').exists():
                time.sleep(5)
            completed = read(upstream/'COMPLETE.json')
            if (not completed['success'] or completed['smoke'] or
                    completed['global_step']!=job['expected_lora_step'] or completed['train']['epoch']!=2.0):
                raise ValueError('Planning epoch2 did not complete the authorized budget')
            adapter = upstream/'final_adapter/adapter_model.safetensors'
            if not adapter.is_file() or adapter.stat().st_size == 0:
                raise ValueError('Final planning adapter was not saved')
            status('waiting_gpu_release',lora_completed=completed['global_step'])
            while True:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    memory=list(pool.map(free_memory,cluster['nodes']))
                if all(x['minimum_free_gib']>=cluster['minimum_free_gib'] for x in memory):break
                status('waiting_gpu_release',memory=memory)
                time.sleep(5)
            if sha(job['config'])!=job['config_sha256'] or sha(job['checkpoint'])!=job['checkpoint_sha256']:
                raise ValueError('Prepared resume inputs changed while waiting')
            if (Path(job['output'])/'STOP').exists():
                raise ValueError('New resume directory has an explicit STOP request')
            cmds=commands(cluster,job['config'],resume=job['checkpoint'])
            for cmd in cmds:
                cmd[cmd.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts/p2_j_duration_train.py')
            status('av_training',resume_step=job['resume_step'],save_every=job['save_every'],
                   output=job['output'],lora_adapter=str(adapter),lora_adapter_sha256=sha(adapter))
            launch(cluster,cmds,WORK/'launch_logs'/str(time.time_ns()))
            final = read(Path(job['output'])/'status.json')
            status('complete' if final['complete'] else 'stopped',av_status=final)
        except Exception as exc:
            status('failed',error=repr(exc),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','watch']);args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:watch()
