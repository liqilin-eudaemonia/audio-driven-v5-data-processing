"""Scheduling adapter around the unchanged, source-only P2 generator."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid
from .common import ROOT, read, sha, digest, atomic_json, complete, artifact_valid, checked_files, code_identity, job_key


def prepare(run, observations, targets, output, seeds=(137,251), *, train_diagnostic=False):
    from p2_listen.contracts import observation, path_at
    from p2_listen.config import source_identity
    from .media import probe
    run, observations, targets, output = map(lambda x:Path(x).resolve(),(run,observations,targets,output))
    trained=read(run/'run.json');c=trained['config']
    if trained['source']!=source_identity(c):raise ValueError('Training source changed')
    if trained['s0']:raise ValueError('Formal evaluation requires a non-S0 training run')
    if not seeds or len(seeds)!=len(set(seeds)) or any(type(s) is not int or s<0 for s in seeds):raise ValueError('Invalid sample seeds')
    source=read(observations);ground_truth=read(targets)
    rows=[observation(o,observations.parent) for o in source]
    if not rows or len({o['sample_id'] for o in rows})!=len(rows):raise ValueError('Nonempty unique observations required')
    training=read(path_at(c['output'])/'manifests/observations.json')
    train_ids={o['sample_id'] for o in training}
    train_groups={g for o in training for g in [o['group_id'],*o['grouping_ids']]}
    if train_diagnostic:
        indexed={o['sample_id']:o for o in training}
        if any(o['experiment_split']!='train' or o['sample_id'] not in indexed or o!=observation(indexed[o['sample_id']]) for o in rows):
            raise ValueError('Train diagnostic requires unchanged observations from this exact training set')
    else:
        if any(o['experiment_split'] not in ('dev','test') for o in rows):raise ValueError('Evaluation requires dev/test observations')
        if any(o['sample_id'] in train_ids or set([o['group_id'],*o['grouping_ids']])&train_groups for o in rows):
            raise ValueError('Evaluation overlaps training samples/groups/participants')
    gt={r['sample_id']:r for r in ground_truth}
    if len(gt)!=len(ground_truth) or set(gt)!={o['sample_id'] for o in rows}:raise ValueError('GT and observation IDs differ')
    evaluation_targets=[]
    for o in rows:
        record=gt[o['sample_id']]
        name=record.get('gt_video',record.get('target_video'))
        if not name:raise ValueError('Each target requires gt_video or target_video')
        video=(targets.parent/name).resolve();g=probe(video)
        if (g['width'],g['height'],g['fps'])!=(o['geometry']['width'],o['geometry']['height'],o['geometry']['fps']):
            raise ValueError('GT and observation geometry differ')
        # Fractional audio durations are valid: retain all frame timestamps < real end.
        expected_frames=math.ceil(o['requested_duration_s']*g['fps']-1e-6)
        if g['frame_count']!=expected_frames:raise ValueError('GT does not exactly cover the declared observation window')
        value=sha(video)
        if record.get('media_hashes',{}).get('target_video',value)!=value:raise ValueError('GT video hash differs')
        evaluation_targets.append(dict(sample_id=o['sample_id'],gt_video=str(video),sha256=value,geometry=g))
    protocol=dict(schema='p2-glare-generation-v1',evaluation_split='train_diagnostic' if train_diagnostic else 'held_out',run=str(run),run_identity_sha256=digest(trained),
        config=c,source=trained['source'],assets=trained['assets'],budget=trained['budget'],observations=rows,
        seeds=list(seeds),code=code_identity(),source_files={str(observations):sha(observations)},
        evaluation_targets_digest=digest(evaluation_targets))
    atomic_json(output/'generation_protocol.json',protocol,immutable=True)
    # Targets are deliberately separate: workers never open this file or GT videos.
    atomic_json(output/'evaluation_targets.json',evaluation_targets,immutable=True)
    return dict(samples=len(rows),seeds=list(seeds),jobs=len(rows)*len(seeds),output=str(output))


def check_idle_device(device):
    if not device.startswith('cuda:'):raise ValueError('Native P2 generation requires an explicit cuda:N device')
    index=int(device.split(':')[1]);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=visible.split(',')[index] if visible else str(index)
    output=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(output.strip())>2048:raise RuntimeError(f'{device} is busy; wait until training releases its GPUs')


def generate(output,checkpoint,*,device='cuda:0',shards=1,shard_index=0,wait=False,wait_timeout=86400):
    output,checkpoint=Path(output).resolve(),Path(checkpoint).resolve()
    p=read(output/'generation_protocol.json')
    if p['schema']!='p2-glare-generation-v1':raise ValueError('Invalid generation protocol')
    if not math.isfinite(wait_timeout) or wait_timeout<=0:raise ValueError('Positive finite wait timeout required')
    if p['code']!=code_identity():raise ValueError('Evaluation code changed; prepare a new version')
    checked_files(p['source_files'])
    checked_files(p['source'])
    for obs in p['observations']:
        checked_files({obs[key]:value for key,value in obs.get('media_hashes',{}).items()})
    if shards<1 or not 0<=shard_index<shards:raise ValueError('Invalid generation shard')
    begin=time.monotonic()
    while not checkpoint.with_suffix('.complete.json').exists():
        if not wait:raise FileNotFoundError('Checkpoint has no complete marker')
        if time.monotonic()-begin>wait_timeout:raise TimeoutError('Training checkpoint wait timed out')
        time.sleep(5)
    from p2_listen.training import checked_checkpoint
    from p2_listen.generation import generate as native_generate
    saved=checked_checkpoint(checkpoint)
    if digest(saved['identity'])!=p['run_identity_sha256'] or saved['step']!=p['budget']:
        raise ValueError('Expected the completed final checkpoint of this exact training run')
    checkpoint_sha=sha(checkpoint);del saved
    atomic_json(output/'selected_checkpoint.json',dict(sha256=checkpoint_sha,
        run_identity_sha256=p['run_identity_sha256'],step=p['budget']),immutable=True)
    run_status=Path(p['run'])/'status.json'
    while not run_status.exists():
        if not wait:raise RuntimeError('Training has not published its final status')
        if time.monotonic()-begin>wait_timeout:raise TimeoutError('Final training status wait timed out')
        time.sleep(5)
    status=read(run_status)
    if status['step']!=p['budget'] or status.get('stopped'):raise ValueError('Training stopped before the planned budget')
    release=dict(config_sha256=digest(p['config']),checkpoint_sha256=[checkpoint_sha],
                 sample_ids=[o['sample_id'] for o in p['observations']],seeds=p['seeds'])
    release_path=output/f'test_release_{checkpoint_sha[:16]}.json'
    atomic_json(release_path,release,immutable=True)
    jobs=[(o,seed) for seed in p['seeds'] for o in p['observations']][shard_index::shards]
    device_checked=False
    results=[]
    import fcntl
    for obs,seed in jobs:
        key=job_key(obs['sample_id'],seed);destination=output/'generated'/key
        ident=dict(checkpoint_sha256=checkpoint_sha,source=p['source'],assets=p['assets'],config=p['config'],
                   observation=obs,sample_seed=seed,mode='normal',code=p['code'])
        lock_path=output/'locks'/(key+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if artifact_valid(destination,ident):
                results.append(dict(sample_id=obs['sample_id'],sample_seed=seed,status='success',reused=True));continue
            if destination.exists():raise ValueError('Incomplete committed generation directory; inspect it before retry')
            if not device_checked:
                while True:
                    try:check_idle_device(device);break
                    except RuntimeError:
                        if not wait or time.monotonic()-begin>wait_timeout:raise
                        time.sleep(5)
                device_checked=True
            attempt=output/'attempts'/(key+'-'+uuid.uuid4().hex)
            attempt.parent.mkdir(parents=True,exist_ok=True)
            try:
                native_generate(p['config'],obs,attempt,checkpoint=checkpoint,seed=seed,device=device,release=release_path)
                complete(attempt,ident,[attempt/n for n in ('generated.mp4','listener.wav','completed.json')],gt_access=False)
                destination.parent.mkdir(parents=True,exist_ok=True);os.replace(attempt,destination)
                record=dict(sample_id=obs['sample_id'],sample_seed=seed,status='success',reused=False)
            except Exception as error:
                record=dict(sample_id=obs['sample_id'],sample_seed=seed,status='failed',
                            error_type=type(error).__name__,error=str(error),attempt=str(attempt))
                atomic_json(output/'generation_failures'/(key+'.json'),record)
            results.append(record)
            atomic_json(output/f'generation_shard_{shard_index:03d}.json',dict(shards=shards,shard_index=shard_index,
                planned=len(jobs),completed=len(results),results=results))
    final=dict(shards=shards,shard_index=shard_index,planned=len(jobs),results=results)
    atomic_json(output/f'generation_shard_{shard_index:03d}.json',final)
    return final


def export_pairs(output,path):
    output=Path(output).resolve();p=read(output/'generation_protocol.json')
    targets=read(output/'evaluation_targets.json')
    if digest(targets)!=p['evaluation_targets_digest']:raise ValueError('Evaluation targets changed after protocol freeze')
    if p['code']!=code_identity():raise ValueError('Evaluation code changed')
    selected=read(output/'selected_checkpoint.json')
    if selected['run_identity_sha256']!=p['run_identity_sha256']:raise ValueError('Checkpoint selection belongs to another run')
    gt={r['sample_id']:r for r in targets}
    samples=[]
    for seed in p['seeds']:
        for obs in p['observations']:
            sid=obs['sample_id'];key=job_key(sid,seed);folder=output/'generated'/key
            target=gt[sid]
            if sha(target['gt_video'])!=target['sha256']:raise ValueError('GT video changed after protocol freeze')
            row=dict(sample_id=sid,sample_seed=seed,gt_video=target['gt_video'])
            if (folder/'complete.json').exists():
                marker=read(folder/'complete.json')
                expected=dict(checkpoint_sha256=selected['sha256'],source=p['source'],assets=p['assets'],config=p['config'],
                    observation=obs,sample_seed=seed,mode='normal',code=p['code'])
                artifact_valid(folder,expected)
                row['generated_video']=str(folder/'generated.mp4')
            elif (output/'generation_failures'/(key+'.json')).exists():
                row['generation_error']=read(output/'generation_failures'/(key+'.json'))['error']
            else:
                raise RuntimeError(f'Generation is still missing/pending: {sid}, seed {seed}')
            samples.append(row)
    manifest=dict(schema='p2-glare-pairs-v1',samples=samples)
    atomic_json(path,manifest,immutable=True)
    return manifest
