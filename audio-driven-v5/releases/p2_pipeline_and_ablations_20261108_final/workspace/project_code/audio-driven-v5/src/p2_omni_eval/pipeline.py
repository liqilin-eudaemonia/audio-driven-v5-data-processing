import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from .common import *
from . import media


def error_record(exc):return dict(status='error',error=f'{type(exc).__name__}: {exc}')


def check_protocol(p):
    if p['schema']!='p2-omni-eval-v1' or p['code']!=code_identity():raise ValueError('Evaluation code/schema changed; use a fresh work directory')
    if sha(p['input_manifest'])!=p['input_sha256']:raise ValueError('Input manifest changed')
    for row in p['rows']:
        checked_files({row[k]:h for k,h in row['hashes'].items()})


def compute_row(row,p,backend,dest):
    record=dict(sample_id=row['sample_id'],sample_seed=row['sample_seed'],metrics={},texts={s:row.get(s+'_text') for s in ('gt','generated')})
    try:
        if row.get('generation_error'):raise ValueError('Generation failed: '+str(row['generation_error']))
        gt=media.geometry(row['gt_video']);generated=media.geometry(row['generated_video']);media.aligned(gt,generated)
        record['geometry']=gt;indexes=media.window(row['sample_id'],gt['frame_count'],p['options'])
    except Exception as error:
        record['error']=str(error)
        for metric in p['metrics']:record['metrics'][metric]=error_record(error) if metric=='text' else {s:error_record(error) for s in ('gt','generated')}
        return record
    audio={}
    if any(m in p['metrics'] for m in ('utmos','sync')):
        for side in ('gt','generated'):
            try:audio[side]=media.audio(row[side+'_audio'],dest/(side+'.wav'),gt['duration_s'])
            except Exception as error:audio[side]=error_record(error)
        record['audio']=audio
    for metric in p['metrics']:
        if metric=='text':
            try:record['metrics'][metric]=backend.text(row.get('generated_text'),row.get('gt_text'))
            except Exception as error:record['metrics'][metric]=error_record(error)
            continue
        record['metrics'][metric]={}
        for side in ('gt','generated'):
            try:
                if metric=='fvd':value=backend.fvd_features(row[side+'_video'],indexes,dest/(side+'_i3d.npy'))
                elif metric=='fd':value=backend.fd_features(row[side+'_features'],dest/(side+'_features.npy'))
                else:
                    if audio[side].get('status')=='error':raise ValueError(audio[side]['error'])
                    if metric=='utmos':value=backend.mos(audio[side])
                    else:value=backend.sync(row[side+'_video'],audio[side],dest/(side+'_sync'))
                record['metrics'][metric][side]=value
            except Exception as error:record['metrics'][metric][side]=error_record(error)
    return record


def has_error(value):
    if isinstance(value,dict):return value.get('status')=='error' or any(has_error(v) for v in value.values())
    if isinstance(value,list):return any(has_error(v) for v in value)
    return False


def worker(work,device):
    from .backends import Backends
    from .scoring import aggregate
    work=Path(work).resolve();p=read(work/'protocol.json');check_protocol(p)
    backend=Backends(p,device);rows=[];protocol_sha=digest(p)
    for i,row in enumerate(p['rows']):
        key=job_key(row['sample_id'],row['sample_seed']);dest=work/'rows'/key;identity=dict(protocol=protocol_sha,row=row,device=device)
        if artifact_valid(dest,identity):record=read(dest/'result.json')
        else:
            # Keep prior errors and any partially written media for diagnosis/retry.
            dest.mkdir(parents=True,exist_ok=True);attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            record=compute_row(row,p,backend,attempt);atomic_json(attempt/'result.json',record);atomic_json(dest/'result.json',record)
            if not has_error(record):
                files=[dest/'result.json',*attempt.glob('*.npy')]
                complete(dest,identity,files)
        rows.append(record)
        atomic_json(work/'progress.json',dict(done=i+1,total=len(p['rows']),sample_id=row['sample_id'],sample_seed=row['sample_seed'],error=has_error(record)))
    # Re-check input/code after the potentially long scoring job.
    check_protocol(p)
    result=aggregate(p,rows);result['protocol_sha256']=protocol_sha;result['device']=device
    atomic_json(work/'results.json',result)
    lines=['# OmniResponse reference metric evaluation','',f"Status: {result['status']}",f"Samples per seed: {p['sample_count']}; seeds: {p['seeds']}",
        '',f"FVD-{p['options']['fvd_frames']}; 25 fps, stride 1, deterministic paired window.",
        'FD is supplied-feature distance, not Inception FID. UTMOS/SyncNet use listener audio.',
        'Full-set scores are null for incomplete coverage; supported subsets and exact IDs remain in JSON.','']
    for seed,scores in result['per_seed'].items():lines.extend([f'## Seed {seed}','','```json',json.dumps(scores,indent=2,ensure_ascii=False),'```',''])
    (work/'report.md').write_text('\n'.join(lines))
    return result


def evaluate(work,device='cpu'):
    work=Path(work).resolve();p=read(work/'protocol.json')
    with (work/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_protocol(p);check_runtime(p['runtime'],full=True);check_device(device)
        started=time.time()
        with (work/'worker.log').open('a') as log:
            completed=subprocess.run([p['runtime']['paths']['metrics_python'],str(ROOT/'scripts/p2_omni_eval.py'),'worker',
                '--work',str(work),'--device',device],env=subprocess_env(p['runtime'],device),stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            result_path=work/'results.json'
            if completed.returncode==2 and result_path.exists() and result_path.stat().st_mtime>=started-1:
                result=read(result_path)
                if result.get('protocol_sha256')==digest(p) and result.get('status')=='incomplete':return result
            result=dict(status='incomplete',error='Metric worker failed; inspect worker.log',returncode=completed.returncode)
            atomic_json(work/'run_failure.json',result);return result
        return read(work/'results.json')
