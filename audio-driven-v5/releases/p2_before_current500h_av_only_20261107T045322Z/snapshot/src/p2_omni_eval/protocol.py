"""Freeze explicit listener pairs and retain every intended sample/seed."""
from pathlib import Path
import math
import uuid
from .common import *

FILE_FIELDS=('gt_video','generated_video','gt_audio','generated_audio','gt_features','generated_features')


def normalize(manifest,base,metrics):
    if manifest.get('schema') not in ('p2-omni-pairs-v1','p2-glare-pairs-v1'):raise ValueError('Unknown pair manifest')
    raw=manifest['samples']
    if not raw:raise ValueError('Empty evaluation set')
    rows=[];seen=set();gt_by_id={};ids={}
    for row in raw:
        r=dict(row);sid=r.get('sample_id');seed=r.get('sample_seed')
        if not isinstance(sid,str) or not sid or type(seed) is not int or seed<0:raise ValueError('Invalid sample ID/seed')
        if (sid,seed) in seen:raise ValueError('Duplicate sample/seed')
        seen.add((sid,seed));ids.setdefault(seed,set()).add(sid)
        if not r.get('gt_video'):raise ValueError('gt_video is required')
        if not r.get('generated_video') and not r.get('generation_error'):raise ValueError('Missing generated video or explicit generation_error')
        r['hashes']={}
        for name in FILE_FIELDS:
            if r.get(name):
                path=(base/r[name]).resolve();r[name]=str(path);r['hashes'][name]=sha(path)
        for side in ('gt','generated'):
            if any(m in metrics for m in ('utmos','sync')) and not r.get(side+'_audio'):
                if side=='gt' or not r.get('generation_error'):raise ValueError('Audio metrics require explicit '+side+'_audio (listener only)')
            if 'fd' in metrics and not r.get(side+'_features'):
                if side=='gt' or not r.get('generation_error'):raise ValueError('FD requires explicit '+side+'_features')
            if 'text' in metrics:
                text=r.get(side+'_text')
                if text is not None and not isinstance(text,str):raise ValueError('Text must be a string or null')
        if 'fd' in metrics and (not isinstance(r.get('feature_schema'),str) or not r['feature_schema'].strip()):raise ValueError('Declare the shared feature_schema for FD')
        if 'text' in metrics and not r.get('text_provenance'):raise ValueError('Declare text_provenance: actual generation/transcript source')
        gt={k:r.get(k) for k in ('gt_video','gt_audio','gt_features','gt_text','feature_schema')}
        gt['hashes']={k:v for k,v in r['hashes'].items() if k.startswith('gt_')}
        if sid in gt_by_id and gt_by_id[sid]!=gt:raise ValueError('GT differs between generation seeds')
        gt_by_id[sid]=gt;rows.append(r)
    if any(s!=next(iter(ids.values())) for s in ids.values()):raise ValueError('Every seed must have the same sample IDs')
    if 'fd' in metrics and len({r['feature_schema'] for r in rows})!=1:raise ValueError('Cannot pool incompatible feature schemas')
    return sorted(rows,key=lambda r:(r['sample_seed'],r['sample_id']))


def prepare(manifest_path,output,runtime_path,metrics=('fvd','utmos','sync'),options=None):
    path=Path(manifest_path).resolve();output=Path(output).resolve();metrics=list(metrics)
    if not metrics or len(set(metrics))!=len(metrics) or set(metrics)-set(METRICS):raise ValueError('Invalid metric selection')
    options=dict(DEFAULTS,**(options or {}))
    if set(options)!=set(DEFAULTS):raise ValueError('Unknown metric option')
    for name in ('fvd_frames','fvd_stride','sync_min_track','sync_vshift','sync_batch_size','bert_layers'):
        if type(options[name]) is not int or options[name]<1:raise ValueError('Invalid '+name)
    if options['fvd_frames'] not in (32,256):raise ValueError('Choose explicit FVD-32 or FVD-256 profile')
    if options['fvd_stride']!=1:raise ValueError('This protocol fixes stride=1')
    for name in ('crop_seed','metric_seed'):
        if type(options[name]) is not int or not 0<=options[name]<2**32:raise ValueError('Invalid '+name)
    if not math.isfinite(options['silence_rms']) or options['silence_rms']<0:raise ValueError('Invalid silence threshold')
    raw=path.read_bytes();manifest=json.loads(raw)
    rows=normalize(manifest,path.parent,metrics);runtime=read(runtime_path);check_runtime(runtime)
    protocol=dict(schema='p2-omni-eval-v1',metrics=metrics,options=options,rows=rows,
        seeds=sorted({r['sample_seed'] for r in rows}),sample_count=len({r['sample_id'] for r in rows}),
        runtime=runtime,code=code_identity(),input_manifest=str(path),input_sha256=__import__('hashlib').sha256(raw).hexdigest(),
        audio_role='listener',frame_policy='25fps aligned timelines; no repair or temporal padding')
    atomic_json(output/'protocol.json',protocol,immutable=True)
    return dict(status='prepared',samples=protocol['sample_count'],seeds=protocol['seeds'],metrics=metrics,work=str(output))


def export_generation(work,gt_targets,output):
    from p2_glare_eval.generation import export_pairs
    work=Path(work).resolve();dest=Path(output).resolve();targets=Path(gt_targets).resolve()
    # The existing exporter validates generation complete markers and all AV hashes.
    tmp=work/'export_attempts'/('omni_visual_'+uuid.uuid4().hex+'.json');manifest=export_pairs(work,tmp)
    raw=read(targets);raw=raw['samples'] if isinstance(raw,dict) else raw
    gt={r['sample_id']:r for r in raw}
    if len(gt)!=len(raw):raise ValueError('Duplicate GT target IDs')
    for r in manifest['samples']:
        t=gt[r['sample_id']];audio=t.get('gt_audio',t.get('target_audio'))
        if not audio:raise ValueError('GT target requires independent listener audio')
        supplied_video=t.get('gt_video',t.get('target_video'))
        if supplied_video and sha((targets.parent/supplied_video).resolve())!=sha(r['gt_video']):
            raise ValueError('GT audio target manifest refers to a different GT video')
        r['gt_audio']=str((targets.parent/audio).resolve())
        expected=t.get('media_hashes',{}).get('target_audio')
        if expected and sha(r['gt_audio'])!=expected:raise ValueError('GT audio hash mismatch')
        if r.get('generated_video'):r['generated_audio']=str(Path(r['generated_video']).parent/'listener.wav')
    manifest['schema']='p2-omni-pairs-v1'
    atomic_json(dest,manifest,immutable=True);return dict(status='exported',rows=len(manifest['samples']),output=str(dest))
