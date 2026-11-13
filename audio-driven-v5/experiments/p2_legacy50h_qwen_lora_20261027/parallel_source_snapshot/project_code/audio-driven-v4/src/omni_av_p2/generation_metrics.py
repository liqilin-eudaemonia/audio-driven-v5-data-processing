"""Post-generation historical ArcFace/FID/FVD/SyncNet evaluation (no model inference).

Reference targets enter only this evaluator, never the generation suite. The
external historical feature extractors are reused and their hashes recorded.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

MODES = ('gt', 'omni')
PROTOCOL = 'p2_p21_legacy_four_metrics_v1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text())


def ratio(n, d):
    return n / d if d else None


def metric_row(case, mode, plan, video, reference, step):
    # Do NOT parse delivery/behavior or fall back to GT for the Omni transcript.
    reply = ' '.join(plan['reply'].split())
    eligible = bool(reply)
    reason = 'reply_present_pending_single_generated_face_gate' if eligible else 'no_main_spoken_reply'
    return dict(id=case['sample_id'], prompt='', transcript=reply, scene_type='live',
        speech_type='single' if eligible else 'none', generated_video=str(video.resolve()),
        reference_video=str(reference.resolve()), seed=case['seed'], checkpoint_step=step,
        mode=mode, metric_applicability={
            'id_similarity': dict(eligible=True, reason='reference_first_frame'),
            'fid': dict(eligible=True, reason='successful_dev_generation'),
            'fvd': dict(eligible=True, reason='successful_dev_generation'),
            'sync_c': dict(eligible=eligible, reason=reason),
            'sync_d': dict(eligible=eligible, reason=reason)})


def prepare(suite_path, generation, metadata_path, output):
    """CPU manifest/audit preparation. Never touches the running training job."""
    suite_path, generation, metadata_path, output = map(Path, (suite_path, generation, metadata_path, output))
    suite = read(suite_path)
    if suite.get('modes') != list(MODES) or not suite.get('cases'):
        raise ValueError('Need a GT/Omni Dev suite')
    if digest(metadata_path) != suite['geometry_metadata_sha256']:
        raise ValueError('Reference metadata differs from generation geometry metadata')
    metadata = {}
    for line in metadata_path.read_text().splitlines():
        row = json.loads(line)
        if row['sample_id'] in metadata:
            raise ValueError('Duplicate reference sample ID')
        metadata[row['sample_id']] = row
    cases = suite['cases']; ids = [c['sample_id'] for c in cases]
    if len(set(ids)) != len(ids) or any(c.get('split') != 'dev' or Path(sid).name != sid
                                     or sid in ('.', '..') for c, sid in zip(cases, ids)):
        raise ValueError('Need unique safe Dev sample IDs')
    workers = [read(p) for p in generation.glob('worker_*.json') if not p.name.endswith('.completed.json')]
    identities = {(w['checkpoint_sha256'], w['step'], w['suite_sha256']) for w in workers}
    if len(identities) > 1 or any(w['suite_sha256'] != digest(suite_path) for w in workers):
        raise ValueError('Mixed checkpoints or generation suites')
    step = workers[0]['step'] if workers else None
    rows = {mode: [] for mode in MODES}; records = []
    reference_hashes = {}
    for case in cases:
        sid = case['sample_id']; ref = metadata[sid]; folder = generation / sid
        if ref.get('split') != 'dev' or any(ref[k] != case['geometry'][g] for k, g in
                [('target_height','height'), ('target_width','width'), ('target_frames','frames'), ('fps','fps')]):
            raise ValueError(f'Reference geometry/split mismatch: {sid}')
        reference = Path(ref['video'])
        if not reference.is_file():
            raise FileNotFoundError(reference)
        reference_hashes[sid] = digest(reference)
        planning = read(folder / 'planning_result.json') if (folder / 'planning_result.json').exists() else {}
        if planning.get('gt_fallback'):
            raise ValueError('GT fallback is forbidden')
        predicted = folder / 'omni/text_plan.json'
        status = planning.get('status', 'valid' if predicted.exists() else
                              'unknown' if (generation / f'{sid}.failed.json').exists() else 'not_attempted')
        if status not in ('valid', 'rejected', 'error', 'unknown', 'not_attempted'):
            raise ValueError(f'Unknown planning status: {status}')
        record = dict(sample_id=sid, omni_plan_status=status, planning_error=planning.get('error'), modes={})
        failure = generation / f'{sid}.failed.json'
        if failure.exists(): record['generation_error'] = read(failure)
        for mode in MODES:
            branch = folder / mode
            video, marker, plan_file = branch/'generated.mp4', branch/'completed.json', branch/'text_plan.json'
            ok = all(p.is_file() for p in (video, marker, plan_file)) and video.stat().st_size > 0
            record['modes'][mode] = 'complete' if ok else 'failed_or_incomplete' if folder.exists() else 'not_attempted'
            if not ok: continue
            if not workers: raise ValueError('Completed video lacks worker checkpoint provenance')
            if read(folder/'case.json') != case: raise ValueError('Generation case differs from suite')
            plan, done = read(plan_file), read(marker)
            if not isinstance(plan.get('reply'), str) or done.get('plan') != plan:
                raise ValueError('Generation plan and completion record differ')
            if mode == 'gt' and plan != case['gt_plan']: raise ValueError('GT plan changed')
            if mode == 'omni' and (status != 'valid' or (planning.get('plan') and planning['plan'] != plan)):
                raise ValueError('Omni completion contradicts planning record')
            if done.get('gt_plan_used') != (mode == 'gt') or done.get('paired_target_media_used') is not False:
                raise ValueError('Generation condition provenance mismatch')
            if any(done['decoded'][k] != case['geometry'][k] for k in ('frames','height','width')):
                raise ValueError('Decoded geometry differs from suite')
            rows[mode].append(metric_row(case, mode, plan, video, reference, step))
        records.append(record)
    common = sorted(set(x['id'] for x in rows['gt']) & set(x['id'] for x in rows['omni']))
    total = len(cases); valid = sum(r['omni_plan_status']=='valid' for r in records)
    attempted = sum(r['omni_plan_status'] in ('valid','rejected','error') for r in records)
    coverage = dict(dataset_count=total, omni_plan_valid_count=valid, omni_plan_attempted_count=attempted,
        omni_plan_valid_rate_all=ratio(valid,total), omni_plan_valid_rate_attempted=ratio(valid,attempted),
        omni_plan_status_counts={s:sum(r['omni_plan_status']==s for r in records)
            for s in ('valid','rejected','error','unknown','not_attempted')}, gt_plan_fallback_count=0,
        generation_success={mode:dict(count=len(rows[mode]), rate_all=ratio(len(rows[mode]),total),
            conditional_on_valid_plan=ratio(len(rows[mode]),total if mode=='gt' else valid)) for mode in MODES},
        common_success_count=len(common), common_success_rate=ratio(len(common),total))
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for mode in MODES:
        selections = [('available',rows[mode])]
        if set(common) != {x['id'] for x in rows[mode]}:
            selections.append(('common',[x for x in rows[mode] if x['id'] in common]))
        for subset, values in selections:
            name=f'{mode}_{subset}'; manifest=output/f'{name}.jsonl'
            manifest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values))
            jobs.append(dict(name=name, mode=mode, subset=subset, sample_ids=[x['id'] for x in values],
                manifest=str(manifest.resolve()), manifest_sha256=digest(manifest),
                generated_sha256={x['id']:digest(x['generated_video']) for x in values}))
    result=dict(protocol=PROTOCOL, suite=str(suite_path.resolve()), suite_sha256=digest(suite_path),
        generation_root=str(generation.resolve()), checkpoint_step=step,
        checkpoint_sha256=workers[0]['checkpoint_sha256'] if workers else None,
        reference_metadata=str(metadata_path.resolve()), reference_metadata_sha256=digest(metadata_path),
        reference_sha256=reference_hashes, coverage=coverage, cases=records, common_sample_ids=common,
        jobs=jobs, common_alias={mode:f'{mode}_available' for mode in MODES
            if {x['id'] for x in rows[mode]}==set(common)},
        note='Fixed-duration Dev48 offline protocol, not old24/protocol-identical or autonomous-duration evaluation')
    write_json(output/'evaluation_plan.json',result)
    return result


def preflight(config):
    c=config; root=Path(c['benchmark_root']); trainer=Path(c['legacy_trainer'])
    files=[trainer/'scripts/compute_arcface_id.py',root/'metrics/visual_distribution.py',
        root/'metrics/syncnet_metrics.py',root/'benchmark_core.py',
        Path(c['syncnet_root'])/'run_vocal_sync_multigpu.py',Path(c['syncnet_root'])/'run_vocal_sync.py',
        Path(c['syncnet_root'])/'data/syncnet_v2.model',
        Path(c['syncnet_root'])/'models/audio-separator/model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        Path(c['syncnet_root'])/'detectors/s3fd/weights/sfd_face.pth',
        Path(c['id_model_root'])/'models'/c['id_model_name']/'det_10g.onnx',
        Path(c['id_model_root'])/'models'/c['id_model_name']/'w600k_r50.onnx',
        Path(c['torch_home'])/'hub/checkpoints/inception_v3_google-0cc3c7bd.pth',
        Path(c['torch_home'])/'hub/checkpoints/I3D_8x8_R50.pyth']
    for p in files+[Path(c[k]) for k in ('id_python','avbench_python','sync_python')]:
        if not p.is_file(): raise FileNotFoundError(p)
    if min(c['id_frames'],c['fid_frames'],c['fvd_frames'],c['timeout_seconds'])<1:
        raise ValueError('Invalid metric configuration')
    return dict(config=c, asset_sha256={str(p):digest(p) for p in files},
        protocol=dict(id='ArcFace buffalo_l cosine, reference first frame vs 8 uniform generated frames',
            fid='historical torchvision ImageNet Inception-v3 pooled features; not pytorch-fid',
            fvd='historical PyTorchVideo I3D-R50 Kinetics pooled features; not TF-I3D FVD',
            sync='historical vocal-separated SyncNet, single generated face gate',
            preprocessing='metric feature copies use legacy resize/crop; source/generation geometry unchanged'))


def ensure_idle(gpu):
    if gpu<0: raise ValueError('GPU index must be nonnegative')
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows={int(x[0]):(int(x[1]),int(x[2])) for line in result.splitlines() if (x:=line.split(','))}
    if gpu not in rows or rows[gpu][0]>1000 or rows[gpu][1]>5:
        raise RuntimeError('Metric GPU is occupied; will not interrupt training')


def run_command(command, cwd, log, env, timeout):
    with log.open('a') as stream:
        stream.write(json.dumps(command)+'\n');stream.flush()
        p=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=max(1,timeout))
            if code: raise subprocess.CalledProcessError(code,command)
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise


def summarize_raw(identity,visual,sync,count):
    # Missing/failed measurements remain null; they are never converted to zeros.
    def finite(value): return isinstance(value,(float,int)) and math.isfinite(value)
    values=dict(id_similarity=identity.get('id_similarity',{}).get('mean'),fid=visual.get('fid'),
        fvd=visual.get('fvd'),sync_c=sync.get('sync_c',{}).get('mean'),sync_d=sync.get('sync_d',{}).get('mean'))
    eligible=sync.get('sync_c',{}).get('eligible_count',0)
    required=['id_similarity','fid']+(['fvd'] if count>=2 else [])+(['sync_c','sync_d'] if eligible else [])
    issues=[f'missing/nonfinite {k}' for k in required if not finite(values[k])]
    for k,v in values.items():
        if v is not None and not finite(v): values[k]=None
    id_count=identity.get('id_similarity',{}).get('count',0)
    sync_count=sync.get('sync_c',{}).get('count',0)
    if id_count!=count: issues.append(f'identity coverage {id_count}/{count}')
    if any(v for k,v in sync.get('status_counts',{}).items() if k in ('error','missing_result','unknown')):
        issues.append('SyncNet reported per-sample errors')
    if sync.get('sync_c',{}).get('dataset_count')!=count: issues.append('SyncNet dataset coverage mismatch')
    if sync_count!=eligible: issues.append(f'SyncNet coverage {sync_count}/{eligible}')
    return dict(status='partial' if issues else 'computed', values=values, issues=issues,
        sample_count=count, id_coverage=identity.get('id_similarity',{}),
        syncnet_coverage=sync.get('sync_c',{}), syncnet_status_counts=sync.get('status_counts',{}),
        fvd_status='computed' if count>=2 and finite(values['fvd']) else 'not_computable' if count<2 else 'failed',
        small_sample_diagnostic=count<8, quality_pass=None)


def score_job(job, config, output, gpu, runner=run_command):
    c=config;manifest=Path(job['manifest']); rows=[json.loads(x) for x in manifest.read_text().splitlines()]
    output.mkdir(parents=True,exist_ok=False)
    if not rows:
        result=dict(status='no_generated_samples',sample_count=0,values={k:None for k in
                    ('id_similarity','fid','fvd','sync_c','sync_d')},quality_pass=None)
        write_json(output/'metrics.json',result);return result
    root=Path(c['benchmark_root']); log=output/'commands.log'; env=dict(os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(root),TORCH_HOME=c['torch_home'],
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    deadline=time.monotonic()+c['timeout_seconds'];errors={}
    commands={
        'identity':[c['id_python'],str(Path(c['legacy_trainer'])/'scripts/compute_arcface_id.py'),
            '--manifest',str(manifest),'--output',str(output/'identity.json'),'--model-root',c['id_model_root'],
            '--model-name',c['id_model_name'],'--sample-frames',str(c['id_frames']),'--cpu'],
        'visual':[c['avbench_python'],str(root/'metrics/visual_distribution.py'),'--manifest',str(manifest),
            '--output',str(output/'visual.json'),'--cache-dir',str(output/'visual_cache'),'--device','cuda:0',
            '--fid-frames',str(c['fid_frames']),'--fvd-frames',str(c['fvd_frames']),
            '--metrics','fid,fvd' if len(rows)>=2 else 'fid','--force'],
        'sync':[c['sync_python'],str(root/'metrics/syncnet_metrics.py'),'--manifest',str(manifest),
            '--output',str(output/'sync.json'),'--per-sample-output',str(output/'sync_per_sample.jsonl'),
            '--work-dir',str(output/'sync_work'),'--syncnet-root',c['syncnet_root'],
            # The legacy inner worker sets CUDA_VISIBLE_DEVICES itself: pass the
            # selected physical index, NOT a hard-coded0 which could steal GPU0.
            '--gpus',str(gpu),'--speech-types','single','--single-face-only']}
    raw={}
    for name,command in commands.items():
        try:
            if time.monotonic()>=deadline: raise TimeoutError('Metric job time budget exhausted')
            runner(command,root,log,env,deadline-time.monotonic())
            raw[name]=read(output/f'{name}.json')
        except Exception as exc:
            raw[name]={};errors[name]=f'{type(exc).__name__}: {exc}'
    result=summarize_raw(raw['identity'],raw['visual'],raw['sync'],len(rows))
    result['errors']=errors
    if errors: result['status']='failed'
    write_json(output/'metrics.json',result);return result


def score(plan_path,config,output,gpu,tensorboard=True):
    plan_path,output=Path(plan_path),Path(output)
    plan=read(plan_path)
    if plan['protocol']!=PROTOCOL: raise ValueError('Evaluation plan protocol mismatch')
    # Verify media/provenance before launching any expensive metric model.
    for job in plan['jobs']:
        if digest(job['manifest'])!=job['manifest_sha256']: raise ValueError('Metric manifest changed')
        for row in map(json.loads,Path(job['manifest']).read_text().splitlines()):
            if digest(row['generated_video'])!=job['generated_sha256'][row['id']]: raise ValueError('Generated video changed')
            if digest(row['reference_video'])!=plan['reference_sha256'][row['id']]: raise ValueError('Reference changed')
    assets=preflight(config);ensure_idle(gpu)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',dict(assets,plan_sha256=digest(plan_path),gpu=gpu))
    results={job['name']:score_job(job,config,output/job['name'],gpu) for job in plan['jobs']}
    for mode,name in plan['common_alias'].items(): results[f'{mode}_common']=dict(results[name],alias_of=name)
    report=dict(protocol=PROTOCOL,step=plan['checkpoint_step'],coverage=plan['coverage'],
        results=results,quality_pass=None,
        complete_dev_set=all(plan['coverage']['generation_success'][m]['count']==plan['coverage']['dataset_count'] for m in MODES),
        note='Available subsets may differ; use common subsets for paired comparison. Dev48 distribution estimates are diagnostic.')
    write_json(output/'metrics.json',report)
    with (output/'metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['mode_subset','status','samples','id_similarity_up','fid_down','fvd_down','sync_c_up','sync_d_down'])
        for name,r in results.items(): writer.writerow([name,r['status'],r['sample_count'],*[r['values'][k] for k in ('id_similarity','fid','fvd','sync_c','sync_d')]])
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter(str(output/'tensorboard')) as writer:
            step=plan['checkpoint_step'] or 0
            for name,r in results.items():
                for key,value in r['values'].items():
                    if value is not None: writer.add_scalar(f'generation/{name}/{key}',value,step)
                writer.add_scalar(f'generation/{name}/scored_samples',r['sample_count'],step)
                writer.add_scalar(f'generation/{name}/metrics_incomplete',r['status']!='computed',step)
                for key in ('dataset_count','eligible_count','success_count','eligible_coverage','dataset_coverage'):
                    value=r.get('syncnet_coverage',{}).get(key)
                    if isinstance(value,(int,float)) and math.isfinite(value):
                        writer.add_scalar(f'generation/{name}/syncnet_{key}',value,step)
                id_coverage=r.get('id_coverage',{}).get('coverage')
                if id_coverage is not None: writer.add_scalar(f'generation/{name}/id_coverage',id_coverage,step)
            for mode,v in plan['coverage']['generation_success'].items():
                writer.add_scalar(f'generation/{mode}/success_rate_all',v['rate_all'],step)
            writer.add_scalar('generation/omni/plan_valid_rate_all',plan['coverage']['omni_plan_valid_rate_all'],step)
    return report
