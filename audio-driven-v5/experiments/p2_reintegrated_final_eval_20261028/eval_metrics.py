"""Pinned OmniResponse and RGB GLARE metrics, with explicit postprocessed GT."""
import sys, os, subprocess, concurrent.futures, fcntl, time, uuid
from pathlib import Path
from eval_common import HERE, ROOT, PY, read, write, sha
CLUSTER_WORK=Path(os.environ.get('P2_EVAL_CLUSTER_WORK',str(HERE/'cluster24')))
sys.path.insert(0,str(ROOT/'src'))

def gt(split,sid):
    out=HERE/'gt'/split/sid;out.mkdir(parents=True,exist_ok=True)
    with (out/'materialize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _gt(split,sid)

def _gt(split,sid):
    from listen_audio_postprocess.recipes import read_listener_audio
    from p2_glare_eval.media import probe
    import soundfile as sf
    if split=='test24':
        # The corrected comparison changes guidance only: reuse the exact GT
        # video and postprocessed listener WAV from the previous test24 run.
        import numpy as np
        rows=read(ROOT/'experiments/p2_test24_fourway_eval_20261027/E0/omni_pairs.json')['samples']
        original=next(r for r in rows if r['sample_id']==sid)
        recipe=read(ROOT/'experiments/p2_test48_backchannel_face_20261027/audio_postprocess/recipes'/f'{sid}.json')
        wave,rate,_=read_listener_audio(recipe);previous,previous_rate=sf.read(original['gt_audio'],dtype='float32',always_2d=True)
        if rate!=previous_rate or not np.array_equal(wave,previous):raise ValueError('Previous test24 GT audio differs from the current recipe')
        geometry=probe(original['gt_video'])
        if (geometry['height'],geometry['width'])!=(512,512):raise ValueError('Original test24 GT is not512x512')
        return {k:original[k] for k in ('gt_video','gt_audio')}
    targets={r['sample_id']:r for r in read(HERE/split/'targets.json')}
    target=targets[sid];out=HERE/'gt'/split/sid;out.mkdir(parents=True,exist_ok=True)
    data_root=ROOT/('experiments/p2_train50h_postprocessed_reselected_20261027' if split=='train24' else 'experiments/p2_test48_backchannel_face_20261027')
    recipe_path=data_root/'audio_postprocess/recipes'/f'{sid}.json';recipe=read(recipe_path)
    source=target['target_video'];ident=dict(recipe_sha256=sha(recipe_path),video_sha256=sha(source),geometry=[512,512],resize='ffmpeg lanczos; CFR timeline unchanged')
    if (out/'complete.json').exists():
        marker=read(out/'complete.json');assert marker['identity']==ident
        assert all(sha(out/n)==h for n,h in marker['files'].items())
    else:
        wave,rate,metadata=read_listener_audio(recipe)
        sf.write(out/'listener.wav',wave,rate,subtype='FLOAT')
        subprocess.run(['ffmpeg','-v','error','-y','-i',source,'-vf','scale=512:512:flags=lanczos','-c:v','ffv1','-an',str(out/'video.mkv')],check=True)
        before=probe(source);after=probe(out/'video.mkv')
        assert {k:v for k,v in before.items() if k not in ('width','height')}=={k:v for k,v in after.items() if k not in ('width','height')}
        write(out/'complete.json',dict(identity=ident,waveform=metadata,files={n:sha(out/n) for n in ['listener.wav','video.mkv']}))
    return dict(gt_video=str(out/'video.mkv'),gt_audio=str(out/'listener.wav'))

def detect(paths):
    from p2_glare_rgb.detector import Engine
    from p2_glare_unfiltered.rules import process
    engine=Engine()
    for video in read(paths):
        h=sha(video);folder=HERE/'detections/rgb'/h
        locks=HERE/'detections/locks';locks.mkdir(parents=True,exist_ok=True)
        with (locks/(h+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            engine.detect(video,folder,h);process(video,folder,HERE/'detections/unfiltered'/h,h)

def score(work):
    from p2_glare_eval.scoring import score_seed,aggregate_seeds
    from p2_glare_eval.media import probe,aligned
    from p2_glare_eval.common import artifact_valid
    rows=read(work/'pairs.json')['samples'];records={mode:[] for mode in ['strict','unfiltered']};coverage=[]
    for row in rows:
        hashes={s:sha(row[s+'_video']) for s in ['gt','generated']}
        geom={s:probe(row[s+'_video']) for s in hashes};aligned(geom['gt'],geom['generated'])
        valid=True
        for h in hashes.values():
            for mode in ['rgb','unfiltered']:
                folder=HERE/'detections'/mode/h;m=read(folder/'complete.json');assert artifact_valid(folder,m['contract'])
            valid=valid and read(HERE/'detections/rgb'/h/'complete.json')['valid']
        coverage.append(dict(sample_id=row['sample_id'],sample_seed=row['sample_seed'],strict_valid=valid,
            quality={s:read(HERE/'detections/unfiltered'/h/'quality.json') for s,h in hashes.items()}))
        for mode in records:
            if mode=='strict' and not valid:continue
            rec={k:row[k] for k in ['sample_id','sample_seed','gt_video','generated_video']};rec['geometry']=geom['gt']
            for side,h in hashes.items():
                p=HERE/'detections'/('rgb' if mode=='strict' else 'unfiltered')/h/'scores.csv';rec[side+'_csv']=str(p);rec[side+'_csv_sha256']=sha(p)
            records[mode].append(rec)
    result={}
    ids={r['sample_id'] for r in rows}
    seeds=sorted({r['sample_seed'] for r in rows})
    for mode,items in records.items():
        by={(r['sample_id'],r['sample_seed']):r for r in items}
        common=sorted(s for s in ids if all((s,seed) in by for seed in seeds))
        scores={str(seed):score_seed([by[s,seed] for s in common],'cuda:0',16) for seed in seeds} if common else {}
        result[mode]=dict(sample_ids=common,samples_per_seed=len(common),planned_samples=len(ids),per_seed=scores,
            aggregation=aggregate_seeds(scores) if scores else None)
    write(work/'glare_results.json',dict(status='complete',results=result,coverage=coverage,
        target_annotation_access=True,scope='Per-condition two-seed common support; strict subsets may differ between conditions'))

def evaluate(arm,split):
    from p2_omni_eval.protocol import prepare
    from p2_omni_eval.common import subprocess_env
    jobs=[j for j in read(HERE/'jobs.json') if j['arm']==arm and j['split']==split]
    expected=2*len(read(HERE/split/'observations.json'))
    assert len(jobs)==expected
    work=HERE/'metrics'/arm/split;work.mkdir(parents=True,exist_ok=True);rows=[]
    for j in jobs:
        out=Path(j['output']);assert (out/'complete.json').exists()
        rows.append(dict(sample_id=j['sample_id'],sample_seed=j['seed'],**gt(split,j['sample_id']),
            generated_video=str(out/'generated.mp4'),generated_audio=str(out/'listener.wav')))
    manifest=dict(schema='p2-omni-pairs-v1',samples=rows)
    if (work/'pairs.json').exists():assert read(work/'pairs.json')==manifest
    else:write(work/'pairs.json',manifest)
    runtime=read(ROOT/'configs/p2_omni_runtime.lock.json');mp=runtime['paths']['metrics_python']
    prepare(work/'pairs.json',work/'omni',ROOT/'configs/p2_omni_runtime.lock.json')
    env=subprocess_env(runtime,'cuda:0')
    # Split the costly I3D/UTMOS/SyncNet rows over the shared GPU task queue.
    # At most eight parent groups exist; sixteen workers remain available for
    # these shards even when all parents are waiting simultaneously.
    if (CLUSTER_WORK/'tasks').is_dir():
        from cluster_eval import publish,state
        shards=[]
        for index in range(3):
            task=dict(id=f'omni_shard_{arm}_{split}_{index}',kind='metric_shard',priority=3,
                cmd=[mp,str(__file__),'omni-shard',str(work/'omni'),str(index)])
            publish(task);shards.append(task)
        while not all(state(t).get('status')=='complete' for t in shards):
            if (CLUSTER_WORK/'STOP').exists() or any(state(t).get('failures',0)>=3 for t in shards):raise RuntimeError('Metric shards stopped/failed')
            time.sleep(3)
    with (work/'omni.log').open('a') as log:
        code=subprocess.run([mp,str(ROOT/'scripts/p2_omni_eval.py'),'worker','--work',str(work/'omni'),'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code not in (0,2) or not (work/'omni/results.json').exists():raise RuntimeError(f'Omni metric worker failed: {code}')
    videos=sorted({r[s+'_video'] for r in rows for s in ['gt','generated']})
    dp=ROOT/'.venv-p2-glare-detector/bin/python'
    denv=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2')
    def shard(i):
        path=work/f'detect_{i}.json';write(path,videos[i::4])
        with (work/f'detect_{i}.log').open('a') as log:
            subprocess.run([str(dp),str(__file__),'detect',str(path)],env=denv,stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(shard,range(4)))
    with (work/'glare.log').open('a') as log:
        subprocess.run([mp,str(__file__),'score',str(work)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    omni=read(work/'omni/results.json')
    write(work/'COMPLETE.json',dict(status='complete',omni_status=omni['status'],outputs=expected,seeds=[137,251],
        checkpoint=str(jobs[0]['checkpoint']),annotation_oracle=True,train_diagnostic=split=='train24',
        results={n:sha(work/n) for n in ['omni/results.json','glare_results.json']}))

def omni_shard(work,index):
    from p2_omni_eval.pipeline import compute_row,check_protocol,has_error
    from p2_omni_eval.backends import Backends
    from p2_omni_eval.common import digest,job_key,artifact_valid,complete
    work=Path(work);p=read(work/'protocol.json');check_protocol(p);backend=Backends(p,'cuda:0');bad=[]
    for row in p['rows'][index::3]:
        dest=work/'rows'/job_key(row['sample_id'],row['sample_seed']);dest.mkdir(parents=True,exist_ok=True)
        ident=dict(protocol=digest(p),row=row,device='cuda:0')
        with (dest/'row.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if artifact_valid(dest,ident):continue
            attempt=dest/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            result=compute_row(row,p,backend,attempt);write(attempt/'result.json',result);write(dest/'result.json',result)
            if has_error(result):bad.append(dict(sample_id=row['sample_id'],sample_seed=row['sample_seed']))
            else:complete(dest,ident,[dest/'result.json',*attempt.glob('*.npy')])
    check_protocol(p);write(work/f'shard{index}.json',dict(complete=not bad,failed_rows=bad))
    if bad:raise RuntimeError('Metric rows failed: '+str(bad))

def smoke():
    from p2_omni_eval.protocol import prepare
    from p2_omni_eval.common import subprocess_env
    from p2_omni_eval.pipeline import has_error
    job=read(HERE/'pilot/final_job.json');out=Path(job['output']);work=HERE/'pilot/metrics';work.mkdir(exist_ok=True)
    row=dict(sample_id=job['sample_id'],sample_seed=job['seed'],**gt(job['split'],job['sample_id']),generated_video=str(out/'generated.mp4'),generated_audio=str(out/'listener.wav'))
    write(work/'pairs.json',dict(schema='p2-omni-pairs-v1',samples=[row]))
    runtime=read(ROOT/'configs/p2_omni_runtime.lock.json');mp=runtime['paths']['metrics_python'];env=subprocess_env(runtime,'cuda:0')
    prepare(work/'pairs.json',work/'omni',ROOT/'configs/p2_omni_runtime.lock.json')
    subprocess.run([mp,str(ROOT/'scripts/p2_omni_eval.py'),'worker','--work',str(work/'omni'),'--device','cuda:0'],env=env,check=True)
    result=read(work/'omni/results.json');assert not has_error(result['rows'])
    paths=work/'videos.json';write(paths,[row['gt_video'],row['generated_video']])
    subprocess.run([str(ROOT/'.venv-p2-glare-detector/bin/python'),str(__file__),'detect',str(paths)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_ENABLE_ONEDNN_OPTS='0'),check=True)
    subprocess.run([mp,str(__file__),'score',str(work)],env=env,check=True)
    write(work/'SMOKE_COMPLETE.json',dict(status='passed',step=1249,samples=1,metrics=['I3D features','UTMOSv2','SyncNet','GLARE strict/unfiltered'],note='One-sample FVD distance is intentionally null; this is runtime validation, not a comparison score'))

if __name__=='__main__':
    if sys.argv[1]=='detect':detect(sys.argv[2])
    elif sys.argv[1]=='score':score(Path(sys.argv[2]))
    elif sys.argv[1]=='evaluate':evaluate(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='omni-shard':omni_shard(sys.argv[2],int(sys.argv[3]))
    else:raise ValueError(sys.argv[1])
