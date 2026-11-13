"""Unified ViCo43 scoring of new AF outputs alongside immutable prior results."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

import numpy as np
from vico_avatarforcing import ROOT, WORK, REPO, DY, OLD, sha, read, save

AF=WORK/'avatarforcing'
BASE=ROOT/'experiments/p2_vico_motion8000_20261106'
L2L=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
COMMON=ROOT/'experiments/common_method_comparison_20261106/vico_deca'
EVAL=ROOT/'evaluation/vico_challenge_baseline/evaluations'


def bfm_source_versions():
    bp=read(BASE/'protocol.json');sp=read(BASE/'SCORING_PROTOCOL.json')
    assert sp['extraction_protocol_sha256']==sha(BASE/'protocol.json')
    files=dict(bp['source_sha256']);changed=[]
    for path,expected in files.items():
        if sha(path)==expected:continue
        assert path in sp['changed_scoring_sources'] and path in sp['implementation_sha256'],path
        archived=BASE/'frozen_source'/Path(path).relative_to(ROOT)
        assert sha(archived)==expected,'Original source archive mismatch'
        assert sha(path)==sp['implementation_sha256'][path],path
        changed.append(dict(path=path,initial=expected,final=sha(path),archive=str(archived)))
        files[path]=sp['implementation_sha256'][path]
    for path,h in sp['implementation_sha256'].items():assert sha(path)==h,path
    files.update(sp['implementation_sha256'])
    return files,dict(initial_protocol_sha256=sha(BASE/'protocol.json'),
        final_scoring_protocol_sha256=sha(BASE/'SCORING_PROTOCOL.json'),
        authorized_prior_revisions=changed,reason=sp['provenance'])


def freeze_evaluation():
    bp=read(BASE/'protocol.json');dp=read(L2L/'DECA_PROTOCOL.json')
    sources,versions=bfm_source_versions()
    files={**sources,**dp['source']}
    files.update({str(q):sha(q) for q in [Path(__file__),Path(__file__).with_name('vico_common_deca.py'),
        EVAL/'compute_base_metrics.py',EVAL/'compute_fid.py',EVAL/'compute_cpbd.py',
        EVAL/'arcface_torch/compute_csim.py']})
    for path,h in files.items():assert sha(path)==h,path
    for sample in bp['samples']:
        for role in ['speaker','gt']:
            info=sample['streams'][role];assert sha(info['path'])==info['sha256']
    out=dict(schema='vico43-unified-external-evaluation-v1',code_and_deca_assets=files,
        bfm_assets=bp['assets'],bfm_prior_versions=versions,arcface_path='/public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth',
        source_frames=16288,source_pairs=43,deca_space='raw50 expression + head axis-angle3 + jaw axis-angle3',
        bfm_space='exp64 + angle3 + translation3; distinct table',
        cadence='30fps; full timeline; no DTW/lag-fitting of predictions',
        motion_aggregation='equal clips, complete120-frame windows stride30; TLCC maxlag60',
        image_aggregation='official frame-weighted; FID whole set',
        bootstrap='paired mean differences; source video_id clusters, 10000 draws seed137')
    out['arcface_sha256']=sha(out['arcface_path'])
    target=WORK/'evaluation_protocol.json'
    if target.exists():assert read(target)==out,'Evaluation changed after freeze'
    else:save(target,out)
    return out


def official_child(metric,index):
    import scipy.ndimage
    import imageio.v2
    scipy.ndimage.imread=imageio.v2.imread
    for k,v in [('float',float),('int',int)]:
        if k not in np.__dict__:setattr(np,k,v)
    spec=read(OLD/'protocol.json')['datasets']['vico']
    target=EVAL/f'compute_{metric}.py'
    args=['--pd_video_folder',str(AF/'scoring/avatarforcing'),'--anno_file',spec['annotation'],'--task','listener']
    if metric in ['base_metrics','fid','csim']:args+=['--gt_video_folder',spec['gt_dir']]
    if metric=='fid':args+=['--batch_size','256']
    if metric=='csim':
        target=EVAL/'arcface_torch/compute_csim.py'
        args+=['--batch_size','256','--weight','/public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth']
    if metric=='cpbd':
        import pandas as pd
        df=pd.read_csv(spec['annotation']);anno=AF/'image_metrics'/f'anno_{index:02d}.csv'
        df.iloc[[index]].to_csv(anno,index=False);args[args.index('--anno_file')+1]=str(anno)
    sys.path.insert(0,str(target.parent));sys.argv=[str(target),*args]
    os.chdir(target.parent);runpy.run_path(str(target),run_name='__main__')


def score_image(metric,index=None):
    folder=AF/'image_metrics';folder.mkdir(parents=True,exist_ok=True)
    name=metric+(f'_{index:02d}' if index is not None else '')
    receipt=folder/(name+'.json')
    if receipt.exists():return read(receipt)
    start=time.time();log=folder/(name+'.log')
    args=[sys.executable,__file__,'child','--metric',metric]
    if index is not None:args+=['--index',str(index)]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if metric=='cpbd':env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with log.open('w') as f:subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    patterns={'base_metrics':{'psnr':r'^psnr:\s*([\d.eE+\-]+)','ssim':r'^ssim:\s*([\d.eE+\-]+)'},
        'fid':{'fid':r'^fid:\s*([\d.eE+\-]+)'},'cpbd':{'cpbd':r'^cpbd:\s*([\d.eE+\-]+)'},
        'csim':{'csim':r'^cosine similarity:\s*([\d.eE+\-]+)'}}[metric]
    values={k:float(re.findall(v,log.read_text(),re.M)[-1]) for k,v in patterns.items()}
    out=dict(metric=metric,index=index,seconds=time.time()-start,values=values)
    save(receipt,out);return out


def images():
    assert read(AF/'GENERATION_COMPLETE.json')['clips']==43
    p=read(AF/'protocol.json');spec=read(OLD/'protocol.json')['datasets']['vico']
    import pandas as pd
    df=pd.read_csv(spec['annotation'])
    assert set(df.uuid)=={s['id'] for s in p['samples']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        pending=[ex.submit(score_image,'cpbd',i) for i in range(43)]
        results=[score_image('base_metrics'),score_image('fid'),score_image('csim')]
        cpbd=[f.result() for f in pending]
    counts={s['id']:s['frames'] for s in p['samples']};total=sum(counts.values())
    vals={k:v for row in results for k,v in row['values'].items()}
    vals['cpbd']=sum(r['values']['cpbd']*counts[df.iloc[r['index']].uuid] for r in cpbd)/total
    save(AF/'IMAGE_RESULTS.json',dict(model='Avatar Forcing source-only adaptation',clips=43,frames=total,
        metrics=vals,reports=results,cpbd=cpbd,scorer='same official ViCo code, frame-weighted; FID set-level'))


def overlap_cpbd(generation_pid):
    """Score completed videos on CPU while one verified generator is live."""
    cmdline=Path(f'/proc/{generation_pid}/cmdline')
    assert cmdline.exists() and b'vico_avatarforcing.py' in cmdline.read_bytes()
    spec=read(OLD/'protocol.json')['datasets']['vico']
    with open(spec['annotation']) as f:rows=list(csv.DictReader(f))
    submitted={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            for i,row in enumerate(rows):
                if i not in submitted and (AF/'generated'/row['uuid']/'complete.json').exists():
                    submitted[i]=pool.submit(score_image,'cpbd',i)
            for f in submitted.values():
                if f.done():f.result()
            done=sum(f.done() for f in submitted.values())
            save(AF/'cpbd_overlap_status.json',dict(submitted=len(submitted),completed=done,total=43,generation_pid=generation_pid,at=time.time()))
            if done==43:return
            if not cmdline.exists():
                if not (AF/'GENERATION_COMPLETE.json').exists():raise RuntimeError('Observed generator exited before completion; no automatic restart')
            elif b'vico_avatarforcing.py' not in cmdline.read_bytes():
                raise RuntimeError('Generator PID was reused; refusing to wait on unrelated process')
            time.sleep(5)


def extract(shard=0,shards=1):
    import torch
    import cv2
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico_motion_extract import Extractor,load_frames
    from vico_common_deca import crop_frame,encode
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.cuda.set_per_process_memory_fraction(.09,0)
    p=read(AF/'protocol.json')
    for f,h in p['old_results'].items():assert sha(f)==h
    old=read(BASE/'protocol.json')
    sources,_=bfm_source_versions()
    for f,h in {**old['assets'],**sources}.items():assert sha(f)==h
    # Same versioned FAN/reconstruction, including detection masks and fallback.
    bf=Extractor()
    dp=read(L2L/'DECA_PROTOCOL.json')
    for f,h in dp['source'].items():assert sha(f)==h
    sys.path.insert(0,str(ROOT/'compare/deployment/vendor/DECA'))
    from decalib.models.encoders import ResnetEncoder
    deca=ResnetEncoder(236)
    deca.load_state_dict(torch.load(ROOT/'compare/deployment/assets/deca_model.tar',map_location='cpu',weights_only=False)['E_flame'],strict=True)
    deca.eval().cuda()
    # Test parity on a previously sealed GT, using the same preprocessing and precision.
    case=old['samples'][shard]
    uid=case['id'];frames=load_frames(case['streams']['gt']['path'])
    landmarks=np.load(BASE/'features/gt'/(uid+'.npz'))['landmarks']
    picks=np.linspace(0,len(frames)-1,8).astype(int)
    torch.backends.cudnn.allow_tf32=True
    crop=[crop_frame((frames[i],landmarks[i])) for i in picks]
    coeff=encode(deca,crop)
    sealed=np.load(L2L/'features/gt'/(uid+'.npz'))
    err=float(np.abs(coeff-sealed['coeff'][picks]).max())
    terr=float(np.abs(np.stack([c[1] for c in crop])-sealed['transform_params'][picks]).max())
    assert err<1e-3 and terr==0,(err,terr)
    save(AF/f'extractor_parity_{shard}.json',dict(passed=True,id=uid,frames=picks.tolist(),deca_max_abs=err,
        crop_max_abs=terr,torch=torch.__version__,face_alignment=__import__('face_alignment').__version__))
    for s in p['samples'][shard::shards]:
        uid=s['id'];receipt=AF/'features'/f'{uid}.json'
        source=AF/'scoring/avatarforcing'/(uid+'.listener.mp4')
        source_sha=sha(source)
        if receipt.exists():
            r=read(receipt);assert source_sha==r['source_sha256']
            for f,h in r['outputs'].items():assert sha(f)==h
            continue
        start=time.time();frames=load_frames(source)
        assert frames.shape==(s['frames'],256,256,3)
        torch.backends.cudnn.allow_tf32=False
        arrays,timing,faces=bf.extract(frames)
        bfpath=AF/'features/bfm'/(uid+'.npz');bfpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(bfpath,**arrays)
        torch.backends.cudnn.allow_tf32=True
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            cropped=list(pool.map(crop_frame,zip(frames,arrays['landmarks'])))
        coeff=encode(deca,cropped)
        motion=np.concatenate([coeff[:,150:200],coeff[:,200:206]],1)
        dpath=AF/'features/deca'/(uid+'.npz');dpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dpath,motion=motion,coeff=coeff,transform_params=np.stack([c[1] for c in cropped]))
        assert np.isfinite(motion).all() and motion.shape==(s['frames'],56)
        save(receipt,dict(id=uid,source_sha256=source_sha,frames=s['frames'],
            missing_face_frames=int((~arrays['detected']).sum()),multi_face_frames=int((arrays['face_counts']>1).sum()),
            outputs={str(f):sha(f) for f in [bfpath,dpath]},seconds=time.time()-start,timing=timing))
        print('extracted',shard,uid,round(time.time()-start,2),flush=True)


def motions():
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico_motion_metrics import score_clip as bfm_metrics
    from vico_common_deca import clip_metrics as deca_metrics, GROUPS
    p=read(AF/'protocol.json');rows=[];bfrows=[]
    for s in p['samples']:
        uid=s['id'];r=read(AF/'features'/(uid+'.json'))
        for f,h in r['outputs'].items():assert sha(f)==h
        native=[np.load(L2L/'features'/role/(uid+'.npz'))['motion'].astype(float) for role in ['speaker','gt']]
        dpath=AF/'features/deca'/(uid+'.npz');pred=np.load(dpath)['motion'].astype(float)
        rows.append(dict(id=uid,frames=len(pred),metrics=deca_metrics(*native,pred),prediction_sha256=sha(dpath)))
        bnative=[np.load(BASE/'features'/role/(uid+'.npz'))['motion'].astype(float) for role in ['speaker','gt']]
        bp=np.load(AF/'features/bfm'/(uid+'.npz'))['motion'].astype(float)
        score,audit=bfm_metrics(*bnative,bp,max_lag=60,window=120,stride=30)
        bfrows.append(dict(id=uid,metrics_raw=score,timing_audit=audit))
    avg={k:{g:float(np.mean([r['metrics'][k][g] for r in rows if r['metrics'][k][g] is not None])) for g in GROUPS} for k in rows[0]['metrics']}
    coverage={k:{g:sum(r['metrics'][k][g] is not None for r in rows) for g in GROUPS} for k in avg}
    save(AF/'DECA_RESULTS.json',dict(method='avatarforcing',clips=43,frames=sum(s['frames'] for s in p['samples']),
        metrics=avg,coverage=coverage,per_clip=rows,readout='video re-extraction'))
    raw={k:{g:float(np.mean([r['metrics_raw'][k][g] for r in bfrows if r['metrics_raw'][k][g] is not None])) for g in v} for k,v in bfrows[0]['metrics_raw'].items()}
    cov={k:{g:sum(r['metrics_raw'][k][g] is not None for r in bfrows) for g in v} for k,v in raw.items()}
    scaled={k:{g:v*(100 if k in ['FD','FID_delta_fm'] else 1) for g,v in groups.items()} for k,groups in raw.items()}
    save(AF/'BFM_RESULTS.json',dict(clips=43,per_clip=bfrows,metrics_raw=raw,metrics_table_scale=scaled,coverage_clips=cov))


def cluster_bootstrap(ids,values,seed=137,n=10000):
    """Pair-level mean; resample whole source dialogues, retaining all their clips."""
    groups={}
    for uid,val in zip(ids,values):
        if val is not None:groups.setdefault(uid.rsplit('_',2)[0],[]).append(float(val))
    if not groups: return dict(mean=None,ci95=None,clips=0,source_groups=0)
    blocks=list(groups.values());sums=np.array([sum(v) for v in blocks]);counts=np.array([len(v) for v in blocks])
    rng=np.random.default_rng(seed);picks=rng.integers(len(blocks),size=(n,len(blocks)))
    estimates=sums[picks].sum(1)/counts[picks].sum(1)
    return dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(estimates,[.025,.975]).tolist(),
        clips=int(counts.sum()),source_groups=len(groups),replicates=n,seed=seed,
        aggregation='equal clips; bootstrap resamples source video_id clusters; conditional on fixed seed/checkpoints')


def summarize():
    p=read(AF/'protocol.json')
    for f,h in p['old_results'].items():assert sha(f)==h,f
    common=read(COMMON/'RESULTS.json')
    common['models']['avatarforcing']=read(AF/'DECA_RESULTS.json')
    image={'joint':read(OLD/'RESULTS.json')['metrics'],'dystream':read(DY/'IMAGE_RESULTS.json')['metrics'],
           'avatarforcing':read(AF/'IMAGE_RESULTS.json')['metrics']}
    main=['joint','dystream','avatarforcing'];reference=common['models']['joint']
    ids=[r['id'] for r in reference['per_clip']]
    assert len(ids)==len(set(ids))==43
    paired={}
    for name in ['dystream','avatarforcing']:
        model=common['models'][name];a={r['id']:r['metrics'] for r in reference['per_clip']};b={r['id']:r['metrics'] for r in model['per_clip']}
        assert set(a)==set(b)
        result={}
        for metric,groups in reference['metrics'].items():
            result[metric]={}
            for group in groups:
                vals=[None if a[u][metric][group] is None or b[u][metric][group] is None else a[u][metric][group]-b[u][metric][group] for u in ids]
                result[metric][group]=cluster_bootstrap(ids,vals)
        paired[name]=result
    source_count=len({u.rsplit('_',2)[0] for u in ids})
    result=dict(schema='vico43-unified-local-v1',test_pairs=43,source_dialogues=source_count,
        images=image,deca=common,bfm={'joint':read(BASE/'RESULTS.json'),
            'dystream':read(DY/'MOTION_RESULTS.json'),'avatarforcing':read(AF/'BFM_RESULTS.json')},
        paired_deca_difference_joint_minus_baseline=paired,
        limitations=['Independent local experiment, not reproduction of DIM published table.',
            'BFM/DECA/EMOCA are distinct parameter spaces.',
            'Zero-self-audio/zero-user-visual adapters are disclosed, not native known-audio protocols.',
            'Training datasets/budgets and video vs native-motion readout differ.',
            'One seed estimates paired test-set uncertainty, not sampling/training variability.',
            'ViCo listener has one reference; distance to that reference is not complete appropriateness evidence.',
            'Image FID is set-level and is not included in per-clip bootstrap.'],
        source_sha256={str(f):sha(f) for f in [AF/'protocol.json',AF/'DECA_RESULTS.json',AF/'IMAGE_RESULTS.json',COMMON/'RESULTS.json']},
        audiovisual_claim=False,l2l_retraining_included=False)
    save(WORK/'RESULTS.json',result)
    with (WORK/'image_metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method',*image['joint']])
        for name in main:w.writerow([name,*[image[name][k] for k in image['joint']]])
    with (WORK/'deca_per_clip.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','uuid','source_video_id','frames','metric','group','value'])
        for name,model in common['models'].items():
            for r in model['per_clip']:
                for metric,groups in r['metrics'].items():
                    for group,val in groups.items():w.writerow([name,r['id'],r['id'].rsplit('_',2)[0],r['frames'],metric,group,val])
    print(json.dumps(dict(test_pairs=43,source_dialogues=source_count,images=image),indent=2))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','child','images','overlap-cpbd','extract','motions','summarize'])
    a.add_argument('--metric');a.add_argument('--index',type=int);a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1)
    a.add_argument('--generation-pid',type=int)
    args=a.parse_args()
    if args.command=='freeze':freeze_evaluation()
    elif args.command=='child':official_child(args.metric,args.index)
    elif args.command=='images':images()
    elif args.command=='overlap-cpbd':overlap_cpbd(args.generation_pid)
    elif args.command=='extract':extract(args.shard,args.shards)
    elif args.command=='motions':motions()
    else:summarize()
