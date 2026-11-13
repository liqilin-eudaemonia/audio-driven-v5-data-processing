"""Unmodified official ViCo image scoring plus existing motion metric definitions."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"experiments/p2_compare_vico43_20261106/dystream"
REF=ROOT/"experiments/p2_vico_motion8000_20261106"
EVAL=ROOT/"evaluation/vico_challenge_baseline/evaluations"


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+f'.tmp.{os.getpid()}');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)


def child(work,metric,index):
    import numpy as np
    import scipy.ndimage
    import imageio.v2
    scipy.ndimage.imread=imageio.v2.imread
    for key,value in [('float',float),('int',int)]:
        if key not in np.__dict__:setattr(np,key,value)
    p=json.loads((work/'protocol.json').read_text());spec=p['datasets']['vico'];anno=Path(spec['annotation'])
    target=EVAL/f'compute_{metric}.py'
    args=['--pd_video_folder',str(work/'scoring/dystream'),'--anno_file',str(anno),'--task','listener']
    if metric in ['base_metrics','fid','csim']:args+=['--gt_video_folder',spec['gt_dir']]
    if metric=='fid':args+=['--batch_size','512']
    if metric=='csim':target=EVAL/'arcface_torch/compute_csim.py';args+=['--batch_size','512','--weight','/public/0_DATA/2_Avatar/ViCo/ms1mv3_arcface_r100_fp16/backbone.pth']
    if metric=='cpbd':
        import pandas as pd
        frame=pd.read_csv(anno);path=work/'image_metrics'/f'anno_{index:02d}.csv';frame.iloc[[index]].to_csv(path,index=False);args[args.index('--anno_file')+1]=str(path)
    sys.path.insert(0,str(target.parent));sys.argv=[str(target),*args];os.chdir(target.parent);runpy.run_path(str(target),run_name='__main__')


def score(work,metric,index=None):
    folder=work/'image_metrics';folder.mkdir(exist_ok=True);name=metric+(f'_{index:02d}' if index is not None else '')
    with (folder/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipt=folder/(name+'.json')
        if receipt.exists():return json.loads(receipt.read_text())
        argv=[sys.executable,str(Path(__file__).resolve()),'child','--work',str(work),'--metric',metric]
        if index is not None:argv+=['--index',str(index)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7' if metric in ['fid','csim'] else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if metric=='cpbd':env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        start=time.time();dest=folder/(name+'.log')
        with dest.open('w') as f:subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        patterns={'base_metrics':{'psnr':r'^psnr:\s*([\d.eE+\-]+)','ssim':r'^ssim:\s*([\d.eE+\-]+)'},'fid':{'fid':r'^fid:\s*([\d.eE+\-]+)'},'cpbd':{'cpbd':r'^cpbd:\s*([\d.eE+\-]+)'},'csim':{'csim':r'^cosine similarity:\s*([\d.eE+\-]+)'}}[metric]
        result=dict(metric=metric,index=index,seconds=time.time()-start,values={k:float(re.findall(v,dest.read_text(),re.M)[-1]) for k,v in patterns.items()})
        save(receipt,result);return result


def images(work):
    p=json.loads((work/'protocol.json').read_text());items=p['samples']
    import pandas as pd
    anno=pd.read_csv(p['datasets']['vico']['annotation']);assert set(anno.uuid)=={s['id'] for s in items}
    submitted={};pending=[];start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        while not (work/'GENERATION_COMPLETE.json').exists():
            if (work/'CONTROLLER_FAILURE.json').exists():raise RuntimeError('Generation failed')
            for i,row in anno.iterrows():
                if i not in submitted and (work/'generated'/row.uuid/'complete.json').exists():
                    submitted[i]=pool.submit(score,work,'cpbd',int(i))
            for f in submitted.values():
                if f.done():f.result()
            save(work/'image_status.json',dict(stage='overlapping_cpbd',submitted=len(submitted),finished=sum(f.done() for f in submitted.values()),at=time.time()))
            time.sleep(3)
        for i,row in anno.iterrows():
            if i not in submitted:submitted[i]=pool.submit(score,work,'cpbd',int(i))
        base=pool.submit(score,work,'base_metrics')
        results=[score(work,'fid'),score(work,'csim'),base.result()]
        cpbd=[submitted[i].result() for i in range(43)]
    frames={s['id']:s['frames'] for s in items};total=sum(frames.values())
    values={k:v for r in results for k,v in r['values'].items()}
    values['cpbd']=sum(r['values']['cpbd']*frames[anno.iloc[r['index']].uuid] for r in cpbd)/total
    out=dict(model='DyStream listener-only adaptation',clips=43,frames=total,metrics=values,seconds=time.time()-start,
             reports=results,cpbd=cpbd,source='original official ViCo scoring scripts, frame-weighted merge; no scorer algorithm changes')
    save(work/'IMAGE_RESULTS.json',out);save(work/'image_status.json',dict(stage='complete',at=time.time()))


def motion_worker(work,shard):
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico_motion_extract import Extractor,load_frames
    p=json.loads((work/'protocol.json').read_text());extractor=Extractor()
    for i,sample in enumerate(p['samples'][shard::4]):
        uid=sample['id'];dest=work/'motion/features/prediction'/(uid+'.npz');receipt=dest.with_suffix('.json')
        if receipt.exists():continue
        source=work/'scoring/dystream'/(uid+'.listener.mp4');frames=load_frames(source);assert len(frames)==sample['frames']
        arrays,timing,faces=extractor.extract(frames)
        if i==0:save(work/f'motion/parity_{shard:02d}.json',input_parity(extractor,frames,arrays,faces))
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
        report=dict(id=uid,frames=len(frames),input_sha256=sha(source),feature_sha256=sha(dest),
            missing_face_frames=int((~arrays['detected']).sum()),multi_face_frames=int((arrays['face_counts']>1).sum()),seconds=timing)
        save(receipt,report);print('motion',shard,uid,flush=True)


def input_parity(ext,frames,arrays,faces):
    """Require identical actual network inputs, retaining all-landmark differences.

    FAN can change unused jaw/eyebrow coordinates between calls. Only the five
    aggregated points enter BFM alignment; do not reject identical RGB inputs
    solely because an unused landmark differs. Network agreement stays strict.
    """
    import numpy as np
    indices=np.flatnonzero(arrays['detected']);picks=indices[np.linspace(0,len(indices)-1,min(8,len(indices))).astype(int)]
    landmark_max=0.;pixel_max=0;coeff_max=0.
    for i in picks:
        lm=ext.fa.get_landmarks_from_image(frames[i],detected_faces=faces[i][:1])[0]
        landmark_max=max(landmark_max,float(np.abs(lm-arrays['landmarks'][i]).max()))
        original,_=ext.align(frames[i:i+1],lm[None]);batched,_=ext.align(frames[i:i+1],arrays['landmarks'][i:i+1])
        pixel_max=max(pixel_max,int(np.abs(original.astype(int)-batched.astype(int)).max()))
        coeff_max=max(coeff_max,float(np.abs(ext.reconstruct(original)[0]-arrays['coeff'][i]).max()))
    result=dict(passed=pixel_max==0 and coeff_max<1e-3,landmark_max_abs=landmark_max,
        aligned_uint8_max_abs=pixel_max,coefficient_max_abs=coeff_max,checked_frames=picks.tolist(),
        criterion='pixel-identical network input and coefficient error<1e-3; unused landmark variations explicitly reported')
    if not result['passed']:raise ValueError(f'Actual network input parity failed: {result}')
    return result


def motion(work):
    assert (work/'GENERATION_COMPLETE.json').exists()
    folder=work/'motion';folder.mkdir(exist_ok=True)
    original=json.loads((REF/'protocol.json').read_text())
    save(folder/'protocol.json',dict(model='DyStream listener-only adaptation',reference_extraction_protocol_sha256=sha(REF/'protocol.json'),
        reference_scoring_protocol_sha256=sha(REF/'SCORING_PROTOCOL.json'),shared_gt_features=str(REF/'features'),
        coefficient_space='BFM expression64 + angle3 + trans3; identical extractor and metrics to joint8000',
        fps=30,metric_parameters=dict(max_lag=60,window=120,stride=30),code={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/p2_vico_motion_extract.py',ROOT/'scripts/p2_vico_motion_metrics.py']}))
    processes=[];handles=[]
    for i in range(4):
        f=(folder/f'worker_{i:02d}.log').open('w');handles.append(f)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        python=ROOT/'compare/.envs/vico/bin/python'
        processes.append(subprocess.Popen([str(python),str(Path(__file__).resolve()),'motion-worker','--work',str(work),'--index',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT))
    try:
        codes=[p.wait() for p in processes];assert not any(codes),codes
    finally:
        for f in handles:f.close()
    import numpy as np
    sys.path.insert(0,str(ROOT/'scripts'));from p2_vico_motion_metrics import score_clip
    p=json.loads((work/'protocol.json').read_text());rows=[]
    for s in p['samples']:
        uid=s['id'];speaker=np.load(REF/'features/speaker'/(uid+'.npz'))['motion'].astype(float)
        gt=np.load(REF/'features/gt'/(uid+'.npz'))['motion'].astype(float)
        pred=np.load(folder/'features/prediction'/(uid+'.npz'))['motion'].astype(float)
        values,audit=score_clip(speaker,gt,pred)
        rows.append(dict(id=uid,frames=len(gt),metrics_raw=values,timing_audit=audit))
    result=aggregate(rows);result.update(model='DyStream listener-only adaptation',coefficient_space='BFM64+3+3',strict_customlistener_table_reproduction=False)
    save(work/'MOTION_RESULTS.json',result)


def aggregate(rows):
    import numpy as np
    raw={k:{g:float(np.mean([r['metrics_raw'][k][g] for r in rows if r['metrics_raw'][k][g] is not None])) for g in rows[0]['metrics_raw'][k]} for k in rows[0]['metrics_raw']}
    coverage={k:{g:sum(r['metrics_raw'][k][g] is not None for r in rows) for g in groups} for k,groups in raw.items()}
    displayed={k:{g:v*(100 if k in ['FD','FID_delta_fm'] else 1) for g,v in groups.items()} for k,groups in raw.items()}
    return dict(clips=len(rows),frames_per_stream=sum(r['frames'] for r in rows),metrics_raw=raw,metrics_table_scale=displayed,coverage_clips=coverage,per_clip=rows)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['child','images','motion','motion-worker']);a.add_argument('--work',type=Path,default=WORK);a.add_argument('--metric');a.add_argument('--index',type=int);p=a.parse_args();work=p.work.resolve()
    if p.command=='child':child(work,p.metric,p.index)
    elif p.command=='images':images(work)
    elif p.command=='motion':motion(work)
    else:motion_worker(work,p.index)
