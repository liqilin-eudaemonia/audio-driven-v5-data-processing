"""Motion readout, paired statistics, and review for the ViCo planner test."""
import argparse
import copy
import csv
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'compare/deployment')]
from vico_avatarforcing import read,save,sha
from p2_vico43_combined8000_sft2752 import WORK,BASE,ARM,ADAPTER
import vico_unified_evaluate as metrics

PREVIOUS=ROOT/'experiments/p2_vico43_unified_20261107'
KEY='joint_sft2752'


def setup_motion():
    assert read(WORK/'GENERATION_AND_IMAGE_COMPLETE.json')['outputs']==43
    p=read(WORK/'protocol.json')
    for path,h in p['execution_code'].items():assert sha(path)==h,path
    reference=read(PREVIOUS/'avatarforcing/protocol.json')
    sources={}
    for s in reference['samples']:
        uid=s['id'];path=WORK/'scoring'/ARM/(uid+'.listener.mp4')
        convert=read(WORK/'conversion'/(uid+'.json'));assert sha(path)==convert['file_sha256']
        sources[uid]=dict(path=str(path),sha256=sha(path))
    for file in [WORK/'RESULTS.json',BASE/'RESULTS.json',PREVIOUS/'RESULTS.json']:
        sources[str(file)]=dict(path=str(file),sha256=sha(file))
    mp=dict(samples=reference['samples'],old_results=reference['old_results'],sources=sources,
        planner_protocol_sha256=sha(WORK/'protocol.json'),
        feature_algorithm='same frozen FAN/BFM/DECA, crops and precision as prior ViCo43',
        source_functions_sha256=sha(ROOT/'compare/deployment/vico_unified_evaluate.py'),
        planner='SFT2752',generation_step=8000)
    save(WORK/'motion_protocol.json',mp)
    image=read(WORK/'RESULTS.json');save(WORK/'IMAGE_RESULTS.json',image)


def scoped_motion(command,shard=0,shards=2):
    # Reuse the already tested extraction/scoring functions without changing
    # their numerical operations or the previous experiment's module files.
    namespace=dict(metrics.__dict__,AF=WORK,WORK=WORK)
    original=metrics.extract if command=='extract' else metrics.motions
    source=inspect.getsource(original)
    assert source.count("p=read(AF/'protocol.json')")==1
    source=source.replace("p=read(AF/'protocol.json')","p=read(AF/'motion_protocol.json')")
    if command=='extract':
        assert source.count("AF/'scoring/avatarforcing'")==1
        source=source.replace("AF/'scoring/avatarforcing'",f"AF/'scoring/{ARM}'")
    else:source=source.replace("method='avatarforcing'",f"method='{KEY}'")
    exec(compile(source,str(SELF)+':scoped_motion','exec'),namespace)
    if command=='extract':namespace['extract'](shard,shards)
    else:namespace['motions']()


def summarize():
    original=read(PREVIOUS/'RESULTS.json')
    result=copy.deepcopy(original)
    result.update(schema='vico43-base-vs-sft2752-fixed8000-v1',
        comparison='Fixed combined8000; replace base Qwen with SFT2752 including its predicted descriptions, H and Gemma; no training',
        planner_protocol_sha256=sha(WORK/'protocol.json'),old_unified_sha256=sha(PREVIOUS/'RESULTS.json'))
    new=read(WORK/'DECA_RESULTS.json')
    result['images'][KEY]=read(WORK/'IMAGE_RESULTS.json')['metrics']
    result['deca']['models'][KEY]=new
    result['bfm'][KEY]=read(WORK/'BFM_RESULTS.json')
    ids=[r['id'] for r in new['per_clip']];a={r['id']:r['metrics'] for r in new['per_clip']}
    paired={}
    for name in ['joint','dystream','avatarforcing']:
        b={r['id']:r['metrics'] for r in result['deca']['models'][name]['per_clip']}
        assert set(a)==set(b) and len(a)==43
        paired[name]={m:{g:metrics.cluster_bootstrap(ids,[None if a[u][m][g] is None or b[u][m][g] is None else a[u][m][g]-b[u][m][g] for u in ids]) for g in groups} for m,groups in new['metrics'].items()}
    result['paired_sft_minus_reference']=paired
    save(WORK/'COMPARISON.json',result)
    with (WORK/'image_comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','psnr','ssim','fid','csim','cpbd'])
        for name in ['joint',KEY,'dystream','avatarforcing']:
            w.writerow([name,*[result['images'][name][m] for m in ['psnr','ssim','fid','csim','cpbd']]])
    # Verify the SFT planner really changed the hidden conditions, with the
    # same underlying source wave/image. Identical text alone is allowed.
    import torch
    from p2_combined_whisper_eval import same_values
    rows=[];p=read(WORK/'protocol.json')
    for s in read(BASE/'shards/00.json'):
        before=BASE/'conditions'/s['key'];after=WORK/'conditions'/s['key']
        assert sha(before/'first_frame.png')==sha(after/'first_frame.png')
        assert sha(before/'input_audit.json')==sha(after/'input_audit.json')
        old=read(before/'qwen/plan.json');current=read(after/'qwen/plan.json')
        assert current['adapter']==p['planner_adapter_identity'] and not current['target_description_access']
        old_h=torch.load(before/'qwen/hidden.pt',weights_only=True,map_location='cpu')
        new_h=torch.load(after/'qwen/hidden.pt',weights_only=True,map_location='cpu')
        same=same_values(old_h['condition'],new_h['condition'])
        rows.append(dict(id=s['id'],duration_s=s['geometry']['duration_s'],
            old_description=old['description'],new_description=current['description'],
            description_changed=old['description']!=current['description'],hidden_changed=not same))
    with (WORK/'planning_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(WORK/'CONDITION_DIFFERENCE.json',dict(pairs=43,description_changed=sum(r['description_changed'] for r in rows),
        hidden_changed=sum(r['hidden_changed'] for r in rows),source_media_identical=True,
        planner_step=2752,backend_step=8000))


def audit():
    import cv2
    r=read(WORK/'COMPARISON.json');p=read(WORK/'protocol.json');rows=[];err=0.
    for item in read(BASE/'shards/00.json'):
        uid=item['id'];g=item['geometry'];folder=WORK/'generated'/ARM/item['key']
        meta=read(folder/'metadata.json')
        assert meta['step']==8000 and meta['planning_adapter']==p['planner_adapter_identity']
        assert meta['checkpoint_sha256']==p['arms'][0]['original_sha256']
        assert not meta['target_description_access'] and not meta['listener_future_input']
        assert meta['seed']==137 and meta['generation']==p['generation']
        conv=read(WORK/'conversion'/(uid+'.json'));score=WORK/'scoring'/ARM/(uid+'.listener.mp4')
        assert sha(score)==conv['file_sha256']
        counts=[]
        for path,size,fps,n in [(folder/'generated.mp4',512,25,g['frames']),(score,256,30,g['gt_frames'])]:
            cap=cv2.VideoCapture(str(path));assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS)==fps
            count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(size,size,3);count+=1
            cap.release();assert count==n;counts.append(count)
        f=read(WORK/'features'/(uid+'.json'))
        assert f['source_sha256']==sha(score)
        for path,h in f['outputs'].items():assert sha(path)==h
        pred=np.load(WORK/'features/deca'/(uid+'.npz'))['motion'].astype(float)
        gt=np.load(ROOT/'experiments/p2_compare_vico43_20261106/learning2listen/features/gt'/(uid+'.npz'))['motion'].astype(float)
        row=next(x for x in r['deca']['models'][KEY]['per_clip'] if x['id']==uid)
        for name,sl in [('exp',slice(0,50)),('head_rotation',slice(50,53)),('jaw_rotation',slice(53,56))]:
            err=max(err,abs(float(np.abs(pred[:,sl]-gt[:,sl]).mean())-row['metrics']['L1'][name]))
        rows.append(dict(id=uid,decoded_native=counts[0],decoded_scoring=counts[1],missing_face_frames=f['missing_face_frames']))
    assert err<1e-12
    for path,h in p['execution_code'].items():assert sha(path)==h,path
    save(WORK/'FINAL_NUMERICAL_AUDIT.json',dict(passed=True,pairs=43,fully_decoded_videos=86,
        scoring_frames=sum(x['decoded_scoring'] for x in rows),independent_L1_max_error=err,
        source_only=True,fixed_generator=True,planner_step=2752,backend_step=8000,rows=rows,
        comparison_sha256=sha(WORK/'COMPARISON.json')))


def finish(pid):
    proc=Path(f'/proc/{pid}/cmdline')
    def observed():
        return proc.exists() and any(name in proc.read_bytes() for name in
            [b'p2_vico43_combined8000_sft2752.py',b'p2_vico43_sft2752_gpu7.py'])
    if not (WORK/'GENERATION_AND_IMAGE_COMPLETE.json').exists():
        assert observed()
    save(WORK/'FOLLOWUP_RUNTIME.json',dict(pid=os.getpid(),generation_pid=pid,at=time.time()))
    while observed():time.sleep(5)
    setup_motion()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    children=[];logs=[]
    try:
        for i in range(2):
            log=(WORK/f'extract_{i}.log').open('w');logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(SELF),'extract','--shard',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
        assert [c.wait() for c in children]==[0,0]
    finally:
        for c in children:
            if c.poll() is None:c.terminate();c.wait()
        for f in logs:f.close()
    scoped_motion('motions');summarize();audit()
    save(WORK/'MOTION_COMPARISON_COMPLETE.json',dict(passed=True,at=time.time(),review_pending=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['finish','extract','motions','summarize','audit']);a.add_argument('--pid',type=int);a.add_argument('--shard',type=int,default=0);args=a.parse_args()
    if args.command=='finish':finish(args.pid)
    elif args.command in ['extract','motions']:scoped_motion(args.command,args.shard)
    elif args.command=='summarize':summarize()
    else:audit()
