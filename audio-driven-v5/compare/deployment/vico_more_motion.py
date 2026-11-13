"""Common BFM and DECA video readout plus separate native BFM motion scores."""
import argparse
import concurrent.futures
from pathlib import Path
import sys
import time

import numpy as np
from vico_more_common import ROOT, WORK, BFM, DECA, read, save, sha, gpu_guard


def verify_scorers():
    old=ROOT/'experiments/p2_vico43_unified_20261107/evaluation_protocol.json'
    p=read(old)
    for group in ['code_and_deca_assets','bfm_assets','fid_weights']:
        for path,h in p[group].items():assert sha(path)==h,path
    assert sha(p['arcface_path'])==p['arcface_sha256']
    return dict(passed=True, sealed_protocol_sha256=sha(old), unchanged=True)


def extract(method):
    import torch
    gpu_guard(18)
    out=WORK/method
    records=read(out/'RENDER_COMPLETE.json')['records']
    audit=verify_scorers();save(out/'SCORER_REUSE_AUDIT.json',audit)
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico_motion_extract import Extractor, load_frames
    from vico_common_deca import crop_frame, encode
    ext=Extractor();torch.set_num_threads(2)
    bfm_records=[]
    for rec in records:
        uid=rec['id'];path=out/'features/bfm'/(uid+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.with_suffix('.json').exists():
            r=read(path.with_suffix('.json'));assert sha(path)==r['sha256'];bfm_records.append(r);continue
        assert sha(rec['path'])==rec['sha256']
        frames=load_frames(rec['path']);assert len(frames)==rec['frames']
        arrays,timing,faces=ext.extract(frames)
        if not bfm_records:
            save(out/'BFM_EXTRACTION_PARITY.json',ext.parity(frames,arrays,faces))
        np.savez_compressed(path,**arrays)
        r=dict(id=uid,path=str(path),sha256=sha(path),frames=len(frames),timing=timing,
            missing_faces=int((~arrays['detected']).sum()),source_sha256=rec['sha256'])
        save(path.with_suffix('.json'),r);bfm_records.append(r)
        print(method,'bfm score',len(bfm_records),uid,flush=True)
    del ext;torch.cuda.empty_cache()
    # Exactly the prior DECA encoder, crop and precision policy.
    torch.backends.cudnn.allow_tf32=True
    sys.path.insert(0,str(ROOT/'compare/deployment/vendor/DECA'))
    from decalib.models.encoders import ResnetEncoder
    model=ResnetEncoder(236)
    model.load_state_dict(torch.load(ROOT/'compare/deployment/assets/deca_model.tar',map_location='cpu',weights_only=False)['E_flame'],strict=True)
    model=model.eval().cuda()
    deca_records=[]
    # A sealed GT clip provides a differential check before new predictions.
    first=records[0]['id']
    sample=next(r for r in read(BFM/'protocol.json')['samples'] if r['id']==first)
    frames=load_frames(sample['streams']['gt']['path'])
    lm=np.load(BFM/'features/gt'/(first+'.npz'))['landmarks']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        crops=list(pool.map(crop_frame,zip(frames,lm)))
    coeff=encode(model,crops)
    old=np.load(DECA/'features/gt'/(first+'.npz'))
    diff=float(np.abs(coeff-old['coeff']).max())
    assert diff<1e-3,diff
    save(out/'DECA_EXTRACTION_PARITY.json',dict(passed=True,clip=first,coefficient_max_abs=diff,
        full_clip=True,frames=len(frames)))
    for rec in records:
        uid=rec['id'];path=out/'features/deca'/(uid+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.with_suffix('.json').exists():
            r=read(path.with_suffix('.json'));assert sha(path)==r['sha256'];deca_records.append(r);continue
        frames=load_frames(rec['path']);landmarks=np.load(out/'features/bfm'/(uid+'.npz'))['landmarks']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            crops=list(pool.map(crop_frame,zip(frames,landmarks)))
        coeff=encode(model,crops)
        motion=np.concatenate([coeff[:,150:200],coeff[:,200:206]],1)
        # DECA order: shape100, texture50, exp50, pose6, camera3, light27.
        assert motion.shape==(len(frames),56)
        np.savez_compressed(path,coeff=coeff,motion=motion,
            transform_params=np.stack([c[1] for c in crops]))
        r=dict(id=uid,path=str(path),sha256=sha(path),frames=len(frames),source_sha256=rec['sha256'])
        save(path.with_suffix('.json'),r);deca_records.append(r)
        print(method,'deca score',len(deca_records),uid,flush=True)
    save(out/'EXTRACTION_COMPLETE.json',dict(success=True,bfm=bfm_records,deca=deca_records))


def aggregate(rows):
    first=rows[0]['metrics']
    means={metric:{group:float(np.mean([r['metrics'][metric][group] for r in rows
        if r['metrics'][metric][group] is not None])) for group in groups} for metric,groups in first.items()}
    coverage={metric:{group:sum(r['metrics'][metric][group] is not None for r in rows)
        for group in groups} for metric,groups in first.items()}
    return dict(metrics=means,coverage=coverage,per_clip=rows,clips=len(rows))


def score(method):
    out=WORK/method
    assert read(out/'EXTRACTION_COMPLETE.json')['success']
    verify_scorers()
    sys.path.insert(0,str(ROOT/'scripts'))
    from p2_vico_motion_metrics import score_clip
    from vico_common_deca import clip_metrics
    deca,bfm,native=[],[],[]
    for rec in read(out/'RENDER_COMPLETE.json')['records']:
        uid=rec['id']
        paths=dict(s=BFM/'features/speaker'/(uid+'.npz'),g=BFM/'features/gt'/(uid+'.npz'),p=out/'features/bfm'/(uid+'.npz'))
        arrays={k:np.load(v)['motion'] for k,v in paths.items()}
        m,audit=score_clip(arrays['s'],arrays['g'],arrays['p'])
        bfm.append(dict(id=uid,frames=rec['frames'],metrics=m,audit=audit))
        p=np.load(out/'predictions'/(uid+'.npz'))['motion']
        m,audit=score_clip(arrays['s'],arrays['g'],p)
        native.append(dict(id=uid,frames=rec['frames'],metrics=m,audit=audit))
        s=np.load(DECA/'features/speaker'/(uid+'.npz'))['motion']
        g=np.load(DECA/'features/gt'/(uid+'.npz'))['motion']
        p=np.load(out/'features/deca'/(uid+'.npz'))['motion']
        m=clip_metrics(s,g,p)
        deca.append(dict(id=uid,frames=rec['frames'],metrics=m))
    result=dict(method=method,deca_video=aggregate(deca),bfm_video=aggregate(bfm),bfm_native=aggregate(native),
        readout_separation='Video re-extraction and native driver coefficients are separate tables',
        original_scorer_code_unchanged=True)
    save(out/'MOTION_RESULTS.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['rlhg','listenformer'])
    p.add_argument('stage',choices=['extract','score']);a=p.parse_args()
    if a.stage=='extract':extract(a.method)
    else:score(a.method)
