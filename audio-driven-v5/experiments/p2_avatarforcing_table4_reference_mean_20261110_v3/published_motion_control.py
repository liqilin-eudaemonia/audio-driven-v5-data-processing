"""Observed published panels through the identical official SPECTRE frontend.

Diagnostic only: two qualitative examples do not establish Table4 membership or
quantitative encodings. No empirical lag, frame selection, seed or metric fit.
"""
import fcntl
import os
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_avatarforcing_table4_generate import read,save,sha,guard


def main():
    import numpy as np
    import torch
    import official_spectre_workers as worker
    out=WORK/'published_motion_control'
    out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=[]
        for base,uid in [('03_listening_01','ClNUxPsUByg_000210_000221.0'),('03_listening_02','Ew2z_sYABE0_000063_000083.0')]:
            for role in ['speaker','avatar']:
                path=ROOT/'experiments/p2_avatarforcing_table4_diagnosis_20261110/published_supplement'/(base+'_'+role+'.mp4')
                sources.append(dict(id=uid,role=role,path=str(path),sha256=sha(path),frames=read(WORK/'native_generated'/uid/'complete.json')['frames']))
        contract=dict(schema='published-motion-control-v1',script_sha256=sha(__file__),
            helper_sha256=sha(WORK/'official_spectre.py'),executor_sha256=sha(WORK/'official_spectre_workers.py'),
            source_protocol_sha256=sha(WORK/'OFFICIAL_SPECTRE_PROTOCOL.json'),sources=sources,
            controls='Same RetinaFace/FAN2/JPEG/context46/FP32 and4warmups; source timeline unchanged; no fitted lag',
            scope='Two published qualitative examples; not a Table4 aggregate and not proof these encodings were scored')
        protocol=out/'PROTOCOL.json'
        if protocol.exists():assert read(protocol)==contract
        else:save(protocol,contract)
        device=guard(.025,32768)
        source,model,tracker,get_landmarks=worker.load_models()
        save(out/'RUNTIME.json',dict(pid=os.getpid(),device=device,started_unix=time.time(),protocol_sha256=sha(protocol)))
        records=[]
        for item in sources:
            dest=out/(item['id']+'_'+item['role']+'.npz');receipt=dest.with_suffix('.json')
            if receipt.exists():
                r=read(receipt);assert r['protocol_sha256']==sha(protocol) and sha(dest)==r['sha256'];records.append(r);continue
            started=time.time()
            frames,grid=source.decode25(item['path'],item['frames'])
            images,landmarks,audit=source.official_crop(frames,tracker,get_landmarks)
            motion=model.chunks(images,batch=46)
            assert motion.shape==(item['frames'],56) and np.isfinite(motion).all()
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,motion=motion,landmarks=landmarks,fps=25,frame_indices=np.arange(len(motion)))
            tmp.replace(dest)
            r=dict(**item,features=str(dest),feature_sha256=sha(dest),sha256=sha(dest),protocol_sha256=sha(protocol),grid=grid,face_audit=audit,seconds=time.time()-started)
            save(receipt,r);records.append(r)
            save(out/'STATUS.json',dict(stage='extracting',pid=os.getpid(),completed=len(records),total=4,updated_unix=time.time()))
            print('Published motion',item['id'],item['role'],r['seconds'],flush=True)
            del frames,images,landmarks,motion
        save(out/'EXTRACTION_COMPLETE.json',dict(passed=True,streams=4,records=records,protocol_sha256=sha(protocol)))
        print('Published control extraction complete',flush=True)


if __name__=='__main__':main()
