"""Evaluation-only DECA GT extraction. No generator opens this directory."""
import argparse,math,time
import numpy as np
from runtime import *

def allocated_deca_extractor(deca):
    # Reuse the original initializer verbatim except its historical GPU7-only guard.
    import inspect
    source=inspect.getsource(deca.extractor)
    replacements={
        "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'":"assert os.environ.get('CUDA_VISIBLE_DEVICES') in {'4','7'}",
        "'-i', '7',":"'-i', os.environ['CUDA_VISIBLE_DEVICES'],",
        "assert actual == GPU_UUID, 'Only authorized node11 GPU7 may run this extractor'":"assert actual == GPU_UUIDS[int(os.environ['CUDA_VISIBLE_DEVICES'])], 'Only authorized node11 GPU4/GPU7 may run this extractor'",
    }
    for before,after in replacements.items():
        assert source.count(before)==1,before
        source=source.replace(before,after)
    namespace=dict(deca.__dict__,GPU_UUIDS=GPU_UUIDS)
    exec(compile(source,str(WORK/'gt_features.py')+':allocated_deca_extractor','exec'),namespace)
    save(WORK/'devices'/f'deca_guard_{os.getpid()}.json',dict(original_source_sha256=sha(deca.__file__),physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),only_changes='GPU7-only guard expanded to user-authorized GPU4/GPU7; original model initialization and TF32 settings preserved',adapted_function=source))
    return namespace['extractor']()

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();activate_readers();device()
    sys.path.insert(0,str(ROOT/'src'))
    import p2_listen.contracts  # Resolve the package before scripts/p2_listen.py can shadow it.
    sys.path.append('/private/liqi_workspace/.cache/paired_video_metrics/python')
    from p2_vico_motion_extract import Extractor
    import vico_common_deca as deca
    from feature_inputs import speaker_frames
    from concurrent.futures import ThreadPoolExecutor
    ext=Extractor();model=allocated_deca_extractor(deca);done=0
    for s in selected():
        resident_checkpoint()
        dest=WORK/'evaluation_gt'/s['dataset']/s['id'];dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (dest/'complete.json').exists():continue
            started=time.time();ref=s['evaluation_reference']
            if s['dataset']=='responsenet':
                fake=dict(s,input_reference=dict(speaker=dict(video=ref['gt'])))
                frames,proof=speaker_frames(fake)
            else:
                from evaluation.seamless.reader import decode_reference
                original,fps,proof=decode_reference(ref['gt_reference']);n=math.ceil(s['duration']*30-1e-7)
                ix=np.minimum(np.floor(np.arange(n)*fps/30+.5).astype(int),len(original)-1);frames=original[ix]
            faces=ext.detect(frames)
            if any(len(f) for f in faces):lm,detected,counts=ext.landmarks(frames,faces)
            else:lm=np.full((len(frames),68,2),-1.,np.float32);detected=np.zeros(len(frames),bool)
            def crop(pair):
                image,landmark=pair
                if np.mean(landmark)!=-1:return deca.crop_frame(pair)
                h,w=image.shape[:2]
                # Original DECA FAN fallback: whole input rectangle, kpt68 scale.
                return deca.crop_frame((image,np.array([[0,0],[h-1,w-1]],np.float32)))
            with ThreadPoolExecutor(max_workers=4) as pool:cropped=list(pool.map(crop,zip(frames,lm)))
            import torch
            torch.backends.cudnn.allow_tf32=True
            c=deca.encode(model,cropped,64);motion=np.concatenate([c[:,150:200],c[:,200:206]],1)
            assert np.isfinite(motion).all();np.savez_compressed(dest/'deca.npz',motion=motion,detected=detected)
            save(dest/'complete.json',dict(role='evaluation_gt_only',id=s['id'],source=proof,frames=len(motion),path=str(dest/'deca.npz'),sha256=sha(dest/'deca.npz'),missing_face_frames=int((~detected).sum()),code_sha256=sha(__file__),seconds=time.time()-started))
            done+=1;print('GT DECA',s['key'],round(time.time()-started,2),flush=True)
            if a.limit and done>=a.limit:return
if __name__=='__main__':main()
