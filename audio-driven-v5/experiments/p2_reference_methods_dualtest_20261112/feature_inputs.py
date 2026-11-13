"""Actual speaker RGB -> original BFM and DECA; listener t0 only for model inputs."""
import argparse,contextlib,math,os,time,fcntl
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime import *

def resized(frame):
    from PIL import Image
    image=frame.to_image().convert('RGB');scale=512/max(image.size)
    return np.asarray(image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BICUBIC))

def speaker_frames(s):
    import importlib.util
    if importlib.util.find_spec('av') is None:
        sys.path.append('/private/liqi_workspace/.cache/paired_video_metrics/python')
    import av
    r=s['input_reference'];n=math.ceil(s['duration']*30-1e-7)
    if s['dataset']=='responsenet':
        source=r['speaker']['video'];opened=contextlib.nullcontext(source);start=0.;proof=dict(path=source,sha256=sha(source))
    else:
        from evaluation.seamless.reader import load_record
        from listen_visual_quality.measurement import SourceBytes
        row,_=load_record(r['gt_reference']);a=row['inputs']['speaker_audio'];pair=read(a['source_result_file'])
        assert sha(a['source_result_file'])==a['source_result_sha256'];clip=pair['clips'][a['clip_index']]
        assert clip['sample_id']==s['id'] and clip['speaker_id']==a['participant_id']
        source=clip['sources']['speaker'];assert Path(source['member']).stem==a['participant_id']
        opened=SourceBytes(source);start=a['start_s'];proof=dict(source=source,start=start,end=a['end_s'],speaker_id=a['participant_id'],pair_sha256=a['source_result_sha256'])
    wanted=start+np.arange(n)/30.;frames=[];pts=[];prev=None;pt=None;last=None;array=None
    with opened as path,av.open(path) as c:
        stream=c.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        fps=float(stream.average_rate);c.seek(int(max(0,start-1)/float(stream.time_base)),stream=stream,backward=True)
        for f in c.decode(video=0):
            assert f.pts is not None;t=float(f.pts*stream.time_base)
            while len(frames)<n and t+1e-8>=wanted[len(frames)]:
                target=wanted[len(frames)];pick=prev if prev is not None and abs(pt-target)<=abs(t-target) else f
                stamp=float(pick.pts*stream.time_base)
                assert abs(stamp-target)<=1/fps+1e-5,(s['key'],stamp,target)
                if last!=pick.pts:array=resized(pick);last=pick.pts
                frames.append(array);pts.append(stamp)
            if len(frames)==n:break
            prev,pt=f,t
        # Last requested timestamp may lie within the final frame's duration.
        while len(frames)<n and prev is not None and wanted[len(frames)]<=pt+1/fps+1e-5:
            frames.append(resized(prev));pts.append(pt)
    assert len(frames)==n,(s['key'],len(frames),n)
    return np.stack(frames),dict(**proof,frames=n,fps=30,source_fps=fps,pts=pts,spatial='aspect-preserving PIL bicubic max-side512 before official face alignment')

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--dataset');a=p.parse_args()
    device()
    import torch
    from PIL import Image
    from p2_vico_motion_extract import Extractor
    from vico_more_features import dynamic,raw_audio
    import vico_common_deca as deca
    ext=Extractor();model=deca.extractor()
    import vico_l2l as l2l
    l2l.WORK=WORK/'features'
    code={str(Path(__file__).resolve()):sha(__file__)};done=0
    for s in selected():
        if a.dataset and s['dataset']!=a.dataset:continue
        dest=WORK/'features'/s['dataset']/s['id'];dest.mkdir(parents=True,exist_ok=True)
        with (dest/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            receipt=dest/'complete.json'
            if receipt.exists():
                assert all(sha(p)==h for p,h in read(receipt)['files'].items());continue
            start=time.time()
            try:
                frames,proof=speaker_frames(s)
                # Upstream BFM uses standard landmarks for an initial undetected frame.
                faces=ext.detect(frames)
                if any(len(f) for f in faces):lm,detected,counts=ext.landmarks(frames,faces)
                else:
                    lm=np.full((len(frames),68,2),-1.,np.float32);detected=np.zeros(len(frames),bool);counts=np.zeros(len(frames),int)
                aligned,transform=ext.align(frames,lm);coeff=ext.reconstruct(aligned)
                speaker=dict(coeff=coeff,transform_params=transform)
                np.savez_compressed(dest/'speaker_bfm.npz',**speaker,dynamic=dynamic(speaker),landmarks=lm,detected=detected)
                # Use unchanged landmark crop; missing leading detections use the official full-image fallback.
                def crop(pair):
                    image,landmarks=pair
                    if np.mean(landmarks)!=-1:return deca.crop_frame(pair)
                    from skimage.transform import estimate_transform,warp
                    h,w=image.shape[:2];old=(h+w-2)/2.*1.1;center=np.array([(h-1)/2.,(w-1)/2.]);size=int(old*1.25)
                    source=np.array([[center[0]-size/2,center[1]-size/2],[center[0]-size/2,center[1]+size/2],[center[0]+size/2,center[1]-size/2]])
                    t=estimate_transform('similarity',source,np.array([[0,0],[0,223],[223,0]]))
                    return warp(image/255.,t.inverse,output_shape=(224,224)).transpose(2,0,1).astype(np.float32),t.params.astype(np.float32)
                with ThreadPoolExecutor(max_workers=4) as pool:cropped=list(pool.map(crop,zip(frames,lm)))
                torch.backends.cudnn.allow_tf32=True
                dc=deca.encode(model,cropped,batch_size=64);motion=np.concatenate([dc[:,150:200],dc[:,200:206]],1)
                assert motion.shape==(len(frames),56) and np.isfinite(motion).all()
                np.savez_compressed(dest/'speaker_deca.npz',motion=motion,detected=detected)
                image=np.asarray(Image.open(s['reference']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC))[None]
                torch.backends.cudnn.allow_tf32=False
                ff=ext.detect(image)
                if any(len(f) for f in ff):ll,dd,cc=ext.landmarks(image,ff)
                else:ll=np.full((1,68,2),-1.,np.float32);dd=np.zeros(1,bool)
                aa,tt=ext.align(image,ll);initial=dict(coeff=ext.reconstruct(aa),transform_params=tt)
                np.savez_compressed(dest/'listener_initial_bfm.npz',**initial,dynamic=dynamic(initial),detected=dd)
                import soundfile as sf
                from scipy.signal import resample_poly
                wave,rate=sf.read(s['audio'],dtype='float32');assert rate==16000
                audio441=dest/'speaker_44100.wav';sf.write(audio441,resample_poly(wave,441,160).astype(np.float32),44100,subtype='FLOAT')
                np.save(dest/'audio45.npy',raw_audio(audio441,len(frames)))
                mel=l2l.audio_features(dict(id=s['dataset']+'_'+s['id'],speaker=s['audio'],frames=len(frames)))
                np.save(dest/'mel128.npy',mel)
                files=[dest/x for x in ['speaker_bfm.npz','speaker_deca.npz','listener_initial_bfm.npz','audio45.npy','mel128.npy','speaker_44100.wav']]
                save(receipt,dict(id=s['id'],key=s['key'],source_only=True,listener_frames_used=1,listener_future_input=False,frames=len(frames),speaker=proof,input_image_sha256=s['reference_sha256'],input_audio_sha256=s['audio_sha256'],protocol_sha256=sha(WORK/'protocol.json'),code_sha256=code,files={str(p):sha(p) for p in files},missing_speaker_faces=int((~detected).sum()),seconds=time.time()-start))
                done+=1;print('features',s['key'],len(frames),round(time.time()-start,2),flush=True)
                if a.limit and done>=a.limit:return
            except BaseException:
                import traceback
                save(dest/'FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
if __name__=='__main__':main()
