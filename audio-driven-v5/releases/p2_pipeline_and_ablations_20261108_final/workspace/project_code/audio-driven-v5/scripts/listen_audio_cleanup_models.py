"""Isolated six-case enhancement experiment; no source/cache mutation."""
import argparse, json, os, sys, types, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy import signal


def write_info(p, info): p.write_text(json.dumps(info,ensure_ascii=False,indent=2))


def deepfilter(run):
    # DeepFilterNet 0.5.6 imports a removed torchaudio annotation. I/O below uses
    # soundfile, so provide only that legacy metadata type (no audio monkeypatch).
    import torchaudio
    if 'torchaudio.backend.common' not in sys.modules:
        import importlib.util
        if importlib.util.find_spec('torchaudio.backend') is None:
            from collections import namedtuple
            backend=types.ModuleType('torchaudio.backend');common=types.ModuleType('torchaudio.backend.common')
            common.AudioMetaData=namedtuple('AudioMetaData','sample_rate num_frames num_channels bits_per_sample encoding')
            backend.common=common;sys.modules['torchaudio.backend']=backend;sys.modules['torchaudio.backend.common']=common
    os.environ['XDG_CACHE_HOME']=str(run/'model_cache')
    from df.enhance import init_df,enhance
    model,state,_=init_df(default_model='DeepFilterNet3',post_filter=False,log_file=None)
    for c in json.loads((run/'cases.json').read_text()):
        p=run/'cases'/c['id'];y,sr=sf.read(p/'context_listener.wav',dtype='float32');assert sr==48000
        for key,limit in [('df_gentle',12),('df_full',None)]:
            started=time.time()
            with torch.inference_mode(): z=enhance(model,state,torch.from_numpy(y[None,:]),pad=True,atten_lim_db=limit).cpu().numpy()[0]
            assert len(z)==len(y) and np.isfinite(z).all()
            off=c['context_offset_samples'];z=z[off:off+c['samples']]
            sf.write(p/f'{key}.wav',z,sr,subtype='FLOAT')
            write_info(p/f'{key}_info.json',dict(model='DeepFilterNet3',sampling_rate=sr,atten_lim_db=limit,post_filter=False,context=True,delay_compensation=True,elapsed=time.time()-started))
            print(c['id'],key,len(z),flush=True)


def av(run):
    import cv2
    from clearvoice.network_wrapper import network_wrapper
    from clearvoice.networks import CLS_AV_MossFormer2_TSE_16K
    from clearvoice.models.av_mossformer2_tse.faceDetector.s3fd import S3FD
    from clearvoice.utils.decode import decode_one_audio_AV_MossFormer2_TSE_16K
    os.chdir(run)
    wrapper=network_wrapper();wrapper.model_name='AV_MossFormer2_TSE_16K';wrapper.load_args_tse()
    args=wrapper.args;net=CLS_AV_MossFormer2_TSE_16K(args);args.device=net.device
    checkpoint_path=Path(args.checkpoint_dir)/(Path(args.checkpoint_dir)/'last_best_checkpoint').read_text().strip()
    cp=torch.load(checkpoint_path,map_location='cpu');state=cp.get('model',cp)
    missing=[]
    for k,v in net.model.state_dict().items():
        candidates=[k,k.replace('module.',''),'module.'+k]
        found=next((j for j in candidates if j in state and state[j].shape==v.shape),None)
        if found is None or not torch.equal(v.cpu(),state[found]): missing.append(k)
    write_info(run/'av_checkpoint_verification.json',dict(total_tensors=len(net.model.state_dict()),unmatched=missing))
    if missing: raise ValueError('Unmatched pretrained AV weights')
    detector=S3FD(device=net.device)
    # Official package's sliding-window values are read as floats; make exact integers.
    args.decode_window=int(args.decode_window)
    for c in json.loads((run/'cases.json').read_text()):
        p=run/'cases'/c['id'];started=time.time()
        if (p/'av.wav').exists(): continue
        try:
            y,sr=sf.read(p/'original.wav',dtype='float32');a=signal.resample_poly(y,1,3).astype('float32')
            cap=cv2.VideoCapture(str(p/'video.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
            while True:
                ret,f=cap.read()
                if not ret: break
                frames.append(f)
            cap.release();assert frames
            # Sample original video at official 25fps without changing audio timeline.
            count=int(np.ceil(len(a)/16000*25));frames=[frames[min(len(frames)-1,round(i*fps/25))] for i in range(count)]
            boxes=[];minor_detections_ignored=0
            with torch.inference_mode():
                for f in frames:
                    b=detector.detect_faces(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),conf_th=.9,scales=[.5])
                    if len(b)>1:
                        areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);order=np.argsort(-areas)
                        # Training crop targets the large foreground listener. Only reject
                        # tiny detections when the primary box has >=4x its area.
                        if areas[order[0]]<4*areas[order[1]]: raise ValueError('multiple comparable faces: target identity ambiguous')
                        minor_detections_ignored+=len(b)-1;b=b[order[:1]]
                    boxes.append(b[0,:4] if len(b)==1 else np.full(4,np.nan))
            boxes=np.asarray(boxes);valid=np.isfinite(boxes).all(1)
            if valid.mean()<.9: raise ValueError(f'insufficient face coverage {valid.mean()}')
            # Interpolate short detection failures; preserve all clip timestamps.
            for k in range(4): boxes[:,k]=np.interp(np.arange(count),np.flatnonzero(valid),boxes[valid,k])
            size=signal.medfilt(np.maximum(boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1])/2,13)
            xx=signal.medfilt((boxes[:,0]+boxes[:,2])/2,13);yy=signal.medfilt((boxes[:,1]+boxes[:,3])/2,13)
            visual=[]
            for i,f in enumerate(frames):
                bs=size[i];cs=.4;bsi=int(bs*(1+2*cs));frame=np.pad(f,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110)
                mx=xx[i]+bsi;my=yy[i]+bsi
                face=frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
                face=cv2.resize(face,(224,224));gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)[56:168,56:168]
                visual.append(gray)
                if i==count//2: cv2.imwrite(str(p/'av_face_crop.png'),face);cv2.imwrite(str(p/'av_model_crop.png'),gray)
            visual=(np.asarray(visual,dtype=np.float32)/255-.4161)/.1688
            # Official evaluation normalizes input peak. Undo that exact gain in output.
            peak=float(np.max(abs(a)));audio=a/max(peak,1e-12)
            with torch.inference_mode(): out=decode_one_audio_AV_MossFormer2_TSE_16K(net.model,(audio[None,:],visual[None,:]),args)
            out=np.asarray(out).ravel()*peak
            if abs(len(out)-len(a))>16: raise ValueError('unexpected model length')
            out=np.pad(out,(0,max(0,len(a)-len(out))))[:len(a)]
            z=signal.resample_poly(out,3,1)[:len(y)].astype('float32');assert len(z)==len(y) and np.isfinite(z).all()
            sf.write(p/'av.wav',z,sr,subtype='FLOAT')
            write_info(p/'av_info.json',dict(model='AV_MossFormer2_TSE_16K',native_sampling_rate=16000,output_sampling_rate=sr,video_fps=25,face_detection_fraction=float(valid.mean()),minor_detections_ignored=minor_detections_ignored,target_face_rule='single detection or largest box >=4x secondary area within listener training crop',audio_peak_gain_restored=peak,preprocessing='official face-box cropScale=.4; 224 gray center112; mean .4161 std .1688; 13-frame median; short missing boxes interpolated',decode_window=args.decode_window,elapsed=time.time()-started))
            (p/'av_error.json').unlink(missing_ok=True)
            print(c['id'],'av done',flush=True)
        except Exception as e:
            import traceback;traceback.print_exc();write_info(p/'av_error.json',dict(error=str(e)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('method',choices=['df','av']);ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4)
    (deepfilter if a.method=='df' else av)(a.run.resolve())
