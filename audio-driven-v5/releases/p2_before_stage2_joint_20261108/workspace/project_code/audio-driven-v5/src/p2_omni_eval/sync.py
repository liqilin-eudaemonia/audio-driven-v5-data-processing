"""Device-explicit adaptation of the reference SyncNet scoring implementation."""
import os
from pathlib import Path
import subprocess
import sys
from .common import module,read,atomic_json,subprocess_env,ROOT


def score_track(track,runtime,options,device,dest):
    import cv2
    import numpy as np
    import torch
    from scipy.io import wavfile
    import python_speech_features
    a=runtime['paths'];root=Path(a['sync_root'])
    sys.path.insert(0,str(root.parent))
    reference=module('p2_sync_reference',Path(a['reference_root'])/'audio/audio_sync/SyncNetInstance_calc_scores.py')
    # Original preprocessing decodes JPEGs from the tracker AVI before resizing.
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(track),'-vsync','0','-threads','1',str(dest/'%06d.jpg')],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(track),'-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_s16le',str(dest/'audio.wav')],check=True)
    images=[cv2.resize(cv2.imread(str(p)),(224,224)) for p in sorted(dest.glob('*.jpg'))]
    rate,audio=wavfile.read(dest/'audio.wav');aligned=min(len(images),len(audio)//640);windows=aligned-5
    support=dict(aligned_frames=aligned,windows=max(0,windows),vshift=options['sync_vshift'],low_support=windows<2*options['sync_vshift']+1)
    if windows<=0:return dict(status='not_applicable',reason='insufficient_sync_windows',**support)
    im=np.transpose(np.expand_dims(np.stack(images,axis=3),0),(0,3,4,1,2))
    im=torch.from_numpy(im.astype(float)).float()
    mfcc=python_speech_features.mfcc(audio,rate).T
    cc=torch.from_numpy(mfcc[None,None].astype(float)).float()
    model=reference.S(num_layers_in_fc_layers=1024).to(device).eval()
    state=torch.load(root/'data/syncnet_v2.model',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    image_features=[];audio_features=[]
    with torch.inference_mode():
        for i in range(0,windows,options['sync_batch_size']):
            indexes=range(i,min(windows,i+options['sync_batch_size']))
            image_features.append(model.forward_lip(torch.cat([im[:,:,j:j+5] for j in indexes],0).to(device)).cpu())
            audio_features.append(model.forward_aud(torch.cat([cc[:,:,:,j*4:j*4+20] for j in indexes],0).to(device)).cpu())
        distances=reference.calc_pdist(torch.cat(image_features),torch.cat(audio_features),vshift=options['sync_vshift'])
        means=torch.stack(distances,1).mean(1);minimum,index=means.min(0);confidence=means.median()-minimum
    values=dict(lse_d=float(minimum),lse_c=float(confidence),offset_frames=options['sync_vshift']-int(index))
    if not all(np.isfinite(x) for x in values.values()):raise ValueError('Nonfinite SyncNet result')
    return dict(status='success',**values,**support,confidence_interpretable=not support['low_support'])


def worker(request):
    import torch
    p=read(request);a=p['runtime']['paths'];root=Path(a['sync_root']);dest=Path(p['output']).parent
    os.environ.update(subprocess_env(p['runtime'],p['device']));torch.set_num_threads(2)
    # The tracker auto-selects CUDA. CPU explicitly hides every CUDA device.
    # CUDA mode restricts its visibility and uses logical cuda:0 inside this worker.
    if p['device'].startswith('cuda:'):
        index=int(p['device'][5:]);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES']=visible.split(',')[index] if visible else str(index)
        p['device']='cuda:0'
    command=[sys.executable,str(ROOT/'scripts/p2_omni_eval.py'),'track-worker','--sync-root',str(root),
        '--videofile',p['video'],'--reference','listener','--data_dir',str(dest/'tracks'),
        '--min_track',str(p['options']['sync_min_track']),'--require_single_face']
    subprocess.run(command,cwd=root,check=True,timeout=1200)
    from .media import geometry,aligned
    aligned(geometry(p['video']),geometry(dest/'tracks/pyavi/listener/video.avi'))
    summary=read(dest/'tracks/pywork/listener/face_track_summary.json')
    if summary['has_multiple_faces']:
        result=dict(status='not_applicable',reason='multiple_listener_faces',tracks=[],face_summary=summary)
    else:
        scores=[score_track(track,p['runtime'],p['options'],p['device'],dest/f'track_{i}')
            for i,track in enumerate(sorted((dest/'tracks/pycrop/listener').glob('0*.avi')))]
        valid=[r for r in scores if r['status']=='success']
        result=dict(status='success' if valid else 'not_applicable',tracks=scores,face_summary=summary,
            reason=None if valid else 'no_supported_listener_face_track')
        if valid:
            result.update(lse_d=sum(r['lse_d'] for r in valid)/len(valid),lse_c=sum(r['lse_c'] for r in valid)/len(valid),
                low_support=any(r['low_support'] for r in valid))
    atomic_json(p['output'],result)


def track_worker(root,args):
    import runpy
    from scipy import signal
    original=signal.medfilt
    def bounded(values,kernel_size=13):
        n=len(values);window=min(kernel_size,n if n%2 else n-1)
        return original(values,kernel_size=max(1,window))
    signal.medfilt=bounded
    # Upstream's implicit AVI audio codec is MP3 on this ffmpeg build: its encoder
    # delay caused 50 original frames to become 52. Preserve PCM and actual frames.
    original_run=subprocess.run
    def explicit_ffmpeg(command,*positional,**kwargs):
        if isinstance(command,list) and command and Path(command[0]).name=='ffmpeg':
            command=list(command)
            if str(command[-1]).endswith('/video.avi'):
                source=command[command.index('-i')+1]
                command=['ffmpeg','-y','-v','error','-threads','1','-i',source,'-map','0:v:0','-map','0:a:0',
                         '-c:v','mpeg4','-qscale:v','2','-c:a','pcm_s16le','-vsync','0','-threads','1',command[-1]]
            else:command=command[:-1]+['-threads','1']+command[-1:]
        return original_run(command,*positional,**kwargs)
    subprocess.run=explicit_ffmpeg
    sys.path.insert(0,str(root));sys.argv=[str(Path(root)/'run_pipeline.py'),*args]
    runpy.run_path(sys.argv[0],run_name='__main__')
