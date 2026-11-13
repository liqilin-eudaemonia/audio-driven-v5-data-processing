"""Official pretrained AF/DyStream forwards; frozen shared input and timestamp adapters."""
import argparse,sys,os,time,math
from pathlib import Path
from runtime import ROOT,WORK,read,save,sha,device,locked_jobs,pixels_video,publish,resident_checkpoint

def avatarforcing():
    import vico_avatarforcing as native
    native.WORK=WORK/'model_load'
    os.chdir(native.REPO)
    model=native.build_model()
    from transformers import Wav2Vec2FeatureExtractor
    proc=Wav2Vec2FeatureExtractor.from_pretrained(str(native.REPO/'pretrained_dir/wav2vec2-base-960h'),local_files_only=True)
    import torch,numpy as np,cv2,soundfile as sf
    def run(s,dest):
        wave,rate=sf.read(s['audio'],dtype='float32');assert rate==16000
        user=proc(wave,sampling_rate=16000,return_tensors='pt').input_values
        silent=proc(np.zeros_like(wave),sampling_rate=16000,return_tensors='pt').input_values
        frame=cv2.cvtColor(cv2.imread(s['reference']),cv2.COLOR_BGR2RGB)
        frame=cv2.resize(frame,(512,512),interpolation=cv2.INTER_AREA)
        ref=torch.from_numpy(((frame.astype(np.float32)-127.5)/127.5).transpose(2,0,1).copy()).unsqueeze(0)
        n=math.ceil(len(wave)*25/16000);data=dict(avatar_ref=ref,avatar_a=silent,user_a=user,user_frame=[None]*n)
        torch.manual_seed(137);np.random.seed(137)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.inference(data,a_cfg_scale=2.,u_cfg_scale=1.,nfe=10,seed=137,use_kv_cache=True)['d_hat']
        assert out.shape==(n,3,512,512) and torch.isfinite(out).all()
        out=out.float().clamp(-1,1).cpu();lo,hi=out.min(),out.max();assert hi>lo
        rgb=((out-lo)/(hi-lo)*255).byte().permute(0,2,3,1).numpy()
        return rgb,dict(native_fps=25,user_visual='zero latent',self_audio='zero waveform',nfe=10,a_cfg=2,u_cfg=1)
    return run

def dystream():
    import torch,numpy as np,soundfile as sf
    from PIL import Image
    import responsenet_dystream_node11 as prior
    prior.EXT=WORK/'model_load/dystream';prior.EXT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(prior.REPO));os.chdir(prior.REPO)
    import app
    prior.install_first_frame_fallback(app,WORK)
    # The original app's audio-muxed duplicate is unused: retain the original raw frames
    # and export native/scoring videos below, as in the already validated fast adapter.
    app.save_video_with_audio=lambda frames,audio,path,fps=25:path
    loads=[];original=torch.nn.Module.load_state_dict
    def audited(module,state,*a,**kw):
        r=original(module,state,*a,**kw)
        if module is app._dystream_model:loads.append(dict(missing=r.missing_keys,unexpected=r.unexpected_keys))
        return r
    torch.nn.Module.load_state_dict=audited
    try:app.load_dystream_model();app.load_visualization_model()
    finally:torch.nn.Module.load_state_dict=original
    assert len(loads)==1 and not loads[0]['missing'] and not loads[0]['unexpected']
    app._dystream_model.cfg.seed=137
    save(prior.EXT/'WEIGHTS_LOADED.json',dict(passed=True,strict_equivalent=loads,checkpoint_sha256=sha(prior.REPO/'checkpoints/last.ckpt')))
    captured={};original_render=app.latents_to_video_frames
    def render(latents,image):
        frames=original_render(latents,image);captured['frames']=frames;return frames
    app.latents_to_video_frames=render
    def run(s,dest):
        torch.manual_seed(137);np.random.seed(137);prior.CURRENT['id']=s['id']
        wave,rate=sf.read(s['audio'],dtype='float32');assert rate==16000
        silent=dest/'self_zero.wav';sf.write(silent,np.zeros_like(wave),16000,subtype='FLOAT')
        ref,mask,latent=app.process_image(Image.open(s['reference']),crop=False)
        ref.save(dest/'processed_reference.png');np.savez_compressed(dest/'reference_motion.npz',motion_latent=latent.numpy())
        path,_,_,info=app.run_inference(None,str(silent),s['audio'],progress=lambda *a,**k:None,
            precomputed_npz_path=str(dest/'reference_motion.npz'),precomputed_ref_img_path=str(dest/'processed_reference.png'),
            video_audio_path=s['audio'],denoising_steps=5,cfg_audio=.5,cfg_audio_other=.5,cfg_anchor=0,cfg_all=1)
        import shutil
        shutil.rmtree(Path(path).parent)
        frames=captured.pop('frames');assert frames.shape[1:]==(512,512,3) and np.isfinite(frames).all()
        assert len(frames)==len(wave)//640-1
        return frames,dict(native_fps=25,self_audio='zero waveform',denoising_steps=5,official_info=info)
    return run

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['avatarforcing','dystream']);p.add_argument('--scope',choices=['responsenet586'],default='responsenet586');p.add_argument('--dataset',choices=['responsenet','seamless']);p.add_argument('--limit',type=int);a=p.parse_args()
    import numpy as np
    code={str(Path(__file__).resolve()):sha(__file__),str(WORK/'runtime.py'):sha(WORK/'runtime.py')}
    jobs=iter(locked_jobs(a.method,a.scope,a.limit,a.dataset));first=next(jobs,None)
    if first is None:print('No pending '+a.method);return
    save(WORK/'processes'/f'{a.method}_{os.getpid()}.json',dict(pid=os.getpid(),device=device(),code=code))
    runner=globals()[a.method]();count=0
    import itertools
    for s,dest in itertools.chain([first],jobs):
        resident_checkpoint()
        assert sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
        started=time.time()
        try:
            frames,metadata=runner(s,dest)
            native=dest/'native.mp4';score=dest/'scoring.mp4'
            pixels_video(native,[frames],25)
            idx=np.minimum(np.floor(np.arange(s['frames'])*25/s['fps']+.5).astype(int),len(frames)-1)
            pixels_video(score,(np.ascontiguousarray(frames[idx[i:i+32]],dtype=np.uint8) for i in range(0,len(idx),32)),s['fps'])
            r=publish(a.method,s,[native,score],native_frames=len(frames),score_frames=len(idx),score_fps=s['fps'],score_path=str(score),mapping=idx.tolist(),code_sha256=code,seconds=time.time()-started,**metadata)
            count+=1;print(s['key'],a.method,'done',count,round(r['seconds'],2),flush=True)
        except BaseException:
            import traceback
            save(dest/'FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
if __name__=='__main__':main()
