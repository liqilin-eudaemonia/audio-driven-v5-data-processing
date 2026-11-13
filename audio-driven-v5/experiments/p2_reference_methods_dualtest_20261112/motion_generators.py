"""Reuse the verified official LF/DualTalk forwards on the new feature cache."""
import argparse,time,os,math
import numpy as np
from runtime import *

def cache(s):return WORK/'features'/s['dataset']/s['id']
def ready(s):return (cache(s)/'complete.json').exists()
def initialize(method):
    import torch
    if method=='listenformer':
        import vico_more_train as native
        model,_=native.model_for(method)
        source=ROOT/'experiments/p2_vico_additional_baselines_20261107/listenformer'
        ck=torch.load(source/'best.pt',map_location='cpu',weights_only=False)
        assert sha(source/'best.pt')==read(source/'TRAIN_COMPLETE.json')['best_checkpoint_sha256']
        model.load_state_dict(ck['model'],strict=True);model.cuda().eval()
        stats=dict(np.load(ROOT/'experiments/p2_vico_additional_baselines_20261107/features/train_stats.npz'))
        def run(s,d):
            f=cache(s);speaker=np.load(f/'speaker_bfm.npz')['dynamic'];initial=np.load(f/'listener_initial_bfm.npz')['dynamic'];audio=np.load(f/'audio45.npy')
            def norm(x,k):return torch.from_numpy(((x-stats[k+'_mean'])/stats[k+'_std']).astype(np.float32)).cuda()
            with torch.inference_mode():pred=native.inference(model,method,norm(audio,'audio'),norm(speaker,'dynamic'),norm(initial,'dynamic')[0]).cpu().numpy()
            pred=pred*stats['dynamic_std']+stats['dynamic_mean'];assert pred.shape==(len(audio),73) and np.isfinite(pred).all()
            np.savez_compressed(d/'motion.npz',dynamic=pred,native_fps=30)
            return dict(frames=len(pred),native_fps=30,checkpoint_sha256=sha(source/'best.pt'),weights='local ViCo200epoch retraining')
    else:
        import vico_more_dualtalk as native
        native.OUT=WORK/'model_load/dualtalk';model=native.construct().cuda().eval();proc=native.processor()
        import soundfile as sf
        def run(s,d):
            f=cache(s);speaker=np.load(f/'speaker_deca.npz')['motion'];wave,rate=sf.read(s['audio'],dtype='float32');assert rate==16000
            n=math.ceil(len(wave)/16000*25);ix=np.minimum(np.floor(np.arange(n)*30/25+.5).astype(int),len(speaker)-1)
            blend=speaker[ix][:,np.r_[0:50,53:56,50:53]]
            with torch.inference_mode():pred=model(*native.inputs(proc,np.zeros_like(wave),wave,blend)).cpu().numpy()[0]
            assert pred.shape==(n,56) and np.isfinite(pred).all()
            motion=pred[:,np.r_[0:50,53:56,50:53]]
            np.savez_compressed(d/'motion.npz',native_exp_jaw_neck=pred,deca_proxy=motion,native_fps=25)
            return dict(frames=n,native_fps=25,checkpoint_sha256=sha(native.WEIGHT),weights='official',rgb_metrics='N/A: no validated RGB/global head mapping')
    return run

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=['listenformer','dualtalk']);p.add_argument('--limit',type=int);p.add_argument('--wait',action='store_true');a=p.parse_args()
    run=None;done=0;code={str(Path(__file__).resolve()):sha(__file__)}
    while True:
        pending=[];processed=0
        for s in selected():
            d=folder(a.method,s)
            if (d/'MOTION_COMPLETE.json').exists():continue
            pending.append(s)
            if not ready(s):continue
            d.mkdir(parents=True,exist_ok=True)
            with (d/'motion.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (d/'MOTION_COMPLETE.json').exists():continue
                if run is None:device();run=initialize(a.method)
                f=cache(s);r=read(f/'complete.json')
                for file,h in r['files'].items():assert sha(file)==h
                start=time.time()
                try:
                    meta=run(s,d);rec=dict(id=s['id'],key=s['key'],protocol_sha256=sha(WORK/'protocol.json'),source_feature_receipt_sha256=sha(f/'complete.json'),files={str(d/'motion.npz'):sha(d/'motion.npz')},source_only=True,listener_future_input=False,seconds=time.time()-start,code_sha256=code,**meta)
                    save(d/'MOTION_COMPLETE.json',rec)
                    if a.method=='dualtalk':publish(a.method,s,[d/'motion.npz'],motion_only=True,code_sha256=code,**meta)
                    done+=1;processed+=1;print(a.method,s['key'],meta['frames'],flush=True)
                    if a.limit and done>=a.limit:return
                except BaseException:
                    import traceback
                    save(d/'MOTION_FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
        if not pending or not a.wait:return
        if not processed:time.sleep(10)
if __name__=='__main__':main()
