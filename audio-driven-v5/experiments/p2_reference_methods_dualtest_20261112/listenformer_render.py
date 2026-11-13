"""Existing PIRender forward at native256/30, explicit timestamp/512 scoring conversion."""
import argparse,time
import numpy as np
from runtime import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--wait',action='store_true');a=p.parse_args()
    model=None;done=0;code={str(Path(__file__).resolve()):sha(__file__)}
    while True:
        pending=[];processed=0
        for s in selected():
            if valid('listenformer',s):continue
            pending.append(s);d=folder('listenformer',s)
            if not (d/'MOTION_COMPLETE.json').exists():continue
            with (d/'render.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if valid('listenformer',s):continue
                if model is None:
                    device()
                    import torch
                    from PIL import Image
                    from vico_render_fast import model_for,encode
                    model=model_for(WORK/'model_load/listenformer')
                receipt=read(d/'MOTION_COMPLETE.json');assert sha(d/'motion.npz')==receipt['files'][str(d/'motion.npz')]
                image=np.asarray(Image.open(s['reference']).convert('RGB').resize((256,256),Image.Resampling.BICUBIC))
                tensor=torch.from_numpy(image.transpose(2,0,1).copy()).float().div(255).sub(.5).div(.5)[None].cuda()
                dyn=np.load(d/'motion.npz')['dynamic'];native=d/'native256.mp4'
                start=time.time();r=encode(model,tensor,dyn,native,fast=True)
                # Preserve native output; resize decoded native frames explicitly and disclose low native resolution.
                from vico_common_deca import load_video
                frames=load_video(native);idx=np.minimum(np.floor(np.arange(s['frames'])*30/s['fps']+.5).astype(int),len(frames)-1)
                def chunks():
                    for i in range(0,len(idx),16):
                        yield np.stack([np.asarray(Image.fromarray(frames[j]).resize((512,512),Image.Resampling.BICUBIC)) for j in idx[i:i+16]])
                score=d/'scoring.mp4';pixels_video(score,chunks(),s['fps'])
                publish('listenformer',s,[native,score,d/'motion.npz'],native_size=256,score_size=512,score_fps=s['fps'],score_frames=s['frames'],score_path=str(score),conversion='decoded native256 H264 -> PIL bicubic512; nearest30fps timestamps',motion_receipt_sha256=sha(d/'MOTION_COMPLETE.json'),code_sha256=code,seconds=time.time()-start)
                done+=1;processed+=1;print('render listenformer',s['key'],round(time.time()-start,2),flush=True)
                if a.limit and done>=a.limit:return
        if not pending or not a.wait:return
        if not processed:time.sleep(10)
if __name__=='__main__':main()
