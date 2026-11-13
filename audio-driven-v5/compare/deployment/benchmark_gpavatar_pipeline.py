"""Compare real GPAvatar forward + identical three-video export, no output reuse."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import sys,time,statistics
from pathlib import Path
import torch,numpy as np
from vico_more_common import ROOT,read,save,sha
from vico_gpavatar_render_fast import pipeline,small_frame,Canvas
sys.path.insert(0,str(ROOT/'scripts'))
import p2_l2l_gpavatar as gp

OUT=ROOT/'experiments/p2_vico_infra_acceptance_20261110/gp_benchmark'

def baseline(model,avatar,motion,indices,dest,scoring):
    start=time.perf_counter();pixels=[];raw=[]
    with torch.inference_mode():
        for index in indices:
            rgb=gp.render_motion(model,avatar,motion[index:index+1]).clamp(0,1)
            assert rgb.shape==(1,3,512,512) and torch.isfinite(rgb).all()
            raw.append((rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
            rgb=gp.restore_canvas(rgb)
            pixels.append((rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
    gp.output_video(dest/'generated.mp4',pixels)
    gp.output_video(dest/'raw_padded.mp4',raw)
    gp.output_video(scoring,[small_frame(x) for x in pixels],256)
    return dict(seconds=time.perf_counter()-start,frames=len(indices),files={str(f):sha(f) for f in [dest/'generated.mp4',dest/'raw_padded.mp4',scoring]})

def main():
    device=gp.guard();gp.save=lambda p,x:save(OUT/Path(p).name,x)
    model=gp.renderer();samples=read(gp.WORK/'protocol.json')['samples'];samples=[min(samples,key=lambda s:s['frames']),max(samples,key=lambda s:s['frames'])]
    parities=[];runs=[]
    for sample in samples:
        folder=gp.WORK/'references'/sample['id']
        avatar=torch.load(folder/'avatar.pt',map_location='cpu',weights_only=True)
        tracked=torch.load(folder/'tracked.pt',map_location='cpu',weights_only=False)
        avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float()
        with torch.inference_mode():
            torch.manual_seed(137);model.build_avatar(avatar)
            for name in gp.NAMES:
                motion=torch.from_numpy(np.load(sample['predictions'][name]['path'])['motion'].astype(np.float32)).cuda()
                canvas=Canvas();errors=[]
                for index in sorted(set([0,len(motion)//2,len(motion)-1])):
                    rgb=gp.render_motion(model,avatar,motion[index:index+1]).clamp(0,1)
                    errors.append(float((canvas(rgb)-gp.restore_canvas(rgb)).abs().max()))
                assert max(errors)==0
                parities.append(dict(id=sample['id'],model=name,max_abs=max(errors)))
        # End-to-end ABBAAB on real 60-frame motion, including encoder startup/drain.
        if sample==samples[-1]:
            motion=torch.from_numpy(np.load(sample['predictions']['conan']['path'])['motion'].astype(np.float32)).cuda()
            indices=np.linspace(0,len(motion)-1,60,dtype=int).tolist();reference=None
            for run,fast in enumerate([False,True,True,False,False,True]):
                dest=OUT/str(run);scoring=dest/'scoring.mp4'
                r=(pipeline if fast else baseline)(*([gp] if fast else []),model,avatar,motion,indices,dest,scoring)
                hashes={Path(f).name:h for f,h in r['files'].items()}
                if reference is None:reference=hashes
                assert hashes==reference,(run,hashes,reference)
                r.update(fast=fast,byte_exact=True);runs.append(r);print(dict(run=run,fast=fast,seconds=r['seconds']),flush=True)
    old=statistics.median(r['seconds'] for r in runs if not r['fast']);new=statistics.median(r['seconds'] for r in runs if r['fast'])
    save(OUT/'BENCHMARK.json',dict(passed=True,device=device,scope='Real FP32 forward + raw512/restored512/scoring256 MP4 export; 60 frames, encoder startup and drain included',
        excluded='Model and avatar initialization, common motion load',runs=runs,canvas_parity=parities,baseline_seconds=old,fast_seconds=new,reduction_pct=100*(1-new/old)))

if __name__=='__main__':main()
