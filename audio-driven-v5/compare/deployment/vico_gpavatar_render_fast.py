"""Bounded GPAvatar render/resize/encode pipeline, preserving official FP32 pixels.

Run on node11 GPU7 in the existing vico_more environment. Archived inputs/results
are read-only. Completed videos are checked before importing torch or GPU admission.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from vico_more_common import ROOT,read,save,sha

SOURCE=ROOT/'experiments/p2_l2l_public_renderer_20261110'
DEFAULT=ROOT/'experiments/p2_vico_infra_acceptance_20261110/gpavatar'


class VideoPipe:
    def __init__(self,path,size):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temp=self.path.with_name(self.path.stem+'.partial.mp4')
        self.process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{size}x{size}','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','1',
            '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(self.temp)],stdin=subprocess.PIPE)

    def write(self,pixels):self.process.stdin.write(pixels.tobytes())

    def finish(self):
        self.process.stdin.close()
        if self.process.wait()!=0:raise RuntimeError(f'ffmpeg failed: {self.path}')

    def abort(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait()
        if not self.process.stdin.closed:self.process.stdin.close()
        self.temp.unlink(missing_ok=True)


class Canvas:
    """Cache the official constant affine grid, at most one shape per renderer."""
    def __init__(self):self.key=None;self.grid=None
    def __call__(self,rgb):
        import torch
        key=(tuple(rgb.shape),rgb.dtype,rgb.device)
        if key!=self.key:
            matrix=rgb.new_tensor([[[.8,0.,0.],[0.,.8,0.]]]).expand(len(rgb),-1,-1)
            self.grid=torch.nn.functional.affine_grid(matrix,rgb.shape,align_corners=False);self.key=key
        return torch.nn.functional.grid_sample(rgb,self.grid,mode='bicubic',padding_mode='border',align_corners=False).clamp(0,1)


def small_frame(pixels):
    import torch
    import numpy as np
    a=torch.from_numpy(pixels.copy()).permute(2,0,1)[None].float()/255.
    a=torch.nn.functional.interpolate(a,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    return (a[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)


def pipeline(gp,model,avatar,motion,indices,dest,scoring):
    """Actual forward on every frame; max two pending RGB pairs, no clip cache."""
    import torch
    import numpy as np
    pipes=[];pending=deque();canvas=Canvas();start=time.perf_counter();count=0
    def export(raw,pixels):
        pipes[0].write(pixels);pipes[1].write(raw);pipes[2].write(small_frame(pixels))
    try:
        for path,size in [(dest/'generated.mp4',512),(dest/'raw_padded.mp4',512),(scoring,256)]:
            pipes.append(VideoPipe(path,size))
        with ThreadPoolExecutor(max_workers=1) as executor,torch.inference_mode():
            for index in indices:
                if len(pending)>=2:pending.popleft().result()
                rgb=gp.render_motion(model,avatar,motion[index:index+1]).clamp(0,1)
                assert rgb.shape==(1,3,512,512) and torch.isfinite(rgb).all()
                raw=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                rgb=canvas(rgb)
                pixels=(rgb[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                pending.append(executor.submit(export,raw,pixels));count+=1
            for future in pending:future.result()
        for pipe in pipes:pipe.finish()
        # Publish only after every encoder has succeeded; the receipt is written last.
        for pipe in pipes:pipe.temp.replace(pipe.path)
    except BaseException:
        for pipe in pipes:pipe.abort()
        raise
    return dict(frames=count,seconds=time.perf_counter()-start,max_pending_frame_pairs=2,
        files={str(pipe.path):sha(pipe.path) for pipe in pipes})


def reusable(receipt,protocol_sha,motion_sha,frames):
    if not receipt.is_file():return None
    try:
        r=read(receipt)
        if r['protocol_sha256']!=protocol_sha or r['motion_sha256']!=motion_sha or r['frames']!=frames:return None
        if len(r['files'])!=3:return None
        if any(not Path(p).is_file() or sha(p)!=h for p,h in r['files'].items()):return None
        return r
    except (KeyError,ValueError,OSError):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=DEFAULT)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--no-archive-reuse',action='store_true',help='Recompute in the new output root; still resume its valid clips')
    parser.add_argument('--smoke',action='store_true',help='First/middle/last frames of shortest/longest clips, all four models')
    args=parser.parse_args();out=args.output_root.resolve()
    if out==SOURCE or SOURCE in out.parents:parser.error('Use a new output root outside the sealed archive')
    p=read(SOURCE/'protocol.json');digest=sha(SOURCE/'protocol.json')
    # Validate official source/model provenance before accepting either old or new outputs.
    sys.path.insert(0,str(ROOT/'scripts'));import p2_l2l_gpavatar as gp
    gp.verify()
    samples=p['samples']
    if args.smoke:samples=[min(samples,key=lambda s:s['frames']),max(samples,key=lambda s:s['frames'])]
    records=[];pending=[];mode='smoke' if args.smoke else 'generated'
    for sample in samples:
        folder=SOURCE/'references'/sample['id']
        for path,h in read(folder/'complete.json')['files'].items():
            assert sha(path)==h,path
        for name in p['methods']:
            pred=sample['predictions'][name];assert sha(pred['path'])==pred['sha256']
            count=len(set([0,sample['frames']//2,sample['frames']-1])) if args.smoke else sample['frames']
            r=None
            for base in ((out,) if args.no_archive_reuse else (out,SOURCE)):
                r=reusable(base/mode/name/sample['id']/'complete.json',digest,pred['sha256'],count)
                if r:break
            if r:records.append(r)
            else:pending.append((sample,name))
    if args.check_only or not pending:
        print(__import__('json').dumps(dict(state='pending' if pending else 'reused_verified',
            complete=len(records),pending=len(pending),gpu_requested=False)));return
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
        os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
        os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH']
        os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(SOURCE/'torch_extensions'))
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0');os.environ.setdefault('MAX_JOBS','2')
        import torch,numpy as np
        save(out/'DEVICE.json',gp.guard())
        gp.save=lambda path,value:save(out/Path(path).name,value)
        model=gp.renderer();current=None;total=len(records)+len(pending)
        for sample,name in pending:
            uid=sample['id'];folder=SOURCE/'references'/uid
            if uid!=current:
                avatar=torch.load(folder/'avatar.pt',map_location='cpu',weights_only=True)
                tracked=torch.load(folder/'tracked.pt',map_location='cpu',weights_only=False)
                avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float()
                torch.manual_seed(137)
                with torch.inference_mode():model.build_avatar(avatar)
                current=uid
            pred=sample['predictions'][name];motion=np.load(pred['path'])['motion'].astype(np.float32)
            assert motion.shape==(sample['frames'],56) and np.isfinite(motion).all()
            tensor=torch.from_numpy(motion).cuda();dest=out/mode/name/uid
            indices=sorted(set([0,len(motion)//2,len(motion)-1])) if args.smoke else range(len(motion))
            with torch.inference_mode():proof=gp.camera_parity(model,avatar,tensor)
            scoring=out/('smoke_scoring' if args.smoke else 'scoring')/('l2l_gpavatar_'+name)/(uid+'.listener.mp4')
            r=pipeline(gp,model,avatar,tensor,indices,dest,scoring)
            r.update(id=uid,model=name,protocol_sha256=digest,motion_sha256=pred['sha256'],camera_parity=proof,
                source_frames=1,future_listener_input=False,fps=30,native_size=512,score_size=256,score_path=str(scoring),
                selected_indices=list(indices) if args.smoke else None,runtime_sha256=sha(__file__))
            save(dest/'complete.json',r);records.append(r)
            save(out/'status.json',dict(complete=len(records),total=total))
            print(uid,name,r['seconds'],flush=True)
        save(out/('SMOKE_COMPLETE.json' if args.smoke else 'RENDER_COMPLETE.json'),dict(passed=True,records=records,protocol_sha256=digest))


if __name__=='__main__':main()
