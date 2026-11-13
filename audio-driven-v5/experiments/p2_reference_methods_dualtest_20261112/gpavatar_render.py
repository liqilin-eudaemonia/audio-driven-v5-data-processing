"""Official GPAvatar rendering of four L2L outputs on original GT timestamps."""
import argparse,os,time
import numpy as np
from runtime import *

def pipelined_video(path,frames,fps):
    """Same FFmpeg arguments and ordered pixels; overlap at most two frame writes."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    path=Path(path);tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.mp4')
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','512x512','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','1',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
    pending=deque()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for pixels in frames:
                if len(pending)>=2:pending.popleft().result()
                pending.append(pool.submit(lambda a:proc.stdin.write(a.tobytes()),pixels))
            for future in pending:future.result()
        proc.stdin.close()
        if proc.wait():raise RuntimeError('FFmpeg rendering export failed')
        tmp.replace(path)
    except BaseException:
        if proc.poll() is None:proc.kill()
        proc.wait()
        if not proc.stdin.closed:proc.stdin.close()
        tmp.unlink(missing_ok=True)
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--person',choices=['conan','fallon','stephen','trevor']);a=p.parse_args()
    model=None;current=None;done=0
    while True:
        pending=False;processed=0
        for s in selected():
            ref=WORK/'gpavatar_references'/s['dataset']/s['id']
            for person in ([a.person] if a.person else ['conan','fallon','stephen','trevor']):
                method='l2l_'+person;d=folder(method,s)
                if valid(method,s):continue
                pending=True
                if not (d/'MOTION_COMPLETE.json').exists() or not (ref/'complete.json').exists():continue
                with (d/'render.lock').open('a') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    if valid(method,s):continue
                    if model is None:
                        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python';os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
                        os.environ['PATH']='/usr/local/cuda/bin:'+os.environ['PATH'];os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(ROOT/'experiments/p2_l2l_public_renderer_20261110/torch_extensions'));os.environ.setdefault('TORCH_CUDA_ARCH_LIST','9.0');os.environ.setdefault('MAX_JOBS','2')
                        device()
                        import torch,p2_l2l_gpavatar as gp
                        from vico_gpavatar_render_fast import Canvas
                        gp.save=lambda p,v:save(WORK/'model_load/gpavatar'/Path(p).name,v)
                        model=gp.renderer();canvas=Canvas()
                    for path,h in read(ref/'complete.json')['files'].items():assert sha(path)==h
                    if current!=s['key']:
                        avatar=torch.load(ref/'avatar.pt',map_location='cpu',weights_only=True);tracked=torch.load(ref/'tracked.pt',map_location='cpu',weights_only=False)
                        avatar['reference_expression']=torch.as_tensor(tracked['emoca_expression']).float();torch.manual_seed(137)
                        with torch.inference_mode():model.build_avatar(avatar)
                        current=s['key']
                    for path,h in read(d/'MOTION_COMPLETE.json')['files'].items():assert sha(path)==h
                    m=np.load(d/'motion.npz')['motion'];tensor=torch.from_numpy(m).cuda();ix=np.minimum(np.floor(np.arange(s['frames'])*30/s['fps']+.5).astype(int),len(m)-1);started=time.time()
                    with torch.inference_mode():proof=gp.camera_parity(model,avatar,tensor)
                    def frames():
                        with torch.inference_mode():
                            for index in ix:
                                rgb=canvas(gp.render_motion(model,avatar,tensor[index:index+1]).clamp(0,1))
                                assert rgb.shape==(1,3,512,512) and torch.isfinite(rgb).all()
                                yield (rgb.permute(0,2,3,1).cpu().numpy()*255).round().astype(np.uint8)
                    score=d/'scoring.mp4';pipelined_video(score,frames(),s['fps'])
                    publish(method,s,[score,d/'motion.npz'],score_path=str(score),score_frames=s['frames'],score_fps=s['fps'],native_motion_fps=30,native_size=512,renderer='public GPAvatar; not original L2L photorealistic renderer',mapping=ix.tolist(),camera_parity=proof,reference_frames=1,source_reference_receipt_sha256=sha(ref/'complete.json'),motion_receipt_sha256=sha(d/'MOTION_COMPLETE.json'),code_sha256=sha(__file__),seconds=time.time()-started)
                    done+=1;processed+=1;print('GPAvatar render',method,s['key'],round(time.time()-started,2),flush=True)
                    if a.limit and done>=a.limit:return
        if not pending or not a.wait:return
        if not processed:time.sleep(10)
if __name__=='__main__':main()
