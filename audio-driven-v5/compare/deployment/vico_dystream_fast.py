"""DyStream official generation with one resident model and bounded export overlap."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from vico_dystream import ROOT, REPO
from vico_more_common import read, save, sha
from vico_fast_registry import BASE, node8_guard, assets_match, dystream_records
ARCHIVE = BASE / 'dystream'


def generate(work):
    work=Path(work).resolve();assert work != ARCHIVE
    work.mkdir(parents=True,exist_ok=True)
    protocol=read(ARCHIVE/'protocol.json');assert assets_match(protocol)
    if (work/'protocol.json').exists():
        assert read(work/'protocol.json') == protocol
    else:shutil.copyfile(ARCHIVE/'protocol.json',work/'protocol.json')
    records={r['id']:r for r in dystream_records(work,protocol)}
    old={r['id']:r for r in dystream_records(ARCHIVE,protocol)}
    pending=[]
    for sample in protocol['samples']:
        uid=sample['id']
        if uid in records:continue
        if uid in old:
            records[uid]=old[uid]
            # Keep original predictions read-only. Do not alter per-clip receipts.
            for src,dest in [(ARCHIVE/'generated'/uid,work/'generated'/uid),
                (ARCHIVE/'scoring/dystream'/(uid+'.listener.mp4'),work/'scoring/dystream'/(uid+'.listener.mp4'))]:
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.is_symlink():dest.unlink()
                if dest.exists():dest.rename(dest.with_name(dest.name+'.invalid.'+str(time.time_ns())))
                dest.symlink_to(src,target_is_directory=src.is_dir())
        else:
            # Detach reused links before computing missing/corrupted results.
            for dest in [work/'generated'/uid,work/'scoring/dystream'/(uid+'.listener.mp4')]:
                if dest.is_symlink():dest.unlink()
            pending.append(sample)
    if pending:
        pending=sorted(pending,key=lambda s:-s['duration'])
        run_workers(work,pending)
    records=dystream_records(work,protocol)
    assert len(records)==len(protocol['samples'])
    save(work/'GENERATION_COMPLETE.json',dict(success=True,clips=len(records),records=records,
        optimization='up to4 resident workers limited by pending clips, bounded asynchronous export per worker'))

def worker(work, samples):
    smoke=False
    node8_guard()
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending=None
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="7":raise ValueError("Physical GPU7 only")
    uuid=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
    assert uuid=="GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc"
    p=json.loads((work/"protocol.json").read_text())
    for path,h in p["code_sha256"].items():assert sha(path)==h,path
    torch.set_num_threads(2);torch.manual_seed(p["seed"]);np.random.seed(p["seed"])
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    import app
    loads=[];load_original=torch.nn.Module.load_state_dict
    def audited_load(module,state,*args,**kwargs):
        report=load_original(module,state,*args,**kwargs)
        if module is app._dystream_model:loads.append(dict(missing=report.missing_keys,unexpected=report.unexpected_keys))
        return report
    torch.nn.Module.load_state_dict=audited_load
    try:app.load_dystream_model();app.load_visualization_model()
    finally:torch.nn.Module.load_state_dict=load_original
    assert len(loads)==1 and not loads[0]["missing"] and not loads[0]["unexpected"],loads
    app._dystream_model.cfg.seed=p["seed"]
    save(work/"model_load"/(str(os.getpid())+".json"),dict(strict_equivalent_load=loads,torch=torch.__version__,gpu_uuid=uuid,seed=137))
    save_video_original=app.save_video_with_audio
    # Native video encoding is deferred to the bounded CPU exporter.
    app.save_video_with_audio=lambda frames,audio,path,fps=25: path
    capture={};render_original=app.latents_to_video_frames
    def render(latents,image):
        frames=render_original(latents,image)
        capture["frames"]=frames;capture["latents"]=latents.detach().float().cpu().numpy()
        return frames
    app.latents_to_video_frames=render
    def process(s):
        nonlocal pending
        uid=s["id"];dest=work/("smoke" if smoke else "generated")/uid;dest.mkdir(parents=True,exist_ok=True)
        start=time.time();torch.manual_seed(p["seed"])
        ref,masked,latent=app.process_image(Image.open(s["image"]),crop=False)
        ref.save(dest/"processed_reference.png");masked.save(dest/"processed_mask.png")
        np.savez_compressed(dest/"reference_motion.npz",motion_latent=latent.numpy())
        path,_,_,info=app.run_inference(None,s["silent"],s["speaker"],progress=lambda *a,**k:None,
            precomputed_npz_path=str(dest/"reference_motion.npz"),precomputed_ref_img_path=str(dest/"processed_reference.png"),
            video_audio_path=s["speaker"],**p["parameters"])
        frames=capture.pop("frames");latents=capture.pop("latents")
        def export():
            save_video_original(frames,s["speaker"],path,fps=25)
            shutil.copyfile(path,dest/"generated.mp4")
            np.save(dest/"motion_latent.npy",latents)
            if not np.isfinite(latents).all():raise ValueError("Nonfinite generated motion")
            native_frames=len(frames);assert frames.shape[1:]==(512,512,3)
            expected=int(round(s["duration"]*16000))//640-1
            assert native_frames==expected,(uid,native_frames,expected)
            mapping=np.minimum(np.floor(np.arange(s["frames"])*25/30+.5).astype(int),native_frames-1)
            score=work/("smoke_scoring" if smoke else "scoring")/"dystream"/(uid+".listener.mp4");score.parent.mkdir(parents=True,exist_ok=True)
            temp=score.with_suffix(".tmp.mp4")
            command=["ffmpeg","-nostdin","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","256x256","-r","30",
                "-i","pipe:0","-an","-c:v","libx264","-threads","1","-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(temp)]
            proc=subprocess.Popen(command,stdin=subprocess.PIPE)
            try:
                for start_index in range(0,len(mapping),32):
                    x=torch.from_numpy(frames[mapping[start_index:start_index+32]].copy()).permute(0,3,1,2).float()/255
                    small=torch.nn.functional.interpolate(x,size=(256,256),mode="bicubic",align_corners=False,antialias=True).clamp(0,1)
                    data=(small.permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
                    proc.stdin.write(data.tobytes())
            finally:proc.stdin.close()
            if proc.wait():raise RuntimeError("256 score export failed")
            temp.replace(score)
            error=np.abs(mapping/25-np.arange(s["frames"])/30)
            record=dict(id=uid,model="DyStream listener adaptation",native_frames=native_frames,native_fps=25,native_size=512,
                score_frames=s["frames"],score_fps=30,score_size=256,mapping=mapping.tolist(),max_timestamp_error_s=float(error.max()),
                official_tail_shortfall_frames=1,seconds=time.time()-start,info=info,source_only=True,
                image_sha256=s["image_sha256"],native_sha256=sha(dest/"generated.mp4"),score_sha256=sha(score),
                listener_audio_generated=False,soundtrack="speaker context only",protocol_sha256=sha(work/"protocol.json"),
                fast_adapter_sha256=sha(__file__),official_model_forward_unchanged=True)
            save(dest/"complete.json",record);print(json.dumps(record),flush=True)
            return record
        if pending is not None:pending.result()
        pending=pool.submit(export)
    try:
        for sample in samples:process(sample)
        if pending is not None:pending.result()
    finally:
        pool.shutdown(wait=True)
        app.save_video_with_audio=save_video_original
        app.latents_to_video_frames=render_original


def run_workers(work, samples):
    # Retain the original maximum parallelism; do not launch idle model replicas.
    node8_guard()
    free=float(subprocess.check_output(['nvidia-smi','-i','7',
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())/1024
    count=min(4,len(samples),max(1,int((free-32)//12)))
    if count==1:
        worker(work,samples)
        return
    processes=[];handles=[]
    try:
        for i in range(count):
            task=work/('worker_samples_%d.json'%i);save(task,samples[i::count])
            log=(work/('worker_fast_%d.log'%i)).open('w');handles.append(log)
            processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                '--work',str(work),'--samples',str(task)],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('DyStream worker failed; see worker_fast_*.log')
            time.sleep(.5)
        if any(p.returncode for p in processes):raise RuntimeError('DyStream worker failed')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for handle in handles:handle.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    args=parser.parse_args()
    worker(args.work,read(args.samples))
