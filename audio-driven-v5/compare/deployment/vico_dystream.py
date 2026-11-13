"""User-approved listener-only ViCo43 adaptation of official DyStream inference."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT/"compare/DyStream"
WORK = ROOT/"experiments/p2_compare_vico43_20261106/dystream"
PARENT = ROOT/"experiments/p2_vico256_combined8000_20261106"
REF = ROOT/"experiments/p2_vico_motion8000_20261106"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n");temp.replace(path)


def prepare(work):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    work.mkdir(parents=True,exist_ok=True)
    if (work/"protocol.json").exists():return json.loads((work/"protocol.json").read_text())
    pre=json.loads((PARENT/"INPUT_PREFLIGHT.json").read_text())["samples_audit"]
    old=json.loads((PARENT/"protocol.json").read_text());samples=[]
    for item in pre:
        uid=item["id"];dest=work/"conditions"/uid;dest.mkdir(parents=True,exist_ok=True)
        source=PARENT/"conditions/vico"/uid/"first_frame.png"
        if not (dest/"first_frame.png").exists():shutil.copyfile(source,dest/"first_frame.png")
        assert sha(source)==sha(dest/"first_frame.png")
        wave,rate=sf.read(item["speaker_audio_path"],dtype="float32",always_2d=True);wave=wave.mean(1)
        gcd=math.gcd(rate,16000);wave=resample_poly(wave,16000//gcd,rate//gcd).astype(np.float32)
        count=round(item["original_gt"]["duration_s"]*16000)
        wave=np.pad(wave,(0,max(0,count-len(wave))))[:count]
        assert hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest()==item["waveform_sha256"]
        sf.write(dest/"speaker.wav",wave,16000,subtype="FLOAT")
        sf.write(dest/"listener_silent.wav",np.zeros(count,np.float32),16000,subtype="FLOAT")
        samples.append(dict(id=uid,duration=item["original_gt"]["duration_s"],frames=item["original_gt"]["frames"],
            image=str(dest/"first_frame.png"),image_sha256=sha(source),speaker=str(dest/"speaker.wav"),
            speaker_waveform_sha256=item["waveform_sha256"],silent=str(dest/"listener_silent.wav"),
            source_input_audit=str(PARENT/"conditions/vico"/uid/"input_audit.json")))
    assets=json.loads((ROOT/"compare/deployment/weights_manifest.json").read_text())["weights"]
    weights={str(ROOT/"compare"/x["path"]):x["sha256"] for x in assets if x["path"].startswith("DyStream/")}
    for path,expected in weights.items():assert sha(path)==expected,path
    protocol=dict(schema="compare-vico43-dystream-listener-v1",model="DyStream",seed=137,samples=samples,
        datasets=old["datasets"],generation_geometry=dict(width=512,height=512,fps=25),
        evaluation_geometry=dict(width=256,height=256,fps=30),
        user_approved=dict(audio_self="all-zero target listener",audio_other="real speaker",image="same listener first frame; process_image(crop=False)",
            no_listener_gt_audio=True,no_listener_future_video=True),
        parameters=dict(denoising_steps=5,cfg_audio=.5,cfg_audio_other=.5,cfg_anchor=0,cfg_all=1),
        temporal_conversion="nearest timestamp round-half-up and clamp at actual native length; no speed change; actual tail extension recorded",
        spatial_conversion="raw official RGB uint8 before native video encoding -> float bicubic antialias align_cornersFalse ->256",
        audio_output="model produces motion/video, no generated vocal waveform; native MP4 soundtrack is speaker context only; not scored as generated listener audio",
        reference_motion_metrics=str(REF),weights=weights,
        code_sha256={str(p):sha(p) for p in [Path(__file__),REPO/"app.py",REPO/"model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py"]},
        authorized_resource="node8 172.29.104.237 SSH22 physical GPU7 only")
    save(work/"protocol.json",protocol);return protocol


def claim(work,samples):
    with (work/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for s in sorted(samples,key=lambda x:-x["duration"]):
            if (work/"generated"/s["id"]/"complete.json").exists():continue
            flag=work/"claims"/(s["id"]+".json")
            if flag.exists():continue
            save(flag,dict(pid=os.getpid(),at=time.time()));return s


def worker(work,smoke=False):
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
    capture={};render_original=app.latents_to_video_frames
    def render(latents,image):
        frames=render_original(latents,image)
        capture["frames"]=frames;capture["latents"]=latents.detach().float().cpu().numpy()
        return frames
    app.latents_to_video_frames=render
    def process(s):
        uid=s["id"];dest=work/("smoke" if smoke else "generated")/uid;dest.mkdir(parents=True,exist_ok=True)
        start=time.time();torch.manual_seed(p["seed"])
        ref,masked,latent=app.process_image(Image.open(s["image"]),crop=False)
        ref.save(dest/"processed_reference.png");masked.save(dest/"processed_mask.png")
        np.savez_compressed(dest/"reference_motion.npz",motion_latent=latent.numpy())
        path,_,_,info=app.run_inference(None,s["silent"],s["speaker"],progress=lambda *a,**k:None,
            precomputed_npz_path=str(dest/"reference_motion.npz"),precomputed_ref_img_path=str(dest/"processed_reference.png"),
            video_audio_path=s["speaker"],**p["parameters"])
        shutil.copyfile(path,dest/"generated.mp4")
        frames=capture.pop("frames");latents=capture.pop("latents");np.save(dest/"motion_latent.npy",latents)
        if not np.isfinite(latents).all():raise ValueError("Nonfinite generated motion")
        native_frames=len(frames);assert frames.shape[1:]==(512,512,3)
        expected=int(round(s["duration"]*16000))//640-1
        assert native_frames==expected,(uid,native_frames,expected)
        mapping=np.minimum(np.floor(np.arange(s["frames"])*25/30+.5).astype(int),native_frames-1)
        score=work/("smoke_scoring" if smoke else "scoring")/"dystream"/(uid+".listener.mp4");score.parent.mkdir(parents=True,exist_ok=True)
        temp=score.with_suffix(".tmp.mp4")
        command=["ffmpeg","-nostdin","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","256x256","-r","30",
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
            listener_audio_generated=False,soundtrack="speaker context only",protocol_sha256=sha(work/"protocol.json"))
        save(dest/"complete.json",record);print(json.dumps(record),flush=True)
        return record
    if smoke:
        shortest=min(p["samples"],key=lambda x:x["duration"]);r=process(shortest)
        save(work/"SMOKE_COMPLETE.json",r)
    else:
        while (s:=claim(work,p["samples"])) is not None:
            try:process(s)
            except Exception as e:
                import traceback
                save(work/"failures"/(s["id"]+".json"),dict(error=str(e),traceback=traceback.format_exc()));raise


def run(work):
    p=json.loads((work/"protocol.json").read_text());assert (work/"SMOKE_COMPLETE.json").exists()
    (work/"claims").mkdir(exist_ok=True)
    with (work/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles=[];processes=[];start=time.time()
        try:
            for i in range(4):
                f=(work/f"worker_{i:02d}.log").open("w");handles.append(f)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="7",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",WANDB_MODE="disabled",TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--work",str(work)],env=env,stdout=f,stderr=subprocess.STDOUT))
            save(work/"LAUNCH.json",dict(controller_pid=os.getpid(),workers=[p.pid for p in processes],at=start,gpu=7))
            with (work/"gpu7.jsonl").open("a",buffering=1) as log:
                while any(proc.poll() is None for proc in processes):
                    if any(proc.poll() not in (None,0) for proc in processes):raise RuntimeError("Generation worker failed")
                    rows=list((work/"generated").glob("*/complete.json"));done=len(rows)
                    save(work/"status.json",dict(stage="generating",completed=done,total=43,elapsed_s=time.time()-start,at=time.time()))
                    gpu=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=memory.used,utilization.gpu,power.draw","--format=csv,noheader,nounits"],text=True)
                    mem,util,power=map(float,gpu.strip().split(','));log.write(json.dumps(dict(at=time.time(),memory_mib=mem,utilization_pct=util,power_w=power))+"\n")
                    time.sleep(2)
            assert len(list((work/"generated").glob("*/complete.json")))==43
            save(work/"GENERATION_COMPLETE.json",dict(clips=43,seconds=time.time()-start))
            save(work/"status.json",dict(stage="generation_complete",completed=43,total=43,at=time.time()))
        except Exception as e:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
            save(work/"CONTROLLER_FAILURE.json",dict(error=str(e),at=time.time()));raise
        finally:
            for proc in processes:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for f in handles:f.close()


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("command",choices=["prepare","smoke","worker","run"]);a.add_argument("--work",type=Path,default=WORK);args=a.parse_args();w=args.work.resolve()
    if args.command=="prepare":prepare(w)
    elif args.command=="run":run(w)
    else:worker(w,args.command=="smoke")
