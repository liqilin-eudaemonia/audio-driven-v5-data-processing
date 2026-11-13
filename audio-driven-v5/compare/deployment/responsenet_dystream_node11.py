"""Node11 GPUs0-6 extension; frozen DyStream generation and ResponseNet scores."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'compare/DyStream'
WORK=ROOT/'experiments/p2_compare_responsenet_random200_20261106/dystream'
EXT=WORK/'node11_extension'
BASE=ROOT/'compare/deployment/responsenet_dystream.py'
CURRENT={}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.name=='complete.json' and path.parent.parent.name=='generated':
        value=dict(value,execution_resource=dict(host=socket.gethostname(),gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
            extension_sha256=sha(EXT/'RESOURCE_EXTENSION.json'),adapter_sha256=sha(Path(__file__))),
            reference_preprocessing=CURRENT.get('preprocessing','official MediaPipe path'))
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def validate_resource():
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');p=json.loads((EXT/'RESOURCE_EXTENSION.json').read_text())
    assert gpu in p['gpus'],f'Only node11 GPUs0-6 authorized, got {gpu}'
    uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==p['gpus'][gpu],(gpu,uuid)
    return uuid


def claim(work,samples):
    with (work/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for s in sorted(samples,key=lambda x:-x['duration']):
            if (work/'generated'/s['id']/'complete.json').exists():continue
            flag=work/'claims'/(s['id']+'.json')
            if flag.exists():continue
            save(flag,dict(pid=os.getpid(),host=socket.gethostname(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],at=time.time()))
            CURRENT.clear();CURRENT['id']=s['id'];return s


def install_first_frame_fallback(app,work):
    # Detection is used only for a diagnostic eye/mouth mask when crop=False.
    # The inference reference and LIA latent depend on the CLEAN image only.
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    original=app.process_image;verified=False
    def clean(image):
        app.load_visualization_model();img=image.convert('RGB')
        transform=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.Normalize([.5],[.5])])
        values=(transform(transforms.functional.to_tensor(img)[None])+1)/2
        pixels=(values[0].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
        resized=Image.fromarray(pixels)
        x=app._vis_ctx['transform'](resized).unsqueeze(0).to(app.DEVICE)
        with torch.no_grad():latent=app._vis_ctx['motion_encoder'](x)[0].cpu()
        return resized,resized.copy(),latent
    def checked(image,*args,**kwargs):
        nonlocal verified
        assert kwargs.get('crop') is False
        try:result=original(image,*args,**kwargs)
        except Exception as error:
            if 'No face detected' not in str(error):raise
            result=clean(image)
            CURRENT['preprocessing']='MediaPipe no face; identical clean-image path and same LIA encoder; diagnostic mask unavailable'
            save(EXT/'first_frame_fallback'/(CURRENT['id']+'.json'),dict(id=CURRENT['id'],reason=str(error),
                only_first_frame=True,listener_future_access=False,clean_reference_unchanged=True,
                diagnostic_mask='clean image placeholder, not a network input',at=time.time()))
        else:
            if not verified:
                other=clean(image)
                pixel_equal=np.array_equal(np.asarray(result[0]),np.asarray(other[0]))
                error=float((result[2]-other[2]).abs().max())
                assert pixel_equal and error<1e-5,(pixel_equal,error)
                save(EXT/'preprocessing_parity'/(str(os.getpid())+'.json'),dict(passed=True,
                    resized_uint8_identical=pixel_equal,latent_max_abs=error,at=time.time()))
                verified=True
        return result
    app.process_image=checked


def worker(work,smoke=False):
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    uuid=validate_resource()
    p=json.loads((work/"protocol.json").read_text())
    for path,h in p["code_sha256"].items():assert sha(path)==h,path
    torch.set_num_threads(2);torch.manual_seed(p["seed"]);np.random.seed(p["seed"])
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    import app
    install_first_frame_fallback(app,work)
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
        mapping=np.minimum(np.floor(np.arange(s["frames"])*25/s["fps"]+.5).astype(int),native_frames-1)
        score=work/("smoke_scoring" if smoke else "scoring")/"dystream"/(uid+".listener.mp4");score.parent.mkdir(parents=True,exist_ok=True)
        temp=score.with_suffix(".tmp.mp4")
        command=["ffmpeg","-nostdin","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","512x512","-r",str(s["fps"]),
            "-i","pipe:0","-an","-c:v","libx264","-threads","1","-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(temp)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for start_index in range(0,len(mapping),32):
                data=frames[mapping[start_index:start_index+32]]
                proc.stdin.write(np.ascontiguousarray(data,dtype=np.uint8).tobytes())
        finally:proc.stdin.close()
        if proc.wait():raise RuntimeError("512 score export failed")
        temp.replace(score)
        error=np.abs(mapping/25-np.arange(s["frames"])/s["fps"])
        record=dict(id=uid,model="DyStream listener adaptation",native_frames=native_frames,native_fps=25,native_size=512,
            score_frames=s["frames"],score_fps=s["fps"],score_size=512,mapping=mapping.tolist(),max_timestamp_error_s=float(error.max()),
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


def score_worker(work,index):
    """Run the unchanged ResponseNet video scorer on one frozen shard."""
    validate_resource()
    p=json.loads((work/'protocol.json').read_text())
    selected=p['samples'][index::p['metric_shards']]
    out=work/'metric_shards'/f'{index:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    manifest=work/'metric_manifests'/f'{index:02d}.jsonl';manifest.parent.mkdir(exist_ok=True)
    rows=[]
    for item in selected:
        r=item['reference']
        rows.append(dict(id=r['id'],dataset='responsenet-j-paired-evaluation-v1',identity='listener',
            gt=r['gt'],generated=str(work/'scoring/dystream'/(r['id']+'.listener.mp4')),
            gt_resize=[512,512],gt_resize_method='pil_bicubic',source_start=r['source_start'],
            source_end=r['source_end'],listening_intervals=r['listening_intervals']))
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    cmd=[sys.executable,'-m','evaluation.video.paired_metrics.run','--manifest',str(manifest),
        '--assets',p['metric_assets'],'--output',str(out),'--device','cuda:0','--threads','4',
        '--batch-size','64','--face-padding','64','--feature-output',str(out/'features.npz'),
        '--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4',
        '--gt-cache',p['metrics_gt_cache']]
    code=subprocess.run(cmd,cwd=ROOT.parent/'OmniResponse').returncode
    summary=json.loads((out/'summary.json').read_text())
    assert not summary['initialization_errors'],summary['initialization_errors']
    actual=[json.loads(x) for x in (out/'pairs.jsonl').read_text().splitlines() if x.strip()]
    assert len(actual)==len(selected) and {x['id'] for x in actual}=={x['id'] for x in selected}
    assert (out/'features.npz').exists()
    save(out/'SUCCESS.json',dict(returncode=code,clips=len(actual),pairs_with_errors=summary['pairs_with_errors'],
        protocol_sha256=sha(work/'protocol.json'),at=time.time()))


def pipeline(work):
    validate_resource()
    spec=importlib.util.spec_from_file_location('original_dy',BASE)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    p=json.loads((work/'protocol.json').read_text());resource=json.loads((EXT/'RESOURCE_EXTENSION.json').read_text())
    dy=ROOT/'compare/.envs/dystream/bin/python';metric=Path('/private/software/conda/envs/AnyTalker/bin/python')
    script=Path(__file__).resolve();start=time.time();active=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='disabled',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT/'compare/deployment/assets/node11_hf_cache'))
    env['PATH']=str(dy.parent)+os.pathsep+env['PATH']
    def status(stage,**kw):
        save(work/'pipeline_status.json',dict(stage=stage,host='node11',allowed_gpus=list(range(7)),at=time.time(),**kw))
    def run_jobs(jobs,python,command,slots):
        nonlocal active
        pending=list(jobs)
        try:
            while pending or active:
                for job in list(active):
                    index,gpu,proc,log=job
                    if proc.poll() is not None:
                        log.close();active.remove(job)
                        if proc.returncode:raise RuntimeError(f'{command} index{index} gpu{gpu} failed; inspect node11_extension/logs')
                occupied={slot for slot,_,_,_ in active}
                for slot,gpu in slots:
                    if not pending:break
                    if slot in occupied:continue
                    index=pending.pop(0);path=EXT/'logs'/f'{command}_{index:02d}.log';path.parent.mkdir(exist_ok=True)
                    log=path.open('a')
                    proc=subprocess.Popen([str(python),str(script),command,'--work',str(work),'--index',str(index)],
                        env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python'),
                        stdout=log,stderr=subprocess.STDOUT)
                    active.append((slot,gpu,proc,log))
                    save(EXT/'processes'/f'{command}_{index:02d}.json',dict(pid=proc.pid,gpu=gpu,index=index,
                        host=socket.gethostname(),log=str(path),at=time.time()))
                completed=len(list((work/'generated').glob('*/complete.json')))
                status('generating' if command=='worker' else 'scoring',completed_videos=completed,total_videos=200,
                    completed_shards=len(list((work/'metric_shards').glob('*/SUCCESS.json'))),total_shards=p['metric_shards'],
                    active=[dict(gpu=gpu,pid=proc.pid) for _,gpu,proc,_ in active])
                if command=='worker':save(work/'status.json',dict(stage='generating',completed=completed,total=200,at=time.time()))
                report=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
                with (EXT/'gpu_usage.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),report=report))+'\n')
                time.sleep(2)
        finally:
            for _,_,proc,log in active:
                if proc.poll() is None:proc.terminate()
                log.close()
            active=[]
    with (work/'pipeline.lock').open('a') as pipeline_lock, (work/'controller.lock').open('a') as controller_lock:
        fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'PIPELINE.json',dict(pid=os.getpid(),host='node11',gpus=list(range(7)),start=start,
            adapter=str(script),resource_extension_sha256=sha(EXT/'RESOURCE_EXTENSION.json')))
        try:
            # The prior controller has exited. Preserve its complete outputs and
            # archive only claims without completed receipts for safe retry.
            history=EXT/'previous';history.mkdir(exist_ok=True)
            for flag in (work/'claims').glob('*.json'):
                if not (work/'generated'/flag.stem/'complete.json').exists():
                    dest=history/'claims'/flag.name;dest.parent.mkdir(exist_ok=True);shutil.move(flag,dest)
            for name in ['PIPELINE_FAILURE.json','CONTROLLER_FAILURE.json','CAPACITY_FAILURE.json']:
                if (work/name).exists():shutil.move(work/name,history/name)
            if (work/'failures').exists():shutil.move(work/'failures',history/'failures')
            status('preflight')
            if not (EXT/'PREFLIGHT_COMPLETE.json').exists():
                with (EXT/'preflight.log').open('a') as log:
                    subprocess.run([str(dy),str(script),'smoke','--work',str(work)],env=dict(env,CUDA_VISIBLE_DEVICES='0'),
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                save(EXT/'PREFLIGHT_COMPLETE.json',dict(passed=True,at=time.time()))
            if not (work/'GENERATION_COMPLETE.json').exists():
                count=resource['generation_workers_per_gpu'];slots=[(i*7+g,g) for i in range(count) for g in range(7)]
                run_jobs(range(len(slots)),dy,'worker',slots)
                assert len(list((work/'generated').glob('*/complete.json')))==200
                save(work/'GENERATION_COMPLETE.json',dict(clips=200,host='node11',seconds=time.time()-start))
                save(work/'status.json',dict(stage='generation_complete',completed=200,total=200,at=time.time()))
            pending=[i for i in range(p['metric_shards']) if not (work/'metric_shards'/f'{i:02d}'/'SUCCESS.json').exists()]
            count=resource['scoring_workers_per_gpu'];slots=[(i*7+g,g) for i in range(count) for g in range(7)]
            run_jobs(pending,metric,'score-worker',slots)
            status('merging')
            subprocess.run([str(metric),str(script),'merge','--work',str(work)],env=dict(env,CUDA_VISIBLE_DEVICES='0'),check=True)
            status('complete',seconds=time.time()-start)
        except Exception as e:
            save(EXT/'PIPELINE_FAILURE.json',dict(error=str(e),at=time.time()));status('failed',error=str(e));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pipeline','worker','smoke','score-worker','merge'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    w=args.work.resolve()
    if args.command=='pipeline':pipeline(w)
    elif args.command=='score-worker':
        (w/'score_locks').mkdir(exist_ok=True)
        with (w/'score_locks'/f'{args.index:02d}.lock').open('a') as metric_lock:
            fcntl.flock(metric_lock,fcntl.LOCK_EX)
            score_worker(w,args.index)
    elif args.command=='merge':
        spec=importlib.util.spec_from_file_location('original_dy',BASE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.score_merge(w)
    else:worker(w,args.command=='smoke')


