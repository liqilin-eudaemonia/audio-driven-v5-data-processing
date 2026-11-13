"""Listener-only DyStream on the frozen ResponseNet stratified random200; original FPS scoring."""
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
WORK = ROOT/"experiments/p2_compare_responsenet_random200_20261106/dystream"
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
    sys.path.insert(0,str(ROOT/'src'))
    from p2_testsets_eval.runtime import conditioning
    from p2_testsets_eval.protocol import geometry
    parent=ROOT/'experiments/p2_responsenet_random200_twelveway_20261105'
    source=parent/'references/references.jsonl'
    old=json.loads((parent/'protocol.json').read_text())
    expected=old['datasets']['responsenet']['references_sha256']
    assert sha(source)==expected
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(rows)==len({s['id'] for s in rows})==200
    work.mkdir(parents=True,exist_ok=True)
    if (work/'protocol.json').exists():
        p=json.loads((work/'protocol.json').read_text())
        assert p['references_sha256']==expected
        return p
    samples=[]
    for r in rows:
        dest=work/'conditions'/r['id'];g=geometry(r,'responsenet')
        result=conditioning(dict(reference=r,geometry=g,dataset='responsenet',id=r['id'],key='responsenet/'+r['id']),dest)
        wave=result['audio'];sf.write(dest/'speaker.wav',wave,16000,subtype='FLOAT')
        sf.write(dest/'listener_silent.wav',np.zeros(len(wave),np.float32),16000,subtype='FLOAT')
        samples.append(dict(id=r['id'],duration=g['duration_s'],frames=g['frames'],fps=g['fps'],
            image=str(dest/'first_frame.png'),image_sha256=sha(dest/'first_frame.png'),
            speaker=str(dest/'speaker.wav'),silent=str(dest/'listener_silent.wav'),
            speaker_waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
            source_input_audit=str(dest/'input_audit.json'),reference=r))
    assets=json.loads((ROOT/'compare/deployment/weights_manifest.json').read_text())['weights']
    weights={str(ROOT/'compare'/x['path']):x['sha256'] for x in assets if x['path'].startswith('DyStream/')}
    for path,h in weights.items():assert sha(path)==h,path
    protocol=dict(schema='responsenet-random200-dystream-listener-v1',model='DyStream',seed=137,samples=samples,
        references=str(source),references_sha256=expected,total_duration_s=sum(s['duration'] for s in samples),
        generation_geometry=dict(width=512,height=512,fps=25),
        evaluation_geometry=dict(width=512,height=512,fps='each reference original FPS',frames='exact reference count'),
        user_approved=dict(audio_self='all-zero target listener',audio_other='real speaker',
            image='same original evaluation conditioning adapter; first listener frame PIL bicubic512; cropFalse',
            no_listener_gt_audio=True,no_listener_future_video=True),
        parameters=dict(denoising_steps=5,cfg_audio=.5,cfg_audio_other=.5,cfg_anchor=0,cfg_all=1),
        temporal_conversion='nearest timestamp round-half-up at original FPS, clamped to native length; no speed change; tail error retained',
        spatial_conversion='native512 RGB before official video encoding; scoring512 H264 CRF18; GT PIL bicubic512 as historical protocol',
        audio_output='No generated listener audio; native MP4 soundtrack is speaker context only, never scored as listener output',
        learning2listen='deferred at user request; no L2L task launched',
        metric_parent_protocol_sha256=sha(parent/'protocol.json'),metric_assets=old['metric_assets'],
        metrics_gt_cache=old['metrics_gt_cache'],metric_shards=16,
        metrics=['psnr','ssim','lpips','fid','fvd','emo_acc','emo_div','delta_resp'],
        weights=weights,code_sha256={str(p):sha(p) for p in [Path(__file__),REPO/'app.py',
            REPO/'model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py',
            ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/protocol.py']},
        authorized_resource='node8 172.29.104.237 SSH22 physical GPU7 only')
    save(work/'protocol.json',protocol)
    save(work/'PRECHECK.json',dict(passed=True,count=200,references_sha256=expected,
        duration_s=protocol['total_duration_s'],fps=sorted({s['fps'] for s in samples}),source_only=True))
    return protocol


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


def run(work):
    p=json.loads((work/"protocol.json").read_text());assert (work/"SMOKE_COMPLETE.json").exists()
    (work/"claims").mkdir(exist_ok=True)
    with (work/"controller.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles=[];processes=[];start=time.time()
        try:
            for i in range(4):
                f=(work/f"worker_{i:02d}.log").open("a");handles.append(f)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="7",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",WANDB_MODE="disabled",TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--work",str(work)],env=env,stdout=f,stderr=subprocess.STDOUT))
            save(work/"LAUNCH.json",dict(controller_pid=os.getpid(),workers=[p.pid for p in processes],at=start,gpu=7))
            with (work/"gpu7.jsonl").open("a",buffering=1) as log:
                while any(proc.poll() is None for proc in processes):
                    if any(proc.poll() not in (None,0) for proc in processes):raise RuntimeError("Generation worker failed")
                    rows=list((work/"generated").glob("*/complete.json"));done=len(rows)
                    save(work/"status.json",dict(stage="generating",completed=done,total=len(p["samples"]),elapsed_s=time.time()-start,at=time.time()))
                    gpu=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=memory.used,utilization.gpu,power.draw","--format=csv,noheader,nounits"],text=True)
                    mem,util,power=map(float,gpu.strip().split(','));log.write(json.dumps(dict(at=time.time(),memory_mib=mem,utilization_pct=util,power_w=power))+"\n")
                    time.sleep(2)
            assert len(list((work/"generated").glob("*/complete.json")))==len(p["samples"])
            save(work/"GENERATION_COMPLETE.json",dict(clips=len(p["samples"]),seconds=time.time()-start))
            save(work/"status.json",dict(stage="generation_complete",completed=len(p["samples"]),total=len(p["samples"]),at=time.time()))
        except Exception as e:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
            save(work/"CONTROLLER_FAILURE.json",dict(error=str(e),at=time.time()));raise
        finally:
            for proc in processes:
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for f in handles:f.close()


def score_worker(work,index):
    """Run the unchanged ResponseNet video scorer on one frozen shard."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
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


def score_merge(work):
    import numpy as np
    sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
    from evaluation.video.paired_metrics.run import summarize,METRICS
    p=json.loads((work/'protocol.json').read_text());rows=[];contracts=[]
    dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
    for index in range(p['metric_shards']):
        folder=work/'metric_shards'/f'{index:02d}';assert (folder/'SUCCESS.json').exists()
        summary=json.loads((folder/'summary.json').read_text())
        contracts.append(json.loads((folder/'contract.json').read_text()))
        rows.extend(json.loads(x) for x in (folder/'pairs.jsonl').read_text().splitlines() if x.strip())
        with np.load(folder/'features.npz') as features:
            for k in dist:
                if features[k+'_gt'].size:
                    dist[k][0].append(features[k+'_gt']);dist[k][1].append(features[k+'_generated'])
                coverage[k]+=summary['coverage'][k]['pairs']
    assert len(rows)==len({x['id'] for x in rows})==len(p['samples'])==200
    assert {r['id'] for r in rows}=={s['id'] for s in p['samples']}
    for c in contracts:assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    summary=summarize(rows,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],len(rows),{})
    for k,n in coverage.items():summary['coverage'][k]['pairs']=n
    summary.update(model='DyStream',dataset='ResponseNet stratified random200',
        references_sha256=p['references_sha256'],protocol_sha256=sha(work/'protocol.json'),
        aggregation='historical ResponseNet scorer; FID/FVD recomputed from all200 features, never mean shard scores',
        audio_metrics=dict(value=None,reason='model generates no listener audio; speaker context never scored as generated listener'),
        learning2listen='deferred by user')
    save(work/'RESULTS.json',summary)
    (work/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit=[]
    for s in p['samples']:
        uid=s['id'];r=json.loads((work/'generated'/uid/'complete.json').read_text())
        assert r['source_only'] and not r['listener_audio_generated']
        assert r['protocol_sha256']==sha(work/'protocol.json')
        assert r['score_frames']==s['frames'] and r['score_fps']==s['fps'] and r['score_size']==512
        for path,expected in [(work/'generated'/uid/'generated.mp4',r['native_sha256']),
                              (work/'scoring/dystream'/(uid+'.listener.mp4'),r['score_sha256'])]:
            assert sha(path)==expected
        audit.append(dict(id=uid,max_timestamp_error_s=r['max_timestamp_error_s']))
    save(work/'FINAL_AUDIT.json',dict(passed=True,clips=200,references_sha256=p['references_sha256'],
        protocol_sha256=sha(work/'protocol.json'),results_sha256=sha(work/'RESULTS.json'),temporal_audit=audit,
        metric_pairs_with_errors=summary['pairs_with_errors'],coverage=summary['coverage'],at=time.time()))
    lines=['# DyStream · ResponseNet分层随机200条','',
        '使用冻结200条清单；512×512/25fps原生生成，最近时间戳映射到每条GT原始FPS及帧数后评分。',
        '仅speaker音频及listener首帧；自身音频置零，关闭二次裁脸；无生成listener音轨。',
        '使用既有ResponseNet视频评分器，FID/FVD全200条特征汇总；Learning2Listen按用户要求暂不启动。','',
        '| 指标 | 值 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in summary['metrics'].items()]
    lines += ['',f"状态：{summary['status']}；有错误的pair：{summary['pairs_with_errors']}。完整覆盖见RESULTS.json。"]
    (work/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(work/'COMPLETE.json',dict(status=summary['status'],clips=200,audit_passed=True,at=time.time()))


def pipeline(work):
    """Detached, locked GPU7-only generation -> scoring -> audited global merge."""
    p=json.loads((work/'protocol.json').read_text());start=time.time()
    script=Path(__file__).resolve();base=ROOT/'compare/.envs/dystream/bin/python'
    metric=Path('/private/software/conda/envs/AnyTalker/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1')
    env['PATH']=str(base.parent)+os.pathsep+env['PATH']
    with (work/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'PIPELINE.json',dict(pid=os.getpid(),host='node8',gpu=7,start=start,learning2listen='deferred'))
        uuid=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        assert uuid=='GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc'
        active=[]
        try:
            for stage in ['smoke','run']:
                if (work/('SMOKE_COMPLETE.json' if stage=='smoke' else 'GENERATION_COMPLETE.json')).exists():continue
                save(work/'pipeline_status.json',dict(stage=stage,at=time.time()))
                subprocess.run([str(base),str(script),stage,'--work',str(work)],env=env,check=True)
            save(work/'pipeline_status.json',dict(stage='scoring',at=time.time()))
            score_env=dict(env,PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python')
            pending=[i for i in range(p['metric_shards']) if not (work/'metric_shards'/f'{i:02d}'/'SUCCESS.json').exists()]
            with (work/'scoring_gpu7.jsonl').open('a',buffering=1) as gpu_log:
                while pending or active:
                    for job in list(active):
                        index,proc,log=job
                        if proc.poll() is not None:
                            log.close();active.remove(job)
                            if proc.returncode:raise RuntimeError(f'score shard{index} failed, see metric_{index:02d}.log')
                    while pending and len(active)<4:
                        index=pending.pop(0);log=(work/f'metric_{index:02d}.log').open('a')
                        proc=subprocess.Popen([str(metric),str(script),'score-worker','--work',str(work)],
                            env=dict(score_env,RESPONSE_DY_METRIC_RANK=str(index)),stdout=log,stderr=subprocess.STDOUT)
                        active.append((index,proc,log))
                    done=len(list((work/'metric_shards').glob('*/SUCCESS.json')))
                    save(work/'pipeline_status.json',dict(stage='scoring',completed_shards=done,total_shards=p['metric_shards'],
                        active_pids=[proc.pid for _,proc,_ in active],at=time.time()))
                    report=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                    gpu_log.write(json.dumps(dict(at=time.time(),report=report))+'\n');time.sleep(3)
            save(work/'pipeline_status.json',dict(stage='merging',at=time.time()))
            subprocess.run([str(metric),str(script),'score-merge','--work',str(work)],env=score_env,check=True)
            save(work/'pipeline_status.json',dict(stage='complete',seconds=time.time()-start,at=time.time()))
        except Exception as e:
            for _,proc,log in active:
                if proc.poll() is None:proc.terminate()
                log.close()
            save(work/'PIPELINE_FAILURE.json',dict(error=str(e),at=time.time()));raise


if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("command",choices=["prepare","smoke","worker","run","pipeline","score-worker","score-merge"]);a.add_argument("--work",type=Path,default=WORK);args=a.parse_args();w=args.work.resolve()
    if args.command=="prepare":prepare(w)
    elif args.command=="run":run(w)
    elif args.command=="pipeline":pipeline(w)
    elif args.command=="score-worker":score_worker(w,int(os.environ['RESPONSE_DY_METRIC_RANK']))
    elif args.command=="score-merge":score_merge(w)
    else:worker(w,args.command=="smoke")
