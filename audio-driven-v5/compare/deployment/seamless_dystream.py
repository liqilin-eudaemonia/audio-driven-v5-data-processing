"""Frozen Seamless200 DyStream listener test on node11 physical GPU6 only."""
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
WORK=ROOT/'experiments/p2_compare_seamless200_20261106/dystream'
PARENT=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
INPUTS=ROOT/'experiments/seamless_quick200_latest_20261105/prepared_200'
REUSE=ROOT/'compare/deployment/responsenet_dystream_node11.py'
REPO=ROOT/'compare/DyStream'
GPU_UUID='GPU-1856694d-3cd3-0855-0e94-c2a259545b97'
GENERATION_WORKERS=12
SCORING_WORKERS=8


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(p)


def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def validate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6','Only node11 physical GPU6 authorized'
    actual=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert actual==GPU_UUID,actual


def prepare(work):
    import numpy as np
    import soundfile as sf
    sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'OmniResponse')]
    from p2_testsets_eval.protocol import geometry
    from p2_testsets_eval.runtime import conditioning
    work.mkdir(parents=True,exist_ok=True)
    parent=read(PARENT/'protocol.json');source=INPUTS/'references.jsonl';seal=read(INPUTS/'seal.json')
    assert sha(source)==parent['datasets']['seamless']['references_sha256']==seal['references_sha256']
    refs=rows(source);assert len(refs)==len({r['id'] for r in refs})==200
    assert all(r['dataset']=='seamless' and r['split']=='test' and r['fps']==25 and r['width']==r['height']==512 for r in refs)
    if (work/'protocol.json').exists():
        p=read(work/'protocol.json');assert p['references_sha256']==sha(source);return p
    start=time.time();done=[]
    def one(r):
        dest=work/'conditions'/r['id'];g=geometry(r,'seamless')
        value=conditioning(dict(reference=r,geometry=g,dataset='seamless',id=r['id'],key='seamless/'+r['id']),dest)
        wave=value['audio'];sf.write(dest/'speaker.wav',wave,16000,subtype='FLOAT')
        sf.write(dest/'listener_silent.wav',np.zeros(len(wave),np.float32),16000,subtype='FLOAT')
        historical=PARENT/'base_standard/conditions/seamless'/r['id']
        assert sha(dest/'first_frame.png')==sha(historical/'first_frame.png'),r['id']
        now=read(dest/'input_audit.json');old=read(historical/'input_audit.json')
        assert now==old,r['id']
        assert now['source_only'] and not now['future_listener_media_in_inputs']
        return dict(id=r['id'],duration=g['duration_s'],frames=g['frames'],fps=g['fps'],reference=r,
            image=str(dest/'first_frame.png'),image_sha256=sha(dest/'first_frame.png'),
            speaker=str(dest/'speaker.wav'),silent=str(dest/'listener_silent.wav'),
            speaker_waveform_sha256=hashlib.sha256(wave.astype('<f4').tobytes()).hexdigest(),
            source_input_audit=str(dest/'input_audit.json'),historical_input_audit_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,r):r['id'] for r in refs}
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result())
            save(work/'prepare_status.json',dict(completed=len(done),total=200,seconds=time.time()-start,at=time.time()))
            if len(done)%20==0:print('prepared',len(done),'/200',flush=True)
    by_id={s['id']:s for s in done};samples=[by_id[r['id']] for r in refs]
    manifest=read(ROOT/'compare/deployment/weights_manifest.json')['weights']
    weights={str(ROOT/'compare'/r['path']):r['sha256'] for r in manifest if r['path'].startswith('DyStream/')}
    for path,expected in weights.items():assert sha(path)==expected,path
    paths=[Path(__file__),REUSE,ROOT/'compare/deployment/responsenet_dystream.py',REPO/'app.py',
        REPO/'model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py',
        ROOT/'src/p2_testsets_eval/runtime.py',ROOT/'src/p2_testsets_eval/protocol.py',
        ROOT/'scripts/p2_cached_video_metrics.py',ROOT/'src/p2_testsets_eval/decoded_gt.py']
    p=dict(schema='dystream-seamless-fixed200-listener-v1',dataset='Seamless fixed200',model='DyStream',seed=137,
        references=str(source),references_sha256=sha(source),seal_sha256=sha(INPUTS/'seal.json'),samples=samples,
        total_duration_s=sum(s['duration'] for s in samples),generation_geometry=dict(width=512,height=512,fps=25),
        evaluation_geometry=dict(width=512,height=512,fps=25,frames='exact frozen reference count'),
        parameters=dict(denoising_steps=5,cfg_audio=.5,cfg_audio_other=.5,cfg_anchor=0,cfg_all=1),
        conditioning='same frozen Seamless adapter; speaker WAV plus listener first frame only; own-listener branch all-zero; cropFalse',
        first_frame_detection='reused verified clean-reference fallback for diagnostic-mask detection failures; clean RGB and LIA input unchanged',
        forbidden_inputs=['listener future frames','listener audio','GT reaction caption'],
        audio_output='no generated listener waveform; native MP4 contains speaker context only; no listener audio metrics',
        temporal_conversion='nearest25fps timestamp identity map, clamp only official native tail shortfall; no speed change',
        video_export='same raw512 RGB -> H264 CRF18 scoring copy as verified ResponseNet adaptation',
        metric_assets=parent['metric_assets'],metrics_gt_cache=parent['metrics_gt_cache'],
        metrics_decoded_gt_cache=parent['metrics_decoded_gt_cache'],metric_shards=16,face_padding=0,
        metrics=['psnr','ssim','lpips','fid','fvd','emo_acc','emo_div','delta_resp'],
        generation_workers=GENERATION_WORKERS,scoring_workers=SCORING_WORKERS,
        reference_protocol_sha256=sha(PARENT/'protocol.json'),weights=weights,code_sha256={str(q):sha(q) for q in paths},
        authorized_resource=dict(host='172.29.189.119',gpu=6,uuid=GPU_UUID),learning2listen='deferred by user')
    save(work/'protocol.json',p)
    save(work/'runtime/RESOURCE_EXTENSION.json',dict(host='172.29.189.119',gpus={'6':GPU_UUID},
        authorized_by='user requested next test after confirming node11 GPU6 idle',excluded_gpus=[0,1,2,3,4,5,7]))
    save(work/'PRECHECK.json',dict(passed=True,clips=200,references_sha256=p['references_sha256'],
        duration_s=p['total_duration_s'],historical_first_frames_and_input_audits_match=200,seconds=time.time()-start))
    return p


def worker(work,smoke=False):
    validate()
    spec=importlib.util.spec_from_file_location('verified_node11_worker',REUSE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.WORK=work;m.EXT=work/'runtime'
    original_save=m.save
    def receipt_save(path,value):
        if path.name=='complete.json' and path.parent.parent.name=='generated':
            value=dict(value,driver_sha256=sha(Path(__file__)))
        original_save(path,value)
    m.save=receipt_save;m.worker(work,smoke)


def score(work,index):
    validate();p=read(work/'protocol.json')
    folder=work/'metric_shards'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    (work/'score_locks').mkdir(exist_ok=True)
    with (work/'score_locks'/f'{index:02d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'SUCCESS.json').exists():return
        selected=p['samples'][index::p['metric_shards']]
        manifest=work/'metric_manifests'/f'{index:02d}.jsonl';manifest.parent.mkdir(exist_ok=True)
        paired=[dict(id=s['id'],dataset='seamless',identity='listener',gt_reference=s['reference']['gt_reference'],
            generated=str(work/'scoring/dystream'/(s['id']+'.listener.mp4')),
            listening_intervals=s['reference']['listening_intervals']) for s in selected]
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in paired))
        cmd=[sys.executable,str(ROOT/'scripts/p2_cached_video_metrics.py'),
            '--decoded-gt-cache',p['metrics_decoded_gt_cache'],'--manifest',str(manifest),
            '--assets',p['metric_assets'],'--output',str(folder),'--device','cuda:0','--threads','4',
            '--batch-size','64','--face-padding','0','--feature-output',str(folder/'features.npz'),
            '--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4',
            '--gt-cache',p['metrics_gt_cache']]
        code=subprocess.run(cmd,cwd=ROOT.parent/'OmniResponse').returncode
        result=read(folder/'summary.json');assert not result['initialization_errors'],result['initialization_errors']
        actual=rows(folder/'pairs.jsonl');assert len(actual)==len(selected) and {r['id'] for r in actual}=={s['id'] for s in selected}
        assert (folder/'features.npz').exists()
        save(folder/'SUCCESS.json',dict(returncode=code,clips=len(actual),pairs_with_errors=result['pairs_with_errors'],
            protocol_sha256=sha(work/'protocol.json'),at=time.time()))


def merge(work):
    import numpy as np
    sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
    from evaluation.video.paired_metrics.run import summarize,METRICS
    p=read(work/'protocol.json');pairs=[];contracts=[];coverage={k:0 for k in ['fid','fvd']}
    dist={k:[[],[]] for k in coverage}
    for index in range(p['metric_shards']):
        f=work/'metric_shards'/f'{index:02d}';assert (f/'SUCCESS.json').exists()
        result=read(f/'summary.json');contracts.append(read(f/'contract.json'));pairs.extend(rows(f/'pairs.jsonl'))
        with np.load(f/'features.npz') as data:
            for k in dist:
                if data[k+'_gt'].size:
                    dist[k][0].append(data[k+'_gt']);dist[k][1].append(data[k+'_generated'])
                coverage[k]+=result['coverage'][k]['pairs']
    expected={s['id']:s for s in p['samples']}
    assert len(pairs)==len({r['id'] for r in pairs})==200 and {r['id'] for r in pairs}==set(expected)
    historical=read(PARENT/'metric_shards/av_only_6000/seamless/00/contract.json')
    for c in contracts:
        assert c['code']==historical['code'] and c['assets']==historical['assets']
        for key in ['batch_size','face_padding','fid_frames','fvd_frames','fvd_stride','face_device','emotion_device','pixel_workers']:
            assert c['arguments'][key]==historical['arguments'][key],key
    result=summarize(pairs,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],200,{})
    for k,n in coverage.items():result['coverage'][k]['pairs']=n
    result.update(model='DyStream',dataset='Seamless fixed200',references_sha256=p['references_sha256'],
        protocol_sha256=sha(work/'protocol.json'),audio_metrics=None,
        aggregation='unchanged Seamless video scorer; full200 FID/FVD from merged features; never mean shard scores')
    save(work/'RESULTS.json',result)
    (work/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    protocol_hash=sha(work/'protocol.json');frames=0;fallback=[]
    for s in p['samples']:
        uid=s['id'];receipt=read(work/'generated'/uid/'complete.json')
        assert receipt['source_only'] and not receipt['listener_audio_generated'] and receipt['protocol_sha256']==protocol_hash
        assert receipt['driver_sha256']==sha(Path(__file__)) and receipt['execution_resource']['gpu']==6
        assert receipt['score_frames']==s['frames'] and receipt['score_fps']==25 and receipt['score_size']==512
        assert sha(work/'generated'/uid/'generated.mp4')==receipt['native_sha256']
        assert sha(work/'scoring/dystream'/(uid+'.listener.mp4'))==receipt['score_sha256']
        if 'diagnostic mask unavailable' in receipt.get('reference_preprocessing',''):fallback.append(uid)
        frames+=s['frames']
    for r in pairs:
        media=r['media'];n=expected[r['id']]['frames']
        assert media['gt_frames']==media['generated_frames']==media['aligned_frames']==n
        assert media['fps']==25 and media['trimmed_gt']==media['trimmed_generated']==0
    for name,digest in p['code_sha256'].items():assert sha(name)==digest,name
    assert sha(p['references'])==p['references_sha256']
    cache=[read(work/'metric_shards'/f'{i:02d}'/'decoded_gt_cache.json') for i in range(p['metric_shards'])]
    audit=dict(passed=True,clips=200,score_frames=frames,source_only=True,gpu=6,host='node11',
        historical_scorer_code_assets_and_parameters_match=True,no_frame_trimming=True,
        references_sha256=p['references_sha256'],results_sha256=sha(work/'RESULTS.json'),
        status=result['status'],coverage=result['coverage'],pairs_with_errors=result['pairs_with_errors'],
        first_frame_diagnostic_mask_fallback_ids=fallback,decoded_gt_cache_hits=sum(c['hits'] for c in cache),
        decoded_gt_cache_misses=sum(c['misses'] for c in cache),at=time.time())
    save(work/'FINAL_AUDIT.json',audit)
    lines=['# DyStream · Seamless固定200条','',
        '固定200条／19.99分钟；speaker音频＋listener首帧，自身音频全零，512×512／25fps，seed137、官方5步。',
        '沿用冻结最终裁框和Seamless评分器，无损GT缓存逐次校验RGB摘要。FID/FVD由全200特征统一重算。',
        '原生官方少帧以尾帧延展到精确GT长度，原生视频保留，逐条时间映射可审计。无生成listener音频，音频指标不适用。','',
        '| 指标 | 值 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in result['metrics'].items()]
    lines += ['',f"状态：{result['status']}；指标错误pair：{result['pairs_with_errors']}；覆盖见RESULTS.json。"]
    (work/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(work/'COMPLETE.json',dict(status=result['status'],clips=200,audit_passed=True,at=time.time()))


def pipeline(work):
    validate();work.mkdir(parents=True,exist_ok=True);start=time.time();active=[]
    script=Path(__file__).resolve();dy=ROOT/'compare/.envs/dystream/bin/python'
    prep=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python';metric=Path('/private/software/conda/envs/AnyTalker/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='disabled',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT/'compare/deployment/assets/node11_hf_cache'))
    env['PATH']=str(dy.parent)+os.pathsep+env['PATH']
    def status(stage,**kw):save(work/'status.json',dict(stage=stage,host='node11',gpu=6,at=time.time(),**kw))
    def run_jobs(command,indices,python,slots):
        nonlocal active
        pending=list(indices)
        while pending or active:
            for item in list(active):
                index,proc,log=item
                if proc.poll() is not None:
                    log.close();active.remove(item)
                    if proc.returncode:raise RuntimeError(f'{command} {index} failed; see logs/{command}_{index:02d}.log')
            while pending and len(active)<slots:
                index=pending.pop(0);path=work/'logs'/f'{command}_{index:02d}.log';path.parent.mkdir(exist_ok=True)
                log=path.open('a');proc=subprocess.Popen([str(python),str(script),command,'--work',str(work),'--index',str(index)],
                    env=dict(env,PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python'),stdout=log,stderr=subprocess.STDOUT)
                active.append((index,proc,log));save(work/'processes'/f'{command}_{index:02d}.json',dict(pid=proc.pid,gpu=6,log=str(path),at=time.time()))
            status('generating' if command=='worker' else 'scoring',completed_videos=len(list((work/'generated').glob('*/complete.json'))),
                total_videos=200,completed_shards=len(list((work/'metric_shards').glob('*/SUCCESS.json'))),total_shards=16,
                active_pids=[p.pid for _,p,_ in active])
            report=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
            with (work/'gpu6.jsonl').open('a') as f:f.write(json.dumps(dict(at=time.time(),report=report.strip()))+'\n')
            time.sleep(2)
    with (work/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'PIPELINE.json',dict(pid=os.getpid(),host='node11',gpu=6,start=start,learning2listen='deferred'))
        try:
            status('preparing')
            if not (work/'protocol.json').exists():subprocess.run([str(prep),str(script),'prepare','--work',str(work)],env=env,check=True)
            p=read(work/'protocol.json')
            if not (work/'SMOKE_COMPLETE.json').exists():
                status('preflight')
                with (work/'smoke.log').open('a') as log:subprocess.run([str(dy),str(script),'smoke','--work',str(work)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            (work/'claims').mkdir(exist_ok=True)
            if not (work/'GENERATION_COMPLETE.json').exists():
                run_jobs('worker',range(p['generation_workers']),dy,p['generation_workers'])
                assert len(list((work/'generated').glob('*/complete.json')))==200
                save(work/'GENERATION_COMPLETE.json',dict(clips=200,seconds=time.time()-start,at=time.time()))
            pending=[i for i in range(p['metric_shards']) if not (work/'metric_shards'/f'{i:02d}'/'SUCCESS.json').exists()]
            run_jobs('score',pending,metric,p['scoring_workers'])
            status('merging');subprocess.run([str(metric),str(script),'merge','--work',str(work)],env=env,check=True)
            status('complete',seconds=time.time()-start)
        except Exception as e:
            for _,proc,log in active:
                if proc.poll() is None:proc.terminate()
                log.close()
            save(work/'PIPELINE_FAILURE.json',dict(error=str(e),at=time.time()));status('failed',error=str(e));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','pipeline','worker','smoke','score','merge'])
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--index',type=int,default=0);args=parser.parse_args();w=args.work.resolve()
    if args.command=='prepare':prepare(w)
    elif args.command=='pipeline':pipeline(w)
    elif args.command=='score':score(w,args.index)
    elif args.command=='merge':merge(w)
    else:worker(w,args.command=='smoke')
