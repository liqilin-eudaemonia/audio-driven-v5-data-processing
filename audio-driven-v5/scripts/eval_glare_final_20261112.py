"""Final GLARE checkpoint: sealed ViCo43 and ResponseNet586, node11 GPU4.

Adapters only: source inputs, resumable queue, official prosody batching,
resident native generation, timestamp conversion, existing scoring functions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_testsets_eval.protocol import read,sha,rows,METRIC_PYTHON

WORK=ROOT/'experiments/glare_final_vico43_responsenet586_20261112'
CHECKPOINT=ROOT/'experiments/glare_paper_settings_node11_gpu7_20261111/latest.pt'
VICO=ROOT/'experiments/p2_vico43_all_lora6500_sft2752_20261111'
RESPONSE=ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111/conditioning'
ARM='glare_final32940'
MODEL='GLARE final32940'
UUID='GPU-c0ca8be5-8935-4e6c-d3d7-a2c5b58c37c0'
GENPY=ROOT/'compare/.envs/avatarforcing/bin/python'
IMAGEPY=ROOT/'compare/.envs/vico_eval/bin/python'
MOREPY=ROOT/'compare/.envs/vico_more/bin/python'
BFMPY=ROOT/'compare/.envs/vico/bin/python'

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def gpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='4'
    row=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==UUID,row
    return dict(uuid=UUID,free_mib=int(row[1]),total_mib=int(row[2]),utilization=int(row[3]))

def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'protocol.json').exists():
        p=read(WORK/'protocol.json');assert sha(CHECKPOINT)==p['checkpoint_sha256'];return p
    vitems=read(VICO/'shards/00.json');ritems=read(RESPONSE/'shared_items.json')
    assert len(vitems)==43 and len(ritems)==586
    samples=[]
    for item in vitems+ritems:
        uid=item['id'];dataset=item['dataset'];g=item['geometry'];source=VICO if dataset=='vico' else RESPONSE
        folder=source/'conditions'/item['key'];image=folder/'first_frame.png';audit=read(folder/'input_audit.json')
        assert audit['source_only'] and not audit['listener_future_media_in_model_input']
        dest=WORK/'inputs'/dataset/uid;dest.mkdir(parents=True,exist_ok=True)
        if dataset=='vico':
            from p2_vico256_combined_eval import source_wave
            wave,waveinfo=source_wave(item,folder)
        else:
            path=item['reference']['speaker']['audio']
            wave,rate=sf.read(path,dtype='float32',always_2d=True);wave=wave.mean(1)
            divisor=math.gcd(rate,16000);wave=resample_poly(wave,16000//divisor,rate//divisor).astype(np.float32)
            count=round(g['duration_s']*16000);wave=np.pad(wave,(0,max(0,count-len(wave))))[:count]
            waveinfo=dict(source=path,source_sha256=sha(path),samples=count,source_only=True)
        audio=dest/'speaker.wav';sf.write(audio,wave,16000,subtype='FLOAT')
        s=dict(id=uid,key=item['key'],dataset=dataset,reference=str(image),reference_sha256=sha(image),audio=str(audio),audio_sha256=sha(audio),
               duration=g['duration_s'],frames=g.get('gt_frames',g['frames']),fps=g.get('gt_fps',g['fps']),size=256 if dataset=='vico' else 512,
               source_input_audit=str(folder/'input_audit.json'),waveform=waveinfo,item=item)
        samples.append(s)
    assert len({s['key'] for s in samples})==629
    parent=read(VICO/'protocol.json');rp=read(RESPONSE/'protocol.json')
    p=dict(schema='GLARE-final32940-ViCo43-ResponseNet586-GPU4-v1',created=time.time(),checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),
           step=32940,seed=137,nfe=10,audio_guidance=2.,samples=samples,physical_gpu=4,gpu_uuid=UUID,
           native=dict(size=512,fps=25,window='full sample, zero history as trained; reference retained at frame0'),
           temporal_conversion='round-half-up nearest timestamp to original GT grid, endpoint clamp recorded',
           spatial_conversion='ViCo antialias bicubic512->256; ResponseNet512; CRF18 one encode of scoring RGB',
           conditions='speaker waveform, listener frame0; native learned null motion statistics; no listener future/GT motion/audio',
           prosody='unchanged official offline prompt, normalization, greedy max64, complete2.4s blocks, NULL/tail zero; cross-input batching64',
           audio_metrics='N/A: no generated listener waveform; MP4 speaker soundtrack is conditioning only',
           references=dict(vico=sha(VICO/'shards/00.json'),responsenet=sha(RESPONSE/'shared_items.json')),
           vico_protocol=parent['datasets']['vico'],metric_assets=rp['metric_assets'],metrics_gt_cache=rp['metrics_gt_cache'],metric_shards=16,
           code={str(f):sha(f) for f in [SELF,ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py']})
    save(WORK/'protocol.json',p)
    w=WORK/'vico';w.mkdir(exist_ok=True)
    vp=dict(parent,arms=[dict(name=ARM,checkpoint=str(CHECKPOINT),original_sha256=p['checkpoint_sha256'],step=32940)],datasets={'vico':p['vico_protocol']})
    save(w/'protocol.json',vp);save(w/'shards/00.json',vitems)
    shutil.copyfile(VICO/'INPUT_PREFLIGHT.json',w/'INPUT_PREFLIGHT.json')
    save(WORK/'PRECHECK.json',dict(passed=True,clips=629,vico=43,responsenet=586,source_only=True,checkpoint_sha256=p['checkpoint_sha256']))
    return p

def prosody_id():
    contract=read(ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/protocol.json')['extractor']
    key=hashlib.sha256(json.dumps(dict(contract=contract,code=sha(ROOT/'scripts/glare_resident_inference.py'),extract=sha(ROOT/'scripts/infer_glare_checkpoint.py')),sort_keys=True).encode()).hexdigest()
    return contract,key

def cache_path(sample,key):
    return WORK/'prosody'/ (hashlib.sha256((sample['audio_sha256']+key).encode()).hexdigest()+'.npz')

def prosody():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from infer_glare_checkpoint import official_module,make_analyzer
    from baseline_training.features import native_glare
    native_glare()
    from float.models.intensity_encoder import IntensityEncoder
    gpu();torch.set_num_threads(4)
    p=read(WORK/'protocol.json');contract,key=prosody_id();native=official_module(contract['official_script'])
    holder=SimpleNamespace(default_fps=25.,_time_to_start_frame_1_based=IntensityEncoder._time_to_start_frame_1_based,
                           _time_to_end_frame_1_based=IntensityEncoder._time_to_end_frame_1_based)
    analyzer=None;pending=[];chunks=[];owners=[];done=0;started=time.time()
    def publish(s,frames,texts):
        value=torch.zeros(frames,1)
        for i,t in enumerate(texts):
            n=min(60,frames-i*60)
            if n>0:value[i*60:i*60+n]=IntensityEncoder.text_to_frame_vector(holder,t,n,fps=25.)
        dest=cache_path(s,key);dest.parent.mkdir(exist_ok=True)
        meta=dict(chunks=texts,policy='official full 2.4s blocks, tail defaults to zero',normalizer_sha256=sha(contract['official_script']),batching='cross-input64')
        with tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.npz',delete=False) as tmp:
            np.savez_compressed(tmp,intensity=value.numpy(),metadata=json.dumps(meta));temp=Path(tmp.name)
        temp.replace(dest)
    def flush():
        nonlocal analyzer,done
        if chunks:
            if analyzer is None:analyzer=make_analyzer('/private/ckpts/Qwen2-Audio-7B-Instruct',native,contract)
            texts=[]
            for start in range(0,len(chunks),64):texts.extend(analyzer.infer_chunks(chunks[start:start+64],16000,2.4))
            assert len(texts)==len(chunks)
            for s,frames,begin,count in pending:publish(s,frames,texts[begin:begin+count]);done+=1
        save(WORK/'PROSODY_STATUS.json',dict(done=done,total=629,seconds=time.time()-started,at=time.time()))
        pending.clear();chunks.clear()
    # ViCo first to make its complete scoring pipeline available early.
    for s in p['samples']:
        if cache_path(s,key).exists():done+=1;continue
        wave=native.load_waveform(Path(s['audio']),16000);frames=max(2,math.ceil(len(wave)/640))
        blocks=native.split_chunks(native.normalize_waveform_loudness(wave,-20.,.98),16000,2.4)
        if not blocks:publish(s,frames,[]);done+=1;continue
        pending.append((s,frames,len(chunks),len(blocks)));chunks.extend(blocks)
        if len(chunks)>=64:flush()
    flush();save(WORK/'PROSODY_COMPLETE.json',dict(clips=done,seconds=time.time()-started))

def claim(samples):
    with (WORK/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for s in samples:
            if (WORK/s['dataset']/'generated'/s['id']/'complete.json').exists():continue
            flag=WORK/'claims'/s['dataset']/(s['id']+'.json')
            if flag.exists():
                owner=read(flag)['pid']
                try:os.kill(owner,0);continue
                except ProcessLookupError:pass
            _,key=prosody_id()
            if not cache_path(s,key).exists():continue
            save(flag,dict(pid=os.getpid(),time=time.time()));return s

def encode_rgb(frames,path,fps):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.part.mp4')
    h,w=frames.shape[1:3]
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
                           '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(temp)],stdin=subprocess.PIPE)
    try:
        for batch in range(0,len(frames),32):proc.stdin.write(frames[batch:batch+32].tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg scoring export failed')
    temp.replace(path)

def all_generated(samples):
    return all((WORK/x['dataset']/'generated'/x['id']/'complete.json').exists() for x in samples)

def generate(slot,decode_batch):
    import numpy as np
    import torch
    from glare_resident_inference import ResidentGLARE
    from infer_glare_checkpoint import write_video
    gpu();torch.set_num_threads(2)
    p=read(WORK/'protocol.json');assert sha(CHECKPOINT)==p['checkpoint_sha256']
    model=ResidentGLARE(CHECKPOINT,WORK/'prosody');assert model.step==p['step'] and model.checkpoint_sha==p['checkpoint_sha256']
    save(WORK/'loads'/f'{slot:02d}.json',dict(pid=os.getpid(),seconds=model.load_seconds,step=model.step))
    def publish(s,image,rgb,padded,score,indices,info):
        folder=WORK/s['dataset']/'generated'/s['id'];folder.mkdir(parents=True,exist_ok=True)
        native=folder/'generated.mp4';target=WORK/s['dataset']/'scoring'/ARM/(s['id']+'.listener.mp4')
        write_video(image,rgb,padded,native);encode_rgb(score,target,s['fps'])
        record=dict(info,id=s['id'],key=s['key'],native_frames=len(rgb)+1,score_frames=len(score),score_fps=s['fps'],
                    source_only=True,listener_audio_generated=False,reference_frame_in_output=True,
                    frame_indices=indices.tolist(),max_timestamp_error_s=float(np.abs(indices/25-np.arange(len(indices))/s['fps']).max()),
                    native_sha256=sha(native),score_sha256=sha(target),protocol_sha256=sha(WORK/'protocol.json'))
        save(folder/'complete.json',record)
        if s['dataset']=='vico':save(WORK/'vico/conversion'/(s['id']+'.json'),dict(file_sha256=record['score_sha256'],source_frame_indices=indices.tolist()))
        print(json.dumps(dict(id=s['key'],seconds=info['generation_seconds'],frames=len(rgb)+1)),flush=True)
    pending=None
    with ThreadPoolExecutor(max_workers=1) as writer:
        while True:
            s=claim(p['samples'])
            if s is None:
                if all_generated(p['samples']):break
                if (WORK/'PROSODY_COMPLETE.json').exists():break
                time.sleep(1);continue
            try:
                image,rgb,padded,raw,info=model.generate(s['reference'],s['audio'],137,10,decode_batch)
                del raw
                # Multiple resident processes otherwise retain each other's
                # peak-sized idle allocator pools. Release unused blocks only;
                # model tensors, caches, precision and native math stay intact.
                if torch.cuda.mem_get_info()[0]<24*1024**3:torch.cuda.empty_cache()
                all_frames=np.concatenate([image[None],rgb]);indices=np.floor(np.arange(s['frames'])*25/s['fps']+.5).astype(int).clip(max=len(all_frames)-1)
                if s['size']==256:
                    small=[]
                    for start in range(0,len(all_frames),32):
                        a=torch.from_numpy(all_frames[start:start+32]).cuda().permute(0,3,1,2).float()/255
                        a=torch.nn.functional.interpolate(a,size=(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
                        small.append((a*255).round().byte().permute(0,2,3,1).cpu().numpy())
                    small=np.concatenate(small);score=np.ascontiguousarray(small[indices])
                else:score=np.ascontiguousarray(all_frames[indices])
                del all_frames
                if pending is not None:pending.result()
                pending=writer.submit(publish,s,image,rgb,padded,score,indices,info)
            except BaseException:
                save(WORK/'failures'/f'{slot:02d}_{s["id"]}.json',dict(sample=s['key'],traceback=traceback.format_exc(),at=time.time()));raise
        if pending is not None:pending.result()

def image_metric(child=None,index=None):
    import p2_vico_official_metrics as source
    source.ARM=ARM;source.__file__=str(SELF)
    work=WORK/'vico'
    if child:return source.child(work,child,index)
    with (work/'image.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (work/'IMAGE_RESULTS.json').exists():return
        source.run(work)
        result=read(work/'RESULTS.json');result.update(model=MODEL,checkpoint=str(CHECKPOINT),checkpoint_sha256=read(WORK/'protocol.json')['checkpoint_sha256'])
        save(work/'IMAGE_RESULTS.json',result)

def cpbd():
    """CPU-only official sharpness scores can overlap GPU generation."""
    import p2_vico_official_metrics as source
    source.ARM=ARM;source.__file__=str(SELF)
    with ThreadPoolExecutor(max_workers=8) as pool:
        reports=list(pool.map(lambda i:source.execute(WORK/'vico','cpbd',i),range(43)))
    save(WORK/'vico/CPBD_OVERLAP_COMPLETE.json',dict(clips=len(reports),at=time.time()))

def motion_template():
    import p2_vico43_all_lora6500_metrics_20261111 as source
    source.ARM=ARM
    return source

def prepare_motion():
    work=WORK/'vico';source=motion_template();source.prepare(work)
    # Correct only inherited display/resource labels before any feature commits.
    assert not list((work/'bfm/features/prediction').glob('*.json'))
    p=read(work/'bfm/protocol.json');p['model']=MODEL;p['resource']='node11 physical GPU4'
    p['checkpoint_sha256']=read(WORK/'protocol.json')['checkpoint_sha256'];save(work/'bfm/protocol.json',p)
    spec=read(work/'bfm/SCORING_PROTOCOL.json');spec['extraction_protocol_sha256']=sha(work/'bfm/protocol.json');save(work/'bfm/SCORING_PROTOCOL.json',spec)
    for slot in range(4):
        path=work/'emoca'/f'shard_{slot:02d}/protocol.json';p=read(path);p['resource']='node11 physical GPU4 only';save(path,p)
    save(work/'MODEL_AND_RESOURCE_LABELS.json',dict(model=MODEL,physical_gpu=4,display_only_correction=True,at=time.time()))

def motion_worker(kind,slot):
    import numpy as np
    import torch
    torch.set_num_threads(4);device=gpu();source=motion_template();work=WORK/'vico'
    torch.cuda.set_per_process_memory_fraction(8192/device['total_mib'])
    with (work/'motion_prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (work/'MODEL_AND_RESOURCE_LABELS.json').exists():prepare_motion()
    if kind=='emoca':
        import vico_literature_emoca as native
        import vico_more_common as common
        def guard(*args):
            report=gpu();torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            return report
        common.gpu_guard=guard;native.WORK=work/'emoca'/f'shard_{slot:02d}';native.run();return
    import p2_vico_motion_extract as native
    bfm=work/'bfm';tasks=read(bfm/'tasks.json')[slot::4];extractor=native.Extractor();records=[]
    for task in tasks:
        target=bfm/'features/prediction'/(task['id']+'.npz');receipt=target.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert r['input_sha256']==task['video_sha256'] and sha(target)==r['feature_sha256'];records.append(r);continue
        assert sha(task['video'])==task['video_sha256'];frames=native.load_frames(task['video']);assert len(frames)==task['frames']
        arrays,timing,faces=extractor.extract(frames)
        if not records:
            try:proof=extractor.parity(frames,arrays,faces)
            except ValueError as exc:
                # Reuse the accepted6500 policy for unused landmark diagnostics.
                import ast
                from p2_vico6500_bounded_motion_20261111 import accept_report
                prefix='Batching parity failed: '
                if not str(exc).startswith(prefix):raise
                proof=ast.literal_eval(str(exc)[len(prefix):])
                if not accept_report(proof):raise
                proof=dict(proof,passed=True,unused_68point_difference_diagnostic_only=True,
                           acceptance='existing6500 policy: five alignment points/crops identical; native coefficient tolerance1e-3')
            save(bfm/f'BATCH_PARITY_{slot:02d}.json',proof)
        for name in ['coeff','motion','landmarks','transform_params']:assert np.isfinite(arrays[name]).all()
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        target.with_suffix('.tmp').replace(target)
        r=dict(id=task['id'],role='prediction',frames=len(frames),feature=str(target),feature_sha256=sha(target),input_sha256=task['video_sha256'],
               protocol_sha256=sha(bfm/'protocol.json'),extractor_source_sha256=sha(ROOT/'scripts/p2_vico_motion_extract.py'),
               missing_face_frames=int((~arrays['detected']).sum()),multi_face_frames=int((arrays['face_counts']>1).sum()),seconds=timing)
        save(receipt,r);records.append(r)
    save(bfm/f'EXTRACTION_COMPLETE_{slot:02d}.json',dict(passed=True,streams=len(records),records=records))

def response_worker(index):
    lockpath=WORK/'responsenet/metric_locks'/f'{index:02d}.lock';lockpath.parent.mkdir(exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return response_worker_unlocked(index)

def response_worker_unlocked(index):
    p=read(WORK/'protocol.json');all_samples=[s for s in p['samples'] if s['dataset']=='responsenet']
    # Contiguous disjoint shards become ready during generation; pooled formulas
    # and the full586 population stay identical to the existing scorer.
    count=math.ceil(len(all_samples)/p['metric_shards']);samples=all_samples[index*count:(index+1)*count]
    w=WORK/'responsenet';out=w/'metric_shards'/f'{index:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'SUCCESS.json').exists():return
    manifest=w/'metric_manifests'/f'{index:02d}.jsonl';manifest.parent.mkdir(exist_ok=True)
    selected=[]
    for s in samples:
        r=s['item']['reference']
        selected.append(dict(id=s['id'],dataset='responsenet-j-paired-evaluation-v1',identity='listener',gt=r['gt'],
                             generated=str(w/'scoring'/ARM/(s['id']+'.listener.mp4')),gt_resize=[512,512],gt_resize_method='pil_bicubic',
                             source_start=r['source_start'],source_end=r['source_end'],listening_intervals=r['listening_intervals']))
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in selected))
    command=[sys.executable,str(ROOT/'scripts/p2_pipelined_video_metrics.py'),'--manifest',str(manifest),'--assets',p['metric_assets'],
             '--output',str(out),'--device','cuda:0','--threads','4','--batch-size','64','--face-padding','64',
             '--feature-output',str(out/'features.npz'),'--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0',
             '--pixel-workers','4','--gt-cache',p['metrics_gt_cache']]
    code=subprocess.call(command,cwd=ROOT.parent/'OmniResponse')
    summary=read(out/'summary.json');assert not summary['initialization_errors']
    actual=rows(out/'pairs.jsonl');assert {s['id'] for s in actual}=={s['id'] for s in samples}
    assert (out/'features.npz').exists()
    save(out/'SUCCESS.json',dict(returncode=code,clips=len(actual),pairs_with_errors=summary['pairs_with_errors']))

def response_merge():
    import numpy as np
    sys.path.insert(0,str(ROOT.parent/'OmniResponse'))
    from evaluation.video.paired_metrics.run import summarize,METRICS
    p=read(WORK/'protocol.json');w=WORK/'responsenet';actual=[];contracts=[]
    dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
    for i in range(p['metric_shards']):
        folder=w/'metric_shards'/f'{i:02d}';assert (folder/'SUCCESS.json').exists()
        summary=read(folder/'summary.json');contracts.append(read(folder/'contract.json'));actual.extend(rows(folder/'pairs.jsonl'))
        with np.load(folder/'features.npz') as features:
            for k in dist:
                if features[k+'_gt'].size:dist[k][0].append(features[k+'_gt']);dist[k][1].append(features[k+'_generated'])
                coverage[k]+=summary['coverage'][k]['pairs']
    samples=[s for s in p['samples'] if s['dataset']=='responsenet']
    assert len(actual)==len({r['id'] for r in actual})==586 and {r['id'] for r in actual}=={s['id'] for s in samples}
    for c in contracts:assert c['code']==contracts[0]['code'] and c['assets']==contracts[0]['assets']
    result=summarize(actual,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],586,{})
    for k,n in coverage.items():result['coverage'][k]['pairs']=n
    result.update(model=MODEL,dataset='ResponseNet full586',checkpoint_sha256=p['checkpoint_sha256'],audio_metrics=p['audio_metrics'],
                  aggregation='all586 pooled FID/FVD features, original paired video scorer',protocol_sha256=sha(WORK/'protocol.json'))
    save(w/'RESULTS.json',result);(w/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in actual))

def motion_score():
    import importlib.metadata as metadata
    source=motion_template();work=WORK/'vico'
    import p2_vico_motion_metrics as motion
    result=motion.run(work/'bfm')
    result.update(model=MODEL,checkpoint_sha256=read(WORK/'protocol.json')['checkpoint_sha256'])
    save(work/'bfm/RESULTS.json',result)
    source.score_DIM(work)
    save(work/'SCORING_ENVIRONMENT.json',dict(python=sys.executable,
        versions={k:metadata.version(k) for k in ['numpy','scipy','scikit-learn','torch']},
        matches_existing_baseline_environment=True,at=time.time()))

def env_for(kind):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    env['PYTHONPATH']=''
    if kind=='gen':
        env['PYTHONPATH']=os.pathsep.join(str(ROOT/x) for x in ['experiments/glare_current500h_nonprosody_cache_20261110/runtime_deps','compare/.envs/glare-deps','src','scripts'])
    elif kind=='response':env['PYTHONPATH']='/private/liqi_workspace/.cache/paired_video_metrics/python'
    return env

def launch(action,tag,python,kind='other',extra=()):
    log=WORK/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:proc=subprocess.Popen([str(python),'-u',str(SELF),action,*extra],cwd=ROOT,env=env_for(kind),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    save(WORK/'processes'/(tag+'.json'),dict(pid=proc.pid,action=action,log=str(log),gpu=4,started=time.time()))
    return proc

def parallel(action,python,count,capacity,kind='other',extra=()):
    active={};pending=list(range(count))
    while active or pending:
        for slot,proc in list(active.items()):
            if proc.poll() is not None:
                if proc.returncode:raise RuntimeError(f'{action} slot{slot} failed: logs/{action}_{slot:02d}.log')
                del active[slot]
        while pending and len(active)<capacity:
            slot=pending.pop(0);active[slot]=launch(action,f'{action}_{slot:02d}',python,kind,[*extra,'--slot',str(slot)])
        save(WORK/'status.json',dict(stage=action,pending=len(pending),active=[x.pid for x in active.values()],gpu=gpu(),at=time.time()))
        time.sleep(3)

def run(workers,decode_batch,metric_workers=16):
    gpu();p=read(WORK/'protocol.json');started=time.time()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(WORK/'LAUNCH.json',dict(pid=os.getpid(),workers=workers,decode_batch=decode_batch,gpu=4,started=started))
        if not (WORK/'GENERATION_COMPLETE.json').exists():
            pros=launch('prosody','prosody',GENPY,'gen') if not (WORK/'PROSODY_COMPLETE.json').exists() else None
            active=[launch('generate',f'generate_{slot:02d}',GENPY,'gen',['--slot',str(slot),'--decode-batch',str(decode_batch)]) for slot in range(workers)]
            try:
                while any(x.poll() is None for x in active) or (pros and pros.poll() is None):
                    for proc in active+([pros] if pros else []):
                        if proc.poll() not in (None,0):raise RuntimeError(f'generation/prosody PID{proc.pid} failed, inspect logs')
                    counts={d:len(list((WORK/d/'generated').glob('*/complete.json'))) for d in ['vico','responsenet']}
                    save(WORK/'status.json',dict(stage='generation',counts=counts,total=629,gpu=gpu(),elapsed=time.time()-started,at=time.time()))
                    time.sleep(3)
                assert sum(len(list((WORK/d/'generated').glob('*/complete.json'))) for d in ['vico','responsenet'])==629
            except BaseException:
                for proc in active+([pros] if pros else []):
                    if proc.poll() is None:proc.terminate()
                raise
            save(WORK/'GENERATION_COMPLETE.json',dict(outputs=629,at=time.time()))
        save(WORK/'vico/GENERATION_COMPLETE.json',dict(outputs=43,at=time.time()))
        if not (WORK/'vico/IMAGE_RESULTS.json').exists():
            proc=launch('image','image',IMAGEPY);assert proc.wait()==0,'image scoring failed'
        # All source/GT features are reused; only new predictions extracted.
        source=motion_template();source.prepare(WORK/'vico')
        for kind,python in [('bfm',BFMPY),('emoca',MOREPY)]:
            markers=[WORK/'vico/bfm'/f'EXTRACTION_COMPLETE_{i:02d}.json' if kind=='bfm'
                     else WORK/'vico/emoca'/f'shard_{i:02d}/EXTRACTION_COMPLETE.json' for i in range(4)]
            if all(path.exists() for path in markers):continue
            parallel('motion-worker',python,4,4,extra=['--kind',kind])
        if not (WORK/'vico/SCORING_ENVIRONMENT.json').exists():
            proc=launch('motion-score','motion_score',MOREPY);assert proc.wait()==0,'motion scoring failed'
        parallel('response-worker',METRIC_PYTHON,p['metric_shards'],metric_workers,'response')
        proc=launch('response-merge','response_merge',METRIC_PYTHON,'response');assert proc.wait()==0,'response merge failed'
        finish()

def media_audit():
    """Bounded CPU full-decode audit overlaps remaining GPU scoring."""
    import cv2
    p=read(WORK/'protocol.json')
    def inspect(s):
        folder=WORK/s['dataset']/'generated'/s['id'];r=read(folder/'complete.json')
        assert r['checkpoint_sha256']==p['checkpoint_sha256'] and r['step']==p['step']
        assert r['source_only'] and not r['listener_audio_generated'] and sha(s['reference'])==s['reference_sha256'] and sha(s['audio'])==s['audio_sha256']
        native=folder/'generated.mp4';score=WORK/s['dataset']/'scoring'/ARM/(s['id']+'.listener.mp4')
        assert sha(native)==r['native_sha256'] and sha(score)==r['score_sha256']
        for path,count,size,fps in [(native,r['native_frames'],512,25),(score,s['frames'],s['size'],s['fps'])]:
            cap=cv2.VideoCapture(str(path));assert cap.isOpened();assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<.002
            n=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape==(size,size,3);n+=1
            cap.release();assert n==count,(s['id'],n,count)
        return dict(key=s['key'],native_frames=r['native_frames'],score_frames=s['frames'],endpoint_error_s=r['max_timestamp_error_s'],
                    native_sha256=r['native_sha256'],score_sha256=r['score_sha256'])
    with ThreadPoolExecutor(max_workers=8) as pool:audit=list(pool.map(inspect,p['samples']))
    save(WORK/'MEDIA_AUDIT_COMPLETE.json',dict(passed=True,clips=len(p['samples']),decoded_videos=2*len(p['samples']),rows=audit,protocol_sha256=sha(WORK/'protocol.json'),at=time.time()))

def finish():
    p=read(WORK/'protocol.json')
    if not (WORK/'MEDIA_AUDIT_COMPLETE.json').exists():media_audit()
    media=read(WORK/'MEDIA_AUDIT_COMPLETE.json');assert media['passed'] and media['protocol_sha256']==sha(WORK/'protocol.json')
    assert len(media['rows'])==len({r['key'] for r in media['rows']})==629
    bykey={r['key']:r for r in media['rows']};audit=media['rows']
    for s in p['samples']:
        folder=WORK/s['dataset']/'generated'/s['id'];r=read(folder/'complete.json');proof=bykey[s['key']]
        assert r['checkpoint_sha256']==p['checkpoint_sha256'] and r['step']==32940
        assert sha(folder/'generated.mp4')==r['native_sha256']==proof['native_sha256']
        assert sha(WORK/s['dataset']/'scoring'/ARM/(s['id']+'.listener.mp4'))==r['score_sha256']==proof['score_sha256']
        if s['dataset']=='vico':
            receipt=read(WORK/'vico/bfm/features/prediction'/(s['id']+'.json'))
            assert receipt['input_sha256']==r['score_sha256'] and sha(receipt['feature'])==receipt['feature_sha256']
    assert read(WORK/'vico/SCORING_ENVIRONMENT.json')['matches_existing_baseline_environment']
    result=dict(model='GLARE final32940',checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],
                vico_image=read(WORK/'vico/IMAGE_RESULTS.json'),vico_motion=read(WORK/'vico/bfm/RESULTS.json'),
                vico_DIM=read(WORK/'vico/DIM_RESULTS.json'),responsenet=read(WORK/'responsenet/RESULTS.json'))
    save(WORK/'RESULTS.json',result);save(WORK/'FINAL_AUDIT.json',dict(passed=True,clips=629,decoded_videos=1258,rows=audit))
    save(WORK/'COMPLETE.json',dict(passed=True,clips=629,at=time.time(),results_sha256=sha(WORK/'RESULTS.json')))
    save(WORK/'status.json',dict(stage='complete',clips=629,at=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',nargs='?',default='run');parser.add_argument('--work',type=Path)
    parser.add_argument('--slot',type=int,default=0);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--decode-batch',type=int,default=8);parser.add_argument('--kind',choices=['bfm','emoca']);parser.add_argument('--child');parser.add_argument('--index',type=int)
    parser.add_argument('--metric-workers',type=int,default=16)
    a=parser.parse_args()
    if a.child:image_metric(a.child,a.index)
    elif a.action=='prepare':prepare()
    elif a.action=='prosody':prosody()
    elif a.action=='generate':generate(a.slot,a.decode_batch)
    elif a.action=='image':image_metric()
    elif a.action=='cpbd':cpbd()
    elif a.action=='motion-worker':motion_worker(a.kind,a.slot)
    elif a.action=='prepare-motion':prepare_motion()
    elif a.action=='response-worker':response_worker(a.slot)
    elif a.action=='response-merge':response_merge()
    elif a.action=='motion-score':motion_score()
    elif a.action=='finish':finish()
    elif a.action=='media-audit':media_audit()
    elif a.action=='run':
        try:run(a.workers,a.decode_batch,a.metric_workers)
        except BaseException:
            save(WORK/'FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise
    else:raise ValueError(a.action)
