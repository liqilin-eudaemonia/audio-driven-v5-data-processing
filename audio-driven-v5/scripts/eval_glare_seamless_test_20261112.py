"""Stream sealed Seamless test6624 through the verified final GLARE/GPU4 path."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import eval_glare_final_20261112 as core
from p2_testsets_eval.protocol import read,rows,sha,METRIC_PYTHON,geometry

SELF=Path(__file__).resolve()
WORK=ROOT/'experiments/glare_final_seamless_test6624_20261112'
PREPARED=ROOT.parent/'OmniResponse/evaluation/seamless/prepared_20261102'
QUICK=ROOT/'experiments/seamless_quick200_latest_20261105/prepared_200'
HISTORY=ROOT/'experiments/p2_compare_seamless200_20261106/dystream'
PARENT=ROOT/'experiments/p2_seamless200_tenway_24gpu_20261105'
core.WORK=WORK;core.SELF=SELF
core.all_generated=lambda samples:(WORK/'GENERATION_COMPLETE.json').exists()
save=core.save

def configure():
    WORK.mkdir(parents=True,exist_ok=True)
    if (WORK/'protocol.json').exists():
        p=read(WORK/'protocol.json');assert p['checkpoint_sha256']==sha(core.CHECKPOINT);return p
    references=rows(PREPARED/'references.jsonl');assert len(references)==len({r['id'] for r in references})==6624
    assert sha(PREPARED/'references.jsonl')==read(PREPARED/'seal.json')['references_sha256']
    quick=rows(QUICK/'references.jsonl');quickids=[r['id'] for r in quick];byid={r['id']:r for r in references}
    assert len(set(quickids))==200 and set(quickids)<=set(byid)
    quickby={r['id']:r for r in quick};ordered=[byid[i] for i in quickids]+[r for r in references if r['id'] not in quickby]
    samples=[]
    for r in ordered:
        uid=r['id'];folder=WORK/'inputs'/uid;g=geometry(r,'seamless')
        if uid in quickby:
            a,b=r['gt_reference'],quickby[uid]['gt_reference']
            for key in ['sample_id','row_sha256','crop_config_sha256','reader_contract']:assert a[key]==b[key],(uid,key)
            assert r['frames']==quickby[uid]['frames'] and r['duration_s']==quickby[uid]['duration_s']
        samples.append(dict(id=uid,key='seamless/'+uid,dataset='seamless',reference=str(folder/'first_frame.png'),audio=str(folder/'speaker.wav'),
            ready=str(folder/'READY.json'),duration=r['duration_s'],frames=r['frames'],fps=25,size=512,quick200=uid in quickby,
            item=dict(id=uid,key='seamless/'+uid,dataset='seamless',reference=r,geometry=g),
            evaluation_reference=quickby.get(uid,r)))
    blocks=[samples[i:i+25] for i in range(0,200,25)]+[samples[i:i+32] for i in range(200,6624,32)]
    assert sum(map(len,blocks))==6624 and len(blocks)==209
    parent=read(PARENT/'protocol.json')
    p=dict(schema='GLARE-final32940-Seamless-test6624-GPU4-v1',created=time.time(),checkpoint=str(core.CHECKPOINT),checkpoint_sha256=sha(core.CHECKPOINT),
           step=32940,seed=137,nfe=10,audio_guidance=2.,physical_gpu=4,gpu_uuid=core.UUID,total=6624,samples=samples,
           references=str(PREPARED/'references.jsonl'),references_sha256=sha(PREPARED/'references.jsonl'),seal_sha256=sha(PREPARED/'seal.json'),
           quick200_ids=quickids,quick200_references_sha256=sha(QUICK/'references.jsonl'),total_duration_s=sum(s['duration'] for s in samples),
           metric_assets=parent['metric_assets'],metrics_gt_cache=parent['metrics_gt_cache'],metrics_decoded_gt_cache=parent['metrics_decoded_gt_cache'],
           metric_shards=len(blocks),metric_batch_size=64,face_padding=0,
           conditions='same official source-only reader: speaker waveform and single listener reference; no target caption/audio/future frames',
           prosody='same official offline prompt/normalizer/greedy64/complete2.4s blocks/NULL zero; cross-input batch64',
           native=dict(size=512,fps=25,seed=137,nfe=10,speaker_CFG=2.,history=0),
           audio_metrics='N/A: GLARE generates no listener waveform; speaker soundtrack is conditioning only',
           acceleration='16 bounded CPU input readers; cached200 inputs; 4 resident GLARE workers/decode8; official prosody batch64; ready32-clip scoring blocks, up to16 scorers; CPU read/pixel prefetch and GT features reused',
           storage='Reuse existing200 decoded GT only; remaining6424 decoded on demand without full RGB persistence',
           previous_completed=str(ROOT/'experiments/glare_final_vico43_responsenet586_20261112'),
           code={str(f):sha(f) for f in [SELF,ROOT/'scripts/eval_glare_final_20261112.py',ROOT/'scripts/glare_resident_inference.py',ROOT/'scripts/infer_glare_checkpoint.py']})
    save(WORK/'protocol.json',p)
    for i,block in enumerate(blocks):save(WORK/'blocks'/f'{i:03d}.json',block)
    return p

def prepare_inputs():
    import numpy as np
    import soundfile as sf
    from p2_testsets_eval.runtime import conditioning
    p=read(WORK/'protocol.json');old={s['id']:s for s in read(HISTORY/'protocol.json')['samples']}
    done=0;reused=0;started=time.time()
    def one(s):
        ready=Path(s['ready'])
        if ready.exists():return read(ready)['reused']
        folder=ready.parent;folder.mkdir(parents=True,exist_ok=True)
        if s['id'] in old:
            cached=old[s['id']];ref=s['item']['reference'];a,b=ref['gt_reference'],cached['reference']['gt_reference']
            for key in ['sample_id','row_sha256','crop_config_sha256','reader_contract']:assert a[key]==b[key]
            assert sha(cached['image'])==cached['image_sha256']
            image=Path(cached['image']);audio=Path(cached['speaker']);audit=read(cached['source_input_audit'])
            wave,rate=sf.read(audio,dtype='float32');assert rate==16000 and len(wave)==round(s['duration']*16000)
            assert __import__('hashlib').sha256(wave.astype('<f4').tobytes()).hexdigest()==cached['speaker_waveform_sha256']
            assert audit['source_only'] and not audit['listener_future_media_in_model_input']
            prepared=dict(s,reference=str(image),audio=str(audio),source_input_audit=cached['source_input_audit']);reuse=True
        else:
            value=conditioning(s['item'],folder)
            audio=Path(s['audio']);temp=audio.with_suffix('.tmp.wav');sf.write(temp,value['audio'],16000,subtype='FLOAT');temp.replace(audio)
            image=Path(s['reference']);prepared=dict(s,source_input_audit=str(folder/'input_audit.json'));reuse=False
        prepared.update(audio_sha256=sha(audio),reference_sha256=sha(image))
        save(ready,dict(sample=prepared,reused=reuse,source_only=True,protocol_sha256=sha(WORK/'protocol.json')))
        return reuse
    pending=iter(p['samples'])
    with ThreadPoolExecutor(max_workers=16) as pool:
        active={}
        def fill():
            while len(active)<32:
                s=next(pending,None)
                if s is None:break
                active[pool.submit(one,s)]=(s,0)
        fill()
        while active:
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                sample,attempt=active.pop(future)
                try:reused+=bool(future.result());done+=1
                except Exception:
                    if attempt<2:active[pool.submit(one,sample)]=(sample,attempt+1);continue
                    save(WORK/'INPUT_FAILURE.json',dict(id=sample['id'],traceback=traceback.format_exc()));raise
            fill();save(WORK/'INPUT_STATUS.json',dict(done=done,total=6624,reused=reused,seconds=time.time()-started,at=time.time()))
    save(WORK/'INPUT_COMPLETE.json',dict(clips=done,reused=reused,seconds=time.time()-started))

def ready_samples(p):
    for s in p['samples']:
        ready=Path(s['ready'])
        if ready.exists():yield read(ready)['sample']

def prosody():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from infer_glare_checkpoint import official_module,make_analyzer
    from baseline_training.features import native_glare
    native_glare()
    from float.models.intensity_encoder import IntensityEncoder
    core.gpu();torch.set_num_threads(4)
    p=read(WORK/'protocol.json');contract,key=core.prosody_id();native=official_module(contract['official_script']);analyzer=None
    holder=SimpleNamespace(default_fps=25.,_time_to_start_frame_1_based=IntensityEncoder._time_to_start_frame_1_based,
                           _time_to_end_frame_1_based=IntensityEncoder._time_to_end_frame_1_based)
    done=set();pending=[];chunks=[];started=time.time();last_flush=time.time()
    def publish(s,frames,texts):
        value=torch.zeros(frames,1)
        for i,text in enumerate(texts):
            n=min(60,frames-i*60)
            if n>0:value[i*60:i*60+n]=IntensityEncoder.text_to_frame_vector(holder,text,n,fps=25.)
        dest=core.cache_path(s,key);dest.parent.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent,suffix='.npz',delete=False) as tmp:
            np.savez_compressed(tmp,intensity=value.numpy(),metadata=json.dumps(dict(chunks=texts,policy='official full2.4s blocks; tail/NULL zero',normalizer_sha256=sha(contract['official_script']))));temp=Path(tmp.name)
        temp.replace(dest);done.add(s['id']);enqueue(s)
    def flush():
        nonlocal analyzer,last_flush
        if chunks:
            if analyzer is None:analyzer=make_analyzer('/private/ckpts/Qwen2-Audio-7B-Instruct',native,contract)
            texts=[]
            for start in range(0,len(chunks),64):texts.extend(analyzer.infer_chunks(chunks[start:start+64],16000,2.4))
            for s,frames,begin,count in pending:publish(s,frames,texts[begin:begin+count])
        pending.clear();chunks.clear();last_flush=time.time()
        if torch.cuda.mem_get_info()[0]<24*1024**3:torch.cuda.empty_cache()
        save(WORK/'PROSODY_STATUS.json',dict(done=len(done),total=6624,seconds=time.time()-started,at=time.time()))
    while len(done)<6624:
        discovered=False
        for s in ready_samples(p):
            if s['id'] in done or any(x[0]['id']==s['id'] for x in pending):continue
            if core.cache_path(s,key).exists():done.add(s['id']);enqueue(s);continue
            wave=native.load_waveform(Path(s['audio']),16000);frames=max(2,math.ceil(len(wave)/640))
            blocks=native.split_chunks(native.normalize_waveform_loudness(wave,-20.,.98),16000,2.4);discovered=True
            if not blocks:publish(s,frames,[]);continue
            pending.append((s,frames,len(chunks),len(blocks)));chunks.extend(blocks)
            if len(chunks)>=64:flush()
        if pending and (time.time()-last_flush>5 or (WORK/'INPUT_COMPLETE.json').exists()):flush()
        if not discovered:time.sleep(1)
    flush();save(WORK/'PROSODY_COMPLETE.json',dict(clips=len(done),seconds=time.time()-started))

def enqueue(s):
    # Append only after the official intensity cache is atomically published.
    with (WORK/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (WORK/'ready_queue.jsonl').open('a') as stream:stream.write(json.dumps(dict(id=s['id'],ready=s['ready']))+'\n')

def claim(samples):
    with (WORK/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=WORK/'ready_queue.jsonl'
        if not queue.exists():return None
        cursor=WORK/'queue_cursor.json';offset=read(cursor)['offset'] if cursor.exists() else 0
        with queue.open() as stream:
            stream.seek(offset)
            while line:=stream.readline():
                s=json.loads(line);save(cursor,dict(offset=stream.tell()))
                if (WORK/'seamless/generated'/s['id']/'complete.json').exists():continue
                flag=WORK/'claims/seamless'/(s['id']+'.json')
                if flag.exists():
                    try:os.kill(read(flag)['pid'],0);continue
                    except ProcessLookupError:pass
                job=read(s['ready'])['sample'];save(flag,dict(pid=os.getpid(),at=time.time()));return job

def score(slot):
    if (WORK/'PAUSED.json').exists():raise RuntimeError('This evaluation was paused by the user; keep existing results')
    p=read(WORK/'protocol.json');block=read(WORK/'blocks'/f'{slot:03d}.json')
    folder=WORK/'metrics'/f'{slot:03d}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'SUCCESS.json').exists():return
        manifest=folder/'manifest.jsonl';paired=[]
        for s in block:
            r=s['evaluation_reference'];target=WORK/'seamless/scoring'/core.ARM/(s['id']+'.listener.mp4')
            assert target.exists() and (WORK/'seamless/generated'/s['id']/'complete.json').exists()
            paired.append(dict(id=s['id'],dataset='seamless',identity='listener',gt_reference=r['gt_reference'],generated=str(target),listening_intervals=r['listening_intervals']))
        manifest.write_text(''.join(json.dumps(x)+'\n' for x in paired))
        cmd=[str(METRIC_PYTHON),str(ROOT/'scripts/p2_pipelined_video_metrics.py'),'--manifest',str(manifest),'--assets',p['metric_assets'],
             '--output',str(folder),'--device','cuda:0','--threads','4','--batch-size','64','--face-padding','0',
             '--feature-output',str(folder/'features.npz'),'--allow-partial','--face-device','cuda:0','--emotion-device','cuda:0','--pixel-workers','4','--gt-cache',p['metrics_gt_cache']]
        if all(s['quick200'] for s in block):cmd+=['--decoded-gt-cache',p['metrics_decoded_gt_cache']]
        with (folder/'video.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT.parent/'OmniResponse',stdout=log,stderr=subprocess.STDOUT)
        result=read(folder/'summary.json');assert not result['initialization_errors']
        actual=rows(folder/'pairs.jsonl');assert {r['id'] for r in actual}=={s['id'] for s in block}
        assert (folder/'features.npz').exists()
        # Native missing-face/support N/A is retained, never interpreted as a negative.
        execution=[r for r in actual if r['errors'].get('pair')]
        # Keep the native allow-partial result for structurally incomplete GT.
        # Never pad a missing source frame or silently drop its requested row.
        unexpected=[r for r in execution if not r['errors']['pair'].startswith('ValueError: Incomplete source decode:')]
        if unexpected:raise RuntimeError(f'Input/decode failure: {unexpected[0]["id"]}')
        if execution:save(WORK/'source_issues'/f'{slot:03d}.json',dict(records=[dict(id=r['id'],error=r['errors']['pair']) for r in execution],
            policy='native allow-partial; preserve requested row and N/A; no padding or invented metrics'))
        save(folder/'SUCCESS.json',dict(clips=len(actual),returncode=code,pairs_with_errors=result['pairs_with_errors'],
            source_unavailable=len(execution),status=result['status'],at=time.time()))

def merge():
    import numpy as np
    from evaluation.video.paired_metrics.run import summarize,METRICS
    p=read(WORK/'protocol.json')
    compatibility_path=WORK/'READER_COMPATIBILITY.json'
    compatibility=read(compatibility_path) if compatibility_path.exists() else None
    unavailable=compatibility.get('unavailable_records',{}) if compatibility else {}
    for label,slots,count in [('test6624',range(p['metric_shards']),6624),('fixed200',range(8),200)]:
        actual=[];contracts=[];dist={k:[[],[]] for k in ['fid','fvd']};coverage={k:0 for k in dist}
        for i in slots:
            f=WORK/'metrics'/f'{i:03d}';assert (f/'SUCCESS.json').exists();s=read(f/'summary.json');contracts.append(read(f/'contract.json'))
            block_rows=rows(f/'pairs.jsonl');actual.extend(unavailable.get(r['id'],r) for r in block_rows)
            with np.load(f/'features.npz') as a:
                for k in dist:
                    keep=[];cursor=0;removed=0
                    for r in block_rows:
                        n=r.get(k,{}).get('feature_count',0)
                        if r['id'] not in unavailable:keep.extend(range(cursor,cursor+n))
                        elif n:removed+=1
                        cursor+=n
                    assert cursor==len(a[k+'_gt'])==len(a[k+'_generated'])
                    if keep:dist[k][0].append(a[k+'_gt'][keep]);dist[k][1].append(a[k+'_generated'][keep])
                    coverage[k]+=s['coverage'][k]['pairs']-removed
        assert len(actual)==len({r['id'] for r in actual})==count
        if label=='fixed200':assert {r['id'] for r in actual}==set(p['quick200_ids'])
        for i,c in zip(slots,contracts):
            assert c['assets']==contracts[0]['assets']
            if c['code']!=contracts[0]['code']:
                compatibility=read(WORK/'READER_COMPATIBILITY.json')
                assert compatibility['passed']
                assert compatibility['contracts_sha256'][str(i)]==sha(WORK/'metrics'/f'{i:03d}'/'contract.json')
                reader='../../seamless/reader.py'
                assert {k:v for k,v in c['code'].items() if k!=reader}=={k:v for k,v in contracts[0]['code'].items() if k!=reader}
                assert c['code'][reader] in compatibility['reader_versions'] and contracts[0]['code'][reader] in compatibility['reader_versions']
        result=summarize(actual,dist,set(METRICS),contracts[0]['schema'],contracts[0]['datasets'],count,{})
        for k,n in coverage.items():result['coverage'][k]['pairs']=n
        result.update(model=p.get('model',core.MODEL),dataset='Seamless '+label,checkpoint_sha256=p['checkpoint_sha256'],audio_metrics=p['audio_metrics'],
                      protocol_sha256=sha(WORK/'protocol.json'),aggregation='global pooled features, never mean shard FID/FVD')
        if compatibility:
            result['reader_compatibility']=dict(audit_sha256=sha(compatibility_path),native_frozen_reader_unavailable_ids=[r['id'] for r in actual if r['id'] in unavailable],raw_score_blocks_unchanged=True)
        save(WORK/(label+'_RESULTS.json'),result)
        (WORK/(label+'_pairs.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in actual))
    audit=[]
    for s in p['samples']:
        r=read(WORK/'seamless/generated'/s['id']/'complete.json');job=read(s['ready'])['sample']
        assert r['step']==p['step'] and r['checkpoint_sha256']==p['checkpoint_sha256'] and r['source_only'] and not r['listener_audio_generated']
        assert r['score_frames']==s['frames'] and r['score_fps']==25
        assert sha(job['reference'])==r['reference_sha256'] and sha(job['audio'])==r['audio_sha256']
        assert sha(WORK/'seamless/generated'/s['id']/'generated.mp4')==r['native_sha256']
        assert sha(WORK/'seamless/scoring'/core.ARM/(s['id']+'.listener.mp4'))==r['score_sha256']
        audit.append(dict(id=s['id'],frames=s['frames']))
    save(WORK/'FINAL_AUDIT.json',dict(passed=True,clips=6624,rows=audit,checkpoint_sha256=p['checkpoint_sha256']))
    final=read(WORK/'test6624_RESULTS.json')
    save(WORK/'COMPLETE.json',dict(clips=6624,passed=True,status=final['status'],pairs_with_errors=final['pairs_with_errors'],
        coverage=final['coverage'],at=time.time(),results_sha256=sha(WORK/'test6624_RESULTS.json')))
    save(WORK/'status.json',dict(stage='complete',clips=6624,status=final['status'],pairs_with_errors=final['pairs_with_errors'],at=time.time()))

def run():
    core.gpu();p=read(WORK/'protocol.json');started=time.time()
    with (WORK/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'PAUSED.json').exists():
            value=read(WORK/'PAUSED.json')
            save(WORK/'status.json',dict(stage='paused',checkpoint_step=value['checkpoint_step'],generated=value['generated'],
                 total=6624,completed_score_blocks=value['completed_score_blocks'],scored_in_complete_blocks=value['scored_in_complete_blocks'],
                 active_metric_workers=0,active_stage_workers=[],at=time.time(),reason='explicit user pause'))
            return
        save(WORK/'queue_cursor.json',dict(offset=0))
        save(WORK/'LAUNCH.json',dict(pid=os.getpid(),gpu=4,workers=p.get('generation_workers',4),decode_batch=p.get('decode_batch',8),max_metric_workers=p.get('score_workers_after_generation',16),started=started))
        jobs={};metrics={};attempts={};logs=[];generation_slots=set()
        try:
            if not (WORK/'INPUT_COMPLETE.json').exists():jobs['inputs']=core.launch('inputs','inputs',core.GENPY,'gen')
            if not (WORK/'PROSODY_COMPLETE.json').exists():jobs['prosody']=core.launch('prosody','prosody',core.GENPY,'gen')
            if not (WORK/'GENERATION_COMPLETE.json').exists():
                for slot in range(p.get('generation_workers',4)):
                    if core.gpu()['free_mib']>=p.get('generation_min_free_mib',0):
                        jobs['generate'+str(slot)]=core.launch('generate',f'generate_{slot:02d}',core.GENPY,'gen',['--slot',str(slot)]);generation_slots.add(slot)
            while True:
                for name,proc in list(jobs.items()):
                    if proc.poll() is None:continue
                    del jobs[name]
                    if proc.returncode:raise RuntimeError(f'{name} failed, inspect logs')
                generated=len(list((WORK/'seamless/generated').glob('*/complete.json')))
                generation_done=generated==6624
                trigger=p.get('expand_generation_after')
                if not generation_done:
                    target=p.get('generation_workers',4)
                    if trigger and Path(trigger).exists():target=p.get('expanded_generation_workers',target)
                    for slot in range(target):
                        if slot not in generation_slots and core.gpu()['free_mib']>=p.get('generation_min_free_mib',0):
                            jobs['generate'+str(slot)]=core.launch('generate',f'generate_{slot:02d}',core.GENPY,'gen',['--slot',str(slot)]);generation_slots.add(slot)
                if generation_done and not (WORK/'GENERATION_COMPLETE.json').exists():save(WORK/'GENERATION_COMPLETE.json',dict(clips=6624,at=time.time()))
                for slot,proc in list(metrics.items()):
                    if proc.poll() is None:continue
                    del metrics[slot]
                    if proc.returncode:
                        if attempts[slot]>=3:raise RuntimeError(f'score block{slot} failed after3 attempts')
                        f=WORK/'metrics'/f'{slot:03d}';f.rename(f.with_name(f.name+f'.failed.{time.time_ns()}'))
                done={i for i in range(p['metric_shards']) if (WORK/'metrics'/f'{i:03d}/SUCCESS.json').exists()}
                if len(done)==p['metric_shards'] and generation_done and not jobs:break
                # Preserve enough room for the proven batch64 Qwen encoder. Once
                # it exits, overlap eight scorers with four GLARE residents;
                # after generation all sixteen scoring slots may be occupied.
                capacity=p.get('score_workers_after_generation',16) if generation_done else (p.get('score_workers_during_generation',8) if (WORK/'PROSODY_COMPLETE.json').exists() else 0)
                priority=WORK/'GPU4_PRIORITY.json'
                if priority.exists():
                    rule=read(priority);dependent=Path(rule['dependent_work'])
                    if time.time()<rule['expires_at'] and not (dependent/'COMPLETE.json').exists() and not (dependent/'FAILURE.json').exists():
                        capacity=min(capacity,rule['while_pending_max_metric_workers'])
                for slot in range(p['metric_shards']):
                    if len(metrics)>=capacity:break
                    if slot in done or slot in metrics:continue
                    block=read(WORK/'blocks'/f'{slot:03d}.json')
                    if not all((WORK/'seamless/generated'/s['id']/'complete.json').exists() for s in block):continue
                    if core.gpu()['free_mib']<20*1024:break
                    attempts[slot]=attempts.get(slot,0)+1
                    metrics[slot]=core.launch('score',f'score_{slot:03d}',METRIC_PYTHON,'response',['--slot',str(slot)])
                score_count=sum(read(WORK/'metrics'/f'{i:03d}/SUCCESS.json')['clips'] for i in done)
                save(WORK/'status.json',dict(stage='running',generated=generated,total=6624,completed_score_blocks=len(done),score_blocks=p['metric_shards'],
                    scored_in_complete_blocks=score_count,active_metric_workers=len(metrics),active_stage_workers=list(jobs),gpu=core.gpu(),elapsed_s=time.time()-started,at=time.time()))
                time.sleep(10)
            proc=core.launch('merge','merge',METRIC_PYTHON,'response');assert proc.wait()==0,'Global merge failed'
        except BaseException:
            for proc in list(jobs.values())+list(metrics.values()):
                if proc.poll() is None:proc.terminate()
            save(WORK/'FAILURE.json',dict(traceback=traceback.format_exc(),at=time.time()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','inputs','prosody','generate','score','merge']);parser.add_argument('--slot',type=int,default=0)
    a=parser.parse_args();core.claim=claim
    if a.action=='prepare':configure()
    elif a.action=='inputs':prepare_inputs()
    elif a.action=='prosody':prosody()
    elif a.action=='generate':core.generate(a.slot,8)
    elif a.action=='score':score(a.slot)
    elif a.action=='merge':merge()
    else:run()
