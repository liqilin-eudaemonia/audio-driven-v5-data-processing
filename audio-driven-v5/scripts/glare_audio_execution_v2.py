#!/usr/bin/env python3
"""Execution overlay for frozen audio: bounded asynchronous verified writes.

Cache identities, numerical methods, precision, input windows and payloads are
exactly those of glare_frozen_audio_cache. No original source is modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_frozen_audio_cache as core
import run_glare_shared_cache as common
import prepare_glare_nonprosody_cache as base


class Engine:
    def __init__(self,gpu,assigned_gpus=(4,5,6,7)):
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from transformers import Wav2Vec2FeatureExtractor
        from baseline_training.features import native_glare
        from baseline_training.training import strict_wav2vec
        assert gpu in assigned_gpus and 0<=gpu<=7 and os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
        self.gpu=gpu;self.torch=torch;self.np=np;self.resample=resample_poly
        self.protocol=core.read(core.WORK/'protocol.json')
        assert self.protocol['code']=={s:core.sha(s) for s in self.protocol['code']}
        for path,h in self.protocol['assets'].items():assert core.sha(path)==h
        self.old=base.load_protocol(base.DEFAULT);self.joint=common.load_protocol()
        torch.set_num_threads(2);torch.cuda.set_device(0)
        assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>12*1024**3
        torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        opt=native_glare()
        from float.models.FLOAT import AudioEncoder,Audio2Emotion
        self.acoustic=AudioEncoder(opt)
        strict_wav2vec(self.acoustic.wav2vec2,core.WAV)
        self.acoustic=self.acoustic.eval().requires_grad_(False).cuda()
        self.emotion=Audio2Emotion(opt).eval().requires_grad_(False).cuda()
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(core.WAV,local_files_only=True)
        db=sqlite3.connect(base.DEFAULT/'jobs.sqlite');db.row_factory=sqlite3.Row
        self.jobs={r['ordinal']:dict(r) for r in db.execute('SELECT * FROM jobs')};db.close()

    def prepare_audio(self,job):
        from p2_j_cache.data import read_speaker
        from p2_j_current_qwen_hidden import item_from_row
        np,torch=self.np,self.torch
        row=base.load_row(job,self.old)
        path=common.path_for(job['sid'],'audio.pt')
        if path.exists() and common.stage_ok(job,'audio',self.joint):
            normalized=torch.load(path,map_location='cpu',weights_only=True)['audio']
        else:
            wave,rate=read_speaker(item_from_row(row));wave=wave.mean(axis=1)
            wave=self.resample(wave,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
            n=math.ceil(job['duration']*25-1e-7)*640
            wave=np.pad(wave,(0,max(0,n-len(wave))))[:n]
            normalized=torch.from_numpy(np.asarray(self.processor(wave,sampling_rate=16000).input_values[0],np.float32))
        return normalized[640:].contiguous(),len(normalized)//640-1

    def compute(self,audio,n):
        torch=self.torch
        with torch.inference_mode():
            x=audio[None].cuda()
            hidden=self.acoustic.get_wav2vec2_feature(x,seq_len=n)[0].cpu()
            emotion=self.emotion.predict_emotion(x)[0].cpu()
        return hidden,emotion

    def write(self,job,audio,n,hidden,emotion,root):
        torch=self.torch;started=time.time()
        assert hidden.shape==(n,9216) and emotion.shape==(7,)
        assert torch.isfinite(hidden).all() and torch.isfinite(emotion).all()
        folder=Path(root)/'cache'/__import__('hashlib').sha256(job['sid'].encode()).hexdigest()[:2]
        folder.mkdir(parents=True,exist_ok=True)
        payload=folder/(job['sid']+'.pt');receipt=folder/(job['sid']+'.json')
        tmp=payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
        torch.save(dict(wav2vec=hidden,emotion=emotion),tmp)
        back=torch.load(tmp,map_location='cpu',weights_only=True)
        assert torch.equal(back['wav2vec'],hidden) and torch.equal(back['emotion'],emotion)
        tmp.replace(payload)
        core.save(receipt,dict(identity=core.identity(job,self.protocol),payload_sha256=core.sha(payload),
            shape=list(hidden.shape),waveform_samples=len(audio),pre_projection=True,
            full_glare_training_admitted=False,completed_at=time.time()))
        error=Path(root)/'errors'/(job['sid']+'.json')
        if error.exists():error.unlink()
        return time.time()-started

    def run(self,jobs,root,asynchronous=True,reuse=True):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        state=dict(gpu=self.gpu,pid=os.getpid(),stage='running',completed=0,reused=0,failed=0,
            total=len(jobs),started=time.time(),gpu_s=0.,write_s=0.,input_wait_s=0.,writer_wait_s=0.,
            execution='async-writer-v2' if asynchronous else 'serial-control',writer_queue_bound=4)
        stopping=[False];signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
        last=[0.]
        def pulse(force=False):
            if asynchronous and not force and time.time()-last[0]<2.:return
            state.update(heartbeat=time.time(),elapsed_s=time.time()-state['started'],
                         peak_cuda_gib=self.torch.cuda.max_memory_allocated()/1024**3)
            core.save(root/'workers'/('gpu%d.json'%self.gpu),state);last[0]=time.time()
        def failure(job,exc):
            state['failed']+=1
            core.save(root/'errors'/(job['sid']+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
        def consume(item):
            job,future=item
            try:
                state['write_s']+=future.result();state['completed']+=1
            except Exception as exc:failure(job,exc)
        pending=[];writes=[];cursor=0;pulse(True)
        original_root=core.WORK
        # Only paths()/exists() consult core.WORK; no numerical method does.
        core.WORK=root
        try:
            with ThreadPoolExecutor(max_workers=2) as prep, ThreadPoolExecutor(max_workers=1) as writer:
                while cursor<len(jobs) or pending:
                    if stopping[0] or (common.WORK/'STOP').exists():break
                    while cursor<len(jobs) and len(pending)<2:
                        job=jobs[cursor];cursor+=1
                        if reuse and core.exists(job,self.protocol):
                            state['reused']+=1
                            if state['reused']%128==0:pulse()
                            continue
                        pending.append((job,prep.submit(self.prepare_audio,job)))
                    if not pending:continue
                    job,future=pending.pop(0)
                    try:
                        tick=time.time();audio,n=future.result();state['input_wait_s']+=time.time()-tick
                        tick=time.time();hidden,emotion=self.compute(audio,n);state['gpu_s']+=time.time()-tick
                        if asynchronous:
                            writes.append((job,writer.submit(self.write,job,audio,n,hidden,emotion,root)))
                            if len(writes)>=4:
                                tick=time.time();consume(writes.pop(0));state['writer_wait_s']+=time.time()-tick
                            while writes and writes[0][1].done():consume(writes.pop(0))
                        else:
                            state['write_s']+=self.write(job,audio,n,hidden,emotion,root);state['completed']+=1
                    except Exception as exc:failure(job,exc)
                    pulse()
                for item in writes:consume(item)
            state['stage']='stopped' if stopping[0] or (common.WORK/'STOP').exists() else ('finished' if not state['failed'] else 'finished_with_errors')
            pulse(True)
            return state
        finally:core.WORK=original_root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
    ap.add_argument('--benchmark',action='store_true');ap.add_argument('--samples',type=int,default=64)
    ap.add_argument('--canary',action='store_true')
    ap.add_argument('--all-partitions',action='store_true');args=ap.parse_args()
    engine=Engine(args.gpu);root=core.WORK
    db=sqlite3.connect(common.WORK/'groups.sqlite')
    rows=db.execute('SELECT ordinals FROM groups ORDER BY gid') if args.all_partitions else db.execute('SELECT ordinals FROM groups WHERE worker=? ORDER BY gid',(args.gpu-4,))
    ordinals=[o for (text,) in rows for o in json.loads(text)];db.close()
    if args.all_partitions:assert len(ordinals)==len(set(ordinals))==len(engine.jobs)
    if args.canary:
        d=core.read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
        lookup={j['sid']:o for o,j in engine.jobs.items()}
        ordinals=[lookup[s] for s in d['diagnostic_train']+d['diagnostic_validation']]
    jobs=[engine.jobs[o] for o in ordinals]
    if args.benchmark:
        jobs=jobs[:args.samples]
        # Warm the exact cohort before ABBA, so the second arm does not win
        # merely because the first arm fetched source WAVs into filesystem cache.
        for job in jobs:engine.prepare_audio(job)
        audio,n=engine.prepare_audio(jobs[0]);engine.compute(audio,n)
        out=common.WORK/'optimization_audio_v2';results=[]
        for i,async_ in enumerate([False,True,True,False]):
            path=out/('round%d_%s'%(i,'async' if async_ else 'serial'))
            result=engine.run(jobs,path,async_,False);assert result['failed']==0 and result['completed']==len(jobs)
            results.append(result)
        # Compare every tensor across the execution variants, not just averages.
        torch=engine.torch
        for job in jobs:
            suffix=__import__('hashlib').sha256(job['sid'].encode()).hexdigest()[:2]+'/'+job['sid']+'.pt'
            values=[torch.load(path/'cache'/suffix,map_location='cpu',weights_only=True) for path in [
                out/'round0_serial',out/'round1_async',out/'round2_async',out/'round3_serial']]
            assert all(torch.equal(v[k],values[0][k]) for v in values[1:] for k in ['wav2vec','emotion'])
        serial=sum(r['elapsed_s'] for r in results if r['execution']=='serial-control')
        async_s=sum(r['elapsed_s'] for r in results if r['execution']=='async-writer-v2')
        core.save(out/'RESULT.json',dict(passed=True,all_tensors_exact=True,samples=len(jobs),
            rounds=results,serial_seconds=serial,async_seconds=async_s,speedup=serial/async_s,
            execution_source_sha256=core.sha(Path(__file__))))
        print(json.dumps(dict(speedup=serial/async_s,serial_seconds=serial,async_seconds=async_s)),flush=True)
    else:
        lock_path=root/'workers'/('gpu%d.lock'%args.gpu)
        lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.save(root/'workers'/('gpu%d.execution.json'%args.gpu),dict(
            execution='async-writer-v2',all_partitions=args.all_partitions,source_sha256=core.sha(Path(__file__)),window=engine.protocol['view']))
        engine.run(jobs,root,True,True)


if __name__=='__main__':main()
