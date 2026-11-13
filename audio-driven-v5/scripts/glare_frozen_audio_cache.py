#!/usr/bin/env python3
"""Frozen GLARE Wav2Vec2/SER features for the existing first-frame/no-history adapter.

Caches PRE-projection Wav2Vec2 features. The trainable projection is never cached.
Window = normalized whole-pair waveform[640:], N-1 video frames. A different
training window/context requires a new cache; no claim of arbitrary-window reuse.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
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
import run_glare_shared_cache as common
import prepare_glare_nonprosody_cache as base
WORK=common.WORK/'frozen_audio'
read,save,sha,canonical=base.read,base.save,base.sha,base.canonical
WAV=ROOT/'compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h'
SER=ROOT/'compare/deployment/assets/wav2vec-english-speech-emotion-recognition'


def prepare():
    WORK.mkdir(parents=True,exist_ok=True)
    old=read(ROOT/'experiments/baseline_training_realdata_20261109/runs/glare/protocol.json')
    p=dict(schema='glare-frozen-audio-firstframe-nohistory-v1',selection_sha256=read(base.DEFAULT/'protocol.json')['selection_sha256'],
           samples=284218,gpus=[4,5,6,7],frame_grid=25,sample_rate=16000,
           view='normalize entire padded pair; remove first640 samples; seq_len=N-1; no history',
           pre_projection=True,wav2vec_features='hidden_states1..12 concatenated, native interpolation',
           wav2vec_shape='[N-1,9216]',ser_shape=[7],dtype='float32',
           numerics=dict(cudnn_allow_tf32=False,matmul_allow_tf32=False,deterministic=True),
           assets={str(WAV/'pytorch_model.bin'):sha(WAV/'pytorch_model.bin'),str(SER/'pytorch_model.bin'):sha(SER/'pytorch_model.bin')},
           code={str(p):sha(p) for p in [Path(__file__),ROOT/'src/baseline_training/training.py',
                  ROOT/'related/glare/float/models/FLOAT.py',ROOT/'related/glare/float/models/wav2vec2.py',ROOT/'related/glare/float/models/wav2vec2_ser.py']},
           prosody='deferred_by_user',full_glare_training_admitted=False)
    for path,digest in p['assets'].items():assert old['implementation']['assets_sha256'][path]==digest
    if (WORK/'protocol.json').exists():assert read(WORK/'protocol.json')==p
    else:save(WORK/'protocol.json',p)
    print('FROZEN_AUDIO_PREPARED',flush=True)


def paths(sid):
    folder=WORK/'cache'/__import__('hashlib').sha256(sid.encode()).hexdigest()[:2]
    return folder/(sid+'.pt'),folder/(sid+'.json')


def identity(job,p):
    return dict(sample_id=job['sid'],row_sha256=job['row_sha'],protocol_sha256=canonical(p))


def exists(job,p):
    payload,receipt=paths(job['sid'])
    if not receipt.exists():return False
    r=read(receipt)
    assert r['identity']==identity(job,p) and sha(payload)==r['payload_sha256'],'Frozen audio cache changed'
    return True


def worker(gpu,canary=False):
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    from scipy.signal import resample_poly
    from p2_j_cache.data import read_speaker
    from p2_j_current_qwen_hidden import item_from_row
    from baseline_training.features import native_glare
    from baseline_training.training import strict_wav2vec
    assert gpu in (4,5,6,7) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
    p=read(WORK/'protocol.json')
    assert p['code']=={s:sha(s) for s in p['code']}
    old=base.load_protocol(base.DEFAULT)
    joint_protocol=common.load_protocol()
    assert p['selection_sha256']==old['selection_sha256']
    for s,h in p['assets'].items():assert sha(s)==h
    lock_path=WORK/'workers'/('gpu%d.lock'%gpu);lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>16*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    opt=native_glare()
    from float.models.FLOAT import AudioEncoder,Audio2Emotion
    acoustic=AudioEncoder(opt)
    strict_wav2vec(acoustic.wav2vec2,WAV)
    acoustic=acoustic.eval().requires_grad_(False).cuda()
    emotion=Audio2Emotion(opt).eval().requires_grad_(False).cuda()
    processor=Wav2Vec2FeatureExtractor.from_pretrained(WAV,local_files_only=True)
    db=sqlite3.connect(base.DEFAULT/'jobs.sqlite');db.row_factory=sqlite3.Row
    jobs={r['ordinal']:dict(r) for r in db.execute('SELECT * FROM jobs')};db.close()
    db=sqlite3.connect(common.WORK/'groups.sqlite')
    ordinals=[o for (raw,) in db.execute('SELECT ordinals FROM groups WHERE worker=? ORDER BY gid',((gpu-4),)) for o in __import__('json').loads(raw)]
    db.close()
    if canary:
        d=read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
        wanted=d['diagnostic_train']+d['diagnostic_validation'];lookup={j['sid']:o for o,j in jobs.items()};ordinals=[lookup[s] for s in wanted]
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    state=dict(gpu=gpu,pid=os.getpid(),stage='running',completed=0,reused=0,failed=0,total=len(ordinals),started=time.time(),gpu_s=0.,io_s=0.)
    def pulse():
        state.update(heartbeat=time.time(),elapsed_s=time.time()-state['started'],peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
        save(WORK/'workers'/('gpu%d.json'%gpu),state)
    def prepare_audio(job):
        row=base.load_row(job,old)
        input_path=common.path_for(job['sid'],'audio.pt')
        if input_path.exists() and common.stage_ok(job,'audio',joint_protocol):
            normalized=torch.load(input_path,map_location='cpu',weights_only=True)['audio']
        else:
            wave,rate=read_speaker(item_from_row(row));wave=wave.mean(axis=1)
            wave=resample_poly(wave,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
            n=math.ceil(job['duration']*25-1e-7)*640
            wave=np.pad(wave,(0,max(0,n-len(wave))))[:n]
            normalized=torch.from_numpy(np.asarray(processor(wave,sampling_rate=16000).input_values[0],np.float32))
        return normalized[640:].contiguous(),len(normalized)//640-1
    pending=[];cursor=0;pulse()
    with ThreadPoolExecutor(max_workers=2) as prep:
        while cursor<len(ordinals) or pending:
            if stopped[0] or (common.WORK/'STOP').exists():break
            while cursor<len(ordinals) and len(pending)<2:
                job=jobs[ordinals[cursor]];cursor+=1
                if exists(job,p):state['reused']+=1;continue
                pending.append((job,prep.submit(prepare_audio,job)))
            if not pending:continue
            job,future=pending.pop(0)
            try:
                tick=time.time();audio,n=future.result();state['io_s']+=time.time()-tick
                tick=time.time()
                with torch.inference_mode():
                    x=audio[None].cuda()
                    hidden=acoustic.get_wav2vec2_feature(x,seq_len=n)[0].cpu()
                    probabilities=emotion.predict_emotion(x)[0].cpu()
                assert hidden.shape==(n,9216) and probabilities.shape==(7,)
                assert torch.isfinite(hidden).all() and torch.isfinite(probabilities).all()
                state['gpu_s']+=time.time()-tick
                payload,receipt=paths(job['sid']);payload.parent.mkdir(parents=True,exist_ok=True)
                tmp=payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
                torch.save(dict(wav2vec=hidden,emotion=probabilities),tmp)
                back=torch.load(tmp,map_location='cpu',weights_only=True)
                assert torch.equal(back['wav2vec'],hidden) and torch.equal(back['emotion'],probabilities)
                tmp.replace(payload)
                save(receipt,dict(identity=identity(job,p),payload_sha256=sha(payload),shape=list(hidden.shape),
                                 waveform_samples=len(audio),pre_projection=True,full_glare_training_admitted=False,completed_at=time.time()))
                error=WORK/'errors'/(job['sid']+'.json')
                if error.exists():error.unlink()
                state['completed']+=1
            except Exception as exc:
                state['failed']+=1;save(WORK/'errors'/(job['sid']+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
            pulse()
    state['stage']='stopped' if stopped[0] or (common.WORK/'STOP').exists() else ('finished' if not state['failed'] else 'finished_with_errors');pulse()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker']);ap.add_argument('--gpu',type=int,choices=[4,5,6,7]);ap.add_argument('--canary',action='store_true');args=ap.parse_args()
    prepare() if args.action=='prepare' else worker(args.gpu,args.canary)
