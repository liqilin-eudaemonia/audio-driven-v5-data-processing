#!/usr/bin/env python3
"""Incremental, provenance-preserving Qwen2-Audio prosody on node11 GPU7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_glare_shared_cache as common
import prepare_glare_nonprosody_cache as base
from glare_prosody_audit import SCHEMA,parse_reply,frame_targets,attach_audit

WORK=common.WORK/'prosody'
MODEL=Path('/private/ckpts/Qwen2-Audio-7B-Instruct')
save,read,sha=common.save,common.read,common.sha


def identity(job,protocol):
    return dict(sample_id=job['sid'],row_sha256=job['row_sha'],protocol_sha256=common.canonical(protocol))


def paths(sid):
    folder=WORK/'cache'/hashlib.sha256(sid.encode()).hexdigest()[:2]
    return folder/(sid+'.npz'),folder/(sid+'.json')


def load_cached(job,protocol):
    """Missing/failed/legacy NULL never silently becomes an intensity vector."""
    import numpy as np
    payload,receipt=paths(job['sid'])
    if not receipt.exists():raise FileNotFoundError(receipt)
    r=read(receipt)
    assert r['identity']==identity(job,protocol) and r['analysis']['accepted']
    assert sha(payload)==r['payload_sha256']
    with np.load(payload,allow_pickle=False) as v:
        scores,valid=v['intensity'],v['intensity_valid']
    n=math.ceil(job['duration']*25-1e-7)
    assert scores.shape==valid.shape==(n,1) and valid.dtype==bool and valid.all()
    assert scores.dtype==np.float32 and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    return dict(intensity=scores,intensity_valid=valid,receipt=r)


def prepare():
    from baseline_training.features import native_glare
    native_glare()
    from float.models.intensity_encoder import IntensityEncoder
    WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (WORK/'protocol.json').exists():
            protocol=read(WORK/'protocol.json')
            assert protocol['code']=={p:sha(p) for p in protocol['code']}
            return protocol
        old=base.load_protocol(base.DEFAULT)
        assets=sorted(MODEL.glob('model-*.safetensors'))+[MODEL/name for name in [
            'model.safetensors.index.json','config.json','generation_config.json',
            'preprocessor_config.json','tokenizer_config.json','tokenizer.json','merges.txt','vocab.json']]
        prompt=IntensityEncoder.get_prompt(None)
        protocol=dict(schema=SCHEMA,samples=old['samples'],selection_sha256=old['selection_sha256'],
            parent_protocol_sha256=sha(base.DEFAULT/'protocol.json'),model=str(MODEL),
            assets_sha256={str(p):sha(p) for p in assets},prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            source_encoder_sha256=sha(ROOT/'related/glare/float/models/intensity_encoder.py'),
            code={str(p):sha(p) for p in [Path(__file__),ROOT/'src/glare_prosody_audit.py']},
            threshold=.5,fps=25,sample_rate=16000,input='final selected speaker waveform; mono; resample16000; before W2V standardization; entire pair',
            frame_contract='N=ceil(pair duration*25); native two-decimal/floor-ceil vector; reference frame retained; training uses frames1..N-1',
            explicit_null='valid model-declared no event; known zero, not human verification',
            invalid_reply='no payload or training admission; preserve raw; at most one format/truncation retry',
            generation=dict(do_sample=False,max_new_tokens=256,retry_max_new_tokens=512),gpu=7,
            model_load=dict(dtype='float16',attention='sdpa',matmul_tf32=False,cudnn_tf32=False,seed=137),
            old_normalized_null='raw unavailable: never reused; re-infer into this separate namespace',
            full_glare_training_admitted=False)
        save(WORK/'protocol.json',protocol)
        return protocol


def jobs_for(args):
    db=sqlite3.connect((base.DEFAULT/'jobs.sqlite').as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    jobs=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY ordinal')];db.close()
    assert len(jobs)==len({j['sid'] for j in jobs})==284218
    if args.canary:
        d=read(ROOT/'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
        ids=d['diagnostic_train']+d['diagnostic_validation'];lookup={j['sid']:j for j in jobs}
        chosen=[lookup[sid] for sid in ids]
        if args.extra_samples:
            remaining=[j for j in jobs if j['sid'] not in ids]
            chosen+=random.Random(137).sample(remaining,args.extra_samples)
        jobs=chosen
    if args.limit:jobs=jobs[:args.limit]
    return jobs


def input_wave(job,old):
    import numpy as np
    from scipy.signal import resample_poly
    from p2_j_cache.data import read_speaker
    from p2_j_current_qwen_hidden import item_from_row
    row=base.load_row(job,old);wave,rate=read_speaker(item_from_row(row));wave=wave.mean(axis=1)
    wave=resample_poly(wave,16000//math.gcd(rate,16000),rate//math.gcd(rate,16000)).astype(np.float32)
    assert wave.ndim==1 and len(wave)>0 and np.isfinite(wave).all()
    assert abs(len(wave)/16000-job['duration'])<.05
    return wave


def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7' and args.gpu==7
    import numpy as np
    import torch
    from baseline_training.features import native_glare
    protocol=prepare()
    for path,digest in protocol['assets_sha256'].items():assert sha(path)==digest
    assert sha(ROOT/'related/glare/float/models/intensity_encoder.py')==protocol['source_encoder_sha256']
    lock=(WORK/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.manual_seed(137);torch.cuda.manual_seed_all(137)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>30*1024**3
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    from float.models.intensity_encoder import IntensityEncoder
    opt=native_glare();opt.intensity_min_score=.5
    encoder=attach_audit(IntensityEncoder(opt,cuda=True))
    original_prompt=encoder.prompt_text
    assert encoder.prompt==protocol['prompt']
    jobs=jobs_for(args);old=base.load_protocol(base.DEFAULT)
    if args.probe_audio:
        probes=[]
        for job in [jobs[0],max(jobs,key=lambda j:j['duration'])]:
            wave=input_wave(job,old)
            encoder.prompt_text=encoder._build_prompt_text('Transcribe the English speech in this audio. Return only the words you hear.')
            encoder.audit_result=None;encoder.audit_generation={};encoder.max_new_tokens=128
            encoder.infer_text(wave,sampling_rate=16000)
            probes.append(dict(sid=job['sid'],duration_s=len(wave)/16000,
                rms=float(np.sqrt(np.mean(wave**2))),raw_transcription=encoder.audit_result['raw_text'],
                generation=encoder.audit_generation,diagnostic_only_not_a_prosody_target=True))
            save(WORK/'AUDIO_INPUT_PROBE.json',dict(probes=probes))
        encoder.prompt_text=original_prompt
    stopping=[False];signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    state=dict(stage='running',pid=os.getpid(),gpu=7,total=len(jobs),completed=0,reused=0,failed=0,
        failed_previous=0,status_counts={},started=time.time(),prosody_only=True,full_glare_training_admitted=False)
    last=[0.]
    def pulse(force=False):
        if not force and time.time()-last[0]<2:return
        state.update(heartbeat=time.time(),elapsed_s=time.time()-state['started'],peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
        save(WORK/'status.json',state);last[0]=time.time()
    def produce(job,wave):
        folder=WORK/'attempts'/hashlib.sha256(job['sid'].encode()).hexdigest()[:2];folder.mkdir(parents=True,exist_ok=True)
        analysis=None;attempt_records=[]
        for attempt in [0,1]:
            encoder.audit_result=None;encoder.audit_generation={}
            encoder.max_new_tokens=256 if attempt==0 else 512
            encoder.prompt_text=original_prompt
            if attempt and analysis and analysis['status']!='truncated':
                prompt=protocol['prompt']+f'\nFormat reminder: audio duration is {len(wave)/16000:.6f} seconds. Return only NULL or numeric start_time, end_time, intensity_score rows within this duration; no explanations or markdown. Keep the same event detection criteria and score threshold.'
                encoder.prompt_text=encoder._build_prompt_text(prompt)
            tick=time.time()
            try:
                encoder.infer_text(wave,sampling_rate=16000)
                analysis=encoder.audit_result
                assert analysis is not None,'Raw generation was not captured'
            except Exception as exc:
                analysis=dict(schema=SCHEMA,status='inference_error',accepted=False,raw_text=None,
                    normalized_text=None,events=[],errors=[repr(exc)],traceback=traceback.format_exc())
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
            record=dict(identity=identity(job,protocol),attempt=attempt+1,analysis=analysis,
                generation=encoder.audit_generation,prompt_text=encoder.prompt_text,
                prompt_sha256=hashlib.sha256(encoder.prompt_text.encode()).hexdigest(),elapsed_s=time.time()-tick,
                waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),waveform_samples=len(wave),sample_rate=16000,
                input_standardized=False,at=time.time())
            attempt_path=folder/(job['sid']+'.'+str(time.time_ns())+'.json');save(attempt_path,record);attempt_records.append(str(attempt_path))
            if analysis['accepted']:
                # For every real accepted reply, prove exact native vector semantics.
                frames=math.ceil(job['duration']*25-1e-7)
                scores,valid=frame_targets(analysis,frames)
                native_text=IntensityEncoder.normalize_result(encoder,analysis['canonical_text'],len(wave)/16000)
                assert native_text==analysis['normalized_text']
                native=encoder.text_to_frame_vector(native_text,frames,fps=25).cpu().numpy()
                assert np.array_equal(scores,native)
                payload,receipt=paths(job['sid']);payload.parent.mkdir(parents=True,exist_ok=True)
                tmp=payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
                with tmp.open('wb') as stream:
                    np.savez_compressed(stream,intensity=scores,intensity_valid=valid)
                    stream.flush();os.fsync(stream.fileno())
                with np.load(tmp,allow_pickle=False) as back:
                    assert np.array_equal(back['intensity'],scores) and np.array_equal(back['intensity_valid'],valid)
                tmp.replace(payload)
                save(receipt,dict(identity=identity(job,protocol),analysis=analysis,payload_sha256=sha(payload),
                    attempts=attempt_records,frames=frames,fps=25,native_vector_exact=True,human_verified=False,
                    completed_at=time.time(),full_glare_training_admitted=False))
                error=WORK/'errors'/(job['sid']+'.json')
                if error.exists():error.unlink()
                return analysis['status']
        save(WORK/'errors'/(job['sid']+'.json'),dict(identity=identity(job,protocol),analysis=analysis,
            attempts=attempt_records,retry_exhausted=True,no_zero_target_written=True))
        return None
    pulse(True);checks=[];consecutive_errors=0
    with ThreadPoolExecutor(max_workers=4) as prep:
        pending=[];cursor=0
        while cursor<len(jobs) or pending:
            if stopping[0] or (WORK/'STOP').exists():break
            while cursor<len(jobs) and len(pending)<8:
                job=jobs[cursor];cursor+=1
                if paths(job['sid'])[1].exists():
                    cached=load_cached(job,protocol);state['reused']+=1
                    status=cached['receipt']['analysis']['status'];state['status_counts'][status]=state['status_counts'].get(status,0)+1
                    checks.append(dict(sid=job['sid'],status=status,reused=True));pulse();continue
                if (WORK/'errors'/(job['sid']+'.json')).exists() and not args.retry_errors:
                    state['failed_previous']+=1;continue
                pending.append((job,prep.submit(input_wave,job,old)))
            if not pending:continue
            job,future=pending.pop(0)
            try:status=produce(job,future.result())
            except Exception as exc:
                status=None;save(WORK/'errors'/(job['sid']+'.json'),dict(identity=identity(job,protocol),
                    input_or_write_error=repr(exc),traceback=traceback.format_exc(),no_zero_target_written=True))
            if status:
                state['completed']+=1;state['status_counts'][status]=state['status_counts'].get(status,0)+1
                checks.append(dict(sid=job['sid'],status=status,reused=False));consecutive_errors=0
            else:state['failed']+=1;consecutive_errors+=1
            pulse()
            if consecutive_errors>=3:
                state['stop_reason']='three_consecutive_failures';break
    state['stage']='stopped' if stopping[0] or (WORK/'STOP').exists() else ('finished' if not state['failed'] and not state['failed_previous'] and state['completed']+state['reused']==len(jobs) else 'needs_error_review')
    pulse(True)
    if args.canary:
        save(WORK/'CANARY_ACCEPTED.json',dict(passed=state['stage']=='finished',state=state,checks=checks,
            real_raw_outputs_saved=True,exact_native_vectors=True,legacy_normalized_null_reinferred=True))
    return state


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run'])
    ap.add_argument('--gpu',type=int,choices=[7],default=7);ap.add_argument('--canary',action='store_true')
    ap.add_argument('--extra-samples',type=int,default=0);ap.add_argument('--limit',type=int)
    ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--probe-audio',action='store_true');args=ap.parse_args()
    result=prepare() if args.action=='prepare' else run(args)
    print(json.dumps(result,ensure_ascii=False),flush=True)
