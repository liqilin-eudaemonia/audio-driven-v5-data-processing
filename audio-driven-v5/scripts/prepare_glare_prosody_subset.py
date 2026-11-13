#!/usr/bin/env python3
"""Frozen-cohort/source-grid prosody extraction and admission; no training launch."""
import argparse
from collections import Counter, defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from glare_prosody_subset import (SCHEMA, ChunkCache, IntegrityError, canonical, frame_count,
    lock, read, required_chunks, save, sha, track_identity, CHUNK_SAMPLES)

BASE = ROOT/'experiments/glare_current500h_prosody_subset_20261110'
PARENT = ROOT/'experiments/glare_current500h_nonprosody_cache_20261110'
OLD = ROOT/'experiments/glare_current500h_shared_cache_20261110'
OFFICIAL = ROOT/'related/visual_reaction_annotation/prosody_extract'


def require(condition, message):
    if not condition:
        raise IntegrityError(message)


def extractor_contract():
    old = read(OLD/'prosody/protocol.json')
    files = [Path(__file__), ROOT/'src/glare_prosody_subset.py', ROOT/'src/glare_prosody_source_runtime.py',
             ROOT/'src/glare_prosody_audit.py', OFFICIAL/'batch_emotion_fluctuation_multi_gpu.py',
             OFFICIAL/'input_prompt.txt', OFFICIAL/'run_qwen_mt.sh']
    return dict(schema=SCHEMA, code={str(p):sha(p) for p in files},
        official_script=str(OFFICIAL/'batch_emotion_fluctuation_multi_gpu.py'),
        prompt=(OFFICIAL/'input_prompt.txt').read_text().strip(), model=old['model'],
        assets_sha256=old['assets_sha256'], chunk_seconds=2.4, sample_rate=16000, fps=25,
        input='full original participant speaker timeline, existing constant-gain LUFS WAV; official librosa resampling and full-track RMS-20dBFS/peak0.98 before chunking',
        min_rms_gate=False, max_new_tokens=64, do_sample=False, batch_cap=16,
        precision='official FP16 CUDA; official TF32 setting',
        admission='strict raw lines or explicit NULL; no format repair; reject whole affected pair on any bad/unsupported required chunk',
        tail='incomplete original source tail unsupported; never pad or call it a semantic failure',
        mapping='half-open source chunks; canonical chunk events clipped in Decimal; no second rounding; native local25fps floor/ceil last-event assignment')


def connect(plan):
    class ClosingConnection(sqlite3.Connection):
        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                self.close()
    db = sqlite3.connect(str(Path(plan)/'index.sqlite'),factory=ClosingConnection); db.row_factory = sqlite3.Row
    return db


def protocol(plan):
    plan = Path(plan)
    p = read(plan/'protocol.json')
    require(sha(plan/'index.sqlite')==p['index_sha256'], 'Frozen plan index changed')
    for path,digest in p['extractor']['code'].items():
        require(sha(path)==digest, 'Extractor code changed: '+path)
    require(sha(p['selection'])==p['selection_sha256'], 'Selected500h manifest changed')
    return p


def source_pair(row, job, result):
    src = row['source']; clip = result['clips'][src['source_clip_index']]
    sid = row['sample_id']; split = row['split']
    require(split in {'train','dev'} and split==row['source_split']==src['split']==clip['split']==job['split'], 'Split mismatch/test leakage')
    require(sid==clip['sample_id']==src['sample_id']==job['sid'], 'Pair identity mismatch')
    for key in ('speaker_id','listener_id'):
        require(row[key]==src[key]==clip[key], 'Participant role mismatch')
    require(row['speaker_id']!=row['listener_id'], 'Speaker equals listener')
    start,end,duration = row['source_start_s'],row['source_end_s'],row['duration_s']
    require(all(math.isfinite(v) for v in (start,end,duration)) and 0<=start<end, 'Invalid source interval')
    rate=clip['normalized_audio']['speaker']['sample_rate']
    sample_duration=(round(end*rate)-round(start*rate))/rate
    require(abs(sample_duration-duration)<1e-6 and abs(duration-job['duration'])<1e-6,
            'Duration differs from original sample-quantized interval')
    require(abs(clip['start']-start)<1e-7 and abs(clip['end']-end)<1e-7 and
            abs(src['source_start_s']-start)<1e-7 and abs(src['source_end_s']-end)<1e-7, 'Source timeline mismatch')
    original = row.get('original_effective_source',src)
    if clip['sources']['listener'] != original['source']:
        # Upright restoration retains the original archive lineage in the caption receipt.
        require(sha(row['caption_file'])==row['caption_file_sha256'], 'Restored caption changed')
        caption=read(row['caption_file'])
        require(caption['source']==src, 'Restored caption/source mismatch')
        original=caption.get('original_effective_source',caption['source'])
    require(clip['sources']['listener']==original['source'], 'Listener source lineage mismatch')
    audio = clip['normalized_audio']['speaker']
    require(audio['contract']['source']==clip['sources']['speaker'] and
            Path(clip['sources']['speaker']['member']).stem==row['speaker_id'], 'Wrong speaker audio lineage')
    require(Path(audio['path'])!=Path(src['normalized_audio']['path']), 'Listener WAV passed as speaker')
    track = dict(track_id=track_identity(audio,row['speaker_id']), speaker_id=row['speaker_id'],
                 audio=audio, split=split, group_id=row['group_id'])
    pair = dict(sid=sid,ordinal=job['ordinal'],offset=job['offset'],row_sha=job['row_sha'],split=split,
                group_id=row['group_id'],start=start,end=end,duration=duration,frames=frame_count(duration),track_id=track['track_id'])
    return pair,track


def verified_sources(staging, workers=4):
    def load(reference):
        path,digest=reference;raw=Path(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==digest,'Source result changed: '+path)
        return path,digest,json.loads(raw)
    refs=iter(staging.execute('SELECT DISTINCT path,digest FROM rows ORDER BY path,digest'))
    pending=deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for reference in refs:
            pending.append(pool.submit(load,reference))
            if len(pending)>=workers:yield pending.popleft().result()
        while pending:yield pending.popleft().result()


def plan(args):
    args.plan.mkdir(parents=True,exist_ok=True)
    with lock(args.plan/'prepare.lock'):
        if (args.plan/'protocol.json').exists():
            p=protocol(args.plan)
            require(p['seed']==args.seed and p['requested_samples']==(None if args.full else args.samples), 'Existing plan has different cohort parameters')
            print(json.dumps({k:p[k] for k in ('pairs','tracks','chunks','hours','index_sha256')})); return
        parent=read(PARENT/'protocol.json'); selection=Path(parent['selection'])
        require(sha(selection)==parent['selection_sha256'], 'Frozen pool changed')
        db=sqlite3.connect(PARENT/'jobs.sqlite'); db.row_factory=sqlite3.Row
        jobs=[dict(j) for j in db.execute('SELECT * FROM jobs ORDER BY ordinal')]; db.close()
        require(len(jobs)==parent['samples'], 'Parent index incomplete')
        # Detect role-track/group split crossing over the whole cohort, not only sampled rows.
        # The frozen native groups index already guarantees split consistency; verify chosen tracks below.
        if not args.full:
            jobs=heapq.nsmallest(args.samples,jobs,key=lambda j:hashlib.sha256((str(args.seed)+':'+j['sid']).encode()).digest())
            jobs.sort(key=lambda j:j['ordinal'])
        # Disk staging bounds memory even for the full284k cohort.
        stage_path=args.plan/'staging.sqlite'
        if stage_path.exists():stage_path.unlink()
        staging=sqlite3.connect(stage_path)
        staging.execute('CREATE TABLE rows(path TEXT,digest TEXT,job TEXT,data TEXT)')
        with selection.open('rb') as stream:
            for job in jobs:
                stream.seek(job['offset']); raw=stream.readline()
                require(hashlib.sha256(raw).hexdigest()==job['row_sha'], 'Selected row changed')
                row=json.loads(raw)
                require(canonical(row['source'])==job['source_sha'], 'Frozen source changed')
                require(row['caption_file_sha256']==job['caption_sha'], 'Frozen caption identity changed')
                src=row['source']; staging.execute('INSERT INTO rows VALUES(?,?,?,?)',
                    (src['source_result'],src['source_result_sha256'],json.dumps(job),json.dumps(row)))
        staging.execute('CREATE INDEX source_path ON rows(path,digest)');staging.commit()
        tmp=args.plan/'index.sqlite.building'
        if tmp.exists(): tmp.unlink()
        db=sqlite3.connect(tmp)
        db.executescript('CREATE TABLE pairs(sid TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,track_id TEXT,data TEXT); CREATE TABLE tracks(track_id TEXT PRIMARY KEY,data TEXT); CREATE TABLE chunks(track_id TEXT,chunk_index INTEGER,PRIMARY KEY(track_id,chunk_index)); CREATE INDEX pair_track ON pairs(track_id);')
        tracks={}; groups={}; paths={}; hours=0.
        for path,digest,result in verified_sources(staging):
            for job_text,row_text in staging.execute('SELECT job,data FROM rows WHERE path=? AND digest=?',(path,digest)):
                job,row=json.loads(job_text),json.loads(row_text)
                pair,track=source_pair(row,job,result); tid=track['track_id']
                for key,table in ((row['group_id'],groups),(track['audio']['sha256'],paths)):
                    require(key not in table or table[key]==pair['split'], 'Train/dev source crossing')
                    table[key]=pair['split']
                if tid in tracks:
                    require(tracks[tid]==track, 'Track has inconsistent group/split/contract')
                else:
                    tracks[tid]=track; db.execute('INSERT INTO tracks VALUES(?,?)',(tid,json.dumps(track)))
                db.execute('INSERT INTO pairs VALUES(?,?,?,?)',(pair['sid'],pair['ordinal'],tid,json.dumps(pair)))
                db.executemany('INSERT OR IGNORE INTO chunks VALUES(?,?)',[(tid,i) for i in required_chunks(pair['start'],pair['end'])])
                hours+=pair['duration']/3600
        db.commit(); chunks=db.execute('SELECT count(*) FROM chunks').fetchone()[0]; db.close(); tmp.replace(args.plan/'index.sqlite')
        staging.close();stage_path.unlink()
        p=dict(schema=SCHEMA,selection=str(selection),selection_sha256=parent['selection_sha256'],
            parent_jobs_sha256=sha(PARENT/'jobs.sqlite'),parent_protocol_sha256=sha(PARENT/'protocol.json'),
            seed=args.seed,requested_samples=None if args.full else args.samples,selection_method='sha256(seed:sample_id) smallest; original ordinal processing',
            pairs=len(jobs),tracks=len(tracks),chunks=chunks,hours=hours,index_sha256=sha(args.plan/'index.sqlite'),
            extractor=extractor_contract(),cache_root=str(args.cache_root.resolve()),created_at=time.time(),full_glare_training_admitted=False)
        save(args.plan/'protocol.json',p)
        print(json.dumps({k:p[k] for k in ('pairs','tracks','chunks','hours','index_sha256')}))


def extract(args):
    from glare_prosody_source_runtime import OfficialRuntime,waveform_sha
    p=protocol(args.plan); cache=ChunkCache(p['cache_root'],p['extractor']); runtime=None
    state=dict(started=time.time(),new=0,reused=0,counts={},stage='running',pid=os.getpid())
    stopping=[False]; signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    counts=Counter()
    # Shared contract lock covers check/commit/archive across every cohort plan.
    with lock(cache.root/'mutation.lock',blocking=False), connect(args.plan) as db:
        save(args.plan/'status.json',state)
        for entry in db.execute('SELECT data FROM tracks ORDER BY track_id'):
            if stopping[0] or (args.plan/'STOP').exists(): break
            track=json.loads(entry['data']); tid=track['track_id']; needed=[]
            for r in db.execute('SELECT chunk_index FROM chunks WHERE track_id=? ORDER BY chunk_index',(tid,)):
                i=r['chunk_index']
                if not args.all and state['new']+len(needed)>=args.limit: break
                if args.retry_execution: cache.retry_execution(track,i)
                old=cache.load(track,i)
                if old is not None: state['reused']+=1; continue
                needed.append(i)
            if not needed:
                if not args.all and state['new']>=args.limit: break
                continue
            if runtime is None: runtime=OfficialRuntime(p['extractor'])
            attempts=[]; wave=None
            for attempt in range(3):
                try: wave=runtime.waveform(track); break
                except IntegrityError: raise
                except (OSError,ValueError,RuntimeError) as exc:
                    attempts.append(dict(attempt=attempt,error=repr(exc),time=time.time()))
            if wave is None:
                for i in needed:
                    cache.error(track,i,'input_error_exhausted','Cannot read source after3 attempts',attempts)
                    counts['input_error_exhausted']+=1; state['new']+=1
                continue
            full=len(wave)//CHUNK_SAMPLES
            todo=[]
            for i in needed:
                if i>=full:
                    cache.error(track,i,'uncovered_source_tail',f'Source has {len(wave)} samples; chunk requires {(i+1)*CHUNK_SAMPLES}')
                    state['new']+=1; counts['uncovered_source_tail']+=1
                else: todo.append(i)
            def infer(indices):
                waves=[wave[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES] for i in indices]
                errors=[]
                # Model/API/identity startup failures abort, never classify the dataset as bad.
                runtime.load()
                for attempt in range(3):
                    try:
                        out=runtime.infer(waves)
                        for i,w,(raw,gen,native) in zip(indices,waves,out):
                            record=cache.commit(track,i,raw,gen,native,waveform_sha(w),errors)
                            counts[record['analysis']['status']]+=1; state['new']+=1
                        return
                    except IntegrityError: raise
                    except Exception as exc:
                        import torch
                        if not isinstance(exc,(torch.cuda.OutOfMemoryError,OSError,TimeoutError)): raise
                        errors.append(dict(attempt=attempt,error=repr(exc),time=time.time()))
                        torch.cuda.empty_cache()
                        if isinstance(exc,torch.cuda.OutOfMemoryError) and len(indices)>1:
                            mid=len(indices)//2; infer(indices[:mid]); infer(indices[mid:]); return
                for i in indices:
                    cache.error(track,i,'inference_error_exhausted','Temporary execution failure after3 attempts',errors)
                    counts['inference_error_exhausted']+=1; state['new']+=1
            for first in range(0,len(todo),args.batch_size):
                if stopping[0] or (args.plan/'STOP').exists(): break
                infer(todo[first:first+args.batch_size])
                state.update(counts=dict(counts),heartbeat=time.time(),elapsed_s=time.time()-state['started'])
                save(args.plan/'status.json',state)
            del wave
        state.update(stage='stopped' if stopping[0] or (args.plan/'STOP').exists() else 'bounded_run_complete',counts=dict(counts),finished=time.time())
        save(args.plan/'status.json',state)
    print(json.dumps(state))


def assemble(args):
    p=protocol(args.plan);cache=ChunkCache(p['cache_root'],p['extractor'])
    with lock(cache.root/'mutation.lock',blocking=False),connect(args.plan) as db:
        stamp=str(time.time_ns()); out=args.plan/'exports'/stamp; out.mkdir(parents=True)
        names=['retained_all','retained_train','retained_dev','excluded','pending']
        streams={n:(out/(n+'.jsonl')).open('w') for n in names}; counts=Counter(); seconds=Counter(); reasons=Counter(); split_counts=Counter()
        split_seconds=Counter(); breakdown=defaultdict(lambda:dict(pairs=0,seconds=0.))
        try:
            tracks={r['track_id']:json.loads(r['data']) for r in db.execute('SELECT * FROM tracks')}
            with Path(p['selection']).open('rb') as selection:
                for entry in db.execute('SELECT data FROM pairs ORDER BY ordinal'):
                    pair=json.loads(entry['data']); decision=cache.admit_pair(pair,tracks[pair['track_id']],p['selection_sha256'])
                    key=decision['decision'];counts[key]+=1;seconds[key]+=pair['duration'];split_counts[pair['split']+':'+key]+=1
                    split_seconds[pair['split']+':'+key]+=pair['duration']
                    family=pair['sid'].split('_',1)[0]
                    length='lt2.4' if pair['duration']<2.4 else '2.4to4' if pair['duration']<4 else '4to8' if pair['duration']<8 else 'ge8'
                    for subgroup in ('source:'+family,'duration:'+length):
                        item=breakdown[subgroup+':'+key];item['pairs']+=1;item['seconds']+=pair['duration']
                    for failure in decision.get('failures',[]): reasons[failure['status']]+=1
                    record=dict(pair=pair,prosody=decision,full_glare_training_admitted=False)
                    if key=='keep':
                        selection.seek(pair['offset']);raw=selection.readline();require(hashlib.sha256(raw).hexdigest()==pair['row_sha'],'Row changed at export')
                        row=json.loads(raw);row['glare_prosody']=decision;row['full_glare_training_admitted']=False
                        for name in ('retained_all','retained_'+pair['split']):streams[name].write(json.dumps(row,ensure_ascii=False)+'\n')
                    else: streams['excluded' if key=='exclude' else 'pending'].write(json.dumps(record,ensure_ascii=False)+'\n')
        finally:
            for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
        chunk_counts=Counter()
        for row in db.execute('SELECT track_id,chunk_index FROM chunks'):
            rec=cache.load(tracks[row['track_id']],row['chunk_index']);chunk_counts[rec['analysis']['status'] if rec else 'pending']+=1
        resolved=counts['keep']+counts['exclude']
        summary=dict(pairs=p['pairs'],counts=dict(counts),hours={k:v/3600 for k,v in seconds.items()},
            split_counts=dict(split_counts),chunk_status_counts=dict(chunk_counts),excluded_chunk_reasons=dict(reasons),
            split_hours={k:v/3600 for k,v in split_seconds.items()},subgroups=dict(breakdown),
            resolved_retention=counts['keep']/resolved if resolved else None,all_pairs_retention=counts['keep']/max(1,p['pairs']),
            pending_are_not_rejected=True,format_validity_is_not_human_quality=True,full_glare_training_admitted=False)
        save(out/'SUMMARY.json',summary)
        seal=dict(protocol_sha256=sha(args.plan/'protocol.json'),files={f.name:sha(f) for f in out.iterdir() if f.is_file()},created_at=time.time())
        save(out/'SEAL.json',seal);save(args.plan/'LATEST.json',dict(export=str(out.resolve()),seal_sha256=sha(out/'SEAL.json')))
        print(json.dumps(dict(export=str(out),summary=summary)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=BASE/'pilot320')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--samples',type=int,default=320);p.add_argument('--full',action='store_true');p.add_argument('--seed',type=int,default=137);p.add_argument('--cache-root',type=Path,default=BASE/'cache')
    p=sub.add_parser('extract');p.add_argument('--limit',type=int,default=16);p.add_argument('--all',action='store_true');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--retry-execution',action='store_true')
    sub.add_parser('assemble');args=parser.parse_args();args.plan=args.plan.resolve()
    if args.command=='plan': require(args.samples>0,'samples must be positive');plan(args)
    elif args.command=='extract':require(args.limit>0 and 1<=args.batch_size<=16,'Invalid bounded batch/limit');extract(args)
    else: assemble(args)


if __name__=='__main__':main()
