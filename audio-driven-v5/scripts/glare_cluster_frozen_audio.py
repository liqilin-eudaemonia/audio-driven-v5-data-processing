"""Reuse/repair frozen Wav2Vec2+SER alongside each existing LIA task.

Complete old caches need no model or media decoding. Receipts bind the source
row, exact audio window and native protocol. Full payload hashing is available
explicitly; normal reuse avoids rereading the entire completed feature corpus.
"""
import hashlib
import math
from pathlib import Path

import glare_cache_cluster as cluster


def cache_paths(root,sid):
    assert Path(sid).name==sid and sid not in ('.','..')
    folder=Path(root)/'cache'/hashlib.sha256(sid.encode()).hexdigest()[:2]
    return folder/(sid+'.pt'),folder/(sid+'.json')


def cached(root,job,protocol,verify_payload=False):
    payload,receipt=cache_paths(root,job['sid'])
    if not receipt.exists():return False
    r=cluster.read(receipt)
    # Match the canonical encoder protocol hash, including non-ASCII settings.
    from glare_prosody_subset import canonical
    assert r['identity']==dict(sample_id=job['sid'],row_sha256=job['row_sha'],protocol_sha256=canonical(protocol)), 'Frozen audio source/protocol mismatch'
    frames=math.ceil(job['duration']*25-1e-7)-1
    assert r['shape']==[frames,9216] and r['waveform_samples']==frames*640, 'Frozen audio window/shape mismatch'
    assert r['pre_projection'] is True and len(r['payload_sha256'])==64, 'Invalid frozen audio receipt'
    if not payload.is_file() or payload.stat().st_size==0:return False
    if verify_payload:assert cluster.sha(payload)==r['payload_sha256'], 'Frozen audio payload checksum mismatch'
    return True


def task_jobs(cfg,m,task):
    import json
    with cluster.ro(Path(cfg['shared_lia'])/'groups.sqlite') as db:
        db.execute('ATTACH DATABASE ? AS routing',(str(Path(cfg['work'])/'routing.sqlite'),))
        ordinals=[o for (text,) in db.execute('SELECT g.ordinals FROM groups g JOIN routing.lia_assign a '
                  'ON a.gid=g.gid WHERE a.task_id=? ORDER BY g.gid',(task['task_id'],)) for o in json.loads(text)]
    with cluster.ro(Path(m['parent'])/'jobs.sqlite') as db:
        import sqlite3
        db.row_factory=sqlite3.Row
        return [dict(db.execute('SELECT * FROM jobs WHERE ordinal=?',(o,)).fetchone()) for o in ordinals]


def ensure(cfg,m,node,gpu,folder,jobs,engines):
    if (Path(cfg['work'])/'STOP').exists():return dict(validated=False,stopped=True)
    root=Path(cfg['shared_lia'])/'frozen_audio';protocol=m['frozen_audio_protocol']
    missing=[j for j in jobs if not cached(root,j,protocol)]
    reused=len(jobs)-len(missing)
    if not missing:
        return dict(validated=True,total=len(jobs),reused=reused,completed=0,
                    validation='source identity + exact window + committed receipt + payload presence; no corpus rehash')
    if cfg.get('benchmark'):raise cluster.TaskDataError('Benchmark requires existing frozen audio; never repair production data during a replay')
    # Import/load CUDA encoders only for genuinely missing entries, preserving
    # the original encoders, FP32, input windows and pre-projection payloads.
    import glare_audio_execution_v2 as native
    import glare_frozen_audio_cache as core
    import torch
    core.WORK=root
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    if not engines:engines.append(native.Engine(gpu,assigned_gpus=tuple(node['gpus'])))
    engine=engines[0]
    original_write=engine.write
    engine.write=lambda job,audio,n,hidden,emotion,_root:original_write(job,audio,n,hidden,emotion,root)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        for job in missing:
            _,receipt=cache_paths(root,job['sid'])
            if receipt.exists():
                history=root/'retry_history'/(job['sid']+'.'+str(__import__('time').time_ns())+'.json')
                history.parent.mkdir(parents=True,exist_ok=True);receipt.replace(history)
        state=engine.run(missing,Path(folder)/'frozen_audio_exec',asynchronous=True,reuse=False)
    finally:
        engine.write=original_write
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
    if state['stage']=='stopped':return dict(validated=False,stopped=True,native_state=state)
    if state['failed']:raise cluster.TaskDataError('Frozen Wav2Vec2/SER execution errors: '+str(state['failed']))
    assert state['completed']==len(missing) and all(cached(root,j,protocol) for j in missing), 'Frozen audio repair incomplete'
    return dict(validated=True,total=len(jobs),reused=reused,completed=len(missing),native_state=state)
