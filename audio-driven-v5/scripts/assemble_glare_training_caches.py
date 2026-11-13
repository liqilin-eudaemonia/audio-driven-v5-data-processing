#!/usr/bin/env python3
"""Publish one cache registry containing every required GLARE training input.

Reads receipts and indices, never decodes media or starts GPU models. Existing
payload hashes are retained; --verify-payloads additionally rereads tensors.
This prepares data, not the separately required production training reader.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster
from glare_cluster_frozen_audio import cached,cache_paths
from glare_prosody_subset import canonical


def payload_descriptor(payload,receipt,record,verify_payload=False):
    payload=Path(payload);receipt=Path(receipt)
    assert payload.is_file() and payload.stat().st_size>0,'Missing/empty payload: '+str(payload)
    assert len(record['payload_sha256'])==64,'Invalid payload digest'
    if verify_payload:assert cluster.sha(payload)==record['payload_sha256'],'Payload checksum mismatch: '+str(payload)
    return dict(payload=str(payload),receipt=str(receipt),payload_sha256=record['payload_sha256'],
                receipt_sha256=cluster.sha(receipt))


def collect(cfg,m,job,row,verify_payload=False):
    sid=job['sid'];old=Path(cfg['shared_lia']);p=m['shared_protocol']
    assert row['sample_id']==sid and row['split']==job['split'] and job['split'] in ('train','dev')
    identity=dict(schema=p['schema'],sample_id=sid,row_sha256=job['row_sha'],
                  selection_sha256=m['selection_sha256'],code_sha256=canonical(p['code']))
    folder=old/'cache'/__import__('hashlib').sha256(sid.encode()).hexdigest()[:2]
    result={}
    for kind in ('lia','audio','reaction','statistics'):
        receipt=folder/(sid+'.'+kind+'.json');r=cluster.read(receipt)
        assert r['identity']==identity,'Nonprosody identity mismatch: '+kind
        result[kind]=payload_descriptor(r['payload'],receipt,r,verify_payload)
    result['lia']['tensor_key']='motion';result['audio']['tensor_key']='audio'
    result['reaction']['array_keys']=['reaction','reaction_valid']
    audio_root=old/'frozen_audio'
    assert cached(audio_root,job,m['frozen_audio_protocol'],verify_payload),'Frozen audio not extracted'
    payload,receipt=cache_paths(audio_root,sid);r=cluster.read(receipt)
    result['frozen_audio']=payload_descriptor(payload,receipt,r)
    result['frozen_audio'].update(wav2vec_shape=r['shape'],emotion_shape=[7],pre_projection=True,
                                 waveform_samples=r['waveform_samples'])
    prosody=row['glare_prosody'];receipt=Path(prosody['intensity_receipt']);r=cluster.read(receipt)
    assert cluster.sha(receipt)==prosody['intensity_receipt_sha256'],'Prosody receipt changed'
    pair=r['identity']['pair'];frames=math.ceil(job['duration']*25-1e-7)
    assert pair['sid']==sid and pair['row_sha']==job['row_sha'] and pair['split']==job['split']
    assert pair['frames']==frames and r['identity']['source_contract_id']==canonical(m['cache_contract'])
    assert r['training_condition_usable'] is True and r['policy']=='official'
    result['prosody']=payload_descriptor(prosody['intensity'],receipt,r,verify_payload)
    result['prosody']['array_keys']=['intensity','intensity_valid']
    result['view']=dict(fps=25,frames=frames,reference_frame=0,target_frames=frames-1,history_frames=0,
                       frozen_audio_protocol_sha256=canonical(m['frozen_audio_protocol']))
    return result


def assemble(cfg,m,verify_payload=False):
    latest=cluster.read(Path(cfg['source_plan'])/'LATEST.json');source=Path(latest['export'])
    assert latest['policy']=='official' and cluster.sha(source/'SEAL.json')==latest['seal_sha256']
    summary=cluster.read(source/'SUMMARY.json')
    assert summary['counts'].get('keep',0)==m['pairs'] and not summary['counts'].get('pending',0) and not summary['counts'].get('failed_execution',0)
    out=Path(cfg['work'])/'training_cache_exports'/str(time.time_ns());out.mkdir(parents=True)
    counts=Counter();seen=set();failed=0
    with cluster.locked(Path(cfg['work'])/'training_cache_assemble.lock'),ExitStack() as stack:
        streams={s:stack.enter_context((out/(s+'.jsonl')).open('w')) for s in ('train','dev','unavailable')}
        db=stack.enter_context(cluster.ro(Path(m['parent'])/'jobs.sqlite'));db.row_factory=sqlite3.Row
        original=stack.enter_context((source/'retained_all.jsonl').open())
        for raw in original:
            row=json.loads(raw);sid=row['sample_id'];assert sid not in seen,'Duplicate pair';seen.add(sid)
            record=db.execute('SELECT * FROM jobs WHERE sid=?',(sid,)).fetchone();assert record is not None,'Unknown pair'
            job=dict(record)
            try:row['glare_caches']=collect(cfg,m,job,row,verify_payload)
            except (OSError,ValueError,AssertionError,KeyError) as exc:
                failed+=1;streams['unavailable'].write(json.dumps(dict(sample_id=sid,error=repr(exc)))+'\n');continue
            counts[row['split']]+=1;streams[row['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
        assert len(seen)==m['pairs'],'Prosody export coverage mismatch'
        for f in streams.values():f.flush();os.fsync(f.fileno())
    result=dict(run_id=m['run_id'],pairs=m['pairs'],counts=dict(counts),unavailable=failed,
                required_caches=m['required_caches'],payload_hashes_reread=verify_payload,
                cached_view='first reference frame; no history; N-1 targets; unchanged existing audio contract',
                cache_registry_complete=failed==0,production_training_reader_ready=False,
                source_prosody_export=str(source),created_at=time.time())
    cluster.save(out/'SUMMARY.json',result)
    cluster.save(out/'SEAL.json',dict(files={p.name:cluster.sha(p) for p in out.iterdir() if p.is_file()},
                                    manifest_sha256=cluster.sha(Path(cfg['work'])/'manifest.json')))
    assert failed==0,'Required caches missing/invalid; see '+str(out/'unavailable.jsonl')
    cluster.save(Path(cfg['work'])/'TRAINING_CACHES_LATEST.json',dict(export=str(out),seal_sha256=cluster.sha(out/'SEAL.json')))
    return dict(export=str(out),summary=result)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,default=cluster.DEFAULT)
    ap.add_argument('--verify-payloads',action='store_true');a=ap.parse_args()
    cfg=cluster.config(a.config);print(json.dumps(assemble(cfg,cluster.load_run(cfg),a.verify_payloads)))
