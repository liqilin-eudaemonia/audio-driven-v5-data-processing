#!/usr/bin/env python3
"""Official normalization/zero defaults; strict raw audit is diagnostic only."""
import argparse
from collections import Counter
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import ChunkCache,IntegrityError,canonical,check_generation,frame_count,lock,read,save,sha,required_chunks
from prepare_glare_prosody_subset import BASE,protocol,connect,require


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class OfficialPolicy:
    def __init__(self,extractor):
        offline=load_module(extractor['official_script'],'official_glare_offline_normalizer')
        encoder=load_module(ROOT/'related/glare/float/models/intensity_encoder.py','official_glare_intensity_vectors')
        self.normalize=offline.normalize_result
        cls=encoder.IntensityEncoder
        self.vector=lambda text,n:cls.text_to_frame_vector(SimpleNamespace(default_fps=25.,
            _time_to_start_frame_1_based=cls._time_to_start_frame_1_based,
            _time_to_end_frame_1_based=cls._time_to_end_frame_1_based),text,n,fps=25.).numpy()
        self.identity=dict(policy='official-normalize-and-vector-v1',
            normalizer_sha256=sha(extractor['official_script']),
            frame_encoder_sha256=sha(ROOT/'related/glare/float/models/intensity_encoder.py'),
            adapter_sha256=sha(__file__),NULL='official zero vector; not raw-output rejection',
            tail='official floor chunks skipped; frame encoder default zero outside annotations')

    def map_pair(self,pair,records):
        from numpy import ones
        start,end=Decimal(str(pair['start'])),Decimal(str(pair['end']));lines=[];tails=[];nulls=[]
        for i in required_chunks(pair['start'],pair['end']):
            r=records.get(i)
            if r is None:return dict(decision='pending',reason='chunk_not_processed',chunk_index=i)
            if r['analysis']['status']=='uncovered_source_tail':tails.append(i);continue
            if r['analysis']['status'] in {'input_error_exhausted','inference_error_exhausted'}:
                return dict(decision='failed_execution',reason=r['analysis']['status'],chunk_index=i)
            require(not r['generation'].get('early_reject'),'Early prefix is not a complete official generation')
            check_generation(r['generation'])
            text=self.normalize(r['analysis']['raw_text'],max_time=2.4)
            require(text==r['official_normalized_text'],'Stored official normalizer output changed')
            if text=='NULL':nulls.append(i);continue
            origin=Decimal(i)*Decimal('2.4')
            for line in text.splitlines():
                a,b,score=[Decimal(v.strip()) for v in line.split(',')]
                a=max(start,origin+a);b=min(end,origin+b)
                if b>a:lines.append(f'{a-start:f}, {b-start:f}, {score:f}')
        mapped='\n'.join(lines) or 'NULL';values=self.vector(mapped,frame_count(pair['duration']))
        return dict(decision='keep',intensity=values,intensity_valid=ones(values.shape,bool),mapped_text=mapped,
                    official_NULL_chunks=nulls,official_skipped_tail_chunks=tails)


def assemble(plan):
    import numpy as np
    p=protocol(plan);raw_cache=ChunkCache(p['cache_root'],p['extractor']);policy=OfficialPolicy(p['extractor'])
    policy_id=canonical(dict(extraction_contract_id=raw_cache.contract_id,policy=policy.identity))
    cache=BASE/'official_conditions'/policy_id;counts=Counter();seconds=Counter();chunk_counts=Counter()
    with lock(raw_cache.root/'mutation.lock',False),connect(plan) as db:
        out=plan/'official_exports'/str(time.time_ns());out.mkdir(parents=True)
        names=['retained_all','retained_train','retained_dev','failed_execution','pending']
        files={name:(out/(name+'.jsonl')).open('w') for name in names}
        try:
            with Path(p['selection']).open('rb') as selected:
                for track_row in db.execute('SELECT * FROM tracks ORDER BY track_id'):
                    track=json.loads(track_row['data']);records={}
                    for row in db.execute('SELECT chunk_index FROM chunks WHERE track_id=? ORDER BY chunk_index',(track['track_id'],)):
                        i=row['chunk_index'];_,receipt=raw_cache.paths(track,i)
                        if receipt.exists():
                            r=read(receipt);require(r['identity']==raw_cache.identity(track,i),'Source receipt identity mismatch')
                            records[i]=r
                            chunk_counts[r['official_normalized_text'] if r.get('official_normalized_text')=='NULL' else r['analysis']['status']]+=1
                    for row in db.execute('SELECT data FROM pairs WHERE track_id=? ORDER BY ordinal',(track['track_id'],)):
                        pair=json.loads(row['data']);result=policy.map_pair(pair,records);decision=result['decision']
                        counts[decision]+=1;seconds[pair['split']+':'+decision]+=pair['duration']
                        if decision!='keep':files[decision].write(json.dumps(dict(pair=pair,official_condition=result))+'\n');continue
                        key=canonical(dict(sample_id=pair['sid'],row_sha256=pair['row_sha']))
                        folder=cache/key[:2];folder.mkdir(parents=True,exist_ok=True);payload=folder/(key+'.npz');receipt=folder/(key+'.json')
                        identity=dict(policy_id=policy_id,source_contract_id=raw_cache.contract_id,pair=pair,
                            chunk_receipts={str(i):sha(raw_cache.paths(track,i)[1]) for i in required_chunks(pair['start'],pair['end'])})
                        if receipt.exists():
                            old=read(receipt);require(old['identity']==identity and sha(payload)==old['payload_sha256'],'Official condition cache changed')
                        else:
                            tmp=payload.with_name(payload.name+'.tmp.'+str(os.getpid()))
                            with tmp.open('wb') as f:np.savez_compressed(f,intensity=result['intensity'],intensity_valid=result['intensity_valid'])
                            tmp.replace(payload)
                            save(receipt,dict(identity=identity,payload_sha256=sha(payload),mapped_text=result['mapped_text'],
                                official_NULL_chunks=result['official_NULL_chunks'],official_skipped_tail_chunks=result['official_skipped_tail_chunks'],
                                policy='official',human_verified=False,training_condition_usable=True))
                        selected.seek(pair['offset']);raw=selected.readline()
                        import hashlib
                        require(hashlib.sha256(raw).hexdigest()==pair['row_sha'],'Selected row changed')
                        original=json.loads(raw);original['glare_prosody']=dict(policy='official',intensity=str(payload),intensity_receipt=str(receipt),
                            intensity_receipt_sha256=sha(receipt),strict_raw_audit_is_diagnostic_only=True)
                        for name in ('retained_all','retained_'+pair['split']):files[name].write(json.dumps(original,ensure_ascii=False)+'\n')
        finally:
            for f in files.values():f.flush();os.fsync(f.fileno());f.close()
        summary=dict(policy=policy.identity,pairs=p['pairs'],counts=dict(counts),hours={k:v/3600 for k,v in seconds.items()},
            chunk_status_counts=dict(chunk_counts),strict_audit_does_not_exclude_pairs=True,human_verified=False,
            training_ready_requires_other_feature_caches=True)
        save(out/'SUMMARY.json',summary);save(out/'SEAL.json',dict(protocol_sha256=sha(plan/'protocol.json'),policy_id=policy_id,
            files={path.name:sha(path) for path in out.iterdir() if path.is_file()},created_at=time.time()))
        save(plan/'LATEST.json',dict(export=str(out),seal_sha256=sha(out/'SEAL.json'),policy='official'))
        print(json.dumps(dict(export=str(out),summary=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=BASE/'full');a=p.parse_args();assemble(a.plan.resolve())
