#!/usr/bin/env python3
"""Paired, raw-token audited generation benchmarks; never writes label caches."""
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import ChunkCache,CHUNK_SAMPLES,audit_raw,lock,read,save,sha,IntegrityError
from prepare_glare_prosody_subset import BASE,OLD,protocol,connect,require
from run_glare_source_prosody_full import PackedRuntime


from glare_prosody_execution_opts import install


def sample_inputs(plan,p,count):
    import numpy as np
    cache=ChunkCache(p['cache_root'],p['extractor']);groups={};all_records=[]
    # Explicit current contract cache only. Bound metadata intake for this diagnostic.
    for path in (cache.root/'chunks').glob('*/*.json'):
        r=read(path)
        if r['analysis']['status'] in {'explicit_null','events','below_threshold','refusal','truncated','parse_error'}:
            item=(r['identity']['track_id'],r['identity']['chunk_index'],str(path))
            groups.setdefault(r['analysis']['status'],[]).append(item);all_records.append(item)
        if len(all_records)>=8192:break
    selected=[]
    for n in range(count):
        for status in sorted(groups):
            if n<len(groups[status]) and len(selected)<count:selected.append(groups[status][n])
    require(len(selected)==count,'Insufficient cached real chunks for paired diagnostic')
    runtime=PackedRuntime(p['extractor']);waves=[];references=[]
    by_track={}
    for tid,index,path in selected:by_track.setdefault(tid,[]).append((index,path))
    with connect(plan) as db:
        for tid,items in sorted(by_track.items()):
            track=json.loads(db.execute('SELECT data FROM tracks WHERE track_id=?',(tid,)).fetchone()['data'])
            wave=runtime.waveform(track)
            for index,path in sorted(items):
                part=wave[index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES].copy()
                require(len(part)==CHUNK_SAMPLES,'Diagnostic source tail')
                waves.append(part);references.append(dict(track_id=tid,chunk_index=index,receipt=path))
    return runtime,waves,references


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=BASE/'full');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=64);a=parser.parse_args();p=protocol(a.plan)
    a.out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock'):
            locks.enter_context(lock(path,False))
        runtime,waves,refs=sample_inputs(a.plan,p,a.samples);runtime.load()
        import torch
        # Preserve other jobs and the production32GiB ceiling throughout this probe.
        torch.cuda.reset_peak_memory_stats();runtime.infer(waves[:16])
        report=dict(stage='running',started=time.time(),samples=len(waves),references=refs,
                    script_sha256=sha(__file__),overlay_source_sha256=sha(ROOT/'src/glare_prosody_execution_opts.py'),lm_source_sha256=sha(inspect.getfile(type(runtime.analyzer.model.language_model))),results=[])
        save(a.out/'RESULT.json',report)
        def evaluate(mode,batch):
            restore=install(runtime,mode)
            outputs=[];tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:
                for first in range(0,len(waves),batch):
                    for raw,gen,native in runtime.infer(waves[first:first+batch]):
                        outputs.append(dict(raw=raw,tokens=gen['token_ids'],native=native,analysis=audit_raw(raw,gen['truncated'])))
                torch.cuda.synchronize()
                result=dict(mode=mode,batch=batch,seconds=time.monotonic()-tick,outputs=outputs,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,error=None)
            except torch.cuda.OutOfMemoryError as exc:
                result=dict(mode=mode,batch=batch,seconds=time.monotonic()-tick,outputs=[],error=repr(exc))
            finally:restore()
            gc.collect();torch.cuda.empty_cache()
            report['results'].append(result);save(a.out/'RESULT.json',report)
            print(json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True)
            return result
        # Alternate order so an incidental GPU-load change is visible.
        native=evaluate('native',16);short=evaluate('last_logits',16)
        short2=evaluate('last_logits',16);native2=evaluate('native',16)
        baseline=native['outputs']
        for result in report['results']:
            result['same_raw_tokens_normalization_and_audit']=not result['error'] and result['outputs']==baseline
        trim=evaluate('trim_audio',16)
        trim['same_raw_tokens_normalization_and_audit']=not trim['error'] and trim['outputs']==baseline
        packed=evaluate('trim_audio',32)
        packed['same_raw_tokens_normalization_and_audit']=not packed['error'] and packed['outputs']==baseline
        valid_native=not native['error'] and native2['outputs']==baseline
        valid_short=valid_native and all(r['same_raw_tokens_normalization_and_audit'] for r in (short,short2))
        base_s=(native['seconds']+native2['seconds'])/2
        opt_s=(short['seconds']+short2['seconds'])/2
        report.update(stage='complete',finished=time.time(),native_repeat_equal=valid_native,
            last_logits_equal=valid_short,last_logits_speedup=base_s/opt_s,
            trim_audio_equal=trim['same_raw_tokens_normalization_and_audit'],
            trim_audio_speedup=base_s/trim['seconds'] if not trim['error'] else None,
            batch32_equal=packed['same_raw_tokens_normalization_and_audit'],
            batch32_speedup_vs_native16=base_s/packed['seconds'] if not packed['error'] else None,
            production_cache_modified=False,official_prompt_or_generation_limit_modified=False)
        save(a.out/'RESULT.json',report)


if __name__=='__main__':main()
