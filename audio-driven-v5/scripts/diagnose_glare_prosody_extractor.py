#!/usr/bin/env python3
"""Isolated bounded ablations; no production-cache writes or full restart."""
import argparse
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import (audit_raw,generation_rows,lock,read,save,sha,CHUNK_SAMPLES,NUMERIC_LINE)
from glare_prosody_audit import parse_reply
from prepare_glare_prosody_subset import BASE,OLD,protocol,connect,require
from run_glare_source_prosody_full import PackedRuntime


def strict_audit(raw,duration,truncated):
    if abs(duration-2.4)<1e-8:return audit_raw(raw,truncated)
    if raw.strip().upper()!='NULL' and any(not NUMERIC_LINE.fullmatch(line.strip()) for line in raw.splitlines() if line.strip()):
        return dict(status='parse_error',accepted=False,raw_text=raw,events=[],normalized_text=None)
    return parse_reply(raw,duration,truncated=truncated)


def samples(plan,p,count):
    refs=read(BASE/'capacity_probe/RESULT.json')['references'];chosen=[];seen=set()
    for ref in refs:
        receipt=read(ref['receipt']);tid=ref['track_id']
        if receipt['analysis']['status'] not in {'refusal','truncated','parse_error'} or tid in seen:continue
        chosen.append(dict(ref,old_raw=receipt['analysis']['raw_text'],old_status=receipt['analysis']['status']));seen.add(tid)
        if len(chosen)==count:break
    require(len(chosen)==count,'Insufficient distinct failed source chunks')
    runtime=PackedRuntime(p['extractor']);waves=[];fullpairs=[];metadata=[]
    with connect(plan) as db,Path(p['selection']).open('rb') as selection:
        for ref in chosen:
            tid,index=ref['track_id'],ref['chunk_index'];track=json.loads(db.execute('SELECT data FROM tracks WHERE track_id=?',(tid,)).fetchone()['data'])
            wave=runtime.waveform(track);part=wave[index*CHUNK_SAMPLES:(index+1)*CHUNK_SAMPLES].copy()
            candidates=[json.loads(r['data']) for r in db.execute('SELECT data FROM pairs WHERE track_id=?',(tid,))]
            pair=max(candidates,key=lambda item:max(0,min(item['end'],(index+1)*2.4)-max(item['start'],index*2.4)))
            selection.seek(pair['offset']);raw=selection.readline();require(hashlib.sha256(raw).hexdigest()==pair['row_sha'],'Pair row changed')
            row=json.loads(raw);clip=read(row['source']['source_result'])['clips'][row['source']['source_clip_index']]
            full=wave[round(pair['start']*16000):round(pair['end']*16000)].copy()
            meta=dict(**ref,sample_id=pair['sid'],speaker_id=track['speaker_id'],pair_start=pair['start'],pair_end=pair['end'],
                source_chunk_start=index*2.4,source_chunk_end=(index+1)*2.4,pair_speaker_transcript=clip['text'],
                rms=float((part.astype('float64')**2).mean()**.5),peak=float(abs(part).max()),duration=len(part)/16000,
                original_speaker_path=track['audio']['path'],speaker_sha256=track['audio']['sha256'])
            waves.append(part);fullpairs.append(full);metadata.append(meta)
    return runtime,waves,fullpairs,metadata


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=BASE/'full');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=8);a=parser.parse_args();p=protocol(a.plan);a.out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as guards:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock'):guards.enter_context(lock(path,False))
        runtime,waves,fullpairs,metadata=samples(a.plan,p,a.samples)
        import soundfile as sf
        import numpy as np
        for i,wave in enumerate(waves):sf.write(a.out/f'sample_{i:02}.wav',wave,16000,subtype='FLOAT')
        native_normalize=runtime.native.normalize_result
        runtime.load();model=runtime.analyzer.model;analyzer=runtime.analyzer
        original_generate=model.generate
        report=dict(stage='running',started=time.time(),code_sha256=sha(__file__),samples=metadata,
            production_cache_modified=False,full_stays_paused=True,original_prompt=p['extractor']['prompt'],
            loaded_generation_config=model.generation_config.to_dict(),experiments=[])
        save(a.out/'RESULT.json',report)
        def run(name,input_waves,prompt,rp,tokens,durations=None):
            analyzer.prompt_text=analyzer._build_prompt_text(prompt);analyzer.max_new_tokens=tokens
            def generate(**kwargs):return original_generate(**dict(kwargs,repetition_penalty=rp))
            model.generate=generate;runtime.raw=[];runtime.rows=[];tick=time.monotonic()
            # Official routine unchanged apart from explicitly named diagnostic variable.
            analyzer.infer_chunks(input_waves,16000,2.4)
            rows=[]
            for i,(raw,gen) in enumerate(zip(runtime.raw,runtime.rows)):
                generation=generation_rows([gen['token_ids']],gen['eos_ids'],tokens)[0]
                duration=durations[i] if durations is not None else len(input_waves[i])/16000
                audit=strict_audit(raw,duration,generation['truncated'])
                rows.append(dict(index=i,raw=raw,generation=generation,analysis=audit,
                    native_normalized=native_normalize(raw,max_time=duration)))
            result=dict(name=name,repetition_penalty=rp,max_new_tokens=tokens,prompt=prompt,
                seconds=time.monotonic()-tick,status_counts=dict(Counter(r['analysis']['status'] for r in rows)),rows=rows)
            report['experiments'].append(result);save(a.out/'RESULT.json',report)
            print(json.dumps({k:v for k,v in result.items() if k not in ['rows','prompt']}),flush=True)
            return result
        original=p['extractor']['prompt']
        run('official_greedy_inherited_rep1.1_max64',waves,original,1.1,64)
        run('only_disable_repetition_penalty',waves,original,1.,64)
        run('only_raise_token_limit',waves,original,1.1,256)
        run('rep1.0_max256',waves,original,1.,256)
        duration_prompt=original+'\nThis audio is exactly2.40 seconds long. All timestamps refer to this audio only: 0.00 through2.40 seconds.'
        run('only_explicit_duration',waves,duration_prompt,1.1,64)
        run('duration_rep1.0_max256',waves,duration_prompt,1.,256)
        short=('Listen to this2.40-second speech recording. Find audible changes in pitch, loudness, stress or rhythm. '
               'Return only numeric start,end,intensity rows using real line breaks. Times must be within0.00 to2.40 seconds; '
               'intensity must be0.50 to1.00. If no clear localized event is audible, return exactlyNULL. Do not invent times.')
        run('short_prompt_rep1.0_max256',waves,short,1.,256)
        asr='Transcribe the speech in this audio. Return only the spoken words.'
        real_asr=run('asr_real_audio',waves[:4],asr,1.,128)
        run('asr_same_duration_silence',[np.zeros_like(w) for w in waves[:4]],asr,1.,128)
        run('describe_prosody_natural_language',waves[:4],'Describe how the person speaks in this audio: tone, loudness, pitch and pace.',1.,128)
        for i in range(2):
            run(f'full_original_pair_rep1.0_{i}',[fullpairs[i]],original,1.,256)
        model.generate=original_generate
        report.update(stage='complete',finished=time.time(),all_inputs_audio_route_verified=True)
        save(a.out/'RESULT.json',report)


if __name__=='__main__':main()
