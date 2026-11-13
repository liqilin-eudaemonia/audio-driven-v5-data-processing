#!/usr/bin/env python3
"""Negative/time-shift controls for prosody grounding; full task stays paused."""
import argparse
from contextlib import ExitStack
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import generation_rows,lock,read,save,sha
from prepare_glare_prosody_subset import BASE,OLD,protocol,require
from run_glare_source_prosody_full import PackedRuntime
from diagnose_glare_prosody_extractor import strict_audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--mixed-only',action='store_true');a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    p=protocol(BASE/'full');old=read(BASE/'extractor_diagnosis/RESULT.json')
    require((BASE/'full/STOP').exists(),'Full task must remain explicitly paused')
    with ExitStack() as guards:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock'):guards.enter_context(lock(path,False))
        import numpy as np
        import soundfile as sf
        waves=[]
        for i in range(len(old['samples'])):
            wave,sr=sf.read(BASE/'extractor_diagnosis'/f'sample_{i:02}.wav',dtype='float32');assert sr==16000
            waves.append(wave)
        silent=[np.zeros_like(w) for w in waves]
        delayed=[np.concatenate([np.zeros(9600,np.float32),w[:-9600]]) for w in waves]
        padded=[np.concatenate([w,np.zeros_like(w)]) for w in waves]
        for i,w in enumerate(delayed):sf.write(a.out/f'delayed_{i:02}.wav',w,16000,subtype='FLOAT')
        sf.write(a.out/'silence.wav',silent[0],16000,subtype='FLOAT')
        runtime=PackedRuntime(p['extractor']);normal=runtime.native.normalize_result;runtime.load()
        analyzer=runtime.analyzer;original_generate=analyzer.model.generate
        def generate(**kwargs):return original_generate(**dict(kwargs,repetition_penalty=1.1))
        analyzer.model.generate=generate;analyzer.max_new_tokens=128
        report=dict(stage='running',started=time.time(),script_sha256=sha(__file__),experiments=[],full_stays_paused=True,
            source_samples=old['samples'],silence_pcm_all_zero=all(not w.any() for w in silent),delay_seconds=.6)
        def run(name,inputs,duration,prompt,prosody=True):
            analyzer.prompt_text=analyzer._build_prompt_text(prompt);runtime.raw=[];runtime.rows=[];tick=time.monotonic()
            tensor_inputs=analyzer._build_model_inputs(inputs,16000)
            feature_shas=[hashlib.sha256(f.detach().cpu().numpy().tobytes()).hexdigest() for f in tensor_inputs['input_features']]
            valid_frames=tensor_inputs['feature_attention_mask'].sum(-1).tolist();del tensor_inputs
            analyzer.infer_chunks(inputs,16000,duration)
            rows=[]
            for raw,gen in zip(runtime.raw,runtime.rows):
                g=generation_rows([gen['token_ids']],gen['eos_ids'],128)[0]
                rows.append(dict(raw=raw,generation=g,analysis=strict_audit(raw,duration,g['truncated']) if prosody else None,
                    native_normalized=normal(raw,max_time=duration) if prosody else None))
            r=dict(name=name,duration=duration,prompt=prompt,repetition_penalty=1.1,max_new_tokens=128,
                seconds=time.monotonic()-tick,feature_sha256=feature_shas,valid_mel_frames=valid_frames,rows=rows)
            if prosody:r['status_counts']=dict(Counter(row['analysis']['status'] for row in rows))
            report['experiments'].append(r);save(a.out/'RESULT.json',report)
            print(json.dumps({k:v for k,v in r.items() if k not in ['rows','prompt','feature_sha256']}),flush=True)
        def duration_prompt(duration):
            return p['extractor']['prompt']+f'\nThis audio is exactly{duration:.2f} seconds long. All timestamps refer to this audio only: 0.00 through{duration:.2f} seconds.'
        if a.mixed_only:
            mixed=[w.copy() for w in waves]
            for w in mixed:w[12800:]=0.
            report['mixed_full_rms']=[float(np.sqrt(np.mean(w.astype('float64')**2))) for w in mixed]
            report['mixed_exact_zero_interval']=[.8,2.4]
            for i,w in enumerate(mixed):sf.write(a.out/f'mixed_{i:02}.wav',w,16000,subtype='FLOAT')
            run('speech0to0.8_silence0.8to2.4',mixed,2.4,duration_prompt(2.4))
            run('same_mixed_original_prompt',mixed,2.4,p['extractor']['prompt'])
            for experiment in report['experiments']:
                for i,row in enumerate(experiment['rows']):
                    row['predicted_event_rms']=[]
                    for event in row['analysis']['events']:
                        chunk=mixed[i][max(0,round(event['start']*16000)):min(38400,round(event['end']*16000))]
                        row['predicted_event_rms'].append(float(np.sqrt(np.mean(chunk.astype('float64')**2))) if len(chunk) else None)
        else:
            run('real2.4s',waves,2.4,duration_prompt(2.4))
            run('silent2.4s',silent,2.4,duration_prompt(2.4))
            run('same_speech_delayed0.6s',delayed,2.4,duration_prompt(2.4))
            run('same_speech_append2.4s_silence',padded,4.8,duration_prompt(4.8))
            run('silent4.8s',[np.zeros_like(w) for w in padded],4.8,duration_prompt(4.8))
            existence=('Listen carefully. Is any human speech audible? Answer SPEECH if speech is audible, or SILENT if the recording is silent. Do not transcribe.')
            run('speech_presence_real',waves,2.4,existence,False)
            run('speech_presence_silence',silent,2.4,existence,False)
        report.update(stage='complete',finished=time.time(),production_cache_modified=False)
        save(a.out/'RESULT.json',report)


if __name__=='__main__':main()
