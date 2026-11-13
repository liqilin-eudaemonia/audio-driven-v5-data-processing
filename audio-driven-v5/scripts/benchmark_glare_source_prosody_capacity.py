#!/usr/bin/env python3
"""Capacity/backend/irreversible-prefix experiments; no production label writes."""
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from types import MethodType
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import audit_raw,lock,read,save,sha,IntegrityError
from prepare_glare_prosody_subset import BASE,OLD,protocol,require
from benchmark_glare_source_prosody_decode import sample_inputs
from glare_prosody_prefix_reject import install as early_install,impossible_prefix


def sdpa_install(runtime):
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioSdpaAttention
    restores=[]
    for layer in runtime.analyzer.model.audio_tower.layers:
        module=layer.self_attn
        if type(module).__name__=='Qwen2AudioAttention':
            old=module.forward
            module.forward=MethodType(Qwen2AudioSdpaAttention.forward,module)
            restores.append((module,old))
    return lambda:[setattr(m,'forward',fn) for m,fn in restores]


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=BASE/'full');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64);a=p.parse_args();protocol_data=protocol(a.plan);a.out.mkdir(parents=True,exist_ok=True)
    with ExitStack() as locks:
        for path in (OLD/'run_all.lock',OLD/'controller.lock',OLD/'workers/gpu7.lock'):locks.enter_context(lock(path,False))
        runtime,waves,refs=sample_inputs(a.plan,protocol_data,a.samples);runtime.load()
        import torch
        free,total=torch.cuda.mem_get_info();cap=min(56*1024**3,torch.cuda.memory_reserved()+free-16*1024**3)
        require(cap>=32*1024**3,'Cannot enlarge memory without16GiB reserve')
        torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
        report=dict(stage='running',samples=len(waves),references=refs,started=time.time(),memory_cap_gib=cap/1024**3,
            script_sha256=sha(__file__),prefix_source_sha256=sha(ROOT/'src/glare_prosody_prefix_reject.py'),
            audio_attention_class=type(runtime.analyzer.model.audio_tower.layers[0].self_attn).__name__,
            text_attention_class=type(runtime.analyzer.model.language_model.model.layers[0].self_attn).__name__,results=[])
        save(a.out/'RESULT.json',report)
        runtime.infer(waves[:16])
        def evaluate(mode,batch):
            restores=[]
            if 'sdpa' in mode:restores.append(sdpa_install(runtime))
            if 'early' in mode:restores.append(early_install(runtime))
            out=[];torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
            try:
                for first in range(0,len(waves),batch):
                    for raw,gen,native in runtime.infer(waves[first:first+batch]):
                        out.append(dict(raw=raw,tokens=gen['token_ids'],native=native,
                            analysis=audit_raw(raw,gen['truncated']),early_reject=gen.get('early_reject',False),
                            early_reject_reason=gen.get('early_reject_reason'),raw_prefix=gen.get('raw_prefix')))
                torch.cuda.synchronize()
                result=dict(mode=mode,batch=batch,seconds=time.monotonic()-tick,outputs=out,error=None,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            except torch.cuda.OutOfMemoryError as exc:
                result=dict(mode=mode,batch=batch,seconds=time.monotonic()-tick,outputs=[],error=repr(exc))
            finally:
                for restore in reversed(restores):restore()
            gc.collect();torch.cuda.empty_cache();report['results'].append(result);save(a.out/'RESULT.json',report)
            print(json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result
        baseline=evaluate('native',16)
        evaluate('native',32);evaluate('native',64)
        evaluate('sdpa',16);evaluate('sdpa',64)
        evaluate('early',16);evaluate('early_sdpa',64)
        repeat=evaluate('native',16)
        for r in report['results']:
            equal=not r['error'] and len(r['outputs'])==len(baseline['outputs']);preserves=equal;early=0
            for old,new in zip(baseline['outputs'],r['outputs']):
                if old['raw']!=new['raw'] or old['tokens']!=new['tokens'] or old['analysis']!=new['analysis']:equal=False
                if new['early_reject']:
                    early+=1
                    if old['analysis']['accepted'] or not impossible_prefix(new['raw_prefix']):preserves=False
                elif old['raw']!=new['raw'] or old['tokens']!=new['tokens'] or old['analysis']!=new['analysis']:
                    preserves=False
            r.update(exact_outputs_equal=equal,preserves_admission_and_unstopped_outputs=preserves,early_rejected=early)
        report.update(stage='complete',finished=time.time(),native_repeat_equal=report['results'][-1]['exact_outputs_equal'],production_cache_modified=False)
        save(a.out/'RESULT.json',report)


if __name__=='__main__':main()
