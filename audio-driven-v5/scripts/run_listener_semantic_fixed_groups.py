#!/usr/bin/env python3
"""Grouped extractive proposal + time-explicit critique; experimental variant."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_fixed_group_prompt import (VERSION,PLAN_PROMPT,REVIEW_PROMPT,digest,source_payload,
    plan_schema,compile_plan,review_schema,validate_proposal,validate_review,materialize,render_prompt,review_payload)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def cached_answer(path, request, stage, attempt, protocol_sha):
    """A cached answer belongs to the exact request; never relabel old evidence."""
    if not path.exists():return None
    cached=json.loads(path.read_text())
    expected=dict(job_id=request['job_id'],stage=stage,attempt=attempt,prompt=request['prompt'],protocol_sha256=protocol_sha)
    if any(cached.get(k)!=v for k,v in expected.items()):
        raise ValueError('Cached response identity/request mismatch; use a fresh output directory')
    if cached.get('finish_reason')!='stop':return None
    return json.loads(cached['raw'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='/private/ckpts/Qwen3-30B-A3B-Instruct-2507');p.add_argument('--batch',type=int,default=16)
    p.add_argument('--memory',type=float,default=.5);p.add_argument('--limit',type=int);p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--decode-graph',action='store_true',help='Validated separately; decode-only CUDA Graph, compilation mode0')
    p.add_argument('--tensor-parallel-size',type=int,default=1)
    p.add_argument('--max-batched-tokens',type=int,default=32768)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=[json.loads(l) for l in a.inputs.open() if l.strip()]
    if a.limit:jobs=jobs[:a.limit]
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate job ID')
    for j in jobs:
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in j['job_id']):raise ValueError('Unsafe job ID')
        source_payload(j['record'])
        if j['provenance']['role']=='current_postprocessed' and 'audio_postprocess' not in j['record']:raise ValueError('Current prompt lacks postprocessed source')
    import hashlib
    code={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/listen_benchmark/semantic_fixed_group_prompt.py',ROOT/'src/listen_benchmark/semantic_group_prompt.py',ROOT/'src/listen_benchmark/semantic_prompt.py',ROOT/'src/p3_listen/prompts.py']}
    model=Path(a.model).resolve()
    model_files=[f for f in model.iterdir() if f.is_file() and f.suffix in ('.safetensors','.json')]
    identity=dict(version=VERSION,implementation="extractive_fixed_groups_v2",code=code,model=str(model),model_files={f.name:dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in sorted(model_files)},
        prompt_sha256=digest(PLAN_PROMPT),review_prompt_sha256=digest(REVIEW_PROMPT),input_sha256=digest(jobs),temperature=0,attempts=a.attempts)
    graph_sizes=[n for n in [1,2,4,8,16,32,64] if n<=a.batch]
    identity['execution']=dict(tensor_parallel_size=a.tensor_parallel_size,max_batched_tokens=a.max_batched_tokens,batch=a.batch,memory=a.memory,decode_graph=a.decode_graph,graph_sizes=graph_sizes if a.decode_graph else [])
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=identity:raise ValueError('Output identity differs; use a new directory')
    write(out/'protocol.json',identity)
    for name in code:
        destination=out/'code'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        content=(ROOT/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=code[name]:raise ValueError('Source changed during snapshot')
        destination.write_bytes(content)
    if (out/'COMPLETE.json').exists():print((out/'COMPLETE.json').read_text());return
    states={};pending=[]
    for j in jobs:
        path=out/'results'/f'{j["job_id"]}.json'
        if path.exists():
            d=json.loads(path.read_text())
            if d['source_record_sha256']!=digest(j['record']) or d['provenance']['protocol_sha256']!=digest(identity):raise ValueError('Stale result')
            states[j['job_id']]=dict(status='accepted',result=d)
        else:pending.append(j)
    llm_holder=[None];tokenizer_holder=[None]
    def get_engine():
        if llm_holder[0] is None:
            from vllm import LLM
            from transformers import AutoTokenizer
            tokenizer_holder[0]=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
            cfg=json.loads((model/'config.json').read_text()) if (model/'config.json').is_file() else {}
            mm_kwargs=dict(limit_mm_per_prompt={'image':0,'video':0},skip_mm_profiling=True) if 'vision_config' in cfg else {}
            llm_holder[0]=LLM(model=str(model),dtype='bfloat16',gpu_memory_utilization=a.memory,max_model_len=32768,
                    max_num_seqs=a.batch,max_num_batched_tokens=a.max_batched_tokens,tensor_parallel_size=a.tensor_parallel_size,
                    enable_chunked_prefill=True,enforce_eager=not a.decode_graph,enable_prefix_caching=True,seed=20261027,
                    **({'compilation_config':{'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':graph_sizes,'max_cudagraph_capture_size':graph_sizes[-1]}} if a.decode_graph else {}),
                    structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},**mm_kwargs)
        return llm_holder[0],tokenizer_holder[0]
    feedback={};start=time.time()
    def infer(requests,stage,attempt):
        results=[];to_run=[]
        for req in requests:
            cached_path=out/'attempts'/f'{req["job_id"]}.{stage}.{attempt}.json'
            if cached_path.exists():
                val=cached_answer(cached_path,req,stage,attempt,digest(identity))
                if val is not None:
                    results.append((req,val,None))
                    continue
            to_run.append(req)
        if not to_run:return results
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        llm,tokenizer=get_engine()
        for offset in range(0,len(to_run),a.batch):
            batch=to_run[offset:offset+a.batch];prompts=[];params=[]
            for req in batch:
                try:
                    text=tokenizer.apply_chat_template([dict(role='user',content=req['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                except TypeError:
                    text=tokenizer.apply_chat_template([dict(role='user',content=req['prompt'])],tokenize=False,add_generation_prompt=True)
                budget=req.get('max_tokens',3072)
                if len(tokenizer.encode(text))+budget>32768:raise ValueError('Context overflow; no truncation allowed')
                prompts.append(text);params.append(SamplingParams(temperature=0,max_tokens=budget,structured_outputs=StructuredOutputsParams(json=req['schema'])))
            responses=llm.generate(prompts,params,use_tqdm=False)
            if len(responses)!=len(batch):raise RuntimeError('Missing engine output')
            for req,response in zip(batch,responses):
                ans=response.outputs[0];entry=dict(job_id=req['job_id'],stage=stage,attempt=attempt,raw=ans.text,finish_reason=ans.finish_reason,tokens=len(ans.token_ids),prompt=req['prompt'],protocol_sha256=digest(identity))
                write(out/'attempts'/f'{req["job_id"]}.{stage}.{attempt}.json',entry)
                try:
                    if ans.finish_reason!='stop':raise ValueError('Truncated model output')
                    value=json.loads(ans.text)
                except Exception as exc:value=None;entry['error']=repr(exc)
                results.append((req,value,entry.get('error')))
        return results
    for attempt in range(a.attempts):
        if not pending:break
        requests=[];by_id={j['job_id']:j for j in pending};retry=[];review_requests=[]
        for j in pending:
            text=PLAN_PROMPT+json.dumps(source_payload(j['record']),ensure_ascii=False)
            if j['job_id'] in feedback:text+='\nPrevious attempt failed validation. Correct it without dropping genuine actions:\n'+feedback[j['job_id']]
            budget=max(8192,min(24000,1024+220*len(source_payload(j['record'])['visual_sources'])))
            try:
                schema=plan_schema(j['record'])
            except ValueError as exc:
                states[j['job_id']]=dict(status='rejected',stage='preflight',error=str(exc))
                continue
            requests.append(dict(job_id=j['job_id'],prompt=text,schema=schema,max_tokens=budget))
        proposals={}
        for req,value,error in infer(requests,'proposal',attempt):
            j=by_id[req['job_id']]
            try:
                if error:raise ValueError(error)
                value=compile_plan(j['record'],value);proposals[j['job_id']]=value
                review_requests.append(dict(job_id=j['job_id'],schema=review_schema(),prompt=REVIEW_PROMPT+json.dumps(review_payload(j['record'],value),ensure_ascii=False)))
            except Exception as exc:
                feedback[j['job_id']]=str(exc);states[j['job_id']]=dict(status='rejected',stage='program',error=str(exc));retry.append(j)
        for req,value,error in infer(review_requests,'review',attempt):
            j=by_id[req['job_id']];answer=proposals[j['job_id']]
            try:
                if error:raise ValueError(error)
                if not validate_review(j['record'],answer,value):raise ValueError(json.dumps(value['issues'],ensure_ascii=False))
                result=materialize(j['record'],answer,value,dict(**j['provenance'],model=str(model),protocol_sha256=digest(identity),attempt=attempt))
                write(out/'results'/f'{j["job_id"]}.json',result)
                prompt_dir=out/'prompts';prompt_dir.mkdir(exist_ok=True)
                (prompt_dir/f'{j["job_id"]}.txt').write_text(render_prompt(result))
                states[j['job_id']]=dict(status='accepted',result=result)
            except Exception as exc:
                feedback[j['job_id']]=str(exc);states[j['job_id']]=dict(status='rejected',stage='semantic_review',error=str(exc));retry.append(j)
        pending=retry
        write(out/'progress.json',dict(attempt=attempt+1,total=len(jobs),accepted=sum(s['status']=='accepted' for s in states.values()),remaining=len(pending),elapsed_seconds=time.time()-start))
        print(json.dumps(json.loads((out/'progress.json').read_text())),flush=True)
    all_passed=sum(s['status']=='accepted' for s in states.values())==len(jobs)
    summary=dict(status='passed' if all_passed else 'review_required',samples=len(jobs),accepted=sum(s['status']=='accepted' for s in states.values()),
        rejected=[dict(job_id=k,**v) for k,v in states.items() if v['status']!='accepted'],
        input_visual=sum(len(source_payload(j['record'])['visual_sources']) for j in jobs),
        output_visual_accepted=sum(sum(e['modality']=='visual' for e in s['result']['events']) for s in states.values() if s['status']=='accepted'),
        elapsed_seconds=time.time()-start,raw_annotations_modified=False,canonical_cache_updated=False,model_review_is_not_human_validation=True)
    write(out/'summary.json',summary)
    if all_passed:write(out/'COMPLETE.json',dict(**summary,protocol_sha256=digest(identity)))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
