"""Sharded, batched vLLM semantic classification; records exact prompts and responses."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_mapping import PROMPT, response_schema, validate_answer, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--shards',type=int,default=8);p.add_argument('--model',default='/private/ckpts/Qwen3-VL-8B-Instruct')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--memory',type=float,default=.5)
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.batch_size<1:raise ValueError('Invalid shard/batch')
    run=Path(a.run);out=run/'responses';out.mkdir(exist_ok=True)
    jobs=json.loads((run/'jobs.json').read_text())[a.shard::a.shards]
    # Only reuse an exact, complete validated response, never a stale or truncated one.
    pending=[]
    for job in jobs:
        path=out/(job['job_id']+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record.get('payload_sha256')==job['payload_sha256'] and record.get('prompt_sha256')==digest(PROMPT) and record.get('valid'):
                continue
            raise ValueError('Existing invalid/stale response; retry in a new run: '+str(path))
        pending.append(job)
    if not pending:return
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    llm=LLM(model=a.model,dtype='bfloat16',gpu_memory_utilization=a.memory,max_model_len=16384,
            max_num_seqs=a.batch_size,max_num_batched_tokens=16384,
            limit_mm_per_prompt={'image':0,'video':0},skip_mm_profiling=True,enforce_eager=True)
    failures=[]
    for first in range(0,len(pending),a.batch_size):
        batch=pending[first:first+a.batch_size];prompts=[];params=[]
        for job in batch:
            text=PROMPT+json.dumps(job['payloads'],ensure_ascii=False)
            prompts.append(processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':text}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
            params.append(SamplingParams(temperature=0,max_tokens=768,
                structured_outputs=StructuredOutputsParams(json=response_schema(job['payloads']))))
        started=time.monotonic();results=llm.generate(prompts,params,use_tqdm=False)
        for job,prompt,result in zip(batch,prompts,results):
            answer=result.outputs[0]
            record=dict(job_id=job['job_id'],payload_sha256=job['payload_sha256'],prompt_sha256=digest(PROMPT),
                        prompt=prompt,raw=answer.text,finish_reason=answer.finish_reason,
                        model=a.model,text_only=True,batch_seconds=time.monotonic()-started,
                        completion_tokens=len(answer.token_ids),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),valid=False)
            try:
                parsed=json.loads(answer.text)
                if answer.finish_reason!='stop' or set(parsed)!=set(job['payloads']):raise ValueError('Incomplete output')
                for key,value in parsed.items():validate_answer(job['payloads'][key],value)
                record.update(parsed=parsed,valid=True)
            except Exception as e:
                record['error']=str(e);failures.append(job['job_id'])
            (out/(job['job_id']+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(dict(job=job['job_id'],valid=record['valid'],finish=answer.finish_reason)),flush=True)
    (run/f'semantic-g{a.shard}.complete.json').write_text(json.dumps(dict(jobs=len(jobs),failures=failures)))
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
