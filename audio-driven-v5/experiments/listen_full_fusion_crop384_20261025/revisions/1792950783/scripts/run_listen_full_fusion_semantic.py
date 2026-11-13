"""Persistent text-only GPU worker; claims immutable ready shards with flock."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,lock,jsonl,cached,cache_put,semantic_state
from listen_benchmark.semantic_mapping import request_prompt,response_schema,validate_answer

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--worker',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--batch',type=int,default=128);p.add_argument('--memory',type=float,default=.85);p.add_argument('--one-shard',action='store_true');a=p.parse_args();run=a.run.resolve();cfg=read(run/'config.json');state=run/f'workers/{a.worker}.json';log=run/f'attempts/{a.worker}.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
 done_jobs=0;start=time.time();llm=None;processor=None
 a.batch=int(cfg.get('semantic_batch_overrides',{}).get(a.worker,a.batch))
 def status(name,**kw):atomic(state,dict(worker=a.worker,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),status=name,heartbeat=time.time(),started_at=start,completed_jobs=done_jobs,**kw))
 status('starting')
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 from transformers import AutoProcessor
 def infer(jobs,attempt=0):
  nonlocal done_jobs
  if not jobs:return []
  prep_start=time.time()
  prompts=[];params=[]
  for job in jobs:
   payloads={'item':job['payload']};text=request_prompt(payloads)+json.dumps(payloads,ensure_ascii=False)
   prompts.append(processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=text)])],tokenize=False,add_generation_prompt=True,enable_thinking=False))
   params.append(SamplingParams(temperature=0,max_tokens=[768,1536,2048][attempt],structured_outputs=StructuredOutputsParams(json=response_schema(payloads))))
  before=time.time()
  try:
   answers=llm.generate(prompts,params,use_tqdm=False)
   if len(answers)!=len(jobs):raise RuntimeError('Inference output count differs')
  except Exception:
   if len(jobs)>1:
    half=len(jobs)//2;return infer(jobs[:half],attempt)+infer(jobs[half:],attempt)
   raise # engine/runtime failures are restarted, not mislabeled as semantic negatives
  generated=time.time();retry=[]
  with log.open('a') as f:
   for job,result in zip(jobs,answers):
    ans=result.outputs[0];record=dict(**job,valid=False,raw=ans.text,finish_reason=ans.finish_reason,model=cfg['model'],model_signature=cfg['model_signature'],mapping_version=cfg['mapping_version'],attempt=attempt,worker=a.worker,utc=time.time(),batch_size=len(jobs),batch_seconds=time.time()-before,completion_tokens=len(ans.token_ids))
    try:
     parsed=json.loads(ans.text)
     if ans.finish_reason!='stop' or set(parsed)!= {'item'}:raise ValueError('Incomplete response')
     validate_answer(job['payload'],parsed['item']);record.update(valid=True,answer=parsed['item'])
    except Exception as e:record['error']=repr(e)
    f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if record['valid'] or attempt==2:cache_put(run,record);done_jobs+=1
    else:retry.append(job)
   f.flush();os.fsync(f.fileno())
  metric=dict(utc=time.time(),worker=a.worker,attempt=attempt,jobs=len(jobs),retry_jobs=len(retry),
              preparation_seconds=before-prep_start,generation_seconds=generated-before,
              persistence_seconds=time.time()-generated,elapsed_seconds=time.time()-prep_start)
  with (run/f'attempts/{a.worker}-timings.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
  status('inference',last_batch_seconds=time.time()-before,last_batch_jobs=len(jobs),retry_jobs=len(retry),timings=metric)
  return retry
 while not (run/'STOP').exists():
  markers=sorted((run/'state/ingest').glob('*.json'))
  if markers:
   off=a.rank%len(markers);markers=markers[off:]+markers[:off]
  claimed=False
  for marker in markers:
   shard=marker.stem
   if read(run/f'state/semantic/{shard}.json').get('done'):continue
   with lock(run/f'locks/semantic/{shard}.lock') as acquired:
    if not acquired or read(run/f'state/semantic/{shard}.json').get('done'):continue
    claimed=True;jobs=list(jsonl(run/f'jobs/{shard}.jsonl'));pending=[j for j in jobs if cached(run,j) is None]
    status('loading_model' if pending and llm is None else 'working',shard=shard,pending=len(pending))
    if pending and llm is None:
     processor=AutoProcessor.from_pretrained(cfg['model'],local_files_only=True)
     llm=LLM(model=cfg['model'],dtype='bfloat16',gpu_memory_utilization=a.memory,max_model_len=16384,max_num_seqs=a.batch,max_num_batched_tokens=32768,limit_mm_per_prompt={'image':0,'video':0},skip_mm_profiling=True,enforce_eager=cfg.get('enforce_eager',False),enable_prefix_caching=True)
    # Keep the same three attempts and token budgets, but batch failures from the
    # whole shard instead of blocking every primary batch on one or two retries.
    for attempt in range(3):
     retry=[]
     for start_job in range(0,len(pending),a.batch):
      if (run/'STOP').exists():status('stopped',shard=shard);return
      batch=[j for j in pending[start_job:start_job+a.batch] if cached(run,j) is None]
      status('inference',shard=shard,batch=len(batch),attempt=attempt);retry.extend(infer(batch,attempt))
     pending=retry
     if not pending:break
    result=semantic_state(run,shard,a.worker)
    if not result:raise RuntimeError('Shard cache incomplete')
    print(json.dumps(result),flush=True);status('shard_complete',shard=shard)
    if a.one_shard:return
   break
  if not claimed:
   if (run/'INGEST_COMPLETE.json').exists() and all(read(run/f'state/semantic/{p.stem}.json').get('done') for p in markers):
    status('finished');return
   status('waiting_for_ready_shard');time.sleep(5)
 status('stopped')
if __name__=='__main__':main()
