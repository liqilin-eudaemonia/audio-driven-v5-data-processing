#!/usr/bin/env python3
"""Node8 eight-GPU ready-first labeling of repaired orientations and all repeated labels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_upright_regen_20261105'
BASE=ROOT/'experiments/listen_j_reaction_vocal_omni_20261103'
OLDWORK=ROOT/'experiments/listen_j_annotation_node8_20261030'
PY='/private/software/conda/envs/vllm/bin/python'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from regenerate_j_loop_captions import hard_failures,read,write

def prepare_task(row):
 os.environ.update(CUDA_VISIBLE_DEVICES='',J_OMNI_FER_GPU='',J_OMNI_CAPTION_NO_MEASURE='0')
 from listen_benchmark.j_omni_reaction_vocal import prepare_input
 parent_raw=Path(row['caption_file']).read_bytes();assert hashlib.sha256(parent_raw).hexdigest()==row['caption_file_sha256'];parent=json.loads(parent_raw)
 restoration=None
 if row.get('_restore_orientation'):
  path=WORK/'restored_rows'/f"{row['sample_id']}.json";started=time.monotonic()
  while not path.exists():
   error=WORK/'restoration_execution_errors'/f"{row['sample_id']}.json"
   if error.exists():raise RuntimeError('Media restoration infrastructure error: '+str(read(error)))
   if time.monotonic()-started>3600:raise TimeoutError('Waiting for verified upright source: '+row['sample_id'])
   time.sleep(1)
  restoration=read(path);assert restoration['parent_caption_sha256']==row['caption_file_sha256']
  if not restoration['restoration_usable']:return row,parent,None,restoration
  prepared=prepare_input(restoration['request'],str(OLDWORK),str(WORK),str(WORK/'resources'))
  assert prepared['evidence']['audio_encoder_sha256']==parent['input_evidence']['audio_encoder_sha256']
  assert prepared['recipe']==row['audio_recipe']
  assert prepared['effective_source']['source_start_s']==row['source_start_s']
  assert prepared['effective_source']['listener_id']==row['listener_id']
 else:
  os.environ['J_OMNI_CAPTION_NO_MEASURE']='1'
  original=parent.get('original_source',parent['source'])
  request=dict(sample_id=row['sample_id'],source=original,split=row['split'],duration_s=row['duration_s'],quality=row['quality'])
  prepared=prepare_input(request,str(OLDWORK),str(BASE),str(OLDWORK))
  for f in ['video_frames_sha256','audio_encoder_sha256']:assert prepared['evidence'][f]==parent['input_evidence'][f]
 return row,parent,prepared,restoration

def generate_batch(llm,processor,tasks,gpu):
 from vllm import SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 from listen_benchmark.j_omni_reaction_vocal import _build_request
 from listen_benchmark.native_caption_quality import generation_failures
 from run_glare_reaction_omni_pilot import cue_gaps,coverage_failure_flags
 pending=[];attempts={};best={};skipped=0
 for i,(row,parent,prepared,restoration) in enumerate(tasks):
  if prepared is None:
   write(WORK/'excluded_visual'/f"{row['sample_id']}.json",dict(sample_id=row['sample_id'],reason='restored_visual_quality_excluded',restoration_file=str(WORK/'restored_rows'/f"{row['sample_id']}.json"),decision=restoration['decision']))
   skipped+=1
  else:pending.append(i);attempts[i]=[]
 for round_index,penalty in enumerate([1.1,1.18],1):
  if not pending:break
  requests=[];prompts={};rendered_prompts={}
  for i in pending:
   prepared=tasks[i][2]
   prompt=prepared['prompt']+'\n\nWrite one concise paragraph of at most 250 words. Describe each sustained state once. Include the supplied observable movements and audible responses in temporal order. Stop when the clip description is complete; do not repeat sentences.'
   if round_index==2:prompt+=' Summarize genuine repeated movements in one sentence. Use fewer than 180 words.'
   prompts[i]=prompt;request=_build_request(processor,prepared,prompt,f'upright-{gpu}-{time.time_ns()}-{i}');requests.append(request);rendered_prompts[i]=request['prompt']
  params=SamplingParams(temperature=0,max_tokens=768,repetition_penalty=penalty,structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][ -~]*'))
  llm.reset_mm_cache();started=time.monotonic();outputs=llm.generate(requests,params,use_tqdm=False);again=[]
  for i,output in zip(pending,outputs):
   prepared=tasks[i][2];gen=output.outputs[0];caption=re.sub(r'^Assistant:\s*','',gen.text).strip();hard=hard_failures(caption,gen.finish_reason)
   gaps=cue_gaps(prepared['vocal_cues'],prepared['reaction_cues'],prepared['nod_events'],caption,silent=prepared['evidence']['listener_audio_is_silent'])
   flags=list(dict.fromkeys(generation_failures(caption,gen.finish_reason)+coverage_failure_flags(gaps)))
   a=dict(caption=caption,finish_reason=gen.finish_reason,hard_failures=hard,review_flags=flags,gaps=gaps,prompt=prompts[i],rendered_prompt=rendered_prompts[i],completion_tokens=len(gen.token_ids),round=round_index,decoding=dict(temperature=0,max_tokens=768,repetition_penalty=penalty),batch_seconds=time.monotonic()-started)
   attempts[i].append(a)
   if hard:again.append(i)
   else:best[i]=a
  pending=again
 accepted=failed=0
 for i,values in enumerate(tasks):
  row,parent,prepared,restoration=values;sid=row['sample_id']
  if prepared is None:continue
  if i not in best:
   write(WORK/'regeneration_rejected'/f'{sid}.json',dict(sample_id=sid,parent_caption_file=row['caption_file'],parent_caption_sha256=row['caption_file_sha256'],reason='hard_generation_failure_after_two_attempts',attempts=attempts[i],gpu=gpu,orientation_restored=bool(restoration)))
   failed+=1;continue
  a=best[i];record=copy.deepcopy(parent);identity=dict(schema='j-upright-and-repeat-caption-v1',parent_caption_sha256=row['caption_file_sha256'],video_frames_sha256=prepared['evidence']['video_frames_sha256'],audio_encoder_sha256=prepared['evidence']['audio_encoder_sha256'],decoding=a['decoding'],prompt_sha256=hashlib.sha256(a['prompt'].encode()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  record.update(pipeline_id='listen-j-upright-repeat-caption-v1-20261105',prompt_version='native-av-upright-concise-v1',caption=a['caption'],prompt=a['prompt'],finish_reason=a['finish_reason'],completion_tokens=a['completion_tokens'],generation_checks_passed=not a['review_flags'],review_flags=a['review_flags'],remaining_gaps=a['gaps'],attempts=attempts[i],input_evidence=prepared['evidence'],completed_at=time.time(),node='node8',gpu=gpu,parent_caption_file=row['caption_file'],parent_caption_sha256=row['caption_file_sha256'],parent_cache_identity=parent.get('cache_identity'),cache_identity=identity,caption_identity_version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),hard_generation_passed=True,training_admitted=False,human_ground_truth=False)
  # Keep rendered text consistent with the actual repair prompt, while preserving the original template in the parent file.
  record['rendered_prompt']=a['rendered_prompt']
  if restoration:
   record.update(source=prepared['effective_source'],crop_provenance=prepared['evidence']['crop_application'],orientation_restoration_file=str(WORK/'restored_rows'/f'{sid}.json'),original_effective_source=parent['source'],quality=restoration['request']['quality'])
   identity['orientation_provenance_sha256']=hashlib.sha256(Path(restoration['provenance_file']).read_bytes()).hexdigest()
   record['caption_identity_version']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
  write(WORK/'regenerated'/f'{sid}.json',record);accepted+=1
 return dict(accepted=accepted,failed=failed,visual_excluded=skipped)

def worker(gpu,max_num_seqs):
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
 rows=[json.loads(x) for x in (WORK/f'regeneration_shards/gpu-{gpu}.jsonl').open()]
 rows=[r for r in rows if not any((WORK/n/f"{r['sample_id']}.json").exists() for n in ['regenerated','regeneration_rejected','excluded_visual'])]
 state=dict(gpu=gpu,pid=os.getpid(),hostname=socket.gethostname(),tasks=len(rows),completed=0,accepted=0,failed=0,visual_excluded=0)
 def status(**kw):state.update(kw,heartbeat=time.time());write(WORK/f'regeneration_workers/gpu-{gpu}.json',state)
 status(stage='loading_and_prefetching')
 if not rows:status(stage='complete');return
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  iterator=iter(rows);pending=set()
  def fill():
   while len(pending)<8:
    try:pending.add(pool.submit(prepare_task,next(iterator)))
    except StopIteration:break
  fill()
  from transformers import AutoProcessor
  from vllm import LLM
  cfg=read(BASE/'config.json');processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
  llm=LLM(model=cfg['model'],trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.86,max_model_len=49152,max_num_seqs=max_num_seqs,max_num_batched_tokens=49152,enable_chunked_prefill=False,limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2,4],'max_cudagraph_capture_size':4})
  while pending:
   ready=[f for f in pending if f.done()]
   if not ready:status(stage='waiting_for_prepared_input');ready=list(wait(pending,return_when=FIRST_COMPLETED)[0])
   ready=ready[:max_num_seqs];tasks=[]
   for f in ready:pending.remove(f);tasks.append(f.result())
   fill();status(stage='generating',batch=len(tasks),prefetch=len(pending));result=generate_batch(llm,processor,tasks,gpu)
   for n,v in result.items():state[n]+=v
   state['completed']+=len(tasks);status(stage='batch_complete');del tasks
  status(stage='complete')

def launch(max_num_seqs):
 assert socket.gethostname()=='dedicated-developjob-zz-dev-machine-1-wx3im'
 rows=[json.loads(x) for x in (WORK/'regenerate_candidates.jsonl').open()];shards=[[] for _ in range(8)];loads=[0.]*8
 # The repeated labels are prepared immediately while the rotation repair pool is being rendered and measured.
 for group in [[r for r in rows if not r.get('_restore_orientation')],[r for r in rows if r.get('_restore_orientation')]]:
  for r in sorted(group,key=lambda r:r['duration_s'],reverse=True):
   g=min(range(8),key=lambda g:loads[g]);shards[g].append(r);loads[g]+=r['duration_s']
 for g,items in enumerate(shards):
  p=WORK/f'regeneration_shards/gpu-{g}.jsonl';p.parent.mkdir(exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
 free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert len(free.splitlines())==8 and all(int(v.strip())<2000 for v in free.splitlines()),free
 env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',J_OMNI_BOUNDED_CROP_V1='1',J_OMNI_CAPTION_NO_MEASURE='0',J_OMNI_FER_GPU='',J_OMNI_REACTION_SLOTS='32',J_OMNI_INPUT_OPTIMIZATIONS=json.dumps(dict(fast_crop=True,roi_crop=True,shared_decode=True,shared_raw_decode=True)),VLLM_USAGE_STATS_ENABLED='0')
 processes=[];logs=WORK/'regeneration_logs';logs.mkdir(exist_ok=True)
 for g in range(8):
  with (logs/f'gpu-{g}.log').open('ab') as log:p=subprocess.Popen([PY,str(Path(__file__).resolve()),'worker','--gpu',str(g),'--max-num-seqs',str(max_num_seqs)],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  processes.append((g,p))
 write(WORK/'node8_launch.json',dict(host='172.29.104.237',ssh_port=22,requested_port_hint=20491,execution='8 independent GPU replicas; no collective port',pids={str(g):p.pid for g,p in processes},tasks_per_gpu={str(g):len(shards[g]) for g in range(8)},max_num_seqs=max_num_seqs,prepare_workers_per_gpu=4,prefetch_per_gpu=8,ready_first=True,started_at=time.time()))
 with (WORK/'node8_gpu_telemetry.jsonl').open('a') as f:
  while any(p.poll() is None for _,p in processes):
   values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);f.write(json.dumps(dict(at=time.time(),gpus=[list(map(int,x.split(','))) for x in values.splitlines()]))+'\n');f.flush();time.sleep(2)
 codes={str(g):p.returncode for g,p in processes};write(WORK/'node8_worker_exit_codes.json',codes)
 if any(codes.values()):raise RuntimeError('Infrastructure error; resume missing jobs after fixing logs, never silently discard')
 print(json.dumps(dict(exit_codes=codes)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','worker']);p.add_argument('--gpu',type=int);p.add_argument('--max-num-seqs',type=int,default=4);a=p.parse_args()
 (launch(a.max_num_seqs) if a.command=='launch' else worker(a.gpu,a.max_num_seqs))
