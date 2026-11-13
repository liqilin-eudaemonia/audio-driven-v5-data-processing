#!/usr/bin/env python3
"""Eight independent node8 GPU workers repair source-bound pathological captions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_clean_regen_20261105'
BASE=ROOT/'experiments/listen_j_reaction_vocal_omni_20261103'
PY='/private/software/conda/envs/vllm/bin/python'


def read(p):return json.loads(Path(p).read_bytes())


def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)


def hard_failures(text,finish):
    flags=[]
    if finish!='stop':flags.append('generation_not_finished')
    if not text.strip():flags.append('empty_caption')
    sentences=[re.sub(r'\s+',' ',x.strip().lower()) for x in re.split(r'[.!?]+',text) if len(x.strip().split())>=8]
    if max(Counter(sentences).values(),default=0)>=8:flags.append('pathological_sentence_loop')
    words=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
    if max(Counter(tuple(words[i:i+16]) for i in range(max(0,len(words)-15))).values(),default=0)>=8:flags.append('pathological_token_loop')
    return flags


def prepare_task(row):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['J_OMNI_FER_GPU']='';os.environ['J_OMNI_CAPTION_NO_MEASURE']='1'
    sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
    from listen_benchmark.j_omni_reaction_vocal import prepare_input
    parent_raw=Path(row['caption_file']).read_bytes();parent=json.loads(parent_raw)
    if hashlib.sha256(parent_raw).hexdigest()!=row['caption_file_sha256']:raise ValueError('Parent caption changed')
    original=parent.get('original_source',parent['source'])
    request=dict(sample_id=row['sample_id'],source=original,split=row['split'],duration_s=row['duration_s'],quality=row['quality'])
    resources=str(ROOT/'experiments/listen_j_annotation_node8_20261030')
    prepared=prepare_input(request,str(ROOT/'experiments/listen_j_annotation_node8_20261030'),str(BASE),resources)
    for key in ['video_frames_sha256','audio_encoder_sha256']:
        if prepared['evidence'][key]!=parent['input_evidence'][key]:raise ValueError('Regeneration input differs: '+key+' '+row['sample_id'])
    return row,parent,prepared


def worker(gpu,max_num_seqs):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
    rows=[json.loads(line) for line in (WORK/f'regeneration_shards/gpu-{gpu}.jsonl').open()]
    rows=[r for r in rows if not (WORK/'regenerated'/f"{r['sample_id']}.json").exists() and not (WORK/'regeneration_rejected'/f"{r['sample_id']}.json").exists()]
    state=dict(gpu=gpu,pid=os.getpid(),hostname=socket.gethostname(),tasks=len(rows),stage='preparing_and_loading',completed=0)
    def status(**kw):state.update(kw,heartbeat=time.time());write(WORK/f'regeneration_workers/gpu-{gpu}.json',state)
    status()
    if not rows:status(stage='complete');return
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(4,len(rows)),mp_context=context) as pool:
        futures=[pool.submit(prepare_task,row) for row in rows]
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from listen_benchmark.j_omni_reaction_vocal import _build_request
        from listen_benchmark.native_caption_quality import generation_failures
        from run_glare_reaction_omni_pilot import cue_gaps,coverage_failure_flags
        cfg=read(BASE/'config.json')
        processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
        kwargs=dict(model=cfg['model'],trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=.86,max_model_len=49152,max_num_seqs=max_num_seqs,max_num_batched_tokens=49152,
            enable_chunked_prefill=False,limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,
            compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY','cudagraph_capture_sizes':[1,2] if max_num_seqs>=2 else [1],'max_cudagraph_capture_size':2 if max_num_seqs>=2 else 1})
        llm=LLM(**kwargs)
        tasks=[f.result() for f in futures]
        status(stage='generating',prepared=len(tasks),max_num_seqs=max_num_seqs)
        pending=list(range(len(tasks)));attempts=[[] for _ in tasks];best={}
        for round_index,penalty in enumerate([1.18,1.25],1):
            if not pending:break
            requests=[];prompts={}
            for i in pending:
                prepared=tasks[i][2]
                prompt=prepared['prompt']+'\n\nWrite one concise paragraph. Describe each sustained state once. Stop when the clip description is complete. Do not repeat sentences or manufacture a repetitive timeline.'
                if round_index==2:prompt+=' Keep the paragraph under 220 words; summarize genuine repeated movements in one sentence.'
                prompts[i]=prompt
                requests.append(_build_request(processor,prepared,prompt,f'regen-{gpu}-{i}-{round_index}-{time.time_ns()}'))
            params=SamplingParams(temperature=0,max_tokens=1024,repetition_penalty=penalty,
                structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][ -~]*'))
            llm.reset_mm_cache();t=time.monotonic();outputs=llm.generate(requests,params,use_tqdm=False)
            again=[]
            for i,output in zip(pending,outputs):
                row,parent,prepared=tasks[i];gen=output.outputs[0];text=re.sub(r'^Assistant:\s*','',gen.text).strip()
                hard=hard_failures(text,gen.finish_reason)
                gaps=cue_gaps(prepared['vocal_cues'],prepared['reaction_cues'],prepared['nod_events'],text,
                    silent=prepared['evidence']['listener_audio_is_silent'])
                review=list(dict.fromkeys(generation_failures(text,gen.finish_reason)+coverage_failure_flags(gaps)))
                item=dict(caption=text,finish_reason=gen.finish_reason,hard_failures=hard,review_flags=review,gaps=gaps,
                    prompt=prompts[i],completion_tokens=len(gen.token_ids),round=round_index,
                    decoding=dict(temperature=0,max_tokens=1024,repetition_penalty=penalty),batch_seconds=time.monotonic()-t)
                attempts[i].append(item)
                if hard:again.append(i)
                else:best[i]=item
            pending=again
        for i,(row,parent,prepared) in enumerate(tasks):
            sid=row['sample_id']
            if i not in best:
                write(WORK/'regeneration_rejected'/f'{sid}.json',dict(sample_id=sid,parent_caption_file=row['caption_file'],
                    parent_caption_sha256=row['caption_file_sha256'],reason='hard_generation_failure_after_two_attempts',attempts=attempts[i],gpu=gpu))
                status(completed=state['completed']+1);continue
            selected=best[i];record=copy.deepcopy(parent)
            identity=dict(schema='j-loop-regenerated-caption-v1',parent_caption_sha256=row['caption_file_sha256'],
                video_frames_sha256=prepared['evidence']['video_frames_sha256'],audio_encoder_sha256=prepared['evidence']['audio_encoder_sha256'],
                decoding=selected['decoding'],prompt_sha256=hashlib.sha256(selected['prompt'].encode()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            record.update(pipeline_id='listen-j-loop-regenerated-v1-20261105',prompt_version='native-av-reaction-vocal-anti-loop-v1',
                caption=selected['caption'],prompt=selected['prompt'],finish_reason=selected['finish_reason'],
                completion_tokens=selected['completion_tokens'],generation_checks_passed=not selected['review_flags'],
                review_flags=selected['review_flags'],remaining_gaps=selected['gaps'],attempts=attempts[i],
                input_evidence=prepared['evidence'],completed_at=time.time(),node='node8',gpu=gpu,
                parent_caption_file=row['caption_file'],parent_caption_sha256=row['caption_file_sha256'],
                parent_cache_identity=parent.get('cache_identity'),cache_identity=identity,
                caption_identity_version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
                hard_generation_passed=True,training_admitted=False,human_ground_truth=False)
            write(WORK/'regenerated'/f'{sid}.json',record)
            status(completed=state['completed']+1)
        status(stage='complete')


def launch(max_num_seqs):
    assert socket.gethostname()=='dedicated-developjob-zz-dev-machine-1-wx3im'
    rows=[json.loads(line) for line in (WORK/'regenerate_candidates.jsonl').open()]
    loads=[0.]*8;shards=[[] for _ in range(8)]
    for row in sorted(rows,key=lambda r:r['duration_s'],reverse=True):
        gpu=min(range(8),key=lambda i:loads[i]);shards[gpu].append(row);loads[gpu]+=row['duration_s']
    for gpu,items in enumerate(shards):
        p=WORK/f'regeneration_shards/gpu-{gpu}.jsonl';p.parent.mkdir(exist_ok=True)
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(free.splitlines())==8 and all(int(v.strip())<2000 for v in free.splitlines()),free
    env=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        VLLM_WORKER_MULTIPROC_METHOD='spawn',J_OMNI_BOUNDED_CROP_V1='1',J_OMNI_CAPTION_NO_MEASURE='1',J_OMNI_FER_GPU='',
        J_OMNI_INPUT_OPTIMIZATIONS=json.dumps(dict(fast_crop=True,roi_crop=True,shared_decode=True,shared_raw_decode=True)),VLLM_USAGE_STATS_ENABLED='0')
    processes=[];logs=WORK/'regeneration_logs';logs.mkdir(exist_ok=True)
    for gpu in range(8):
        log=(logs/f'gpu-{gpu}.log').open('ab')
        p=subprocess.Popen([PY,str(Path(__file__).resolve()),'worker','--gpu',str(gpu),'--max-num-seqs',str(max_num_seqs)],
            cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((gpu,p))
    write(WORK/'node8_launch.json',dict(host='172.29.104.237',ssh_port=22,requested_port_hint=20491,
        execution='eight independent single-GPU workers; no inter-rank collective port',hostname=socket.gethostname(),
        pids={str(g):p.pid for g,p in processes},tasks_per_gpu={str(g):len(shards[g]) for g in range(8)},max_num_seqs=max_num_seqs,started_at=time.time()))
    with (WORK/'node8_gpu_telemetry.jsonl').open('a') as telemetry:
        while any(p.poll() is None for _,p in processes):
            values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            telemetry.write(json.dumps(dict(at=time.time(),gpus=[list(map(int,v.split(','))) for v in values.splitlines()]))+'\n');telemetry.flush()
            time.sleep(2)
    codes={str(g):p.returncode for g,p in processes};write(WORK/'node8_worker_exit_codes.json',codes)
    print(json.dumps(dict(exit_codes=codes)),flush=True)
    if any(codes.values()):raise RuntimeError('GPU worker failed; inspect logs and resume without dropping infrastructure failures')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['launch','worker']);p.add_argument('--gpu',type=int);p.add_argument('--max-num-seqs',type=int,default=2);a=p.parse_args()
    if a.command=='launch':launch(a.max_num_seqs)
    else:worker(a.gpu,a.max_num_seqs)
