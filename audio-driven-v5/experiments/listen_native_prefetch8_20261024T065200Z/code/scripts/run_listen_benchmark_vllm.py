#!/usr/bin/env python3
"""Batched vLLM queue worker for Qwen/Joy, preserving native AV and raw answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--systems',required=True);p.add_argument('--worker-id',required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--clients',type=int,default=16)
    p.add_argument('--memory',type=float,default=.86);p.add_argument('--limit',type=int,default=50);a=p.parse_args()
    from p2_listen.contracts import atomic_json,sha
    from listen_benchmark.runner import select_samples,run_system
    from listen_benchmark.assets import inspect_weights
    from listen_benchmark.media import timestamped_images,uniform_video
    from listen_labeling.guided import response_format
    from listen_benchmark.simple_timeline import response_schema
    from transformers import AutoProcessor
    import torch
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    cfg=json.loads(Path(a.config).read_text());selected=[next(s for s in cfg['systems'] if s['name']==n) for n in a.systems.split(',')]
    if len({(s['backend'],s['model_path']) for s in selected})!=1:raise ValueError('One model family required')
    native=any(s['branch']=='omni' and s['representation']=='native' for s in selected)
    if native:
        if any(s['branch']!='omni' or s['representation']!='native' for s in selected):
            raise ValueError('Native AV must use a separate engine; mixed-modality batches are unsupported here')
        # vLLM 0.19's interleaved embedding merger fails for some chunked/mixed
        # batches. Preserve native AV by using complete single-request prefill.
        # Keep the GPU batch single-request, but allow caller-selected CPU
        # preparation clients to overlap decoding with the current generation.
        a.batch_size=1
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);base=selected[0];samples=select_samples(a.manifest,a.limit)
    state=root/'workers'/f'{a.worker_id}.json'
    record=dict(worker_id=a.worker_id,pid=os.getpid(),host=os.uname().nodename,systems=a.systems.split(','),engine='vllm',
                gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),batch_size=a.batch_size,clients=a.clients,
                script_sha256=sha(__file__))
    def status(**kw):record.update(kw,utc=datetime.now(timezone.utc).isoformat());atomic_json(state,record)
    status(status='loading_model');asset=inspect_weights(base['model_path'])
    if not asset['ready']:raise ValueError('Weights incomplete')
    processor=AutoProcessor.from_pretrained(base['model_path'],trust_remote_code=True,local_files_only=True)
    limits={'image':128,'audio':1,'video':1} if base['backend'] in ('qwen3_omni','captioner') else (
        {'audio':1} if base['backend']=='qwen2_audio' else {'image':128,'video':0})
    length=8192 if base['backend']=='qwen2_audio' else 49152
    llm=LLM(model=base['model_path'],trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=a.memory,
            max_model_len=length,max_num_seqs=a.batch_size,
            max_num_batched_tokens=length if native else 8192,enable_chunked_prefill=not native,
            limit_mm_per_prompt=limits,skip_mm_profiling=True,enforce_eager=True)
    requests=queue.Queue();shutdown=object()
    def generate_loop():
        while True:
            first=requests.get()
            if first is shutdown:return
            batch=[first];deadline=time.monotonic()+.15
            while len(batch)<a.batch_size:
                try:batch.append(requests.get(timeout=max(.001,deadline-time.monotonic())))
                except queue.Empty:break
            start=time.monotonic()
            try:
                outputs=llm.generate([b[0] for b in batch],[b[1] for b in batch],use_tqdm=False)
                for b,out in zip(batch,outputs):
                    item=out.outputs[0]
                    b[2].set_result(dict(text=item.text,finish_reason=item.finish_reason,
                        generated_token_ids=list(item.token_ids),prompt_tokens=len(out.prompt_token_ids),
                        completion_tokens=len(item.token_ids),rendered_prompt=b[0]['prompt'],
                        representation=b[3],engine='vllm',batch_size=len(batch),batch_wall_seconds=time.monotonic()-start,
                        guided_json=b[1].structured_outputs is not None))
            except Exception as exc:
                for b in batch:b[2].set_exception(RuntimeError(f'vLLM batch failure: {type(exc).__name__}: {exc}'))
    engine_thread=threading.Thread(target=generate_loop,daemon=True);engine_thread.start();status(status='ready')
    class Proxy:
        def __init__(self,system):self.system=system
        def infer(self,media,text,stage,fps):
            s=self.system;parts=[] if s['auxiliary'] else [dict(type='text',text=text)];mm={};options={};evidence={'representation':s['representation']}
            if s['branch']=='omni' and s['representation']=='native':
                frames,evidence=uniform_video(media,fps)
                parts.append(dict(type='video',video='listener://decoded'))
                mm.update(video=[frames],audio=[media['audio']]);options.update(use_audio_in_video=True,fps=fps,do_sample_frames=False)
                # vLLM 0.19's cached prompt replacement reads this explicit
                # kwarg, not HF's computed video_second_per_grid output. Its
                # fallback is 2 seconds, which mis-interleaves an 8 fps input.
                seconds_per_grid=float(processor.image_processor.temporal_patch_size)/fps
                options['second_per_grid_ts']=[seconds_per_grid]
                evidence.update(temporal_patch_size=processor.image_processor.temporal_patch_size,
                                second_per_grid_ts=[seconds_per_grid],
                                alignment_revision='explicit_vllm_grid_time_v1')
            else:
                if media['images']:parts.extend(timestamped_images(media));mm['image']=media['images']
                if media['audio'] is not None:
                    parts.append(dict(type='audio',audio='listener://independent',audio_url='listener://independent'));mm['audio']=[media['audio']]
                if s['backend']=='qwen3_omni':options['use_audio_in_video']=False
            template_options={} if s['backend']=='captioner' else dict(enable_thinking=False)
            prompt=processor.apply_chat_template([dict(role='user',content=parts)],tokenize=False,add_generation_prompt=True,**template_options)
            schema=(response_schema(media['duration_s']) if s.get('prompt_variant')=='backchannel_simple_v2'
                    else response_format(s['branch'],media['duration_s'],stage)['json_schema']['schema']) if not s['auxiliary'] else None
            guided=None if schema is None else StructuredOutputsParams(json=schema)
            params=SamplingParams(temperature=0,max_tokens=cfg['generation']['max_new_tokens'],structured_outputs=guided)
            payload=dict(prompt=prompt,multi_modal_data=mm,mm_processor_kwargs=options)
            future=Future();requests.put((payload,params,future,evidence));return future.result(timeout=1800)
    def job(system,row):
        name,sid=system['name'],row['sample_id'];folder=root/'items'/sid;summary=folder/name/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('complete'):return
        lockpath=root/'claims'/name/(sid+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
        with lockpath.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            if summary.exists() and json.loads(summary.read_text()).get('complete'):return
            proxy=Proxy(system)
            return run_system(cfg,name,[row],folder,'cuda:0',backend_factory=lambda *_:proxy,asset_identity=asset)
    with ThreadPoolExecutor(max_workers=a.clients) as pool:
        futures=[pool.submit(job,s,row) for row in samples for s in selected]
        for f in futures:f.result()
    requests.put(shutdown);engine_thread.join();status(status='complete')


if __name__=='__main__':main()
