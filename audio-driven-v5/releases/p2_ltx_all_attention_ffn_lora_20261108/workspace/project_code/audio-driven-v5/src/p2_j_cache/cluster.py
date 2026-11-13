"""Shared-filesystem queue for the four-node J cache extraction."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager, ExitStack
import csv
import copy
import fcntl
import json
import multiprocessing
import os
import queue
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from p2_listen.cache import cpu, load_part
from p2_listen.config import activate, extractor_config, load as load_config
from p2_listen.contracts import ROOT, atomic_json, digest, read, sha
from p2_listen.features import extract_input, make_observation, weights_identity
from p2_listen.layout import geometry
from p2_plan_cache.encoding import extract_planned_input, encode_text
from p2_plan_cache.runtime import dependencies, validate_payload
from p3_listen.prompts import GEMMA, OMNI, FixedPrompt, observation_messages, contract as prompt_contract
from . import VERSION
from .data import (code_identity, freeze_sample, open_manifest, sample, read_media,
                   observation, SourceVideoGap)
from .runtime import cache_identity, persist


def freeze_record(args):
    row,work,crop_config=args
    path=Path(work)/'samples'/(row['sid']+'.json')
    if path.exists():
        item=read(path)
        if item['selection']!=row:raise ValueError('Frozen selection differs')
    else:
        item=freeze_sample(row,crop_config);atomic_json(path,item,immutable=True)
    return row['sid'],digest(item)


def prepare_full(work, config, workers=64):
    """Freeze exactly the documented main selection and saved supplement trial."""
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    root=ROOT/'experiments/listen_j_final_quality_selection_20261101'
    selection=root/'selection.sqlite'
    supplement=root/'train500h_audio_supplement_feasibility/supplement_trial.jsonl'
    crop_path=ROOT/'experiments/listen_j_omni_20261030/config.json'
    c=load_config(config);c=dict(c,output=str(work),data=dict(c['data']));c['data'].pop('audio_postprocess',None)
    if c['data']['fps']!=25 or c['data']['size']!=512:raise ValueError('Unexpected training geometry')
    con=sqlite3.connect('file:'+str(selection)+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    main=[dict(row) for row in con.execute("SELECT * FROM samples WHERE split='train' AND face_p10>=224 ORDER BY pair,start,sid")]
    extra=[json.loads(line) for line in supplement.read_text().splitlines() if line.strip()]
    for entry in extra:
        selected=con.execute('SELECT * FROM samples WHERE sid=?',(entry['sid'],)).fetchone()
        if selected is None or any(entry[k]!=selected[k] for k in selected.keys()):
            raise ValueError('Supplement no longer matches final selection: '+entry['sid'])
    con.close();rows=main+extra;ids=[r['sid'] for r in rows]
    if len(main)!=264054 or len(extra)!=21584 or len(set(ids))!=285638 or any(r['split']!='train' for r in rows):
        raise ValueError('Documented 500h selection changed')
    list_path=work/'sample_list.csv'
    with list_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['sample_id','split','duration_s','selection_group'])
        writer.writeheader()
        for i,row in enumerate(rows):writer.writerow(dict(sample_id=row['sid'],split=row['split'],duration_s=row['duration'],selection_group='main_face224' if i<len(main) else 'saved_audio_supplement'))
    crop_config=read(crop_path);entries={};begin=time.time()
    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for start in range(0,len(rows),8192):
            entries.update(pool.map(freeze_record,((r,str(work),crop_config) for r in rows[start:start+8192]),chunksize=32))
            progress=dict(samples=len(entries),total=len(rows),elapsed_s=time.time()-begin)
            atomic_json(work/'prepare_progress.json',progress)
            print(json.dumps(progress),flush=True)
    value=dict(schema=VERSION,config=c,code=code_identity(),prompts=prompt_contract(),
        sources={str(Path(p).resolve()):sha(p) for p in (config,list_path,crop_path,supplement)},
        entries=entries,sample_ids=ids,selection_sqlite=str(selection),splits=dict(train=ids,dev=[],test=[]),
        guidance='original_whole_clip_caption_verbatim',event_labels='not_mapped_from_open_statistics',
        source_admission_flags_preserved=True,intermediate_wav=False,intermediate_video=False,
        selection_summary=dict(main=len(main),supplement=len(extra),hours=sum(r['duration'] for r in rows)/3600))
    atomic_json(work/'manifest.json',value,immutable=True)
    return value


def prepare_queue(work, feature_chunk=32, text_chunk=256):
    work=Path(work);m=open_manifest(work);ids=m['sample_ids']
    for phase,size in [('text',text_chunk),('features',feature_chunk)]:
        jobs=[]
        for i,start in enumerate(range(0,len(ids),size)):
            name=f'{i:06d}';atomic_json(work/'jobs'/phase/(name+'.json'),ids[start:start+size],immutable=True);jobs.append(name)
        atomic_json(work/'jobs'/(phase+'.json'),jobs,immutable=True)
    atomic_json(work/'cluster_config.json',dict(text_batch_size=32,text_projection_batch=32,feature_workers_per_gpu=2,
        cpu_threads_per_worker=2,media_prefetch_per_worker=4,writer_threads_per_worker=2,
        feature_chunk=feature_chunk,text_chunk=text_chunk,vae_workspace_slots=0,
        allocator='expandable_segments:True'),immutable=True)


@contextmanager
def job_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:yield False;return
        try:yield True
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


class FeatureSession:
    def __init__(self,c,prepare_workers=4,media_reader=read_media):
        from omni_av_p2.data import load_omni
        from ltx_trainer.model_loader import load_video_vae_encoder,load_audio_vae_encoder
        from ltx_core.model.audio_vae import AudioProcessor
        self.c=c;self.cfg=extractor_config(c);self.media_reader=media_reader
        self.runtime,self.processor=load_omni(self.cfg,'cuda:0')
        # Fast tokenizers mutate their padding/truncation state. Give every CPU
        # producer its own processor, separate from the GPU thread's tokenizer.
        self.cpu_processors=queue.SimpleQueue()
        for _ in range(prepare_workers):self.cpu_processors.put(copy.deepcopy(self.processor))
        rendered=self.processor.apply_chat_template(observation_messages(),tokenize=False,add_generation_prompt=True)
        if not rendered.startswith('<|im_start|>system\n') or rendered.index(OMNI)>rendered.index('<|vision_bos|>'):
            raise ValueError('System task was not rendered before media')
        ve=load_video_vae_encoder(self.cfg.ltx_path,device='cuda:0',dtype=torch.bfloat16).requires_grad_(False).eval()
        ae=load_audio_vae_encoder(self.cfg.ltx_path,device='cuda:0',dtype=torch.float32).requires_grad_(False).eval()
        ap=AudioProcessor(target_sample_rate=ae.sample_rate,mel_bins=ae.mel_bins,mel_hop_length=ae.mel_hop_length,n_fft=ae.n_fft).to('cuda:0')
        self.encoders=dict(video=ve,audio=ae,processor=ap)

    def prepare(self,item,work):
        from p3_listen.features import prepare_prefix
        raw=self.media_reader(item,work);obs=observation(item,raw)
        # Only the reference image and paired speaker audio enter Qwen preprocessing.
        processor=self.cpu_processors.get()
        try:native=prepare_prefix(self.cfg,processor,make_observation(obs))
        finally:self.cpu_processors.put(processor)
        return raw,obs,native

    @torch.no_grad()
    def encode(self,item,prepared):
        from p3_listen.features import prefix
        from p2_listen.runtime import encode_vae
        raw,obs,native=prepared;times={};start=time.monotonic()
        pref=prefix(self.runtime,self.processor,None,prepared=native)
        direct=extract_input(self.c,obs,self.runtime,self.processor,pref=pref)
        guided=extract_planned_input(self.c,obs,item['prompt'],self.runtime,self.processor,32768,pref=pref)
        validate_payload(guided,dict(guidance=dict(prompt=item['prompt'],prompt_sha256=digest(item['prompt']))),'input_plan')
        if guided['prefix_tokens']!=direct['condition']['hidden'].shape[2]:raise ValueError('Direct/guided prefixes differ')
        recovered=self.processor.tokenizer.decode(guided['plan_token_ids'][0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if recovered!=item['prompt']:raise ValueError('Qwen plan text changed')
        del pref;times['qwen']=time.monotonic()-start;start=time.monotonic()
        g=geometry(obs);pixels=raw['pixels'];padding=g['frames']-len(pixels)
        if not 0<=padding<=7:raise ValueError('Unexpected video padding')
        if padding:pixels=np.concatenate([pixels,np.repeat(pixels[-1:],padding,axis=0)])
        vae=encode_vae(self.c,obs,None,'cuda:0',pixels=pixels,waveform=(raw['listener'],raw['rate']),encoders=self.encoders)
        times['vae']=time.monotonic()-start
        return dict(input=direct,input_plan=guided,vae=vae),times


def cluster_worker(work,phase,name,rank,text_batch_size=32,prefetch=4,media_processes=0):
    work=Path(work);m=open_manifest(work);c=m['config'];activate(c);cfg=extractor_config(c)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    started=time.time();completed=0;session=None;timings={};count_jobs=0;active_job=None;active_sample=None
    q=dependencies(c,'input_plan');t=dependencies(c,'text');v=dict(ltx=weights_identity(c,'ltx'))
    assets=dict(input=q,input_plan=q,text_guided=t,text=t,vae=v)
    atomic_json(work/'assets'/('gpu-'+name+'.json'),assets)
    def status(state,**kw):
        atomic_json(work/'workers'/(name+'.json'),dict(worker=name,pid=os.getpid(),phase=phase,
            status=state,started_at=started,heartbeat=time.time(),completed_samples=completed,
            completed_jobs=count_jobs,timing_s=timings,**kw))
    def exists(item,part):
        folder=work/'cache'/part/item['sample_id']
        if not (folder/'complete.json').exists():return False
        load_part(folder,cache_identity(m,item,part,assets[part]));return True
    def write_values(item,values):
        for part,value in values.items():
            persist(work,part,item['sample_id'],cache_identity(m,item,part,assets[part]),value)
        return item['sample_id']
    jobs=read(work/'jobs'/(phase+'.json'));offset=rank%len(jobs);jobs=jobs[offset:]+jobs[:offset]
    status('starting')
    try:
        with ExitStack() as stack:
            media_pool=stack.enter_context(ProcessPoolExecutor(max_workers=media_processes,
                mp_context=multiprocessing.get_context('spawn'))) if media_processes else None
            producers=stack.enter_context(ThreadPoolExecutor(max_workers=prefetch))
            writers=stack.enter_context(ThreadPoolExecutor(max_workers=2))
            def media_reader(item,folder):
                return media_pool.submit(read_media,item,folder).result() if media_pool else read_media(item,folder)
            for jid in jobs:
                done=work/'state'/phase/(jid+'.json')
                quarantined=work/'quarantine'/phase/(jid+'.json')
                if done.exists() or quarantined.exists():continue
                with job_lock(work/'locks'/phase/(jid+'.lock')) as acquired:
                    if not acquired or done.exists() or quarantined.exists():continue
                    active_job=jid;active_sample=None
                    pending=deque();writes=deque()
                    try:
                        ids=read(work/'jobs'/phase/(jid+'.json'));items=[sample(work,m,sid) for sid in ids]
                        parts=('text_guided',) if phase=='text' else ('input','input_plan','vae')
                        missing=[item for item in items if not all(exists(item,p) for p in parts)]
                        start=time.monotonic();job_times={};writes=deque()
                        if missing and session is None:
                            status('loading_models',job=jid)
                            if phase=='text':
                                from omni_av.ltx_runtime import NativeTextConditioner
                                session=NativeTextConditioner(cfg,'cuda:0')
                                fixed=dict(zip(('positive','negative'),map(cpu,session.encode_batch([FixedPrompt(GEMMA),FixedPrompt('')]))))
                                ident=cache_identity(m,None,'text',t)
                                persist(work,'text_replicas',name,ident,fixed)
                                with job_lock(work/'locks'/'fixed.lock') as got:
                                    if got and not (work/'cache/text/shared/complete.json').exists():persist(work,'text','shared',ident,fixed)
                                del fixed
                            else:session=FeatureSession(c,prefetch,media_reader)
                        if phase=='text':
                            for first in range(0,len(missing),text_batch_size):
                                batch=missing[first:first+text_batch_size];prompts=[item['prompt'] for item in batch]
                                tokens=[len(session.encoder.tokenizer.tokenizer(p.strip(),truncation=False)['input_ids']) for p in prompts]
                                status('encoding',job=jid,batch=len(batch),maximum_tokens=max(tokens))
                                if max(tokens)<=1024:values=session.encode_batch([FixedPrompt(p) for p in prompts])
                                else:values=[encode_text(session,p,8192)[0] for p in prompts]
                                for item,value,n in zip(batch,values,tokens):
                                    value=dict(positive=cpu(value),prompt_sha256=digest(item['prompt']),tokens=n,sequence_length=value['mask'].shape[1],truncated=False)
                                    validate_payload(value,dict(guidance=dict(prompt=item['prompt'],prompt_sha256=digest(item['prompt']))),'text')
                                    writes.append(writers.submit(write_values,item,dict(text_guided=value)))
                                    if len(writes)>=2*text_batch_size:writes.popleft().result()
                                del values,value
                        else:
                            pending=deque();iterator=iter(missing)
                            def submit():
                                item=next(iterator,None)
                                if item is not None:pending.append((item,producers.submit(session.prepare,item,work)))
                            for _ in range(prefetch):submit()
                            while pending:
                                item,future=pending.popleft();active_sample=item['sample_id'];wait=time.monotonic();prepared=future.result();wait=time.monotonic()-wait
                                submit();status('encoding',job=jid,sample_id=item['sample_id'],cpu_wait_s=wait)
                                values,elapsed=session.encode(item,prepared);del prepared,future
                                elapsed['cpu_wait']=wait
                                for key,seconds in elapsed.items():job_times[key]=job_times.get(key,0.)+seconds
                                writes.append(writers.submit(write_values,item,values));del values
                                if len(writes)>=4:writes.popleft().result()
                        for future in writes:future.result()
                        completed+=len(ids);count_jobs+=1
                        for key,seconds in job_times.items():timings[key]=timings.get(key,0.)+seconds
                        atomic_json(done,dict(job=jid,worker=name,samples=len(ids),sample_ids=ids,
                            success=True,elapsed_s=time.monotonic()-start,timing_s=job_times,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,finished_at=time.time()))
                        status('job_complete',job=jid)
                    except SourceVideoGap as exc:
                        # Drain this job before releasing its lock; successful writes
                        # remain reusable, but never mark a partial job complete.
                        for _,future in pending:
                            if future.cancel():continue
                            try:future.result()
                            except SourceVideoGap:pass
                        for future in writes:future.result()
                        atomic_json(quarantined,dict(job=jid,worker=name,sample_id=active_sample,
                            error=str(exc),time=time.time(),success=False))
                        status('job_quarantined',job=jid,sample_id=active_sample,error=str(exc))
        status('finished',elapsed_s=time.time()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except Exception as exc:
        import traceback
        status('failed',job=active_job,sample_id=active_sample,error=repr(exc),traceback=traceback.format_exc())
        raise


def queue_status(work,phase):
    work=Path(work);jobs=read(work/'jobs'/(phase+'.json'))
    done=list((work/'state'/phase).glob('*.json'))
    values=[read(p) for p in done]
    return dict(phase=phase,jobs=len(jobs),finished_jobs=len(done),
        samples=sum(v['samples'] for v in values),complete=len(done)==len(jobs))
