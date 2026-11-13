"""Parallel native Qwen, Gemma and AV caches for frozen J source manifests."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch

from p2_listen.cache import cpu, load_part, save_part, tree_to
from p2_listen.config import activate, extractor_config
from p2_listen.contracts import atomic_json, digest, read
from p2_listen.features import extract_input, unload, weights_identity
from p2_listen.layout import geometry
from p2_plan_cache.encoding import extract_planned_input, encode_text
from p2_plan_cache.runtime import dependencies, validate_payload
from p3_listen.prompts import OMNI, GEMMA, FixedPrompt, observation_messages
from .data import open_manifest, sample, read_media, read_input, observation


def cache_identity(manifest, item, part, assets):
    code=manifest.get('cache_code',{}).get(part,manifest['code'])
    previous=manifest.get('sample_cache_code',{}).get(part)
    if previous and item['sample_id'] in previous['sample_ids']:code=previous['code']
    value = dict(schema=manifest['schema'], part=part, code=code, assets=assets)
    if part=='text':
        return dict(value, prompts=manifest['prompts'], negative_prompt='')
    value.update(sample_id=item['sample_id'], source=item['source'], speaker=item['speaker'])
    if part in ('input','input_plan'):
        value['messages'] = manifest['prompts']['omni_messages']
    if part in ('text_guided','input_plan'):
        value.update(prompt=item['prompt'],caption_sha256=item['caption_sha256'])
    if part=='vae':
        value.update(audio_recipe=item['audio_recipe'], mask_policy='causal-prefix-v1',
                     geometry=dict(fps=25,height=512,width=512))
        override = manifest.get('video_sampling_overrides',{}).get(item['sample_id'])
        if override is not None:
            value['video_sampling_override'] = override
            value['code'] = manifest['code']
    return value


def persist(work, part, sid, identity, value):
    folder = Path(work)/'cache'/part/sid
    save_part(folder,identity,value)
    reread = load_part(folder,identity)
    def equal(a,b):
        if isinstance(a,torch.Tensor):
            return torch.equal(a.detach().cpu(),b)
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):
            return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(value,reread):
        raise ValueError('Cache serialization changed a value: '+part+'/'+sid)
    return reread


def prefetched(work, manifest, ids, reader, workers=2):
    """Bound decoded media to the active item and at most two pending items."""
    iterator=iter(ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque()
        def submit():
            sid=next(iterator,None)
            if sid is None:return False
            item=sample(work,manifest,sid)
            pending.append((item,pool.submit(reader,item,work)))
            return True
        for _ in range(workers):submit()
        while pending:
            item,future=pending.popleft()
            yield item,future.result()
            del future,item
            submit()


def worker(work, gpu, sample_ids, text_batch_size=5, memory_gib=None):
    work=Path(work);m=open_manifest(work);c=m['config'];activate(c)
    if text_batch_size<1:raise ValueError('Text batch size must be positive')
    cfg=extractor_config(c);device='cuda:0';torch.cuda.set_device(0);torch.set_num_threads(4)
    if memory_gib is not None:
        torch.cuda.set_per_process_memory_fraction(memory_gib*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    report=dict(gpu=gpu,pid=__import__('os').getpid(),sample_ids=sample_ids,stages=[],success=False,model_loads=[])
    def progress(stage,**extra):
        row=dict(stage=stage,elapsed_s=time.monotonic()-start,**extra)
        report['stages'].append(row);atomic_json(work/'progress'/f'gpu{gpu}.json',dict(report,current=stage))
        print(__import__('json').dumps(dict(gpu=gpu,**row)),flush=True)
    progress('check_cache')
    dep_qwen=dependencies(c,'input_plan');dep_text=dependencies(c,'text');dep_vae=dict(ltx=weights_identity(c,'ltx'))
    assets=dict(input=dep_qwen,input_plan=dep_qwen,text_guided=dep_text,text=dep_text,vae=dep_vae)
    def missing(item,part):
        ident=cache_identity(m,item,part,assets[part])
        return load_part(work/'cache'/part/item['sample_id'],ident,missing_ok=True) is None
    todo={part:[] for part in ('input','input_plan','text_guided','vae')}
    for sid in sample_ids:
        item=sample(work,m,sid)
        for part in todo:
            if missing(item,part):todo[part].append(sid)
    report['new_parts']={part:len(ids) for part,ids in todo.items()}
    pending_input=set(todo['input']);pending_plan=set(todo['input_plan'])
    input_ids=[sid for sid in sample_ids if sid in pending_input or sid in pending_plan]
    if input_ids:
        progress('load_qwen');report['model_loads'].append('qwen')
        from omni_av_p2.data import load_omni
        runtime,processor=load_omni(cfg,device)
        rendered=processor.apply_chat_template(observation_messages(),tokenize=False,add_generation_prompt=True)
        if (rendered.count(OMNI)!=1 or not rendered.startswith('<|im_start|>system\n') or
            rendered.index(OMNI)>rendered.index('<|vision_bos|>') or
            not rendered.endswith('<|im_start|>assistant\n')):
            raise ValueError('Native chat template does not place the task before media')
        atomic_json(work/'requests'/f'gpu{gpu}.json',dict(messages=observation_messages(),rendered=rendered))
        for item,raw in prefetched(work,m,input_ids,read_input):
            sid=item['sample_id'];progress('qwen',sample_id=sid)
            direct=None
            if sid in pending_input:
                direct=extract_input(c,observation(item,raw),runtime,processor)
                persist(work,'input',sid,cache_identity(m,item,'input',dep_qwen),direct)
            if sid in pending_plan:
                planned=extract_planned_input(c,observation(item,raw),item['prompt'],runtime,processor,32768)
                guided=dict(guidance=dict(prompt=item['prompt'],prompt_sha256=digest(item['prompt'])))
                validate_payload(planned,guided,'input_plan')
                if direct is not None and planned['prefix_tokens']!=direct['condition']['hidden'].shape[2]:
                    raise ValueError('Direct and guided Qwen prefixes differ')
                recovered=processor.tokenizer.decode(planned['plan_token_ids'][0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if recovered!=item['prompt']:raise ValueError('Qwen plan text changed in tokenization')
                persist(work,'input_plan',sid,cache_identity(m,item,'input_plan',dep_qwen),planned)
                del planned
            del raw,direct
        del runtime,processor;unload()
    fixed_id=cache_identity(m,None,'text',dep_text)
    fixed=load_part(work/'cache/text_replicas'/f'gpu{gpu}',fixed_id,missing_ok=True)
    need_fixed=fixed is None
    if need_fixed or todo['text_guided']:
        progress('load_gemma');report['model_loads'].append('gemma')
        from omni_av.ltx_runtime import NativeTextConditioner
        encoder=NativeTextConditioner(cfg,device)
        def requests():
            if need_fixed:
                yield 'positive',None,GEMMA
                yield 'negative',None,''
            for sid in todo['text_guided']:
                item=sample(work,m,sid)
                yield 'guided',item,item['prompt']
        iterator=iter(requests());fixed={} if need_fixed else fixed
        while True:
            chunk=list(islice(iterator,text_batch_size))
            if not chunk:break
            prompts=[r[2] for r in chunk]
            tokens=[len(encoder.encoder.tokenizer.tokenizer(p.strip(),truncation=False)['input_ids']) for p in prompts]
            progress('gemma_batch',batch=len(chunk),tokens=tokens)
            if max(tokens)<=1024:
                values=encoder.encode_batch([FixedPrompt(p) for p in prompts])
            else:
                values=[encode_text(encoder,p,8192)[0] if p else encoder.encode(FixedPrompt('')) for p in prompts]
            for (kind,item,prompt),positive,n in zip(chunk,values,tokens):
                positive=cpu(positive)
                if kind!='guided':fixed[kind]=positive;continue
                value=dict(positive=positive,prompt_sha256=digest(prompt),tokens=n,
                           sequence_length=positive['mask'].shape[1],truncated=False)
                validate_payload(value,dict(guidance=dict(prompt=prompt,prompt_sha256=digest(prompt))),'text')
                persist(work,'text_guided',item['sample_id'],cache_identity(m,item,'text_guided',dep_text),value)
            del values,chunk,positive
        encoder.close();del encoder;unload()
    if need_fixed:persist(work,'text_replicas',f'gpu{gpu}',fixed_id,fixed)
    if gpu==0:persist(work,'text','shared',fixed_id,fixed)
    del fixed
    if todo['vae']:
        progress('load_vae');report['model_loads'].append('vae')
        from ltx_trainer.model_loader import load_video_vae_encoder,load_audio_vae_encoder
        from ltx_core.model.audio_vae import AudioProcessor
        from p2_listen.runtime import encode_vae
        ve=load_video_vae_encoder(cfg.ltx_path,device=device,dtype=torch.bfloat16).requires_grad_(False).eval()
        ae=load_audio_vae_encoder(cfg.ltx_path,device=device,dtype=torch.float32).requires_grad_(False).eval()
        ap=AudioProcessor(target_sample_rate=ae.sample_rate,mel_bins=ae.mel_bins,mel_hop_length=ae.mel_hop_length,n_fft=ae.n_fft).to(device)
        encoders=dict(video=ve,audio=ae,processor=ap)
        for item,raw in prefetched(work,m,todo['vae'],read_media):
            sid=item['sample_id'];obs=observation(item,raw);g=geometry(obs)
            pixels=raw['pixels'];padding=g['frames']-len(pixels)
            if padding<0 or padding>7:raise ValueError('Unexpected 8n+1 video padding')
            if padding:pixels=np.concatenate([pixels,np.repeat(pixels[-1:],padding,axis=0)])
            progress('vae',sample_id=sid,frames=g['frames'])
            value=encode_vae(c,obs,None,device,pixels=pixels,waveform=(raw['listener'],raw['rate']),encoders=encoders)
            persist(work,'vae',sid,cache_identity(m,item,'vae',dep_vae),value)
            del raw,pixels,value,obs
        del ve,ae,ap,encoders;unload()
    atomic_json(work/'assets'/f'gpu{gpu}.json',assets)
    progress('cache_complete')
    report.update(success=True,elapsed_s=time.monotonic()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    atomic_json(work/'reports'/f'gpu{gpu}.json',report)
    return report


class CacheReader:
    """Return the existing condition/AV/text types; split selection stays explicit."""
    def __init__(self,work):
        self.work=Path(work);self.manifest=open_manifest(work);activate(self.manifest['config'])
        paths=sorted((self.work/'assets').glob('gpu*.json'))
        if not paths:
            raise ValueError('No completed worker assets')
        self.assets=read(paths[0])
        if any(read(p)!=self.assets for p in paths[1:]):
            raise ValueError('Workers used different model assets')
        c=self.manifest['config'];q=dependencies(c,'input_plan');t=dependencies(c,'text')
        current=dict(input=q,input_plan=q,text_guided=t,text=t,vae=dict(ltx=weights_identity(c,'ltx')))
        if current!=self.assets:
            raise ValueError('J cache model/dependency identity changed')

    def ids(self,split='train'):
        return list(self.manifest['splits'][split])

    def part(self,sid,part,device='cpu'):
        item=None if part=='text' else sample(self.work,self.manifest,sid)
        ident=cache_identity(self.manifest,item,part,self.assets[part])
        return load_part(self.work/'cache'/part/('shared' if part=='text' else sid),ident,device)

    def training_sample(self,sid,variant='direct',device='cpu'):
        from omni_av_p2.connector import ConditionInputs
        if variant not in ('direct','guided'):
            raise ValueError('Expected direct or guided conditions')
        item=sample(self.work,self.manifest,sid)
        cond=self.part(sid,'input' if variant=='direct' else 'input_plan',device)
        text=self.part(sid,'text',device)
        if variant=='guided':
            guided=self.part(sid,'text_guided',device)
            if cond['prompt']!=guided['positive']['prompt'] or cond['prompt']!=item['prompt']:
                raise ValueError('Qwen/Gemma/original caption prompts differ')
            text=dict(text,positive=guided['positive'])
        x=ConditionInputs(**cond['condition']);x.validate();av=self.part(sid,'vae',device)
        return dict(condition=x,layout=cond['layout'],text=text,**av,
                    metadata=dict(sample_id=sid,split=item['split'],variant=variant,
                                  caption_origin=item['caption_origin'],training_admitted=item['training_admitted']))


def verify(work):
    reader=CacheReader(work);m=reader.manifest
    from omni_av.ltx_runtime import with_text
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig,TextToVideoStrategy
    from ltx_trainer.timestep_samplers import UniformTimestepSampler
    strategy=TextToVideoStrategy(TextToVideoConfig(with_audio=True,first_frame_conditioning_p=1,i2av_only=True))
    sampler=UniformTimestepSampler();rows=[]
    fixed=reader.part(None,'text')
    for path in sorted((Path(work)/'cache/text_replicas').iterdir()):
        replica=load_part(path,cache_identity(m,None,'text',reader.assets['text']))
        for branch in ('positive','negative'):
            for key in ('video','audio','mask'):
                torch.testing.assert_close(replica[branch][key],fixed[branch][key],rtol=1e-3,atol=1e-3)
    for sid in m['sample_ids']:
        row=dict(sample_id=sid,success=True)
        for variant in ('direct','guided'):
            pack=reader.training_sample(sid,variant)
            prepared=strategy.prepare_training_inputs(with_text(pack['av'],pack['text']['positive']),sampler)
            for mod in ('video','audio'):
                loss_mask=getattr(prepared,mod+'_loss_mask')
                if loss_mask.shape!=pack['support'][mod].shape or not (loss_mask & pack['support'][mod]).any():
                    raise ValueError('AV training support mask does not match latent tokens')
                if not torch.isfinite(getattr(prepared,mod).latent).all():
                    raise ValueError('Nonfinite native training inputs')
            row[variant]=dict(hidden=list(pack['condition'].hidden.shape),
                video_latent=list(pack['av']['latents']['latents'].shape),
                audio_latent=list(pack['av']['audio_latents']['latents'].shape),
                text_tokens=int(pack['text']['positive']['mask'].sum()))
        audit=read(Path(work)/'media_audit'/(sid+'.json'))
        row.update(split=audit['split'],source_audit=audit.get('review_pixels_audio_pts_exact',False),
                   listener_full_silence=audit['listener_full_silence'])
        rows.append(row)
    report=dict(success=True,samples=len(rows),splits={k:len(v) for k,v in m['splits'].items()},
        results=rows,native_training_packing_passed=True,event_labels_generated=False,
        source_admission_flags_preserved=True)
    atomic_json(Path(work)/'verification.json',report)
    return report
