"""Independent guided-H overlay using the verified native J cache functions."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from p2_listen.contracts import ROOT, read, sha, digest, atomic_json
from p2_listen.config import extractor_config
from p2_listen.cache import load_part
from p2_j_cache.runtime import CacheReader, persist
from p2_j_cache.data import sample, read_speaker, observation
from p2_plan_cache.runtime import dependencies, validate_payload


def code_identity():
    paths=[Path(__file__), ROOT/'scripts/p2_j_sft_hidden.py',
           ROOT/'src/p2_plan_cache/encoding.py', ROOT/'src/p2_j_teacher/runner.py']
    return {str(p):sha(p) for p in paths}


def adapter_identity(path):
    path=Path(path)
    files=['adapter_config.json','adapter_model.safetensors']
    result={name:sha(path/name) for name in files}
    c=read(path/'adapter_config.json')
    if c['peft_type']!='LORA' or c.get('modules_to_save'):
        raise ValueError('Expected language-only planning LoRA')
    return dict(path=str(path.resolve()),files=result)


def prepare_work(source,teacher,adapter,work,canary_ids):
    source=Path(source).resolve();teacher=Path(teacher).resolve();work=Path(work).resolve()
    reader=CacheReader(source);complete=read(source/'COMPLETE.json');ids=reader.ids('train')
    tp=read(teacher/'protocol.json')
    if not complete['success'] or complete['samples']!=len(ids):raise ValueError('Incomplete original cache')
    if set(tp['sample_ids'])!=set(ids) or Path(tp['source'])!=source:raise ValueError('Teacher source/member mismatch')
    if len(canary_ids)<8 or len(set(canary_ids))!=len(canary_ids) or not set(canary_ids)<=set(ids):
        raise ValueError('Invalid canary membership')
    training_complete=Path(adapter).parent/'COMPLETE.json'
    if not read(training_complete)['success']:raise ValueError('Planning SFT is incomplete')
    if Path(read(Path(adapter)/'adapter_config.json')['base_model_name_or_path']).resolve()!=Path(extractor_config(reader.manifest['config']).omni_path).resolve():
        raise ValueError('Planning adapter base differs from source Qwen')
    p=dict(schema='p2-j-planning-sft-guided-hidden-v1',source=str(source),teacher=str(teacher),
           source_manifest_sha256=sha(source/'manifest.json'),source_complete_sha256=sha(source/'COMPLETE.json'),
           teacher_protocol_sha256=sha(teacher/'protocol.json'),adapter=adapter_identity(adapter),
           training_complete=str(training_complete),training_complete_sha256=sha(training_complete),
           dependencies=dependencies(reader.manifest['config'],'input_plan'),code=code_identity(),
           sample_ids=ids,samples=len(ids),ids_sha256=digest(ids),canary_ids=canary_ids,
           part='input_plan',plan_source='unchanged_original_annotation',qwen_frozen=True,
           adapter_execution='unmerged PEFT language LoRA on native Thinker',
           input_policy='verified final-crop first PNG + original paired speaker audio',
           reuse_parts=['text','text_guided','vae','independent_teacher'],
           hidden_scope='all input and exact annotation prompt tokens plus native im_end',
           hidden_layers=list(extractor_config(reader.manifest['config']).hidden_layers))
    atomic_json(work/'protocol.json',p,immutable=True)
    atomic_json(work/'jobs.json',[ids[i:i+128] for i in range(0,len(ids),128)],immutable=True)
    return p


class Reader:
    def __init__(self,work):
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol
        self.source=CacheReader(p['source']);self.ids=set(p['sample_ids']);self.protocol_sha=sha(self.work/'protocol.json')
        if (p['schema']!='p2-j-planning-sft-guided-hidden-v1' or p['code']!=code_identity()
                or adapter_identity(p['adapter']['path'])!=p['adapter']
                or sha(Path(p['source'])/'manifest.json')!=p['source_manifest_sha256']
                or sha(Path(p['source'])/'COMPLETE.json')!=p['source_complete_sha256']
                or sha(p['training_complete'])!=p['training_complete_sha256']
                or dependencies(self.source.manifest['config'],'input_plan')!=p['dependencies']):
            raise ValueError('SFT hidden source/weights/code changed')
    def item(self,sid):
        if sid not in self.ids:raise ValueError('Sample outside frozen train selection')
        return sample(self.source.work,self.source.manifest,sid)
    def identity(self,item):
        return dict(schema=self.protocol['schema'],protocol_sha256=self.protocol_sha,
                    sample_id=item['sample_id'],source_digest=self.source.manifest['entries'][item['sample_id']],
                    part='input_plan',adapter=self.protocol['adapter'],prompt_sha256=digest(item['prompt']))
    def load(self,sid,device='cpu',missing_ok=False):
        item=self.item(sid)
        value=load_part(self.work/'cache/input_plan'/sid,self.identity(item),device,missing_ok)
        if value is not None:validate_payload(value,dict(guidance=dict(prompt=item['prompt'],prompt_sha256=digest(item['prompt']))),'input_plan')
        return value
    def training_sample(self,sid,device='cpu',include_teacher=True):
        # Reuse the original assembler without loading the old H first.
        owner=self
        class View:
            work=owner.source.work
            manifest=owner.source.manifest
            def part(self,sid,part,device='cpu'):
                return owner.load(sid,device) if part=='input_plan' else owner.source.part(sid,part,device)
        result=CacheReader.training_sample(View(),sid,'guided',device)
        result['metadata']['condition_overlay']=str(self.work)
        result['metadata']['planning_adapter_sha256']=self.protocol['adapter']['files']['adapter_model.safetensors']
        if include_teacher:
            from p2_j_teacher.core import Reader as TeacherReader,validate_geometry
            if not hasattr(self,'_teacher'):self._teacher=TeacherReader(self.protocol['teacher'])
            t=self._teacher.load(sid,device);validate_geometry(t,result['layout'],2048)
            result['teacher']={k:t[k] for k in ('video','audio','video_valid','audio_valid')}
        return result


def predecessor_ready(protocol):
    root=Path(protocol['teacher']);f=root/'COMPLETE.json'
    if not f.exists():return False
    result=read(f)
    if (not result['success'] or result['samples']!=protocol['samples']
            or result['protocol_sha256']!=protocol['teacher_protocol_sha256']
            or sha(root/'protocol.json')!=protocol['teacher_protocol_sha256']
            or Path(result['source'])!=Path(protocol['source'])):
        raise ValueError('Predecessor completion identity mismatch')
    return True


def require_ready(reader,canary=False):
    if not predecessor_ready(reader.protocol):raise RuntimeError('Wait for complete teacher audit before GPU extraction')
    if not canary:
        c=read(reader.work/'CANARY.json')
        if not c['success'] or c['protocol_sha256']!=reader.protocol_sha:
            raise ValueError('GPU canary has not passed for this exact protocol')


def verified_input(item,source):
    from listen_benchmark.j_omni import check_file
    source=Path(source);sid=item['sample_id']
    for path,h in item['source_files'].items():check_file(path,h)
    audit=read(source/'media_audit'/(sid+'.json'));reference=source/'references'/(sid+'.png')
    with Image.open(reference) as im:pixels=np.asarray(im.convert('RGB'))
    first_hash=hashlib.sha256(pixels.tobytes()).hexdigest()
    measure=read(item['source']['quality_screening']['measurement'])
    if (pixels.shape!=(512,512,3) or first_hash!=audit['first_rgb_sha256']
            or audit['sample_id']!=sid or audit['final_crop']!=item['source']['video_transform']['crop']
            or audit['final_crop']!=measure['crop']['new'] or item['split']!='train'
            or audit['caption_sha256']!=item['caption_sha256']):
        raise ValueError('Source reference is not the verified final listener first frame')
    wave,rate=read_speaker(item)
    wave_hash=hashlib.sha256(np.asarray(wave,dtype='<f4').tobytes()).hexdigest()
    if wave_hash!=audit['speaker_raw_sha256']:raise ValueError('Original paired speaker audio changed')
    return dict(speaker=wave,rate=rate,reference=str(reference)),dict(
        first_rgb_sha256=first_hash,speaker_raw_sha256=wave_hash,final_crop=audit['final_crop'],
        first_source_pts=audit['source_frame_pts_s'][0],media_audit_sha256=sha(source/'media_audit'/(sid+'.json')),
        reused_verified_first_frame=True,listener_target_in_input=False)


_cpu=None
def initialize_cpu(config,source):
    global _cpu
    torch.set_num_threads(1)
    from transformers import Qwen2_5OmniProcessor
    cfg=extractor_config(config)
    _cpu=(cfg,Qwen2_5OmniProcessor.from_pretrained(cfg.omni_path,local_files_only=True),source)


def prepare_input(item):
    from p3_listen.features import prepare_prefix
    from p2_listen.features import make_observation
    cfg,processor,source=_cpu;t=time.monotonic()
    raw,audit=verified_input(item,source);obs=observation(item,raw)
    native=prepare_prefix(cfg,processor,make_observation(obs))
    return dict(observation=obs,native=native,audit=audit,seconds=time.monotonic()-t)


def load_session(reader):
    from omni_av_p2.data import load_omni
    from peft import PeftModel
    cfg=extractor_config(reader.source.manifest['config'])
    runtime,processor=load_omni(cfg,'cuda:0')
    peft=PeftModel.from_pretrained(runtime.thinker,reader.protocol['adapter']['path'],is_trainable=False)
    peft.requires_grad_(False).eval();runtime.thinker=peft.get_base_model()
    names=[n for n,p in runtime.thinker.named_parameters() if 'lora_' in n]
    if not names or any('.model.layers.' not in '.'+n or p.requires_grad for n,p in runtime.thinker.named_parameters() if 'lora_' in n):
        raise ValueError('Expected frozen language-only PEFT modules')
    if any(p.requires_grad for p in runtime.thinker.parameters()):raise ValueError('Hidden extraction must be frozen')
    if not hasattr(runtime.thinker.model,'layers'):raise ValueError('Native Thinker text-model entry was wrapped incorrectly')
    return runtime,processor,peft


@torch.inference_mode()
def encode(reader,session,item,prepared):
    from p3_listen.features import prefix
    from p2_plan_cache.encoding import extract_planned_input
    runtime,processor,_=session
    pref=prefix(runtime,processor,None,prepared=prepared['native'])
    value=extract_planned_input(reader.source.manifest['config'],prepared['observation'],item['prompt'],runtime,processor,32768,pref=pref)
    validate_payload(value,dict(guidance=dict(prompt=item['prompt'],prompt_sha256=digest(item['prompt']))),'input_plan')
    recovered=processor.tokenizer.decode(value['plan_token_ids'][0],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if recovered!=item['prompt']:raise ValueError('Full original annotation prompt must be preserved')
    value['input_audit']=prepared['audit'];value['planning_adapter']=reader.protocol['adapter']
    return value


def write_value(reader,item,value):
    persist(reader.work,'input_plan',item['sample_id'],reader.identity(item),value)
    folder=reader.work/'cache/input_plan'/item['sample_id'];marker=read(folder/'complete.json')
    return dict(sample_id=item['sample_id'],payload_sha256=marker['payload_sha256'],identity=marker['identity'],bytes=(folder/'payload.pt').stat().st_size)


def cpu_check(work):
    """Check the real inputs without allocating GPU memory or waiting for teacher."""
    from p2_j_cache.data import read_input
    from p3_listen.features import prepare_prefix
    from p2_listen.features import make_observation
    reader=Reader(work);initialize_cpu(reader.source.manifest['config'],reader.protocol['source'])
    cfg,processor,_=_cpu;rows=[]
    for i,sid in enumerate(reader.protocol['canary_ids']):
        item=reader.item(sid);prepared=prepare_input(item);old=reader.source.part(sid,'input_plan')
        tokens=processor.tokenizer.encode(item['prompt'],add_special_tokens=False)
        if tokens!=old['plan_token_ids'][0].tolist() or old['prompt']!=item['prompt']:
            raise ValueError('Exact original prompt tokens differ from old H')
        if prepared['native']['input_ids'].shape[1]!=old['prefix_tokens']:
            raise ValueError('Original multimodal prefix length changed')
        source_exact=None
        if i<8:
            raw=read_input(item,reader.work/'cpu_reference')
            baseline=prepare_prefix(cfg,processor,make_observation(observation(item,raw)))
            source_exact=(baseline.keys()==prepared['native'].keys() and
                          all(torch.equal(baseline[k],prepared['native'][k]) for k in baseline))
            if not source_exact:raise ValueError('Optimized CPU inputs differ from native source decoding')
        rows.append(dict(sample_id=sid,source_processor_exact=source_exact,original_prompt_tokens_exact=True,
                         prefix_tokens=old['prefix_tokens'],plan_tokens=len(tokens),input_audit=prepared['audit']))
        atomic_json(reader.work/'cpu_progress.json',dict(completed=len(rows),total=len(reader.protocol['canary_ids'])))
    result=dict(success=True,count=len(rows),protocol_sha256=reader.protocol_sha,rows=rows,time=time.time(),gpu_used=False)
    atomic_json(reader.work/'CPU_CHECK.json',result);return result


def canary(work):
    from p2_j_cache.data import read_input
    from p3_listen.features import prefix,prepare_prefix
    from p2_listen.features import make_observation
    from p2_plan_cache.encoding import extract_planned_input
    reader=Reader(work);require_ready(reader,True);torch.set_num_threads(2);torch.cuda.set_device(0)
    initialize_cpu(reader.source.manifest['config'],reader.protocol['source']);session=load_session(reader)
    runtime,processor,peft=session;rows=[]
    for i,sid in enumerate(reader.protocol['canary_ids']):
        item=reader.item(sid);prepared=prepare_input(item);value=encode(reader,session,item,prepared)
        old=reader.source.part(sid,'input_plan')
        for k in ('plan_token_ids','plan_mask','position_ids'):
            if not torch.equal(value[k],old[k]):raise ValueError('Source prompt/token geometry differs: '+sid)
        if value['condition']['hidden'].shape!=old['condition']['hidden'].shape:raise ValueError('H shape differs')
        reference_exact=None;disabled_exact=None
        if i<8:
            raw=read_input(item,reader.work/'canary_reference');obs=observation(item,raw)
            native=prepare_prefix(runtime.config,processor,make_observation(obs))
            if native.keys()!=prepared['native'].keys() or any(not torch.equal(native[k],prepared['native'][k]) for k in native):
                raise ValueError('Optimized verified-PNG inputs differ from source decode')
            # Independent native path re-prepares input and re-encodes prefix.
            obs=observation(item,raw)
            ref=extract_planned_input(reader.source.manifest['config'],obs,item['prompt'],runtime,processor,32768)
            reference_exact=torch.equal(ref['condition']['hidden'],value['condition']['hidden'])
            if not reference_exact:raise ValueError('Optimized SFT hidden differs from native extraction')
            with peft.disable_adapter():
                base=encode(reader,session,item,prepared)
            disabled_exact=torch.equal(base['condition']['hidden'],old['condition']['hidden'])
            if not disabled_exact:raise ValueError('Disabling new LoRA does not reproduce the old H')
        changed=not torch.equal(value['condition']['hidden'],old['condition']['hidden'])
        if not changed:raise ValueError('Planning adapter did not affect H')
        write_value(reader,item,value)
        joined=reader.training_sample(sid)
        if not torch.equal(joined['condition'].hidden,value['condition']['hidden']):raise ValueError('Overlay read lost new H')
        rows.append(dict(sample_id=sid,native_reference_exact=reference_exact,disabled_adapter_old_hidden_exact=disabled_exact,
                         adapter_changes_hidden=changed,geometry_and_reader_ok=True,shape=list(value['condition']['hidden'].shape)))
        atomic_json(reader.work/'canary_progress.json',dict(completed=len(rows),total=len(reader.protocol['canary_ids']),rows=rows))
    result=dict(success=True,protocol_sha256=reader.protocol_sha,count=len(rows),rows=rows,time=time.time())
    atomic_json(reader.work/'CANARY.json',result,immutable=True);return result


def prepared_stream(reader,ids,pool,prefetch):
    iterator=iter(ids);pending={};cached=deque();exhausted=False
    def fill():
        nonlocal exhausted
        while not exhausted and len(pending)+len(cached)<prefetch:
            sid=next(iterator,None)
            if sid is None:exhausted=True;break
            item=reader.item(sid)
            if reader.load(sid,missing_ok=True) is not None:cached.append((item,None))
            else:pending[pool.submit(prepare_input,item)]=item
    fill()
    while cached or pending:
        if cached:
            item,value=cached.popleft();fill();yield item,value;continue
        ready,_=wait(pending,timeout=180,return_when=FIRST_COMPLETED)
        if not ready:raise TimeoutError('No prepared input within 180 seconds')
        for f in ready:
            item=pending.pop(f);value=f.result();fill();yield item,value


def worker(work,worker_id,producers=8,prefetch=8):
    from p2_j_teacher.runner import claim
    reader=Reader(work);require_ready(reader);work=reader.work;jobs=read(work/'jobs.json')
    torch.set_num_threads(2);torch.cuda.set_device(0);session=None;count=0;start=time.time()
    pool=ProcessPoolExecutor(max_workers=producers,mp_context=multiprocessing.get_context('spawn'),
                            initializer=initialize_cpu,initargs=(reader.source.manifest['config'],reader.protocol['source']))
    writers=ThreadPoolExecutor(max_workers=2)
    def state(status,**kw):atomic_json(work/'workers'/f'worker{worker_id}.json',dict(status=status,worker=worker_id,
        gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),pid=os.getpid(),time=time.time(),samples=count,elapsed_s=time.time()-start,**kw))
    state('ready')
    try:
        while True:
            job=claim(work,jobs)
            if job is None:break
            name,ids,lock=job;pending=deque();receipts=[];t=time.time();wait_s=encode_s=0
            try:
                state('running',job=name)
                previous=time.monotonic()
                for item,prepared in prepared_stream(reader,ids,pool,prefetch):
                    wait_s+=time.monotonic()-previous
                    if prepared is None:
                        folder=work/'cache/input_plan'/item['sample_id'];marker=read(folder/'complete.json')
                        receipts.append(dict(sample_id=item['sample_id'],payload_sha256=marker['payload_sha256'],identity=marker['identity'],bytes=(folder/'payload.pt').stat().st_size))
                    else:
                        if session is None:session=load_session(reader)
                        tick=time.monotonic();value=encode(reader,session,item,prepared);encode_s+=time.monotonic()-tick
                        pending.append(writers.submit(write_value,reader,item,value));count+=1
                        if len(pending)>=4:receipts.append(pending.popleft().result())
                        if count%8==0:state('running',job=name,cpu_wait_s=wait_s,encode_s=encode_s)
                    previous=time.monotonic()
                while pending:receipts.append(pending.popleft().result())
                if len(receipts)!=len(ids) or {r['sample_id'] for r in receipts}!=set(ids):raise ValueError('Incomplete task membership')
                atomic_json(work/'done'/(name+'.json'),dict(success=True,samples=len(ids),sample_ids=ids,receipts=receipts,
                    seconds=time.time()-t,worker=worker_id,protocol_sha256=reader.protocol_sha,cpu_wait_s=wait_s,encode_s=encode_s))
                state('running',job=name)
            except Exception:
                while pending:
                    try:pending.popleft().result()
                    except Exception:pass
                atomic_json(work/'failed'/(name+'.json'),dict(success=False,error=traceback.format_exc(),sample_ids=ids,worker=worker_id,time=time.time()))
                raise
            finally:lock.close()
        state('finished')
    except BaseException:state('failed',error=traceback.format_exc());raise
    finally:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive():p.terminate()
        pool.shutdown(wait=False,cancel_futures=True);writers.shutdown(wait=True)


def status(work):
    work=Path(work);p=read(work/'protocol.json');rows=[read(f) for f in (work/'done').glob('*.json')]
    return dict(time=time.time(),samples=sum(r['samples'] for r in rows),total=p['samples'],finished_jobs=len(rows),
                failures=[str(f) for f in (work/'failed').glob('*.json')],workers=[read(f) for f in (work/'workers').glob('*.json')])


def audit(work):
    reader=Reader(work);require_ready(reader);work=reader.work;seen=set();total_bytes=0
    if list((work/'failed').glob('*.json')):raise ValueError('Unresolved extraction failures')
    for i,ids in enumerate(read(work/'jobs.json')):
        r=read(work/'done'/f'{i:06d}.json')
        if (r['protocol_sha256']!=reader.protocol_sha or not r['success'] or r['sample_ids']!=ids
                or len(r['receipts'])!=len(ids) or {x['sample_id'] for x in r['receipts']}!=set(ids) or seen & set(ids)):
            raise ValueError('Invalid completed membership')
        seen.update(ids)
        for v in r['receipts']:
            sid=v['sample_id'];folder=work/'cache/input_plan'/sid;m=read(folder/'complete.json');expected=reader.identity(reader.item(sid))
            if m['contract']!=expected or m['identity']!=digest(expected) or m['identity']!=v['identity'] or m['payload_sha256']!=v['payload_sha256'] or (folder/'payload.pt').stat().st_size!=v['bytes']:
                raise ValueError('Invalid payload identity/receipt: '+sid)
            total_bytes+=v['bytes']
    if seen!=reader.ids:raise ValueError('Final member mismatch')
    result=dict(success=True,samples=len(seen),payload_bytes=total_bytes,time=time.time(),protocol_sha256=reader.protocol_sha,
                adapter=reader.protocol['adapter'],writer_validation='SHA256 and exact tensor roundtrip',
                reused_parts=reader.protocol['reuse_parts'])
    atomic_json(work/'COMPLETE.json',result,immutable=True);return result
