#!/usr/bin/env python3
"""Native frozen Qwen H for the current, audited train+dev 500h selection.

Only selection/media adapters and orchestration are new. Both encoders, prefix,
processor, sample-grid repair and checked atomic tensor writers are reused.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import atomic_json,read,sha,digest
from p2_listen.config import activate,extractor_config
from p3_listen.prompts import contract as prompt_contract

SELECTION=ROOT/'experiments/listen_j_upright_av500h_train_dev_v3_20261105'
OLD=ROOT/'experiments/p2_j_train500h_cache_20261101'
DEFAULT_WORK=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
PARTS=('input','input_plan')
EXPECTED='d1ad15af128fb31833e1fd18ebbaae374839c21a45cce382288171d8ed7eb971'

def ensure(ok,message):
    if not ok:raise ValueError(message)


def code_identity():
    files=[Path(__file__),ROOT/'src/p2_j_cache/data.py',ROOT/'src/p2_j_cache/runtime.py',
           ROOT/'src/p2_plan_cache/encoding.py',ROOT/'src/p3_listen/prompts.py',
           ROOT/'src/listen_benchmark/frame_crop.py',ROOT/'src/p2_listen/cache.py']
    return {str(p):sha(p) for p in files}


def prepare(work):
    from p2_plan_cache.runtime import dependencies
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    audit=read(SELECTION/'selection/INDEPENDENT_FINAL_AUDIT.json')
    delivery=read(SELECTION/'DELIVERY_COMPLETE.json')
    manifest=SELECTION/'selection/selected.jsonl';index=SELECTION/'selection/index.sqlite'
    ensure(audit['passed'] and delivery['status']=='complete' and delivery['samples']==284218,'Unverified current selection')
    ensure(sha(manifest)==EXPECTED,'Current selection SHA differs')
    cfg=read(OLD/'manifest.json')['config'];activate(cfg)
    ec=extractor_config(cfg)
    ensure(list(ec.hidden_layers)==[8,18,28,35] and Path(ec.omni_path).name=='Qwen2.5-Omni-3B','Wrong original Qwen configuration')
    ids=[];counts=Counter();hours=0.;candidates={};rows=[]
    con=sqlite3.connect(f'file:{index}?mode=ro',uri=True);con.row_factory=sqlite3.Row
    offsets={r['sid']:dict(r) for r in con.execute('SELECT * FROM samples')};con.close()
    with manifest.open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            r=json.loads(line);sid=r['sample_id'];ix=offsets[sid]
            ensure(r['split'] in ('train','dev') and r['source_split']==r['split'],'Invalid source split')
            ensure(ix['offset']==offset and Path(ix['file'])==manifest and ix['caption_sha256']==r['caption_file_sha256'],'Selection index mismatch')
            ids.append(sid);counts[r['split']]+=1;hours+=r['duration_s']/3600
            key=(r['split'],r['selection_group'],'restored' if 'orientation_restoration' in r else 'upright')
            item=(r['duration_s'],len(r['caption']),sid)
            candidates.setdefault(key,[]).append(item)
            rows.append((sid,offset,hashlib.sha256(line).hexdigest(),r['split'],r['duration_s'],r['caption_file_sha256'],r['selection_group']))
    ensure(len(ids)==len(set(ids))==284218 and dict(counts)=={'train':281060,'dev':3158},'Wrong current members')
    canary=[]
    def add(sid):
        if sid not in canary:canary.append(sid)
    for key,group in sorted(candidates.items()):
        for v in (min(group),max(group),max(group,key=lambda x:x[1])):add(v[2])
    for sid in sorted(ids,key=lambda s:hashlib.sha256(('current-qwen-canary-v1'+s).encode()).digest()):
        if len(canary)>=32:break
        add(sid)
    ensure(len(canary)==32,'Canary strata exceed planned size')
    dbpath=work/'inputs.sqlite'
    if not dbpath.exists():
        temp=work/'inputs.sqlite.tmp';db=sqlite3.connect(temp)
        db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,offset INTEGER,row_sha TEXT,split TEXT,duration REAL,caption_sha TEXT,selection_group TEXT)')
        db.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?)',rows);db.commit();db.close();temp.replace(dbpath)
    p=dict(schema='current-j-train-dev-native-qwen-h-v1',selection=str(manifest),selection_sha256=EXPECTED,
           selection_audit=str(SELECTION/'selection/INDEPENDENT_FINAL_AUDIT.json'),selection_audit_sha256=sha(SELECTION/'selection/INDEPENDENT_FINAL_AUDIT.json'),
           index_sha256=sha(dbpath),samples=len(ids),sample_ids=ids,source_split_counts=dict(counts),hours=hours,
           canary_ids=canary,config=cfg,dependencies=dependencies(cfg,'input_plan'),code=code_identity(),prompts=prompt_contract(),
           model_policy='original Qwen2.5-Omni-3B; frozen; no planning adapter',parts=list(PARTS),
           guidance='full latest caption verbatim in existing shared Gemma/Qwen prompt; native im_end; no truncation',
           input_policy='final-crop first frame and paired speaker audio; listener audio is never Qwen input',
           old_reference_source=str(OLD),hidden_layers=[8,18,28,35],hidden_width=2048,
           cache_scope='Qwen H only; no Gemma, AV latent or teacher extraction; no training',
           original_flags_preserved=True,test_excluded=True)
    atomic_json(work/'protocol.json',p,immutable=True)
    atomic_json(work/'jobs.json',[ids[i:i+128] for i in range(0,len(ids),128)],immutable=True)
    c=read(ROOT/'configs/p2_planning_cluster_filtered_20261102.json')
    for n in c['nodes']:
        if n['name']=='node8':n['host']='172.29.104.237'
    atomic_json(work/'cluster.json',c,immutable=True)
    atomic_json(work/'PREPARED.json',dict(samples=len(ids),hours=hours,source_split_counts=dict(counts),canary_ids=canary,time=time.time()))
    print(json.dumps(dict(prepared=True,samples=len(ids),hours=hours,source_split_counts=dict(counts))),flush=True)


class Reader:
    def __init__(self,work,verify=True):
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol
        self.protocol_sha=sha(self.work/'protocol.json');self.ids=set(p['sample_ids']);self.config=p['config'];activate(self.config)
        ensure(p['code']==code_identity() and p['prompts']==prompt_contract(),'Code/prompt changed; use a new run revision')
        if verify:
            from p2_plan_cache.runtime import dependencies
            ensure(dependencies(self.config,'input_plan')==p['dependencies'],'Original Qwen weights/dependencies changed')
            ensure(sha(self.work/'inputs.sqlite')==p['index_sha256'],'Frozen local row index changed')
        self.db=sqlite3.connect(f'file:{self.work}/inputs.sqlite?mode=ro',uri=True);self.db.row_factory=sqlite3.Row
        self.stream=Path(p['selection']).open('rb')
    def row(self,sid):
        ix=self.db.execute('SELECT * FROM samples WHERE sid=?',(sid,)).fetchone();ensure(ix is not None,'Unknown sample')
        self.stream.seek(ix['offset']);raw=self.stream.readline();ensure(hashlib.sha256(raw).hexdigest()==ix['row_sha'],'Frozen selection row changed')
        row=json.loads(raw);ensure(row['sample_id']==sid and row['split']==ix['split'] and row['caption_file_sha256']==ix['caption_sha'],'Row/index differs')
        return row
    def identity(self,row,part):
        ensure(part in PARTS,'Unknown hidden part')
        return dict(schema=self.protocol['schema'],protocol_sha256=self.protocol_sha,part=part,sample_id=row['sample_id'],
                    source_sha256=digest(row['source']),caption_sha256=row['caption_file_sha256'],split=row['split'])
    def load(self,sid,part='input_plan',device='cpu',missing_ok=False):
        from p2_listen.cache import load_part
        row=self.row(sid);value=load_part(self.work/'cache'/part/sid,self.identity(row,part),device,missing_ok)
        if value is not None:validate(row,value,part)
        return value


@lru_cache(maxsize=64)
def pair_result(path,expected):
    from listen_benchmark.j_omni import check_file
    check_file(path,expected);return read(path)


def item_from_row(row):
    from p2_j_cache.data import caption_prompt
    raw=Path(row['caption_file']).read_bytes();ensure(hashlib.sha256(raw).hexdigest()==row['caption_file_sha256'],'Latest caption changed')
    cap=json.loads(raw)
    for k in ('sample_id','caption','source','audio_recipe','crop_provenance','finish_reason','review_flags'):
        ensure(cap[k]==row[k],'Caption/selected row differs: '+k)
    ensure(row['finish_reason']=='stop' and 'repeated_generation' not in row['review_flags'],'Unrepaired generation failure')
    src=row['source'];clip=pair_result(src['source_result'],src['source_result_sha256'])['clips'][src['source_clip_index']]
    ensure(clip['sample_id']==row['sample_id'] and clip['split']==row['split'],'Wrong paired clip/split')
    ensure(clip['listener_id']==src['listener_id'] and clip['speaker_id']==src['speaker_id'] and src['listener_id']!=src['speaker_id'],'Swapped participant roles')
    ensure(abs(clip['start']-src['source_start_s'])<1e-7 and abs(clip['end']-src['source_end_s'])<1e-7,'Paired time mismatch')
    original=cap.get('original_effective_source',cap['source'])
    ensure(clip['sources']['listener']==original['source'],'Wrong original video lineage')
    speaker=clip['normalized_audio']['speaker']
    ensure(speaker['contract']['source']==clip['sources']['speaker'] and Path(clip['sources']['speaker']['member']).stem==src['speaker_id'],'Wrong paired speaker')
    return dict(sample_id=row['sample_id'],split=row['split'],source=src,speaker=speaker,
                caption=row['caption'],prompt=caption_prompt(row['caption']),caption_sha256=row['caption_file_sha256'])


def input_media(reader,row,item,fresh=False):
    import numpy as np
    from PIL import Image
    from p2_j_cache.data import decode_frames,read_speaker
    sid=item['sample_id'];old=Path(reader.protocol['old_reference_source']);ref=old/'references'/(sid+'.png')
    audit_path=old/'media_audit'/(sid+'.json');olditem=old/'samples'/(sid+'.json');reused=False
    if not fresh and ref.exists() and audit_path.exists() and olditem.exists():
        olddata=read(olditem);audit=read(audit_path)
        if olddata['source']==item['source'] and audit['final_crop']==item['source']['video_transform']['crop']:
            with Image.open(ref) as im:pixels=np.asarray(im.convert('RGB'))
            ensure(hashlib.sha256(pixels.tobytes()).hexdigest()==audit['first_rgb_sha256'],'Reused reference pixels changed')
            pts=audit['source_frame_pts_s'][0];reused=True
    if not reused:
        pixels,ptslist=decode_frames(item['source'],limit=1);pixels=pixels[0];pts=ptslist[0]
        ref=reader.work/'references'/(sid+'.png');ref.parent.mkdir(parents=True,exist_ok=True)
        if fresh:ref=reader.work/'canary_fresh_references'/(sid+'.png');ref.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(pixels).save(ref)
    ensure(pixels.shape==(512,512,3),'Wrong final reference geometry')
    speaker,rate=read_speaker(item)
    evidence=dict(sample_id=sid,split=row['split'],selected_row_sha256=digest(row),caption_sha256=row['caption_file_sha256'],
                  source_sha256=digest(item['source']),reference_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                  first_source_pts_s=pts,reference_reused=reused,final_crop=item['source']['video_transform']['crop'],
                  speaker_raw_sha256=hashlib.sha256(np.asarray(speaker,dtype='<f4',order='C').tobytes()).hexdigest(),
                  speaker_source_sha256=item['speaker']['sha256'],speaker_rate=rate,
                  restored='orientation_restoration' in row,listener_audio_used=False)
    return dict(speaker=speaker,rate=rate,reference=str(ref)),evidence


def initialize_cpu(work):
    global CPU_READER,CPU_PROCESSOR
    import torch
    from transformers import Qwen2_5OmniProcessor
    torch.set_num_threads(1);CPU_READER=Reader(work,verify=False)
    CPU_PROCESSOR=Qwen2_5OmniProcessor.from_pretrained(extractor_config(CPU_READER.config).omni_path,local_files_only=True)


def prepare_input(sid,fresh=False):
    from p2_j_cache.data import observation
    from p2_listen.features import make_observation
    from p3_listen.features import prepare_prefix
    r=CPU_READER;row=r.row(sid);item=item_from_row(row);raw,audit=input_media(r,row,item,fresh)
    obs=observation(item,raw)
    native=prepare_prefix(extractor_config(r.config),CPU_PROCESSOR,make_observation(obs))
    return dict(row=row,item=item,observation=obs,native=native,audit=audit)


def validate(row,value,part):
    import torch
    from p2_j_cache.data import caption_prompt
    from p2_plan_cache.runtime import validate_payload
    from omni_av_p2.connector import ConditionInputs
    ConditionInputs(**value['condition']).validate()
    h=value['condition']['hidden'];ensure(h.shape[1]==4 and h.shape[-1]==2048 and torch.isfinite(h).all(),'Invalid H tensor')
    ensure(value['condition']['layer_ids'].tolist()==[8,18,28,35],'Wrong hidden layers')
    if part=='input_plan':
        prompt=caption_prompt(row['caption']);validate_payload(value,dict(guidance=dict(prompt=prompt,prompt_sha256=digest(prompt))),part)


def load_session(reader):
    import torch
    from omni_av_p2.data import load_omni
    runtime,processor=load_omni(extractor_config(reader.config),'cuda:0')
    ensure(not runtime.thinker.training and not any(p.requires_grad for p in runtime.thinker.parameters()),'Qwen not frozen')
    ensure(not any('lora_' in n for n,_ in runtime.thinker.named_parameters()),'Unexpected SFT/LoRA adapter')
    return runtime,processor


def encode(reader,session,prepared):
    import torch
    from p3_listen.features import prefix
    from p2_listen.features import extract_input
    from p2_plan_cache.encoding import extract_planned_input
    runtime,processor=session
    with torch.inference_mode():
        pref=prefix(runtime,processor,None,prepared=prepared['native'])
        direct=extract_input(reader.config,prepared['observation'],runtime,processor,pref=pref)
        guided=extract_planned_input(reader.config,prepared['observation'],prepared['item']['prompt'],runtime,processor,32768,pref=pref)
    ensure(guided['prefix_tokens']==direct['condition']['hidden'].shape[2],'Direct/guided prefixes differ')
    ensure(processor.tokenizer.decode(guided['plan_token_ids'][0],skip_special_tokens=False,clean_up_tokenization_spaces=False)==prepared['item']['prompt'],'Guidance truncated or changed')
    values=dict(input=direct,input_plan=guided)
    for part,value in values.items():
        validate(prepared['row'],value,part);value['input_audit']=prepared['audit']
    return values


def write_values(reader,prepared,values):
    from p2_j_cache.runtime import persist
    receipts={};sid=prepared['row']['sample_id']
    for part,value in values.items():
        persist(reader.work,part,sid,reader.identity(prepared['row'],part),value)
        folder=reader.work/'cache'/part/sid;m=read(folder/'complete.json')
        receipts[part]=dict(identity=m['identity'],payload_sha256=m['payload_sha256'],bytes=(folder/'payload.pt').stat().st_size)
    return dict(sample_id=sid,split=prepared['row']['split'],input_audit=prepared['audit'],parts=receipts)


def canary(work):
    import torch
    from p2_listen.features import extract_input
    from p2_plan_cache.encoding import extract_planned_input
    r=Reader(work);initialize_cpu(work);torch.set_num_threads(2);session=load_session(r);rows=[]
    for i,sid in enumerate(r.protocol['canary_ids']):
        p=prepare_input(sid);baseline=prepare_input(sid,True)
        ensure(p['native'].keys()==baseline['native'].keys() and all(torch.equal(p['native'][k],baseline['native'][k]) for k in p['native']),'Reused/native processor inputs differ')
        ensure(p['audit']['reference_rgb_sha256']==baseline['audit']['reference_rgb_sha256'] and p['audit']['speaker_raw_sha256']==baseline['audit']['speaker_raw_sha256'],'Input pixel/audio SHA differs')
        values=encode(r,session,p);refexact=None
        if i<8 or p['audit']['restored']:
            runtime,processor=session
            with torch.inference_mode():
                baseline['observation']['speaker_audio'].seek(0)
                direct=extract_input(r.config,baseline['observation'],runtime,processor)
                baseline['observation']['speaker_audio'].seek(0)
                guided=extract_planned_input(r.config,baseline['observation'],p['item']['prompt'],runtime,processor,32768)
            refexact=all(torch.equal(values[part]['condition']['hidden'],ref['condition']['hidden']) for part,ref in [('input',direct),('input_plan',guided)])
            ensure(refexact,'Shared-prefix extraction differs from native independent extraction')
        receipt=write_values(r,p,values)
        for part in PARTS:r.load(sid,part)
        rows.append(dict(sample_id=sid,split=p['row']['split'],restored=p['audit']['restored'],processor_exact=True,
                         native_hidden_exact=refexact,shapes={k:list(v['condition']['hidden'].shape) for k,v in values.items()},receipt=receipt))
        atomic_json(r.work/'canary_progress.json',dict(completed=len(rows),total=32,time=time.time()))
    atomic_json(r.work/'CANARY.json',dict(success=True,count=len(rows),protocol_sha256=r.protocol_sha,rows=rows,time=time.time()))
    print(json.dumps(dict(canary_passed=True,samples=len(rows))),flush=True)


def claim(work,jobs):
    for i,ids in enumerate(jobs):
        name=f'{i:06d}'
        if (work/'done'/(name+'.json')).exists() or (work/'failed'/(name+'.json')).exists():continue
        path=work/'locks'/(name+'.lock');path.parent.mkdir(parents=True,exist_ok=True);lock=path.open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:lock.close();continue
        if (work/'done'/(name+'.json')).exists() or (work/'failed'/(name+'.json')).exists():lock.close();continue
        return name,ids,lock
    return None


def ready_stream(reader,ids,pool,prefetch):
    pending={};iterator=iter(ids)
    def fill():
        while len(pending)<prefetch:
            sid=next(iterator,None)
            if sid is None:break
            pending[pool.submit(prepare_input,sid)]=sid
    fill()
    while pending:
        ready,_=wait(pending,timeout=240,return_when=FIRST_COMPLETED)
        if not ready:raise TimeoutError('CPU preparation timed out')
        for f in ready:
            pending.pop(f);p=f.result();fill();yield p


def worker(work,wid,producers,prefetch):
    import torch
    r=Reader(work);ensure(read(r.work/'CANARY.json')['protocol_sha256']==r.protocol_sha,'Matching canary required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    pool=ProcessPoolExecutor(max_workers=producers,mp_context=mp.get_context('spawn'),initializer=initialize_cpu,initargs=(work,))
    writers=ThreadPoolExecutor(max_workers=2);jobs=read(r.work/'jobs.json');count=0;session=None;start=time.time()
    def state(status,**kw):atomic_json(r.work/'workers'/f'{wid}.json',dict(worker=wid,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),status=status,samples=count,time=time.time(),started=start,**kw))
    state('starting')
    try:
        while True:
            job=claim(r.work,jobs)
            if job is None:break
            name,ids,lock=job;pending=deque();receipts=[];tick=time.time();state('preparing',job=name)
            try:
                for p in ready_stream(r,ids,pool,prefetch):
                    sid=p['row']['sample_id'];row=p['row'];values={};cached=True
                    for part in PARTS:
                        if r.load(sid,part,missing_ok=True) is None:cached=False
                    if not cached:
                        if session is None:state('loading_model',job=name);session=load_session(r)
                        values=encode(r,session,p)
                        pending.append(writers.submit(write_values,r,p,values));count+=1
                    else:
                        for part in PARTS:values[part]=r.load(sid,part)
                        pending.append(writers.submit(write_values,r,p,values))
                    if len(pending)>=4:receipts.append(pending.popleft().result())
                    if count%8==0:state('extracting',job=name,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                while pending:receipts.append(pending.popleft().result())
                ensure(len(receipts)==len(ids) and {x['sample_id'] for x in receipts}==set(ids),'Incomplete block')
                atomic_json(r.work/'done'/(name+'.json'),dict(success=True,samples=len(ids),sample_ids=ids,receipts=receipts,
                            worker=wid,seconds=time.time()-tick,protocol_sha256=r.protocol_sha))
                state('running',job=name)
            except Exception:
                while pending:
                    try:pending.popleft().result()
                    except Exception:pass
                atomic_json(r.work/'failed'/(name+'.json'),dict(error=traceback.format_exc(),worker=wid,sample_ids=ids,time=time.time()))
                state('block_failed',job=name,error=traceback.format_exc())
            finally:lock.close()
        state('finished')
    except BaseException:state('failed',error=traceback.format_exc());raise
    finally:
        for process in list((getattr(pool,'_processes',None) or {}).values()):
            if process.is_alive():process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);writers.shutdown(wait=True)


def env(gpu=None):
    e=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
           HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    return e


def node(work,name,per_gpu):
    folder=Path(work)/'nodes'/name;folder.mkdir(parents=True,exist_ok=True);children={};logs=[]
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for gpu in range(8):
                for replica in range(per_gpu):
                    wid=f'{name}-g{gpu}-r{replica}';producers=6 if replica<2 else 2
                    cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',work,'--worker-id',wid,'--producers',str(producers)]
                    log=(folder/(wid+'.log')).open('a');logs.append(log)
                    children[wid]=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic_json(folder/(wid+'.pid.json'),dict(pid=children[wid].pid,gpu=gpu,replica=replica))
            while True:
                states={k:p.poll() for k,p in children.items()}
                atomic_json(folder/'status.json',dict(node=name,time=time.time(),states=states,per_gpu=per_gpu,pid=os.getpid()))
                snap=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=snap.stdout))+'\n')
                if all(v is not None for v in states.values()):break
                time.sleep(10)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
        finally:
            for p in children.values():
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for f in logs:f.close()


def status(work):
    work=Path(work);p=read(work/'protocol.json');rows=[read(x) for x in (work/'done').glob('*.json')]
    return dict(time=time.time(),samples=sum(x['samples'] for x in rows),total=p['samples'],finished_jobs=len(rows),
                failures=[str(x) for x in (work/'failed').glob('*.json')],workers=[read(x) for x in (work/'workers').glob('*.json')])


@lru_cache(maxsize=1)
def audit_reader(work):
    return Reader(work,verify=False)


def audit_block(args):
    work,index,ids=args;r=audit_reader(work);receipt=read(r.work/'done'/f'{index:06d}.json')
    ensure(receipt['success'] and receipt['protocol_sha256']==r.protocol_sha and receipt['sample_ids']==ids,'Invalid block receipt')
    ensure(len(receipt['receipts'])==len(ids) and {x['sample_id'] for x in receipt['receipts']}==set(ids),'Bad block membership')
    total=0;counts=Counter()
    for v in receipt['receipts']:
        row=r.row(v['sample_id']);counts[row['split']]+=1
        ensure(v['input_audit']['source_sha256']==digest(row['source']) and v['input_audit']['caption_sha256']==row['caption_file_sha256'],'Input source/caption binding differs')
        for part in PARTS:
            folder=r.work/'cache'/part/row['sample_id'];m=read(folder/'complete.json');expected=r.identity(row,part);rr=v['parts'][part]
            ensure(m['contract']==expected and m['identity']==digest(expected)==rr['identity'],'Invalid payload identity')
            ensure(m['payload_sha256']==rr['payload_sha256'] and (folder/'payload.pt').stat().st_size==rr['bytes'],'Invalid payload receipt/size')
            total+=rr['bytes']
    return dict(samples=len(ids),source_split_counts=dict(counts),payload_bytes=total)


def audit(work):
    r=Reader(work);ensure(sha(r.protocol['selection'])==EXPECTED,'Upstream selected manifest changed')
    ensure(not list((r.work/'failed').glob('*.json')),'Unresolved failed blocks')
    jobs=read(r.work/'jobs.json');ids=[sid for block in jobs for sid in block]
    ensure(len(ids)==len(set(ids))==r.protocol['samples'] and ids==r.protocol['sample_ids'],'Invalid queue member coverage')
    counts=Counter();total=0;samples=0;start=time.time()
    with ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for value in pool.map(audit_block,[(work,i,ids) for i,ids in enumerate(jobs)],chunksize=1):
            samples+=value['samples'];total+=value['payload_bytes'];counts.update(value['source_split_counts'])
    ensure(samples==r.protocol['samples'] and dict(counts)==r.protocol['source_split_counts'],'Final source split coverage differs')
    result=dict(success=True,samples=samples,source_split_counts=dict(counts),payload_bytes=total,protocol_sha256=r.protocol_sha,
                parts=list(PARTS),model=r.protocol['model_policy'],selection_sha256=EXPECTED,
                writer_validation='Every payload SHA256 and exact tensor serialization roundtrip at write; final full marker/receipt/size/source check',
                parallel_audit_seconds=time.time()-start,time=time.time(),training_started=False)
    atomic_json(r.work/'COMPLETE.json',result,immutable=True);return result


def remote(cluster,n,args):
    from p2_planning_cluster import ssh
    p=subprocess.run(ssh(n,cluster)+[shlex.join(args)],text=True,capture_output=True,timeout=60)
    ensure(p.returncode==0,n['name']+': '+p.stderr[-3000:]);return p.stdout


def probe(cluster,n):
    try:
        gpu=remote(cluster,n,['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
        apps=remote(cluster,n,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
        rows=[[float(x) for x in line.split(',')] for line in gpu.strip().splitlines()]
        return dict(node=n['name'],available=len(rows)==8 and all(x[1]>=100*1024 for x in rows) and not apps.strip(),gpu=rows,apps=apps)
    except Exception:return dict(node=n['name'],available=False,error=traceback.format_exc())


def launch_remote(cluster,n,work,per_gpu):
    code='''import sys,subprocess,json,os,time
from pathlib import Path
root,work,name,per_gpu,script=sys.argv[1:]
f=Path(work)/'nodes'/name;f.mkdir(parents=True,exist_ok=True)
pidfile=f/'launch.json'
if pidfile.exists():
 old=json.loads(pidfile.read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
 if p.exists() and script.encode() in p.read_bytes() and work.encode() in p.read_bytes():
  print(json.dumps(dict(already_running=True,pid=old['pid'])));sys.exit()
cmd=[sys.executable,script,'node','--work',work,'--node',name,'--per-gpu',per_gpu]
e=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (f/'supervisor.log').open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,command=cmd,time=time.time());pidfile.write_text(json.dumps(r));print(json.dumps(r))'''
    return json.loads(remote(cluster,n,[cluster['python'],'-c',code,str(ROOT),str(work),n['name'],str(per_gpu),str(Path(__file__).resolve())]))


def chain(work,per_gpu):
    work=Path(work);r=Reader(work);cluster=read(work/'cluster.json')
    def state(stage,**kw):atomic_json(work/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (work/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (work/'COMPLETE.json').exists():state('complete');return
            if not (work/'production_started.json').exists():
                stable=0
                while stable<2:
                    with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(lambda n:probe(cluster,n),cluster['nodes']))
                    stable=stable+1 if all(v['available'] for v in checks) else 0
                    state('resource_check',checks=checks,consecutive_clear=stable)
                    if stable<2:time.sleep(10)
                if not (work/'CANARY.json').exists():
                    state('gpu_canary')
                    with (work/'canary.log').open('a') as log:
                        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'canary','--work',str(work)],cwd=ROOT,env=env(0),stdout=log,stderr=subprocess.STDOUT)
                        ensure(child.wait()==0,'Canary failed; see canary.log')
                proof=read(work/'CANARY.json');ensure(proof['success'] and proof['protocol_sha256']==r.protocol_sha,'Wrong canary')
                atomic_json(work/'execution.json',dict(per_gpu=per_gpu,model_workers=32*per_gpu,producers_per_replica=[6,6,2][:per_gpu],prefetch=8,
                            writer_threads=2,job_size=128,native_batch=1,source_split_preserved=True),immutable=True)
                atomic_json(work/'production_started.json',dict(time=time.time(),protocol_sha256=r.protocol_sha),immutable=True)
            with ThreadPoolExecutor(max_workers=4) as pool:launches=list(pool.map(lambda n:launch_remote(cluster,n,work,per_gpu),cluster['nodes']))
            state('extracting',launches=launches);last=-1;change=time.monotonic()
            while True:
                s=status(work);atomic_json(work/'cluster_status.json',s)
                with (work/'history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in s.items() if k!='workers'})+'\n')
                state('extracting',samples=s['samples'],total=s['total'],failures=s['failures'])
                if s['samples']==s['total']:
                    state('auditing');result=audit(str(work));state('complete',result=result);return
                if s['samples']!=last:last=s['samples'];change=time.monotonic()
                nodes=[work/'nodes'/n['name']/'EXIT.json' for n in cluster['nodes']]
                if all(p.exists() for p in nodes):raise RuntimeError('All nodes exited before full coverage; inspect failed blocks and worker logs')
                if time.monotonic()-change>1800:raise RuntimeError('No block completion for 30 minutes')
                time.sleep(20)
        except BaseException:state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','canary','worker','node','chain','status','audit'])
    p.add_argument('--work',default=str(DEFAULT_WORK));p.add_argument('--worker-id',default='local');p.add_argument('--node',default='node12')
    p.add_argument('--producers',type=int,default=6);p.add_argument('--prefetch',type=int,default=8);p.add_argument('--per-gpu',type=int,default=3)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.work)
    elif a.command=='canary':canary(a.work)
    elif a.command=='worker':worker(a.work,a.worker_id,a.producers,a.prefetch)
    elif a.command=='node':node(a.work,a.node,a.per_gpu)
    elif a.command=='chain':chain(a.work,a.per_gpu)
    elif a.command=='audit':print(json.dumps(audit(a.work)))
    else:
        s=status(a.work);s.pop('workers');print(json.dumps(s))

if __name__=='__main__':main()
