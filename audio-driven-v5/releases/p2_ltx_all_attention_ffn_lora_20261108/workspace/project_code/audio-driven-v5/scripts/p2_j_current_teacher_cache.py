#!/usr/bin/env python3
"""Incremental native listener teacher, train+dev, with old audio reuse."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import lru_cache
import fcntl,hashlib,json,multiprocessing as mp,os,signal,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import extractor_config
from p2_j_current_qwen_hidden import Reader as HReader,ensure,item_from_row,env,remote
from p2_j_current_av_cache import free_gpus
import p2_j_teacher.core as core
import p2_j_teacher.media as media
import p2_j_teacher.runner as runner
NativeReader=core.Reader
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
AV=ROOT/'experiments/p2_j_train_dev500h_av_cache_20261105'
AUDIT=ROOT/'experiments/p2_j_train_dev500h_aux_reuse_audit_20261105'
OLD=ROOT/'experiments/p2_j_teacher500h_node8_20261102/production'
WHISPER=ROOT/'experiments/p2_j_train_dev500h_whisper_cache_20261105'
WORK=ROOT/'experiments/p2_j_train_dev500h_teacher_cache_20261105'
ACTIVE=None


def code_identity():
    paths=[Path(__file__),ROOT/'scripts/p2_j_current_qwen_hidden.py',ROOT/'src/p2_j_teacher/core.py',ROOT/'src/p2_j_teacher/media.py',ROOT/'src/p2_j_teacher/runner.py',ROOT/'src/p2_j_teacher_accel.py']
    return {str(p):sha(p) for p in paths}


def prepare(work):
    w=Path(work);w.mkdir(parents=True,exist_ok=True);a=read(AUDIT/'SUMMARY.json');ensure(a['status']=='complete' and not a['errors'],'Unverified teacher reuse map')
    old=NativeReader(OLD);h=HReader(SOURCE);rows=[];increments=[]
    for line in (AUDIT/'reuse_map.jsonl').open():
        p=json.loads(line);rows.append((p['sample_id'],line.strip()))
        if p['teacher']!='reuse_both':increments.append((p['duration_s'],p['sample_id']))
    ensure(len(increments)==7438,'Wrong teacher increment')
    dbpath=w/'mapping.sqlite'
    if not dbpath.exists():
        tmp=w/'mapping.sqlite.tmp';db=sqlite3.connect(tmp);db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,record TEXT)');db.executemany('INSERT INTO samples VALUES(?,?)',rows);db.commit();db.close();tmp.replace(dbpath)
    ids=[sid for _,sid in sorted(increments,reverse=True)];canary=read(AV/'protocol.json')['canary_ids']
    p=dict(schema='current-j-teacher-increment-v1',source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),source_complete_sha256=sha(SOURCE/'COMPLETE.json'),config=h.config,
           sample_ids=ids,samples=7438,full_samples=284218,source_split_counts=h.protocol['source_split_counts'],selection_sha256=h.protocol['selection_sha256'],model=old.protocol['model'],target_dim=2048,
           old=str(OLD),old_protocol_sha256=sha(OLD/'protocol.json'),old_complete_sha256=sha(OLD/'COMPLETE.json'),old_dependencies=old.protocol['dependencies'],
           reuse_map_sha256=sha(AUDIT/'reuse_map.jsonl'),reuse_summary_sha256=sha(AUDIT/'SUMMARY.json'),mapping_sha256=sha(dbpath),code=code_identity(),canary_ids=canary,
           video_encoding_needed=7438,audio_encoding_needed=6542,reuse_both=276780,reuse_audio_only=896,teacher_frozen=True,no_lora=True,video_policy=old.protocol['video_policy'],audio_policy=old.protocol['audio_policy'],mask_policy=old.protocol['mask_policy'])
    atomic_json(w/'protocol.json',p,immutable=True);atomic_json(w/'jobs.json',[ids[i:i+32] for i in range(0,len(ids),32)],immutable=True);atomic_json(w/'cluster.json',read(SOURCE/'cluster.json'),immutable=True)
    print(json.dumps(dict(prepared=True,video=7438,audio=6542,reuse_both=276780,reuse_audio=896)))


class IncrementReader:
    def __init__(self,work,verify=True):
        global ACTIVE
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol;self.protocol_sha=sha(self.work/'protocol.json');self.ids=set(p['sample_ids'])
        ensure(p['code']==code_identity() and sha(SOURCE/'protocol.json')==p['source_protocol_sha256'],'Teacher code/current source changed')
        self.h=HReader(SOURCE,verify=False);self.source=SimpleNamespace(manifest=dict(config=p['config'],video_sampling_overrides={}))
        if verify:
            from p2_plan_cache.runtime import dependencies
            ensure(dependencies(p['config'],'input_plan')==p['old_dependencies'],'Original teacher weights/numerics changed')
            ensure(sha(self.work/'mapping.sqlite')==p['mapping_sha256'],'Teacher reuse map changed')
        self.conn=sqlite3.connect(f'file:{self.work}/mapping.sqlite?mode=ro',uri=True);ACTIVE=self
    def plan(self,sid):
        x=self.conn.execute('SELECT record FROM samples WHERE sid=?',(sid,)).fetchone();ensure(x is not None,'Outside teacher map');return json.loads(x[0])
    def item(self,sid):
        ensure(sid in self.ids,'Outside teacher increment');row=self.h.row(sid);cap=read(row['caption_file']);item=item_from_row(row);plan=self.plan(sid)
        item.update(audio_recipe=row['audio_recipe'],listener_audio_identity=cap['input_evidence']['audio_identity'],crop_provenance=row['crop_provenance'],source_files={row['caption_file']:row['caption_file_sha256']},_row=row,_plan=plan)
        return item
    def identity(self,item):
        return dict(schema=self.protocol['schema'],protocol_sha256=self.protocol_sha,sample_id=item['sample_id'],split=item['split'],source_sha256=digest(item['source']),audio_recipe_sha256=digest(item['audio_recipe']))
    def load(self,sid,device='cpu',missing_ok=False):
        from p2_listen.cache import load_part
        item=self.item(sid);v=load_part(self.work/'cache'/sid,self.identity(item),device,missing_ok)
        if v is not None:
            from p2_listen.layout import make_layout
            core.validate_geometry(v,make_layout(item['source']['duration_s'],(8,8),8,8),2048)
        return v
    def old_value(self,plan):
        from p2_listen.cache import load_part
        folder=Path(plan['teacher_old_payload']).parent;m=read(folder/'complete.json');ensure(m['identity']==plan['teacher_identity'] and m['payload_sha256']==plan['teacher_payload_sha256'],'Old teacher payload changed');return load_part(folder,m['contract'])


class Reader:
    def __init__(self,work=WORK,require_complete=True):
        self.work=Path(work);self.new=IncrementReader(work);p=self.new.protocol
        ensure(sha(OLD/'protocol.json')==p['old_protocol_sha256'] and sha(OLD/'COMPLETE.json')==p['old_complete_sha256'],'Accepted old teacher changed');self.old=NativeReader(OLD);self.h=HReader(SOURCE,verify=False)
        if require_complete:ensure(read(self.work/'MIXED_COMPLETE.json')['samples']==284218,'Teacher full mixed coverage incomplete')
    def load(self,sid,device='cpu'):
        ensure(sid in self.h.ids,'Outside current 500h');v=self.new.load(sid,device) if sid in self.new.ids else self.old.load(sid,device)
        from p2_listen.layout import make_layout
        core.validate_geometry(v,make_layout(self.h.row(sid)['duration_s'],(8,8),8,8),2048);return v


def prepare_media(item,override,size,grid,reference=False):
    # Native processor/timeline and selected-frame functions; current admission
    # replaces only the old train-only/crop-origin guard, not media operations.
    import numpy as np
    from p2_listen.layout import make_layout
    from p2_j_teacher_accel import selected_frames
    row=item['_row'];source=item['source'];ensure(item['split'] in ('train','dev') and row['quality']['decision']=='retain' and source['video_transform']['output_size']==[512,512],'Not accepted current member')
    frames,pts=selected_frames(item,override,size,reference=reference,threads=4)
    wave,identity=media.listener_wave(item);processor=media._processor
    vi=dict(processor.video_processor(videos=[frames],return_tensors='pt',do_sample_frames=False,do_resize=False))
    ai=dict(processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt',return_attention_mask=True,padding='max_length'))
    layout=make_layout(source['duration_s'],grid,8,8)
    return dict(sample_id=item['sample_id'],video=vi,audio=ai,layout=layout,plan=item['_plan'],
                audit=dict(final_crop=source['video_transform']['crop'],crop_revision=item['crop_provenance']['revision'],listener_audio_identity=identity,teacher_frame_indices_25fps=media.frame_indices(source['duration_s']),source_pts=pts,teacher_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),audio_16k_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),duration_s=source['duration_s'],split=item['split'],source_video_used=True,source_sha256=digest(source)))


def extract_batch(runtime,prepared):
    import torch
    from p2_listen.cache import cpu
    full=[p for p in prepared if p['plan']['teacher']!='reuse_audio'];values={p['sample_id']:v for p,v in zip(full,core.extract_batch(runtime,full))} if full else {}
    model=runtime.thinker;device=next(model.parameters()).device
    for p in prepared:
        if p['plan']['teacher']!='reuse_audio':continue
        old=ACTIVE.old_value(p['plan'])
        ensure(old['audit']['audio_16k_sha256']==p['audit']['audio_16k_sha256'],'Reused teacher audio waveform differs')
        with torch.inference_mode():v=model.get_video_features(p['video']['pixel_values_videos'].to(device),p['video']['video_grid_thw'].to(device))
        value=dict(video=v[None],audio=old['audio'],video_valid=p['layout']['video_target_valid'].to(device),audio_valid=old['audio_valid'],audit=p['audit']);core.validate_geometry(value,p['layout'],2048);values[p['sample_id']]=cpu(value)
    return [values[p['sample_id']] for p in prepared]


def canary(work):
    import torch
    r=IncrementReader(work);torch.set_num_threads(2);torch.cuda.set_device(0);runtime,processor,size,grid=core.load_model(r);media.initialize_processor(r.protocol['model']);rows=[]
    for sid in r.protocol['canary_ids']:
        item=r.item(sid);p=prepare_media(item,None,size,grid);ref=prepare_media(item,None,size,grid,True)
        ensure(p['audit']['teacher_rgb_sha256']==ref['audit']['teacher_rgb_sha256'] and p['audit']['source_pts']==ref['audit']['source_pts'],'Accelerated teacher pixels/PTS differ from complete native decode')
        v=extract_batch(runtime,[p])[0];baseline=core.extract_batch(runtime,[ref])[0]
        ensure(all(torch.equal(v[k],baseline[k]) for k in ('video','audio','video_valid','audio_valid')),'Incremental teacher differs from full native frozen towers')
        core.persist(r,item,v);r.load(sid);rows.append(dict(sample_id=sid,split=item['split'],teacher_disposition=item['_plan']['teacher'],native_exact=True))
        atomic_json(r.work/'canary_progress.json',dict(samples=len(rows),total=len(r.protocol['canary_ids']),time=time.time()))
    atomic_json(r.work/'CANARY.json',dict(success=True,protocol_sha256=r.protocol_sha,count=len(rows),rows=rows,time=time.time()));print(json.dumps(dict(canary_passed=True,count=len(rows))),flush=True)


def worker(work,wid):
    global ACTIVE
    r=IncrementReader(work);ensure(read(r.work/'CANARY.json')['protocol_sha256']==r.protocol_sha,'Matching teacher canary required')
    runner.Reader=IncrementReader;runner.prepare=prepare_media;runner.extract_batch=extract_batch
    runner.worker(work,wid,batch=8,producers=4,prefetch=16)


def node(work,name):
    folder=Path(work)/'nodes'/name;folder.mkdir(parents=True,exist_ok=True);c=read(Path(work)/'cluster.json');node_id=next(i for i,n in enumerate(c['nodes']) if n['name']==name);children={};logs=[];started=set()
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                for gpu in free_gpus():
                    if gpu in started:continue
                    for replica in range(2):
                        wid=node_id*100+gpu*2+replica;cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',work,'--worker-id',str(wid)]
                        log=(folder/f'g{gpu}-r{replica}.log').open('a');logs.append(log);children[wid]=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        atomic_json(folder/f'g{gpu}-r{replica}.pid.json',dict(pid=children[wid].pid,gpu=gpu,replica=replica,worker_id=wid))
                    started.add(gpu)
                states={str(k):p.poll() for k,p in children.items()};atomic_json(folder/'status.json',dict(time=time.time(),states=states,gpus_started=sorted(started),node=name))
                snap=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=snap.stdout))+'\n')
                if len(started)==8 and all(v is not None for v in states.values()):break
                time.sleep(5)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
        finally:
            for p in children.values():
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for log in logs:log.close()


@lru_cache(maxsize=1)
def audit_reader(work):return IncrementReader(work,verify=False)


def audit_block(args):
    work,ids=args;r=audit_reader(work);total=0
    for sid in ids:v=r.load(sid);total+=(r.work/'cache'/sid/'payload.pt').stat().st_size
    return dict(samples=len(ids),bytes=total)


def finalize(work):
    r=IncrementReader(work);ensure(not list((r.work/'failed').glob('*.json')),'Unresolved teacher failures');jobs=read(r.work/'jobs.json');seen=set()
    for i,ids in enumerate(jobs):
        d=read(r.work/'done'/f'{i:06d}.json');ensure(d['success'] and d['sample_ids']==ids and d['protocol_sha256']==r.protocol_sha and not seen&set(ids),'Wrong teacher receipt coverage');seen.update(ids)
        ensure(len(d['receipts'])==len(ids) and {x['sample_id'] for x in d['receipts']}==set(ids),'Wrong teacher receipt members')
        for v in d['receipts']:
            m=read(r.work/'cache'/v['sample_id']/'complete.json');ensure(m['identity']==v['identity'] and m['payload_sha256']==v['payload_sha256'],'Teacher receipt/hash differs')
    ensure(seen==r.ids and len(seen)==7438,'Teacher increment incomplete');n=total=0
    with ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for x in pool.map(audit_block,[(work,r.protocol['sample_ids'][i:i+128]) for i in range(0,7438,128)]):n+=x['samples'];total+=x['bytes']
    ensure(n==7438,'Teacher payload coverage incomplete');result=dict(success=True,samples=7438,video_encoded=7438,audio_encoded=6542,reuse_audio_only=896,payload_bytes=total,protocol_sha256=r.protocol_sha,time=time.time())
    atomic_json(r.work/'COMPLETE.json',result,immutable=True)
    full=dict(success=True,samples=284218,reused_both=276780,new_payloads=7438,reuse_audio_only=896,video_encoded=7438,audio_encoded=6542,source_split_counts=r.protocol['source_split_counts'],protocol_sha256=r.protocol_sha,selection_sha256=r.protocol['selection_sha256'],increment_complete_sha256=sha(r.work/'COMPLETE.json'),old_complete_sha256=r.protocol['old_complete_sha256'],time=time.time(),training_started=False)
    atomic_json(r.work/'MIXED_COMPLETE.json',full,immutable=True);return full


def launch(c,n,work,role,gpu=None):
    code='''import sys,subprocess,json,os,time
from pathlib import Path
root,work,node,script,role,gpu=sys.argv[1:];f=Path(work)/'nodes'/node;f.mkdir(parents=True,exist_ok=True);pf=f/(role+'_launch.json')
if pf.exists():
 d=json.loads(pf.read_text());p=Path('/proc')/str(d['pid'])/'cmdline'
 if p.exists() and script.encode() in p.read_bytes() and work.encode() in p.read_bytes():print(json.dumps(dict(already_running=True,pid=d['pid'])));sys.exit()
cmd=[sys.executable,script,role,'--work',work,'--node',node];e=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True')
if gpu!='none':e['CUDA_VISIBLE_DEVICES']=gpu
with (f/(role+'.log')).open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,command=cmd,time=time.time());pf.write_text(json.dumps(d));print(json.dumps(d))'''
    return json.loads(remote(c,n,[c['python'],'-c',code,str(ROOT),str(work),n['name'],str(Path(__file__).resolve()),role,str(gpu) if gpu is not None else 'none']))


def chain(work):
    w=Path(work);r=IncrementReader(work);c=read(w/'cluster.json')
    def state(stage,**kw):atomic_json(w/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (w/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (w/'MIXED_COMPLETE.json').exists():state('complete');return
            while not (WHISPER/'MIXED_COMPLETE.json').exists():state('waiting_whisper_increment');time.sleep(10)
            if not (w/'CANARY.json').exists():
                selected=None
                while selected is None:
                    for n in c['nodes']:
                        free=json.loads(remote(c,n,[c['python'],str(Path(__file__).resolve()),'free']))
                        if free:selected=(n,free[0]);break
                    if selected is None:state('waiting_first_free_gpu');time.sleep(5)
                state('gpu_canary',node=selected[0]['name'],gpu=selected[1]);launch(c,selected[0],w,'canary',selected[1])
                while not (w/'CANARY.json').exists():
                    if (w/'CANARY_FAILED.json').exists():raise RuntimeError('Teacher GPU canary failed')
                    time.sleep(5)
                while not (w/'CANARY_EXIT.json').exists():time.sleep(2)
            ensure(read(w/'CANARY.json')['protocol_sha256']==r.protocol_sha,'Wrong teacher canary')
            with ThreadPoolExecutor(max_workers=4) as pool:launches=list(pool.map(lambda n:launch(c,n,w,'node'),c['nodes']))
            state('extracting',launches=launches)
            while True:
                s=runner.status(w);atomic_json(w/'cluster_status.json',s);state('extracting',new_samples=s['samples'],new_total=7438,total=284218,failures=s['failures'])
                if s['samples']==7438:state('auditing');full=finalize(str(w));state('complete',result=full);return
                if all((w/'nodes'/n['name']/'EXIT.json').exists() for n in c['nodes']):raise RuntimeError('Teacher nodes exited before complete coverage')
                time.sleep(15)
        except BaseException:state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','canary','worker','node','chain','free']);p.add_argument('--work',default=str(WORK));p.add_argument('--worker-id',type=int,default=0);p.add_argument('--node',default='node12');a=p.parse_args()
    if a.command=='prepare':prepare(a.work)
    elif a.command=='worker':worker(a.work,a.worker_id)
    elif a.command=='node':node(a.work,a.node)
    elif a.command=='chain':chain(a.work)
    elif a.command=='free':print(json.dumps(free_gpus()))
    elif a.command=='canary':
        try:canary(a.work)
        except BaseException:atomic_json(Path(a.work)/'CANARY_FAILED.json',dict(error=traceback.format_exc(),time=time.time()));raise
        finally:atomic_json(Path(a.work)/'CANARY_EXIT.json',dict(time=time.time()))

if __name__=='__main__':main()
