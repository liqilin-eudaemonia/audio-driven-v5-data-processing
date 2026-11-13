#!/usr/bin/env python3
"""Incremental AV VAE: preserve 276780 old payloads, encode 7438 video / 6542 audio.

Use native encode_vae for new AV. Video-only uses the identical native video
operations and keeps the checked old audio tensors. Per-card resource handoff.
"""
import argparse
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,wait,FIRST_COMPLETED
from functools import lru_cache
import fcntl,hashlib,json,multiprocessing as mp,os,signal,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,digest,atomic_json
from p2_listen.config import activate,extractor_config
from p2_j_current_qwen_hidden import Reader as HReader,ensure,claim,env,remote,item_from_row
SOURCE=ROOT/'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
TEXT=ROOT/'experiments/p2_j_train_dev500h_gemma_cache_20261105'
REUSE=ROOT/'experiments/p2_j_train_dev500h_vae_reuse_audit_20261105'
OLD=ROOT/'experiments/p2_j_train500h_cache_20261101'
DEFAULT_WORK=ROOT/'experiments/p2_j_train_dev500h_av_cache_20261105'


def code_identity():
    paths=[Path(__file__),ROOT/'scripts/p2_j_current_qwen_hidden.py',ROOT/'scripts/p2_j_current_vae_reuse_audit.py',ROOT/'src/p2_j_cache/data.py',ROOT/'src/p2_j_cache/runtime.py',ROOT/'src/p2_listen/runtime.py',ROOT/'src/p2_listen/layout.py',ROOT/'src/listen_benchmark/j_omni.py',ROOT/'src/listen_benchmark/frame_crop.py',ROOT/'src/listen_benchmark/official_audio_gate.py']
    return {str(p):sha(p) for p in paths}


def prepare(work):
    from p2_j_cache.runtime import CacheReader
    from p2_listen.features import weights_identity
    w=Path(work);w.mkdir(parents=True,exist_ok=True);h=HReader(SOURCE);old=CacheReader(OLD);s=read(REUSE/'SUMMARY.json');plan=read(REUSE/'CACHE_PLAN.json')
    ensure(s['all_common_payloads_sha256_verified'] and s['video_encoding_needed']==7438 and s['audio_encoding_needed']==6542,'Unverified reuse plan')
    ensure(sha(REUSE/'cache_plan.jsonl')==plan['cache_plan_sha256'],'Reuse plan changed')
    tasks=[];index=[];partial=[];new=[];count=Counter()
    for line in (REUSE/'cache_plan.jsonl').open():
        r=json.loads(line);sid=r['sample_id'];index.append((sid,line.strip(),r['disposition']));count[r['disposition']]+=1
        if r['disposition']!='reuse_both':
            row=h.row(sid);tasks.append((row['duration_s'],sid))
            (partial if r['disposition']=='reuse_audio_reencode_video' else new).append((row['duration_s'],sid,row['split'],r['restored']))
    ensure(len(index)==284218 and len(tasks)==7438,'Wrong AV mapping')
    dbpath=w/'mapping.sqlite'
    if not dbpath.exists():
        tmp=w/'mapping.sqlite.tmp';db=sqlite3.connect(tmp);db.execute('CREATE TABLE samples(sid TEXT PRIMARY KEY,record TEXT,disposition TEXT)');db.executemany('INSERT INTO samples VALUES(?,?,?)',index);db.commit();db.close();tmp.replace(dbpath)
    canary=[]
    for group in [partial,new,[x for x in new if x[2]=='dev'],[x for x in new if x[3]]]:
        for x in [min(group),max(group)]:
            if x[1] not in canary:canary.append(x[1])
    for x in sorted(tasks,reverse=True):
        if len(canary)>=8:break
        if x[1] not in canary:canary.append(x[1])
    ids=[sid for _,sid in sorted(tasks,reverse=True)]
    p=dict(schema='current-j-incremental-native-av-v1',source=str(SOURCE),source_protocol_sha256=sha(SOURCE/'protocol.json'),source_complete_sha256=sha(SOURCE/'COMPLETE.json'),text_work=str(TEXT),text_protocol_sha256=sha(TEXT/'protocol.json'),
           reuse=str(REUSE),reuse_plan_sha256=sha(REUSE/'cache_plan.jsonl'),reuse_summary_sha256=sha(REUSE/'SUMMARY.json'),old=str(OLD),old_manifest_sha256=sha(OLD/'manifest.json'),old_complete_sha256=sha(OLD/'COMPLETE.json'),mapping_sha256=sha(dbpath),
           sample_ids=h.protocol['sample_ids'],samples=284218,source_split_counts=h.protocol['source_split_counts'],selection_sha256=h.protocol['selection_sha256'],config=h.config,
           dispositions=dict(count),new_samples=7438,video_encoding_needed=7438,audio_encoding_needed=6542,canary_ids=canary,
           assets=dict(ltx=weights_identity(h.config,'ltx')),code=code_identity(),geometry=dict(fps=25,height=512,width=512),
           video_policy='exact native nearest-source-PTS final crop; repeat final frame to 8n+1; native BF16 encoder and first-frame replacement',audio_policy='exact existing listener official union recipe; native FP32 encoder; BF16 storage; original endpoint and support masks',
           old_payloads_read_only=True,copy_old_payloads=False,only_av_cache=True)
    ensure(p['assets']==old.assets['vae'],'VAE model assets differ')
    atomic_json(w/'protocol.json',p,immutable=True);atomic_json(w/'jobs.json',[ids[i:i+16] for i in range(0,len(ids),16)],immutable=True);atomic_json(w/'cluster.json',read(SOURCE/'cluster.json'),immutable=True)
    print(json.dumps(dict(prepared=True,total=284218,new=7438,video=7438,audio=6542,canary_ids=canary)),flush=True)


class Reader:
    def __init__(self,work,verify=True):
        self.work=Path(work);self.protocol=read(self.work/'protocol.json');p=self.protocol;self.protocol_sha=sha(self.work/'protocol.json');self.config=p['config'];activate(self.config)
        ensure(p['code']==code_identity(),'AV code changed; use a new run revision')
        ensure(sha(SOURCE/'protocol.json')==p['source_protocol_sha256'] and sha(SOURCE/'COMPLETE.json')==p['source_complete_sha256'],'Frozen H source changed')
        if verify:
            from p2_listen.features import weights_identity
            ensure(dict(ltx=weights_identity(self.config,'ltx'))==p['assets'],'VAE weights changed')
            ensure(sha(self.work/'mapping.sqlite')==p['mapping_sha256'] and sha(REUSE/'cache_plan.jsonl')==p['reuse_plan_sha256'],'Reuse mapping changed')
            ensure(sha(OLD/'manifest.json')==p['old_manifest_sha256'] and sha(OLD/'COMPLETE.json')==p['old_complete_sha256'],'Audited old source changed')
        self.source=HReader(SOURCE,verify=False);self.conn=sqlite3.connect(f'file:{self.work}/mapping.sqlite?mode=ro',uri=True)
    def plan(self,sid):
        row=self.conn.execute('SELECT record FROM samples WHERE sid=?',(sid,)).fetchone();ensure(row is not None,'Outside AV selection');return json.loads(row[0])
    def identity(self,row):
        return dict(schema=self.protocol['schema'],protocol_sha256=self.protocol_sha,part='vae',sample_id=row['sample_id'],split=row['split'],source_sha256=digest(row['source']),audio_recipe_sha256=digest(row['audio_recipe']))
    def old_value(self,plan,device='cpu'):
        from p2_listen.cache import load_part
        folder=Path(plan['old_payload']).parent;m=read(folder/'complete.json');ensure(m['identity']==plan['old_identity'] and m['payload_sha256']==plan['payload_sha256'],'Old reuse payload binding changed')
        return load_part(folder,m['contract'],device)
    def load(self,sid,device='cpu',missing_ok=False):
        from p2_listen.cache import load_part
        row=self.source.row(sid);plan=self.plan(sid);ensure(digest(row['source'])==plan['new_source_sha256'],'Current media/reuse plan differs')
        if plan['disposition']=='reuse_both':v=self.old_value(plan,device)
        else:v=load_part(self.work/'cache/vae'/sid,self.identity(row),device,missing_ok)
        if v is not None:validate(row,v)
        return v
    def ids(self,source_split=None):
        ids=list(self.protocol['sample_ids'])
        return ids if source_split is None else [sid for sid in ids if self.source.row(sid)['split']==source_split]
    def training_sample(self,sid,variant='guided',device='cpu'):
        from p2_j_current_gemma_cache import Reader as TextReader
        from omni_av_p2.connector import ConditionInputs
        ensure(variant in ('guided','direct'),'Unknown conditioning variant')
        if not hasattr(self,'text'):self.text=TextReader(TEXT)
        q=self.source.load(sid,'input_plan' if variant=='guided' else 'input',device);text=self.text.fixed(device)
        if variant=='guided':
            t=self.text.load(sid,device);ensure(q['prompt']==t['positive']['prompt'],'Qwen/Gemma prompt mismatch');text=dict(text,positive=t['positive'])
        row=self.source.row(sid);return dict(condition=ConditionInputs(**q['condition']),layout=q['layout'],text=text,**self.load(sid,device),metadata=dict(sample_id=sid,split=row['split'],training_admitted=row['training_admitted'],variant=variant))


def validate(row,v):
    import torch
    from p2_listen.layout import geometry
    g=geometry(dict(requested_duration_s=row['duration_s'],geometry=dict(fps=25,height=512,width=512)))
    ensure(v['geometry']==g,'AV real/padded duration or geometry changed')
    z=v['av']['latents']['latents'];a=v['av']['audio_latents']['latents'];ensure(tuple(z.shape)==(1,128,(g['frames']-1)//8+1,16,16),'Wrong video latent shape')
    ensure(a.ndim==4 and a.shape[0]==1 and a.shape[1]==8 and a.shape[3]==16,'Wrong audio latent shape')
    ensure(z.dtype==a.dtype==torch.bfloat16 and torch.isfinite(z).all() and torch.isfinite(a).all(),'Invalid AV latent values')
    ensure(v['support']['video'].shape==(1,z.shape[2]*16*16) and v['support']['audio'].shape==(1,a.shape[2]),'AV support mask shape differs')
    ensure(v['support']['video'].dtype==v['support']['audio'].dtype==torch.bool,'Wrong AV mask dtype')
    from p2_listen.runtime import support_masks
    masks=support_masks(g,z.shape,a.shape,v['audio_support_margin_s'])
    ensure(all(torch.equal(masks[k].to(v['support'][k].device),v['support'][k]) for k in masks),'Changed causal support policy')


def initialize_cpu(work):
    global CPU_READER
    import torch
    torch.set_num_threads(1);CPU_READER=Reader(work,verify=False)


def prepare_media(sid):
    import numpy as np
    from PIL import Image
    from p2_j_cache.data import decode_frames
    from p2_listen.layout import geometry
    from listen_benchmark.j_omni import read_audio
    r=CPU_READER;row=r.source.row(sid);item=item_from_row(row);plan=r.plan(sid)
    cap=read(row['caption_file']);wave,sr,identity=read_audio(row['audio_recipe']);ensure(identity['processed_waveform_sha256']==cap['input_evidence']['audio_identity']['processed_waveform_sha256'],'Listener waveform differs from accepted caption input')
    override=plan.get('reviewed_pts_override');pixels,pts=decode_frames(row['source'],gap_overrides=override['frames'] if override else None)
    first=hashlib.sha256(pixels[0].tobytes()).hexdigest();h_ref=SOURCE/'references'/(sid+'.png')
    if not h_ref.exists():h_ref=OLD/'references'/(sid+'.png')
    with Image.open(h_ref) as im:ensure(hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()==first,'AV first frame differs from Qwen reference')
    ref=r.work/'references'/(sid+'.png');ref.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels[0]).save(ref)
    obs=dict(sample_id=sid,listener_image=str(ref),requested_duration_s=row['duration_s'],geometry=dict(fps=25,height=512,width=512));g=geometry(obs)
    audit=dict(sample_id=sid,source_sha256=digest(row['source']),audio_recipe_sha256=digest(row['audio_recipe']),processed_waveform_sha256=identity['processed_waveform_sha256'],source_frame_pts_s=pts,first_rgb_sha256=first,rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),frames=len(pixels),padded_frames=g['frames'],split=row['split'],restored='orientation_restoration' in row)
    pad=g['frames']-len(pixels);ensure(0<=pad<=7,'Wrong video padding')
    if pad:pixels=np.concatenate([pixels,np.repeat(pixels[-1:],pad,axis=0)])
    return dict(row=row,plan=plan,obs=obs,pixels=pixels,wave=wave,rate=sr,audit=audit)


def load_session(reader):
    import torch
    from ltx_trainer.model_loader import load_video_vae_encoder,load_audio_vae_encoder
    from ltx_core.model.audio_vae import AudioProcessor
    cfg=extractor_config(reader.config)
    video=load_video_vae_encoder(cfg.ltx_path,device='cuda:0',dtype=torch.bfloat16).requires_grad_(False).eval()
    audio=load_audio_vae_encoder(cfg.ltx_path,device='cuda:0',dtype=torch.float32).requires_grad_(False).eval()
    ap=AudioProcessor(target_sample_rate=audio.sample_rate,mel_bins=audio.mel_bins,mel_hop_length=audio.mel_hop_length,n_fft=audio.n_fft).to('cuda:0')
    return dict(video=video,audio=audio,processor=ap)


def encode(reader,session,p):
    import torch
    from p2_listen.cache import cpu
    from p2_listen.layout import geometry
    from p2_listen.runtime import encode_vae,support_masks
    if p['plan']['disposition']=='reuse_audio_reencode_video':
        # Exactly the native encode_vae video path; audio is loaded, never encoded.
        from PIL import Image
        import numpy as np
        old=reader.old_value(p['plan']);g=geometry(p['obs']);ensure(old['geometry']==g,'Cached audio padding/time differs')
        with Image.open(p['obs']['listener_image']) as im:first=np.asarray(im.convert('RGB')).copy()
        with torch.no_grad():
            value=torch.from_numpy(p['pixels']).permute(3,0,1,2)[None].to('cuda:0',torch.bfloat16)/127.5-1
            ref=torch.from_numpy(first).permute(2,0,1)[None,:,None].to('cuda:0',torch.bfloat16)/127.5-1
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=session['video'](value);z[:,:,:1]=session['video'](ref)
        scalar=lambda x:torch.tensor([x]);v=dict(old);v['av']=dict(old['av'],latents=dict(latents=z,num_frames=scalar(z.shape[2]),height=scalar(z.shape[3]),width=scalar(z.shape[4]),fps=scalar(g['fps'])))
        v['support']=dict(old['support'],video=support_masks(g,z.shape,old['av']['audio_latents']['latents'].shape,old['audio_support_margin_s'])['video']);v=cpu(v)
        ensure(torch.equal(v['av']['audio_latents']['latents'],old['av']['audio_latents']['latents']) and torch.equal(v['support']['audio'],old['support']['audio']),'Reused audio changed')
    else:v=encode_vae(reader.config,p['obs'],None,'cuda:0',pixels=p['pixels'],waveform=(p['wave'],p['rate']),encoders=session)
    validate(p['row'],v);v['input_audit']=p['audit'];return v


def write_value(reader,p,value):
    from p2_j_cache.runtime import persist
    sid=p['row']['sample_id'];persist(reader.work,'vae',sid,reader.identity(p['row']),value);f=reader.work/'cache/vae'/sid;m=read(f/'complete.json')
    atomic_json(reader.work/'media_audit'/(sid+'.json'),p['audit'],immutable=True)
    return dict(sample_id=sid,split=p['row']['split'],video_encoded=True,audio_encoded=p['plan']['disposition']!='reuse_audio_reencode_video',identity=m['identity'],payload_sha256=m['payload_sha256'],bytes=(f/'payload.pt').stat().st_size)


def same(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    return a==b


def canary(work):
    import torch
    from p2_listen.runtime import encode_vae
    r=Reader(work);initialize_cpu(work);torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();session=load_session(r);results=[]
    for i,sid in enumerate(r.protocol['canary_ids']):
        p=prepare_media(sid);v=encode(r,session,p);exact=None
        if p['plan']['disposition']=='reuse_audio_reencode_video' or i<2:
            baseline=encode_vae(r.config,p['obs'],None,'cuda:0',pixels=p['pixels'],waveform=(p['wave'],p['rate']),encoders=session)
            exact=all(same(v[k],baseline[k]) for k in baseline);ensure(exact,'Incremental VAE differs from full native AV encoding')
        write_value(r,p,v);r.load(sid);results.append(dict(sample_id=sid,split=p['row']['split'],disposition=p['plan']['disposition'],native_av_exact=exact,frames=p['audit']['frames'],restored=p['audit']['restored']))
        atomic_json(r.work/'canary_progress.json',dict(samples=len(results),total=len(r.protocol['canary_ids']),time=time.time()))
    peak=torch.cuda.max_memory_reserved()/2**30
    result=dict(success=True,protocol_sha256=r.protocol_sha,rows=results,count=len(results),peak_reserved_gib=peak,production_workers_per_gpu=2 if peak<62 else 1,time=time.time());atomic_json(r.work/'CANARY.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def stream(ids,pool,prefetch):
    iterator=iter(ids);pending={}
    def fill():
        while len(pending)<prefetch:
            sid=next(iterator,None)
            if sid is None:break
            pending[pool.submit(prepare_media,sid)]=sid
    fill()
    while pending:
        ready,_=wait(pending,timeout=300,return_when=FIRST_COMPLETED)
        if not ready:raise TimeoutError('AV media preparation timed out')
        for f in ready:pending.pop(f);p=f.result();fill();yield p


def worker(work,wid,producers=4):
    import torch
    r=Reader(work);proof=read(r.work/'CANARY.json');ensure(proof['success'] and proof['protocol_sha256']==r.protocol_sha,'Matching AV canary required');jobs=read(r.work/'jobs.json')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();session=None;count=0;start=time.time()
    def state(status,**kw):atomic_json(r.work/'workers'/(wid+'.json'),dict(worker=wid,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),status=status,samples=count,time=time.time(),started_at=start,**kw))
    pool=ProcessPoolExecutor(max_workers=producers,mp_context=mp.get_context('spawn'),initializer=initialize_cpu,initargs=(work,));writers=ThreadPoolExecutor(max_workers=2);state('starting')
    try:
        while True:
            task=claim(r.work,jobs)
            if task is None:break
            name,ids,lock=task;pending=deque();receipts=[];tick=time.time();state('preparing',job=name)
            try:
                for p in stream(ids,pool,4):
                    sid=p['row']['sample_id'];v=r.load(sid,missing_ok=True)
                    if v is None:
                        if session is None:state('loading_model',job=name);session=load_session(r)
                        v=encode(r,session,p);count+=1
                    pending.append(writers.submit(write_value,r,p,v))
                    if len(pending)>=2:receipts.append(pending.popleft().result())
                    state('encoding',job=name,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                while pending:receipts.append(pending.popleft().result())
                ensure(len(receipts)==len(ids) and {x['sample_id'] for x in receipts}==set(ids),'Incomplete AV task')
                atomic_json(r.work/'done'/(name+'.json'),dict(success=True,samples=len(ids),sample_ids=ids,receipts=receipts,worker=wid,seconds=time.time()-tick,protocol_sha256=r.protocol_sha));state('running',job=name)
            except Exception:
                while pending:
                    try:pending.popleft().result()
                    except Exception:pass
                atomic_json(r.work/'failed'/(name+'.json'),dict(error=traceback.format_exc(),sample_ids=ids,worker=wid,time=time.time()));state('block_failed',job=name,error=traceback.format_exc())
            finally:lock.close()
        state('finished')
    except BaseException:state('failed',error=traceback.format_exc());raise
    finally:
        for p in list((getattr(pool,'_processes',None) or {}).values()):
            if p.is_alive():p.terminate()
        pool.shutdown(wait=False,cancel_futures=True);writers.shutdown(wait=True)


def free_gpus():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines()};return [int(v[0]) for line in out.splitlines() if (v:=[z.strip() for z in line.split(',')]) and v[1] not in busy and float(v[2])>=110*1024]


def node(work,name):
    folder=Path(work)/'nodes'/name;folder.mkdir(parents=True,exist_ok=True);children={};logs=[];started=set();per_gpu=read(Path(work)/'CANARY.json')['production_workers_per_gpu']
    with (folder/'node.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                if not (Path(work)/'COMPLETE.json').exists():
                    for gpu in free_gpus():
                        if gpu in started:continue
                        for replica in range(per_gpu):
                            wid=f'{name}-g{gpu}-r{replica}';cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--work',work,'--worker-id',wid]
                            log=(folder/(wid+'.log')).open('a');logs.append(log);children[wid]=subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                            atomic_json(folder/(wid+'.pid.json'),dict(pid=children[wid].pid,gpu=gpu,replica=replica))
                        started.add(gpu)
                states={k:p.poll() for k,p in children.items()};atomic_json(folder/'status.json',dict(time=time.time(),node=name,states=states,gpus_started=sorted(started),per_gpu=per_gpu,pid=os.getpid()))
                snap=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
                with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu_csv=snap.stdout))+'\n')
                if len(started)==8 and all(v is not None for v in states.values()):break
                if (Path(work)/'COMPLETE.json').exists() and all(v is not None for v in states.values()):break
                time.sleep(10)
            atomic_json(folder/'EXIT.json',dict(time=time.time(),states=states))
        finally:
            for p in children.values():
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for log in logs:log.close()


def status(work):
    w=Path(work);p=read(w/'protocol.json');rows=[read(f) for f in (w/'done').glob('*.json')];done=sum(x['samples'] for x in rows)
    return dict(time=time.time(),new_samples=done,new_total=7438,full_ready=276780+done,total=284218,finished_jobs=len(rows),video_encoded=done,audio_encoded=sum(v['audio_encoded'] for x in rows for v in x['receipts']),failures=[str(f) for f in (w/'failed').glob('*.json')],workers=[read(f) for f in (w/'workers').glob('*.json')])


@lru_cache(maxsize=1)
def audit_reader(work):return Reader(work,verify=False)


def audit_block(args):
    work,ids=args;r=audit_reader(work);counts=Counter();bytes_new=bytes_old=0
    for sid in ids:
        row=r.source.row(sid);plan=r.plan(sid);ensure(plan['new_source_sha256']==digest(row['source']),'Stale AV mapping');counts[row['split']]+=1;counts[plan['disposition']]+=1
        if plan['disposition']=='reuse_both':
            f=Path(plan['old_payload']);m=read(f.parent/'complete.json');ensure(m['identity']==plan['old_identity'] and m['payload_sha256']==plan['payload_sha256'] and f.stat().st_size==plan['payload_bytes'],'Old reused VAE changed');bytes_old+=f.stat().st_size
        else:
            folder=r.work/'cache/vae'/sid;m=read(folder/'complete.json');expected=r.identity(row);ensure(m['contract']==expected and m['identity']==digest(expected),'New AV contract mismatch');ensure((folder/'payload.pt').stat().st_size>0,'Missing AV payload');bytes_new+=(folder/'payload.pt').stat().st_size
            a=read(r.work/'media_audit'/(sid+'.json'));ensure(a['source_sha256']==digest(row['source']) and a['audio_recipe_sha256']==digest(row['audio_recipe']),'New media audit mismatch')
    return dict(samples=len(ids),counts=dict(counts),bytes_new=bytes_new,bytes_old=bytes_old)


def audit(work):
    r=Reader(work);ensure(not list((r.work/'failed').glob('*.json')),'Unresolved AV failures');jobs=read(r.work/'jobs.json');seen=set();video=audio=0
    for i,ids in enumerate(jobs):
        d=read(r.work/'done'/f'{i:06d}.json');ensure(d['success'] and d['sample_ids']==ids and d['protocol_sha256']==r.protocol_sha and not seen&set(ids),'AV receipt membership mismatch');seen.update(ids)
        ensure(len(d['receipts'])==len(ids) and {x['sample_id'] for x in d['receipts']}==set(ids),'Incomplete AV receipt')
        for v in d['receipts']:
            row=r.source.row(v['sample_id']);m=read(r.work/'cache/vae'/row['sample_id']/'complete.json');ensure(m['identity']==v['identity'] and m['payload_sha256']==v['payload_sha256'],'AV writer receipt mismatch');video+=v['video_encoded'];audio+=v['audio_encoded']
    ensure(len(seen)==video==7438 and audio==6542,'Incremental video/audio coverage differs')
    start=time.time();ids=r.protocol['sample_ids'];counts=Counter();total=0;bn=bo=0
    with ProcessPoolExecutor(max_workers=32,mp_context=mp.get_context('fork')) as pool:
        for x in pool.map(audit_block,[(work,ids[i:i+256]) for i in range(0,len(ids),256)],chunksize=1):total+=x['samples'];counts.update(x['counts']);bn+=x['bytes_new'];bo+=x['bytes_old']
    ensure(total==284218 and {k:counts[k] for k in ('train','dev')}==r.protocol['source_split_counts'],'Full AV train+dev coverage mismatch')
    result=dict(success=True,samples=284218,source_split_counts=r.protocol['source_split_counts'],reuse_both=276780,reuse_audio_only=896,video_encoded=7438,audio_encoded=6542,new_payload_bytes=bn,old_payload_bytes_referenced=bo,protocol_sha256=r.protocol_sha,selection_sha256=r.protocol['selection_sha256'],parallel_audit_seconds=time.time()-start,time=time.time(),training_started=False)
    atomic_json(r.work/'COMPLETE.json',result,immutable=True);return result


def launch(cluster,n,work,role,gpu=None):
    code='''import sys,json,subprocess,os,time
from pathlib import Path
root,work,node,script,role,gpu=sys.argv[1:];folder=Path(work)/'nodes'/node;folder.mkdir(parents=True,exist_ok=True);pidfile=folder/(role+'_launch.json')
if pidfile.exists():
 old=json.loads(pidfile.read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
 if p.exists() and script.encode() in p.read_bytes() and work.encode() in p.read_bytes():print(json.dumps(dict(already_running=True,pid=old['pid'])));sys.exit()
cmd=[sys.executable,script,role,'--work',work,'--node',node]
e=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
if gpu!='none':e['CUDA_VISIBLE_DEVICES']=gpu
with (folder/(role+'.log')).open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,command=cmd,time=time.time());pidfile.write_text(json.dumps(d));print(json.dumps(d))'''
    return json.loads(remote(cluster,n,[cluster['python'],'-c',code,str(ROOT),str(work),n['name'],str(Path(__file__).resolve()),role,str(gpu) if gpu is not None else 'none']))


def chain(work):
    work=Path(work);r=Reader(work);cluster=read(work/'cluster.json')
    def state(stage,**kw):atomic_json(work/'chain_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    with (work/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (work/'COMPLETE.json').exists():state('complete');return
            if not (work/'CANARY.json').exists():
                selected=None
                while selected is None:
                    for n in cluster['nodes']:
                        free=json.loads(remote(cluster,n,[cluster['python'],str(Path(__file__).resolve()),'free']))
                        if free:selected=(n,free[0]);break
                    if selected is None:state('waiting_first_free_gpu');time.sleep(10)
                state('gpu_canary',node=selected[0]['name'],gpu=selected[1]);launch(cluster,selected[0],work,'canary',selected[1])
                while not (work/'CANARY.json').exists():
                    if (work/'CANARY_FAILED.json').exists():raise RuntimeError('AV canary failed; see canary node log')
                    time.sleep(5)
                # Wait for validation process to exit and release its CUDA context.
                while not (work/'CANARY_EXIT.json').exists():time.sleep(2)
            proof=read(work/'CANARY.json');ensure(proof['success'] and proof['protocol_sha256']==r.protocol_sha,'Wrong AV canary')
            if not (work/'production_started.json').exists():atomic_json(work/'production_started.json',dict(time=time.time(),protocol_sha256=r.protocol_sha),immutable=True)
            atomic_json(work/'execution.json',dict(workers_per_gpu=proof['production_workers_per_gpu'],cpu_producers_per_worker=4,prefetch=4,writer_threads=2,job_size=16,per_card_handoff=True),immutable=True)
            with ThreadPoolExecutor(max_workers=4) as pool:launches=list(pool.map(lambda n:launch(cluster,n,work,'node'),cluster['nodes']))
            state('extracting',launches=launches);last=-1;change=time.monotonic()
            while True:
                s=status(work);atomic_json(work/'cluster_status.json',s)
                with (work/'history.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in s.items() if k!='workers'})+'\n')
                state('extracting',new_samples=s['new_samples'],full_ready=s['full_ready'],total=284218,failures=s['failures'])
                if s['new_samples']==7438:state('auditing');result=audit(str(work));state('complete',result=result);return
                if s['new_samples']!=last:last=s['new_samples'];change=time.monotonic()
                if all((work/'nodes'/n['name']/'EXIT.json').exists() for n in cluster['nodes']):raise RuntimeError('AV nodes exited before full coverage')
                if time.monotonic()-change>1800:raise RuntimeError('AV extraction made no progress for 30 minutes')
                time.sleep(15)
        except BaseException:state('attention_required',error=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','canary','worker','node','chain','status','audit','free']);p.add_argument('--work',default=str(DEFAULT_WORK));p.add_argument('--node',default='node12');p.add_argument('--worker-id',default='local');a=p.parse_args()
    if a.command=='prepare':prepare(a.work)
    elif a.command=='canary':
        try:canary(a.work)
        except BaseException:atomic_json(Path(a.work)/'CANARY_FAILED.json',dict(error=traceback.format_exc(),time=time.time()));raise
        finally:atomic_json(Path(a.work)/'CANARY_EXIT.json',dict(time=time.time()))
    elif a.command=='worker':worker(a.work,a.worker_id)
    elif a.command=='node':node(a.work,a.node)
    elif a.command=='chain':chain(a.work)
    elif a.command=='audit':print(json.dumps(audit(a.work)))
    elif a.command=='free':print(json.dumps(free_gpus()))
    else:s=status(a.work);s.pop('workers');print(json.dumps(s))

if __name__=='__main__':main()
