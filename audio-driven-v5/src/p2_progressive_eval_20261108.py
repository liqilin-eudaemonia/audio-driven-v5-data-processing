"""Release GPUs individually into ready decode/metric jobs; preserve native math."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from p2_testsets_eval.protocol import ROOT,read,save,sha,METRIC_PYTHON
from p2_testsets_eval.runtime import ready
from listen_cluster import transport
import p2_current_dual_eval_20261108 as D

BASELINE=ROOT/'experiments/p2_current500h_av6000_sft2752_dual200_20261107'


class ResourcePoller:
    """A slow/unreachable node never blocks reports from the other nodes."""
    def __init__(self,cluster,probe,interval=5):
        self.cluster=cluster;self.probe=probe;self.interval=interval
        self.pool=ThreadPoolExecutor(max_workers=4);self.jobs={};self.latest={};self.started={};self.lock=threading.Lock()
    def snapshot(self):
        with self.lock:
            now=time.time()
            for node in self.cluster['nodes']:
                name=node['name'];future=self.jobs.get(name)
                if future is not None and future.done():
                    try:self.latest[name]=(now,future.result())
                    except Exception as exc:self.latest[name]=(now,dict(error=repr(exc)))
                    del self.jobs[name]
                if name not in self.jobs and now-self.started.get(name,0)>=self.interval:
                    self.jobs[name]=self.pool.submit(self.probe,node);self.started[name]=now
            return {n:v if now-t<45 else dict(error='stale resource sample') for n,(t,v) in self.latest.items()}
    def close(self):self.pool.shutdown(wait=False,cancel_futures=True)


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    temp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));temp.replace(path)


def binding_inputs_ready(work):
    # protocol.json is written before DATA_CONFIG and shards during preparation.
    # Wait for the final acceptance receipt to avoid observing a partial setup.
    if not (work/'protocol.json').exists() or not (work/'CPU_ACCEPTANCE.json').exists():return False
    if not read(work/'CPU_ACCEPTANCE.json').get('success'):return False
    p=read(work/'protocol.json')
    return (work/'shared_items.json').exists() and all(
        (work/'shards'/f'{rank:02d}.json').exists() for rank in range(p['workers']))


class ShardBinder:
    """Reuse sealed native GT rows, changing only generated media paths.

    Canonical full native bind is rerun and compared before final aggregation.
    This permits early scoring without silently reducing the fixed200 set.
    """
    def __init__(self,work):
        self.work=work;self.p=read(work/'protocol.json');self.arm=self.p['arms'][0]['name']
        parent=read(BASELINE/'protocol.json');a=parent['arms'][0]['name']
        self.templates={};self.refs={};self.source={}
        for ds,spec in self.p['datasets'].items():
            assert spec['references_sha256']==parent['datasets'][ds]['references_sha256']
            f=Path(spec['prepared'])/'references.jsonl';assert sha(f)==spec['references_sha256']
            refs=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
            t=BASELINE/'bindings'/a/ds/'pairs.jsonl'
            rows=[json.loads(s) for s in t.read_text().splitlines() if s.strip()]
            assert len(rows)==len(refs)==200 and {x['id'] for x in rows}=={x['id'] for x in refs}
            assert read(t.parent/'binding_audit.json')['status']=='ready_for_scoring'
            self.templates[ds]=rows;self.refs[ds]=refs;self.source[ds]=dict(path=str(t),sha256=sha(t))
        self.items=read(work/'shared_items.json');self.by_key={x['key']:x for x in self.items}
        self.shards=[read(work/'shards'/f'{rank:02d}.json') for rank in range(self.p['workers'])]
        self.published=set();self.committed=set()
    def refresh(self):
        for key in self.by_key.keys()-self.committed:
            if ready(self.work/'generated'/self.arm/key):self.committed.add(key)
    def rows(self,ds,ids):
        from evaluation.listener_metrics.suite import enrich
        chosen=[];mapping=[]
        for original in self.templates[ds]:
            if original['id'] not in ids:continue
            key=ds+'/'+original['id'];assert key in self.committed
            item=self.by_key[key];folder=self.work/'generated'/self.arm/key;m=read(folder/'metadata.json');g=item['geometry']
            assert m['step']==6000 and m['checkpoint_sha256']==self.p['arms'][0]['original_sha256']
            assert m['planning_adapter']==self.p['planner_adapter_identity']
            assert not m['listener_future_input'] and not m['target_description_access']
            e=m['export'];assert e['frames']==g['frames'] and abs(e['fps']-g['fps'])<1e-3
            assert (e['width'],e['height'])==(g['width'],g['height'])==(512,512)
            row=copy.deepcopy(original);row['generated']=str(folder/'generated.mp4');chosen.append(row)
            mapping.append(dict(id=row['id'],generated_audio=str(folder/'listener.wav')))
        assert {x['id'] for x in chosen}==set(ids)
        return enrich(chosen,self.refs[ds],mapping,self.work)
    def publish_ready(self):
        for rank,items in enumerate(self.shards):
            if rank in self.published or not all(x['key'] in self.committed for x in items):continue
            for ds in self.p['datasets']:
                ids={x['id'] for x in items if x['dataset']==ds};assert ids
                path=self.work/'metric_manifests'/self.arm/ds/f'{rank:02d}.jsonl'
                write_rows(path,self.rows(ds,ids))
            save(self.work/'progressive_bind'/f'{rank:02d}.json',dict(at=time.time(),rank=rank,
                complete_shard=True,ids=[x['key'] for x in items],gt_templates=self.source,
                no_test_subset_reduction=True))
            self.published.add(rank)
    def smoke(self):
        paths=[]
        for ds in self.p['datasets']:
            selected=[x['id'] for x in self.templates[ds] if ds+'/'+x['id'] in self.committed][:2]
            if len(selected)!=2:return False
            paths.append((ds,selected))
        for ds,ids in paths:
            write_rows(self.work/'smoke_metric_manifests'/self.arm/ds/'00.jsonl',self.rows(ds,set(ids)))
        save(self.work/'PROGRESSIVE_SMOKE_INPUTS.json',dict(at=time.time(),samples=dict(paths),
             rule='First two committed IDs in sealed reference order per dataset; validation only'))
        return True
    def canonical_audit(self):
        from p2_testsets_eval.cluster import bind
        assert len(self.committed)==400
        previous={}
        for rank in range(self.p['workers']):
            for ds in self.p['datasets']:
                f=self.work/'metric_manifests'/self.arm/ds/f'{rank:02d}.jsonl'
                previous[str(f)]=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
        bind(self.work)
        for path,expected in previous.items():
            actual=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
            assert actual==expected,'Progressive rows differ from full native binding: '+path
        save(self.work/'PROGRESSIVE_BIND_AUDIT.json',dict(success=True,at=time.time(),rows=400,
             all32_manifests_equal_to_native_full_bind=True,templates=self.source))


class Postprocessor:
    def __init__(self,root,cluster,names,resources,launch_decode):
        self.root=root;self.cluster=cluster;self.names=names;self.resources=resources;self.launch_decode=launch_decode
        self.released=set();self.lock=threading.Lock();self.active={};self.attempts={};self.failures={};self.binders={}
        self.smoke_active=set();self.error=None;self.finished=False;self.stop=threading.Event()
        self.decode_limits={rank:4 for rank in range(32)}
        self.final_pool=ThreadPoolExecutor(max_workers=2);self.final_jobs={}
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start()
    def release(self,rank):
        with self.lock:self.released.add(rank)
    def release_all(self):
        with self.lock:self.released.update(range(32))
    def check(self):
        if self.error:raise RuntimeError('Progressive postprocessing failed: '+self.error)
    def join(self):
        while self.thread.is_alive():self.thread.join(timeout=1);self.check()
        self.check();assert self.finished
    def launch_metric(self,name,logical,physical,smoke=False):
        work=D.INFO[name]['work'];b=self.binders[name];node=self.cluster['nodes'][physical//8]
        cmd=['env',f'CUDA_VISIBLE_DEVICES={physical%8}','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',
             'PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',str(METRIC_PYTHON),'-u',
             str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics','--work',str(work),'--rank',str(logical),'--arms',b.arm]
        if smoke:cmd.append('--metric-smoke')
        f=self.root/'logs'/f'progressive_{name}_{logical:02d}_{physical:02d}_{smoke}.log'
        f.parent.mkdir(parents=True,exist_ok=True)
        with f.open('ab',buffering=0) as out:p=subprocess.Popen(transport(self.cluster,node,cmd),stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        save(self.root/'progressive_processes'/f'{name}_{logical:02d}_{smoke}.json',dict(pid=p.pid,at=time.time(),
             rank=logical,physical_rank=physical,smoke=smoke,model=name,log=str(f)))
        return p
    def completed(self,name,rank):
        b=self.binders[name];p=b.work/'progress/metrics'/f'{b.arm}_{rank:02d}.json'
        return p.exists() and read(p).get('status')=='complete'
    def run(self):
        try:self.loop()
        except Exception as exc:
            import traceback
            self.error=repr(exc);save(self.root/'PROGRESSIVE_FAILURE.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()))
    def loop(self):
        next_status=0
        while True:
            if (self.root/'STOP').exists():raise RuntimeError('User STOP')
            for key,job in list(self.active.items()):
                code=job['process'].poll()
                if code is None:continue
                del self.active[key]
                if code:
                    if job['stage']=='decode':
                        log=self.root/'logs'/f'decode_{job["name"]}_{job["physical"]:02d}_{key[3]}_attempt{self.attempts[key]}.log'
                        if log.exists() and 'out of memory' in log.read_text(errors='replace').lower():
                            self.decode_limits[job['physical']]=2
                    self.failures[key]=self.failures.get(key,0)+1
                    if self.failures[key]>=(8 if code==255 else 3):raise RuntimeError('Progressive job failed after retries: '+str(key))
                elif job['stage']=='smoke':
                    save(self.binders[job['name']].work/'METRICS_SMOKE_COMPLETE.json',dict(status='passed',at=time.time(),physical_rank=job['physical']))
            for name in self.names:
                work=D.INFO[name]['work']
                if name not in self.binders and binding_inputs_ready(work):self.binders[name]=ShardBinder(work)
                if name in self.binders:
                    b=self.binders[name];b.refresh();b.publish_ready()
            with self.lock:released=sorted(self.released)
            reports=self.resources.snapshot();per_node={i:0 for i in range(4)}
            for physical in released:
                report=reports.get(self.cluster['nodes'][physical//8]['name'],{})
                if physical%8 not in report:continue
                free=report[physical%8]['free'];on_gpu=[x for x in self.active.values() if x['physical']==physical]
                if per_node[physical//8]>=2:continue
                # Give ready score shards priority once their real-media smoke passed.
                metric_jobs=[]
                for name,b in self.binders.items():
                    smoke_done=(b.work/'METRICS_SMOKE_COMPLETE.json').exists()
                    if not smoke_done:
                        key=('smoke',name,0)
                        if key not in self.active and b.smoke():metric_jobs.append((key,name,0,True))
                    else:
                        for rank in sorted(b.published):
                            key=('metric',name,rank)
                            if key not in self.active and not self.completed(name,rank):metric_jobs.append((key,name,rank,False))
                if metric_jobs and not any(x['stage']=='decode' for x in on_gpu) and len(on_gpu)<2 and free>=24*1024:
                    key,name,logical,smoke=metric_jobs[0]
                    self.attempts[key]=self.attempts.get(key,0)+1
                    self.active[key]=dict(process=self.launch_metric(name,logical,physical,smoke),
                        name=name,stage='smoke' if smoke else 'metric',physical=physical)
                    per_node[physical//8]+=1;continue
                if on_gpu and any(x['stage']!='decode' for x in on_gpu):continue
                if len(on_gpu)>=self.decode_limits[physical] or free<40*1024:continue
                for name,b in self.binders.items():
                    latent=sum((b.work/'latents'/b.arm/x['key']/'complete.json').exists() for x in b.items)
                    backlog=latent-len(b.committed)
                    decoding=sum(x['stage']=='decode' and x['name']==name for x in self.active.values())
                    if backlog<=decoding:continue
                    for slot in range(4):
                        key=('decode',name,physical,slot)
                        if key in self.active:continue
                        self.attempts[key]=self.attempts.get(key,0)+1
                        p=self.launch_decode(self.root,self.cluster,name,'decode',physical,slot,self.attempts[key])
                        self.active[key]=dict(process=p,name=name,stage='decode',physical=physical)
                        per_node[physical//8]+=1;break
                    break
            progress={name:dict(decoded=len(b.committed),ready_shards=len(b.published),
                      scored_shards=sum(self.completed(name,r) for r in range(b.p['workers']))) for name,b in self.binders.items()}
            for name,b in self.binders.items():
                if name not in self.final_jobs and not any(j['name']==name for j in self.active.values()) and (
                        len(b.committed)==400 and all(self.completed(name,r) for r in range(b.p['workers']))):
                    self.final_jobs[name]=self.final_pool.submit(self.finalize,name,b)
            for future in self.final_jobs.values():
                if future.done():future.result()
            if time.time()>=next_status:
                save(self.root/'PROGRESSIVE_STATUS.json',dict(at=time.time(),released_gpus=released,active=len(self.active),models=progress))
                next_status=time.time()+10
            if len(self.final_jobs)==len(self.names) and all(f.done() for f in self.final_jobs.values()) and not self.active:break
            time.sleep(1)
        self.final_pool.shutdown(wait=True)
        self.finished=True

    def finalize(self,name,b):
        # Only final full-manifest verification and global distribution statistics
        # require every sample. No earlier machine-wide stage barrier is needed.
        b.canonical_audit()
        work=b.work
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time(),world=32))
        cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
             'PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',str(METRIC_PYTHON),
             str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(work)]
        with (work/'metrics_merge.log').open('a') as log:
            subprocess.run(transport(self.cluster,self.cluster['nodes'][0],cmd),check=True,stdout=log,stderr=subprocess.STDOUT)
        D.final(work,name)
