"""Node11 four-GPU progressive evaluation of the expanded-LoRA5000 model."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,PYTHON,METRIC_PYTHON
from listen_cluster import transport
from p2_progressive_eval_20261108 import ResourcePoller,ShardBinder
import p2_all_lora5000_eval_20261109 as E


class Binder(ShardBinder):
    def rows(self,ds,ids):
        from evaluation.listener_metrics.suite import enrich
        chosen=[];mapping=[]
        for original in self.templates[ds]:
            if original['id'] not in ids:continue
            key=ds+'/'+original['id'];assert key in self.committed
            item=self.by_key[key];folder=self.work/'generated'/self.arm/key;m=read(folder/'metadata.json');g=item['geometry']
            assert m['step']==self.p['arms'][0]['step'] and m['checkpoint_sha256']==self.p['arms'][0]['original_sha256']
            assert m['planning_adapter']==self.p['planner_adapter_identity']
            assert not m['listener_future_input'] and not m['target_description_access']
            e=m['export'];assert e['frames']==g['frames'] and abs(e['fps']-g['fps'])<1e-3
            assert (e['width'],e['height'])==(g['width'],g['height'])==(512,512)
            row=copy.deepcopy(original);row['generated']=str(folder/'generated.mp4');chosen.append(row)
            mapping.append(dict(id=row['id'],generated_audio=str(folder/'listener.wav')))
        assert {x['id'] for x in chosen}==set(ids)
        return enrich(chosen,self.refs[ds],mapping,self.work)
    def canonical_audit(self):
        from p2_testsets_eval.cluster import bind
        assert len(self.committed)==self.p['total_samples']==200
        old={}
        for rank in range(self.p['workers']):
            p=self.work/'metric_manifests'/self.arm/'seamless'/f'{rank:02d}.jsonl'
            old[p]=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        bind(self.work)
        for p,rows in old.items():assert rows==[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        save(self.work/'PROGRESSIVE_BIND_AUDIT.json',dict(success=True,outputs=200,at=time.time(),
             all4_manifests_equal_to_native_full_bind=True))


def worker_environment(stage,physical):
    assert stage in ('render','decode','metric') and 0<=physical<4
    return dict(CUDA_VISIBLE_DEVICES=str(physical+4),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',PYTORCH_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH='/private/liqi_workspace/.cache/paired_video_metrics/python' if stage=='metric' else '')


def launch(work,cluster,stage,physical,slot=0,logical=None,smoke=False,attempt=1):
    node=cluster['nodes'][0]
    if stage=='metric':
        cmd=[str(METRIC_PYTHON),'-u',str(ROOT/'scripts/p2_pipelined_testsets_eval.py'),'metrics',
             '--work',str(work),'--rank',str(logical),'--arms',E.ARM]
        if smoke:cmd.append('--metric-smoke')
    else:
        cmd=[str(PYTHON),'-u',str(SELF),'worker','--work',str(work),'--stage',stage,
             '--rank',str(physical),'--slot',str(slot)]
    cmd=['env',*[f'{k}={v}' for k,v in worker_environment(stage,physical).items()],*cmd]
    guard=ROOT.parent/'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
    cmd=['env','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1',str(PYTHON),str(guard),'--guard',*cmd]
    log=work/'logs'/f'{stage}_gpu{physical:02d}_slot{slot}_logical{logical}_smoke{smoke}_attempt{attempt}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('ab',buffering=0) as stream:
        p=subprocess.Popen(transport(cluster,node,cmd),stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT)
    save(work/'processes'/f'{stage}_{physical:02d}_{slot}_{logical}_{smoke}.json',dict(pid=p.pid,
         at=time.time(),node=node['name'],gpu=physical+4,logical=logical,stage=stage,log=str(log),attempt=attempt))
    return dict(process=p,physical=physical,stage=stage,log=log,smoke=smoke)


def queue(work):
    work.mkdir(parents=True,exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(work/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),host=os.uname().nodename))
        p=E.prepare(work);cluster=read(work/'cluster.json');binder=Binder(work)
        def probe(node):
            proc=subprocess.run(transport(cluster,node,['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=15)
            if proc.returncode:raise ConnectionError(proc.stderr[-500:])
            return {int(v[0]):dict(used=v[1],free=v[2]-v[1],util=v[3])
                    for line in proc.stdout.splitlines() if (v:=[float(x.strip()) for x in line.split(',')])}
        resources=ResourcePoller(cluster,probe);active={};attempts={};failures={};rendered=set()
        decode_limit={rank:4 for rank in range(4)};score_limit={rank:2 for rank in range(4)};next_status=0
        def scored(rank):
            f=work/'progress/metrics'/f'{E.ARM}_{rank:02d}.json'
            return f.exists() and read(f).get('status')=='complete'
        def start(key,stage,physical,slot=0,logical=None,smoke=False):
            attempts[key]=attempts.get(key,0)+1
            active[key]=launch(work,cluster,stage,physical,slot,logical,smoke,attempts[key])
        try:
            while True:
                if (work/'STOP').exists():
                    save(work/'status.json',dict(stage='stopped',at=time.time()));return
                for key,job in list(active.items()):
                    rc=job['process'].poll()
                    if rc is None:continue
                    if job['process'].stdin:job['process'].stdin.close()
                    del active[key]
                    if rc:
                        failures[key]=failures.get(key,0)+1
                        if 'out of memory' in job['log'].read_text(errors='replace').lower():
                            if job['stage']=='decode':decode_limit[job['physical']]=2
                            elif job['stage']=='metric':score_limit[job['physical']]=1
                        if failures[key]>=(8 if rc==255 else 3):raise RuntimeError('Repeated evaluation failure: '+str(job['log']))
                    elif job['stage']=='render':rendered.add(job['physical'])
                    elif job['smoke']:save(work/'METRICS_SMOKE_COMPLETE.json',dict(status='passed',at=time.time()))
                binder.refresh();binder.publish_ready()
                latents=sum((work/'latents'/E.ARM/x['key']/'complete.json').exists() for x in binder.items)
                reports=resources.snapshot();started={i:0 for i in range(4)}
                for physical in range(4):
                    report=reports.get(cluster['nodes'][0]['name'],{})
                    if physical+4 not in report or started[0]>=2:continue
                    free=report[physical+4]['free'];jobs=[j for j in active.values() if j['physical']==physical]
                    if physical not in rendered:
                        key=('render',physical)
                        if latents==200 and key not in active:
                            rendered.add(physical);continue
                        if key not in active and free>=80*1024:
                            start(key,'render',physical);started[0]+=1
                        continue
                    smoke_done=(work/'METRICS_SMOKE_COMPLETE.json').exists()
                    metric=None
                    if not smoke_done and ('smoke',0) not in active and binder.smoke():metric=(('smoke',0),0,True)
                    elif smoke_done:
                        for logical in sorted(binder.published):
                            key=('metric',logical)
                            if key not in active and not scored(logical):metric=(key,logical,False);break
                    if metric and not any(j['stage']=='decode' for j in jobs) and len(jobs)<score_limit[physical] and free>=24*1024:
                        key,logical,smoke=metric;start(key,'metric',physical,logical=logical,smoke=smoke)
                        started[0]+=1;continue
                    if any(j['stage']!='decode' for j in jobs) or len(jobs)>=decode_limit[physical] or free<40*1024:continue
                    decoding=sum(j['stage']=='decode' for j in active.values())
                    if latents-len(binder.committed)<=decoding:continue
                    for slot in range(4):
                        key=('decode',physical,slot)
                        if key in active:continue
                        start(key,'decode',physical,slot);started[0]+=1;break
                complete_scores=sum(scored(rank) for rank in range(4))
                if time.time()>=next_status:
                    save(work/'status.json',dict(stage='progressive_evaluation',at=time.time(),total=200,
                        latents=latents,decoded=len(binder.committed),scored_shards=complete_scores,total_shards=4,
                        active=len(active),render_gpus_finished=len(rendered)))
                    save(work/'RESOURCES.json',dict(at=time.time(),nodes=reports));next_status=time.time()+10
                if len(binder.committed)==200 and complete_scores==4 and not active:break
                time.sleep(1)
            binder.canonical_audit();save(work/'GENERATION_COMPLETE.json',dict(total_outputs=200,at=time.time()))
            cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
                 'PYTHONPATH=/private/liqi_workspace/.cache/paired_video_metrics/python',str(METRIC_PYTHON),
                 str(ROOT/'scripts/p2_testsets_eval.py'),'merge','--work',str(work)]
            with (work/'metrics_merge.log').open('a') as log:
                subprocess.run(cmd,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
            E.final(work)
            save(work/'status.json',dict(stage='complete',at=time.time(),outputs=200))
        finally:
            resources.close()
            for job in active.values():
                if job['process'].stdin:job['process'].stdin.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','queue','worker'])
    parser.add_argument('--work',type=Path,default=E.WORK);parser.add_argument('--stage',choices=['render','decode'])
    parser.add_argument('--rank',type=int);parser.add_argument('--slot',type=int,default=0);a=parser.parse_args()
    try:
        if a.action=='prepare':E.prepare(a.work.resolve())
        elif a.action=='worker':E.worker(a.work.resolve(),a.stage,a.rank,a.slot)
        else:queue(a.work.resolve())
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
