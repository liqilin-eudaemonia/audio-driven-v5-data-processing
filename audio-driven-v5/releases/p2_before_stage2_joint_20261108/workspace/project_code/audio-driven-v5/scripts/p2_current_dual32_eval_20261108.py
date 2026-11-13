"""Four-node shared-claim generation and pipelined scores after full6000 save."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON,METRIC_PYTHON
import p2_current_dual_eval_20261108 as D
from listen_cluster import transport
WORK=ROOT/'experiments/p2_current_feature_whisper6000_eval32_20261108'


def state(root,stage,**kw):
    save(root/'status.json',dict(stage=stage,at=time.time(),**kw))


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    c=read(D.INFO['whisper']['train']/'cluster.json')
    assert [n['name'] for n in c['nodes']]==['node12','node8','node4','node11']
    for n in c['nodes']:
        n['gpus']=list(range(8))
        if n['name']=='node11':n['ssh_key']='/private/ssh_keys/root_dedicated-developjob-js-infer-machine-ws3qe_ed25519'
    c['training_script']=str(SELF);save(root/'cluster.json',c)
    import torch
    torch.set_num_threads(2)
    feature=D.export_model(root,'feature')
    save(root/'PLAN.json',dict(at=time.time(),models=['feature','whisper'],world=32,
        datasets={'seamless':200,'responsenet':200},planner_step=2752,
        feature_work=str(D.FEATURE),whisper_work=str(D.WHISPER),
        feature_original_sha256=feature['original_sha256'],checkpoint_gate=6000,
        preserve_finished_feature_outputs=True,decode_processes_per_gpu=4,
        scoring_shards_each=32,cpu_prefetch_depth=2,metric_batch_size=64))


def snapshot(cluster,node,pid):
    code="""import sys,json;from pathlib import Path
p=Path('/proc',sys.argv[1]);r={'pid':int(sys.argv[1]),'alive':False}
try:
 s=(p/'stat').read_text().split();r.update(alive=s[2]!='Z',start_ticks=s[21],command=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
except FileNotFoundError:pass
print(json.dumps(r))"""
    r=subprocess.run(transport(cluster,node,['/private/software/conda/bin/python','-c',code,str(pid)]),capture_output=True,text=True,timeout=30)
    if r.returncode:raise RuntimeError(r.stderr[-500:])
    return json.loads(r.stdout)


def resources(cluster):
    def one(n):
        try:
            p=subprocess.run(transport(cluster,n,['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits']),capture_output=True,text=True,timeout=30)
            if p.returncode:raise RuntimeError(p.stderr[-500:])
            return n['name'],{int(v[0]):dict(used=v[1],free=v[2]-v[1],util=v[3])
                for s in p.stdout.splitlines() if (v:=[float(x.strip()) for x in s.split(',')])}
        except Exception as exc:return n['name'],dict(error=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:return dict(pool.map(one,cluster['nodes']))


def wait_training_exit(root,cluster):
    tr=D.INFO['whisper']['train'];record=read(tr/'CONTROLLER.json')
    while True:
        if (root/'STOP').exists():raise RuntimeError('User stopped evaluation before takeover')
        s=snapshot(cluster,cluster['nodes'][0],record['pid'])
        if not s['alive'] or s.get('start_ticks')!=record['start_ticks']:break
        state(root,'waiting_training_exit',checkpoint_published=True,controller_pid=record['pid'])
        time.sleep(5)
    save(root/'TRAINING_EXITED.json',dict(success=True,at=time.time(),controller=record,
         observed=s,checkpoint=str(tr/'step_006000.pt'),training_status=read(tr/'status.json')))


def adopt_feature(root,cluster):
    # Only the old CPU coordinator is replaced. Its active GPU7 sample process
    # keeps running and retains its native per-sample claim and state.
    work=D.FEATURE
    if (work/'HANDOFF_COMPLETE.json').exists():return None,True
    status=read(work/'status.json')
    if status['stage']!='render':
        save(root/'FEATURE_EXTERNAL_FLOW.json',dict(at=time.time(),status=status,
            reason='Feature passed render before migration; preserve its existing postprocessing'))
        return None,True
    c=read(work/'CONTROLLER.json');r=read(work/'processes/render_0.json');node=cluster['nodes'][3]
    current=snapshot(cluster,node,c['pid']);worker=snapshot(cluster,node,r['pid'])
    expected=str(ROOT/'scripts/p2_current_feature6000_sft2752_eval_20261108.py')
    if current['alive']:
        assert expected in current['command'] and ' queue ' in current['command'] and str(work) in current['command']
        code="""import os,signal,sys;from pathlib import Path
p=int(sys.argv[1]);s=Path('/proc',str(p),'stat').read_text().split();assert s[21]==sys.argv[2]
os.kill(p,signal.SIGTERM)"""
        subprocess.run(transport(cluster,node,['/private/software/conda/bin/python','-c',code,str(c['pid']),current['start_ticks']]),check=True,timeout=30)
    save(root/'ADOPTED_FEATURE.json',dict(at=time.time(),coordinator=current,worker=worker,
         cpu_coordinator_only_terminated=True,active_gpu_process_untouched=True))
    for _ in range(20):
        if not snapshot(cluster,node,c['pid'])['alive']:break
        time.sleep(.5)
    else:raise RuntimeError('Old Feature controller still alive')
    if worker['alive']:
        assert expected in worker['command'] and '--stage render' in worker['command'] and str(work) in worker['command']
        worker.update(node=3,model='feature',rank=31,legacy=True)
        return worker,False
    return None,False


def count(name,part):
    work=D.INFO[name]['work'];p=read(work/'protocol.json');arm=p['arms'][0]['name']
    return sum((work/part/arm/x['key']/'complete.json').exists() for x in read(work/'shared_items.json'))


class Adopted:
    def __init__(self,cluster,record):self.cluster=cluster;self.record=record;self.returncode=None;self.next_poll=0
    def poll(self):
        if self.returncode is not None:return self.returncode
        if time.time()<self.next_poll:return None
        self.next_poll=time.time()+10
        s=snapshot(self.cluster,self.cluster['nodes'][self.record['node']],self.record['pid'])
        if s['alive'] and s.get('start_ticks')==self.record['start_ticks']:return None
        f=D.FEATURE/'progress/render/07.json'
        self.returncode=0 if f.exists() and read(f).get('status')=='complete' else 1
        return self.returncode


def launch(root,cluster,name,stage,rank,slot=0,attempt=1):
    node=cluster['nodes'][rank//8];work=D.INFO[name]['work']
    command=['env',f'CUDA_VISIBLE_DEVICES={rank%8}','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',
             'TOKENIZERS_PARALLELISM=false','PYTORCH_ALLOC_CONF=expandable_segments:True',str(PYTHON),'-u',str(SELF),
             'worker','--work',str(root),'--model',name,'--stage',stage,'--rank',str(rank),'--slot',str(slot)]
    path=root/'logs'/f'{stage}_{name}_{rank:02d}_{slot}_attempt{attempt}.log';path.parent.mkdir(exist_ok=True)
    with path.open('ab',buffering=0) as log:
        p=subprocess.Popen(transport(cluster,node,command),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    save(root/'processes'/f'{stage}_{name}_{rank:02d}_{slot}.json',dict(pid=p.pid,at=time.time(),
         model=name,stage=stage,rank=rank,slot=slot,node=node['name'],gpu=rank%8,log=str(path),attempt=attempt))
    return p


def render_both(root,cluster,adopted,external,whisper_future):
    active={};todo={rank:(['whisper'] if external else ['feature','whisper']) for rank in range(32)}
    attempts={};last_resources=0;reports={};next_status=0
    if adopted:
        active[31]=('feature',Adopted(cluster,adopted));attempts[(31,'feature')]=1
    while active or any(todo.values()):
        if (root/'STOP').exists():raise RuntimeError('User STOP; retained active samples may commit')
        for rank,(name,process) in list(active.items()):
            code=process.poll()
            if code is None:continue
            del active[rank]
            if code==0:
                assert todo[rank][0]==name;todo[rank].pop(0)
            elif attempts.get((rank,name),0)>=3:raise RuntimeError(f'Render {name} rank{rank} failed3 times')
        if whisper_future.done():whisper_future.result()
        counts={'feature':count('feature','latents')}
        if (D.WHISPER/'protocol.json').exists():counts['whisper']=count('whisper','latents')
        if time.time()-last_resources>=15:
            reports=resources(cluster);save(root/'RESOURCES.json',dict(at=time.time(),nodes=reports));last_resources=time.time()
        per_node={i:0 for i in range(4)}
        for rank in range(32):
            if rank in active:continue
            while todo[rank]:
                name=todo[rank][0]
                if name=='whisper' and not whisper_future.done():break
                if counts.get(name)==400:todo[rank].pop(0);continue
                node=cluster['nodes'][rank//8];report=reports.get(node['name'],{})
                if rank%8 not in report or report[rank%8]['free']<80*1024:break
                if per_node[rank//8]>=2:break
                key=(rank,name);attempts[key]=attempts.get(key,0)+1
                active[rank]=(name,launch(root,cluster,name,'render',rank,attempt=attempts[key]))
                per_node[rank//8]+=1;break
        if time.time()>=next_status:
            state(root,'render_both',completed=counts,active=len(active),
                active_by_model={n:sum(x[0]==n for x in active.values()) for n in ['feature','whisper']})
            for name,completed in counts.items():
                if name=='feature' and external:continue
                save(D.INFO[name]['work']/'status.json',dict(stage='render32',completed=completed,total=400,
                     active=sum(x[0]==name for x in active.values()),at=time.time(),controller_work=str(root)))
            next_status=time.time()+20
        time.sleep(1)
    whisper_future.result()
    for name in ['feature','whisper']:
        assert count(name,'latents')==400
        if name=='feature' and external:continue
        save(D.INFO[name]['work']/'stages/render.json',dict(complete=True,world=32,at=time.time(),preserved_existing=True))


def decode(root,cluster,name):
    if count(name,'generated')==400:return
    work=D.INFO[name]['work'];active={};attempts={};done=set();next_status=0
    last_resources=0;reports={};gpu_limits={rank:4 for rank in range(32)}
    jobs=[(rank,slot) for slot in range(4) for rank in range(32)]
    while len(done)<len(jobs):
        if (root/'STOP').exists():raise RuntimeError('User STOP during decode')
        for key,p in list(active.items()):
            code=p.poll()
            if code is None:continue
            del active[key]
            if code==0:done.add(key)
            else:
                rank,slot=key
                log=root/'logs'/f'decode_{name}_{rank:02d}_{slot}_attempt{attempts[key]}.log'
                if log.exists() and 'out of memory' in log.read_text(errors='replace').lower():gpu_limits[rank]=2
                if attempts[key]>=3:raise RuntimeError(f'Decode {name} {key} failed3 times')
        completed=count(name,'generated')
        if time.time()-last_resources>=15:
            reports=resources(cluster);last_resources=time.time()
        per_node={i:0 for i in range(4)}
        for key in jobs:
            if key in active or key in done:continue
            rank,slot=key
            if completed==400:done.add(key);continue
            if per_node[rank//8]>=2:continue
            if sum(r==rank for r,s in active)>=gpu_limits[rank]:continue
            observed=reports.get(cluster['nodes'][rank//8]['name'],{})
            if rank%8 not in observed or observed[rank%8]['free']<40*1024:continue
            attempts[key]=attempts.get(key,0)+1
            active[key]=launch(root,cluster,name,'decode',rank,slot,attempts[key]);per_node[rank//8]+=1
        if time.time()>=next_status:
            state(root,'decode_'+name,completed=completed,total=400,active=len(active),max_per_gpu=4)
            save(work/'status.json',dict(stage='decode32',completed=completed,total=400,active=len(active),
                 at=time.time(),controller_work=str(root)))
            next_status=time.time()+20
        time.sleep(1)
    assert count(name,'generated')==400
    save(work/'stages/decode.json',dict(complete=True,world=32,max_per_gpu=4,at=time.time()))


def scores(root,cluster,names):
    from p2_testsets_eval.cluster import bind
    from p2_testsets_eval import metrics
    from p2_current_av6000_score_overlay_20261108 import launch_budget
    # Two score jobs per GPU,32 fixed shards per model. If both models score
    # simultaneously there are at most64 production jobs across32 GPUs.
    for name in names:
        work=D.INFO[name]['work']
        save(work/'GENERATION_COMPLETE.json',dict(total_outputs=400,at=time.time(),world=32))
        bind(work)
        save(work/'SMOKE_COMPLETE.json',dict(status='generation_passed',outputs=400,at=time.time()))
        metrics.prepare_smoke(work)
        save(work/'status.json',dict(stage='metrics32',at=time.time(),controller_work=str(root)))
    def available(report):
        result=[]
        for line in report.splitlines():
            index,used,total,util=[float(v.strip()) for v in line.split(',')]
            if launch_budget(total-used,0)>0:result.append(int(index))
        return result
    def one(name):
        work=D.INFO[name]['work'];metrics.queue(work)
        if (work/'STOP').exists():raise RuntimeError('Work STOP during scoring: '+name)
        D.final(work,name)
    state(root,'metrics_both',models=names,workers_each=32,cpu_prefetch_depth=2)
    with patch.object(metrics,'CLI',ROOT/'scripts/p2_pipelined_testsets_eval.py'),patch.object(metrics,'idle_gpus',available):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one,names))


def run(root):
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time()))
        ready=read(root/'CODE_READY.json')
        assert all(sha(p)==h for p,h in ready['files'].items())
        prepare(root);cluster=read(root/'cluster.json')
        wait_training_exit(root,cluster)
        adopted,external=adopt_feature(root,cluster)
        owned=[]
        for name in (['whisper'] if external else ['feature','whisper']):
            work=D.INFO[name]['work'];work.mkdir(parents=True,exist_ok=True)
            file=(work/'controller.lock').open('a');fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB);owned.append(file)
            save(work/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),host=cluster['nodes'][0]['host'],
                 controller_work=str(root),world=32))
        if not external:D.update_feature_topology(root)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(D.prepare_whisper,root)
            render_both(root,cluster,adopted,external,future)
        names=['whisper'] if external else ['feature','whisper']
        for name in names:decode(root,cluster,name)
        scores(root,cluster,names)
        if external:
            while not (D.FEATURE/'HANDOFF_COMPLETE.json').exists():
                if (root/'STOP').exists():return
                state(root,'waiting_preserved_feature_scoring');time.sleep(15)
        save(root/'FINAL_AUDIT.json',dict(success=True,at=time.time(),outputs=800,
             models=['feature','whisper'],step=6000,planner_step=2752,world=32))
        state(root,'complete',outputs=800)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker'])
    p.add_argument('--work',type=Path,default=WORK);p.add_argument('--model',choices=['feature','whisper'])
    p.add_argument('--stage',choices=['render','decode']);p.add_argument('--rank',type=int);p.add_argument('--slot',type=int,default=0)
    a=p.parse_args()
    try:
        if a.action=='prepare':prepare(a.work.resolve())
        elif a.action=='run':run(a.work.resolve())
        else:D.worker(a.work.resolve(),a.model,a.stage,a.rank,a.slot)
    except Exception as exc:
        save(a.work/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
