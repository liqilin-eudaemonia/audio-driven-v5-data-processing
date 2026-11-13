"""Distributed candidate-only media export using the frozen V8 exporter."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import run_listen_v8_cluster as cluster
import render_listen_v8_full as renderer
from p3_listen.common import write,sha
read=cluster.read
OUT=cluster.OUT
ROOT=cluster.ROOT
SCHED=OUT/'cluster_media'
SCRIPT=Path(__file__).resolve()
_QUEUE=None

def verify():
    cluster.pipeline.install();cluster.full.verify();renderer.OUT=OUT
    c=read(SCHED/'contract.json')
    assert c['script_sha256']==sha(SCRIPT) and c['quality_contract_sha256']==sha(OUT/'contract.json')
    assert c['queue_sha256']==sha(SCHED/'queue.json')

def eligible(row):return row['release']['pool']=='candidate'

def export_pair(pair):
    path=OUT/'results'/(pair+'.json');result=read(path);outputs=[];verified=set()
    for row in result['clips']:
        if not eligible(row):continue
        item=renderer.export(dict(row,effective_decision='candidate'),verified)
        item['release']=row['release'];outputs.append(item)
    value=dict(pair=pair,complete=True,clips=outputs,source_result_sha256=sha(path),utc=cluster.full.now())
    write(OUT/'render_results'/path.name,value);return value

def claim():
    global _QUEUE
    if _QUEUE is None:_QUEUE=read(SCHED/'queue.json')['pairs']
    with (SCHED/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=read(SCHED/'state.json')
        queue=_QUEUE
        if state['cursor']>=len(queue) or read(SCHED/'control.json')['stop']:return None
        pair=queue[state['cursor']];state['cursor']+=1;write(SCHED/'state.json',state);return pair

def worker(name):
    verify();counts=Counter();previous=cluster.full.cpu_sample();last_status=0
    with (SCHED/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=64) as pool:
            pending={};done_reading=False
            while True:
                control=read(SCHED/'control.json');target=0 if control['stop'] else control['workers'][name]
                assert 0<=target<=64
                while len(pending)<target and not done_reading:
                    pair=claim()
                    if pair is None:done_reading=True;break
                    pending[pool.submit(export_pair,pair)]=pair
                if not pending:break
                finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    pair=pending.pop(future)
                    try:
                        value=future.result();counts['complete_pairs']+=1;counts['candidate_clips']+=len(value['clips'])
                    except Exception as e:
                        counts['error_pairs']+=1
                        write(OUT/'render_errors'/(pair+'.json'),dict(pair=pair,error=repr(e),traceback=traceback.format_exc(),node=name))
                if time.monotonic()-last_status>10:
                    previous,metrics=cluster.sample_node(previous)
                    write(SCHED/'nodes'/(name+'.json'),dict(node=name,unix=time.time(),utc=cluster.full.now(),state='running',pid=os.getpid(),counts=dict(counts),inflight=len(pending),**metrics));last_status=time.monotonic()
            previous,metrics=cluster.sample_node(previous)
            write(SCHED/'nodes'/(name+'.json'),dict(node=name,unix=time.time(),utc=cluster.full.now(),state='finished',pid=os.getpid(),counts=dict(counts),inflight=0,**metrics))

def detach_node(name):
    verify()
    with (SCHED/'logs'/(name+'.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPT),'worker','--node',name],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(json.dumps(dict(node=name,pid=p.pid)),flush=True)

def index():
    handles={};counts=Counter()
    try:
        for path in (OUT/'render_results').glob('*.json'):
            result=read(path)
            for row in result['clips']:
                assert eligible(row)
                key='rendered_'+row['split']+'_candidate_'+row['admission_pool']
                if key not in handles:handles[key]=(OUT/'manifests'/(key+'.jsonl.pending')).open('w')
                handles[key].write(json.dumps(row,ensure_ascii=False)+'\n');counts[key]+=1
    finally:
        for h in handles.values():h.close()
    for key in handles:(OUT/'manifests'/(key+'.jsonl.pending')).replace(OUT/'manifests'/(key+'.jsonl'))
    return dict(counts)

def main():
    cluster.pipeline.install();cluster.full.verify();renderer.OUT=OUT;cfg=cluster.configuration()
    for d in ('nodes','logs'):(SCHED/d).mkdir(parents=True,exist_ok=True)
    with (OUT/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # This stage starts only after all screening nodes acknowledged completion.
        nodes=[read(p) for p in (cluster.SCHED/'nodes').glob('*.json')]
        assert len(nodes)==3 and all(n['state']=='finished' for n in nodes)
        old_nodes=[read(p) for p in (SCHED/'nodes').glob('*.json')]
        if any(n['state']!='finished' for n in old_nodes):raise RuntimeError('Previous media workers may still be live; inspect before resuming')
        for old in (SCHED/'nodes').glob('*.json'):old.unlink()
        queue=[]
        for path in sorted((OUT/'results').glob('*.json')):
            r=read(path)
            if any(eligible(row) for row in r['clips']):queue.append(r['pair'])
        random.Random(2026102305).shuffle(queue)
        write(SCHED/'queue.json',dict(pairs=queue));write(SCHED/'state.json',dict(cursor=0,total=len(queue)))
        write(SCHED/'contract.json',dict(utc=cluster.full.now(),script_sha256=sha(SCRIPT),quality_contract_sha256=sha(OUT/'contract.json'),queue_sha256=sha(SCHED/'queue.json'),pools=['candidate'],max_workers_per_node=64,codec_threads=1))
        write(SCHED/'control.json',dict(stop=False,workers={n['name']:16 for n in cfg['nodes']}))
        try:
            for node in cfg['nodes']:cluster.ssh(node,[cfg['python'],str(SCRIPT),'detach-node','--node',node['name']])
            levels=[16,32,64];phase=0;began=time.time();baseline=None;trials=[];best=None;safety=False
            while True:
                nodes={p.stem:read(p) for p in (SCHED/'nodes').glob('*.json')};now=time.time();counts=Counter()
                for n in nodes.values():
                    counts.update(n['counts'])
                    if n['state']!='finished' and now-n['unix']>90:raise RuntimeError('Stale media worker '+n['node'])
                    if cluster.pressure(n) or n['iowait_percent']>15:
                        control=read(SCHED/'control.json');control['workers'][n['node']]=max(4,control['workers'][n['node']]//2);write(SCHED/'control.json',control);safety=True
                if len(nodes)==3 and all(n['state']=='finished' for n in nodes.values()):break
                if now-began>90 and len(nodes)!=3:raise RuntimeError('Missing media node')
                if baseline is None and now-began>=30:baseline=(now,counts['candidate_clips'])
                if baseline and now-began>=120 and phase<len(levels):
                    trial=dict(workers_per_node=levels[phase],clips_per_minute=(counts['candidate_clips']-baseline[1])*60/(now-baseline[0]),utc=cluster.full.now(),resource_backoff=safety);trials.append(trial)
                    if not safety and (best is None or trial['clips_per_minute']>best['clips_per_minute']*1.05):best=trial
                    phase+=1
                    if safety:phase=len(levels)
                    control=read(SCHED/'control.json')
                    if phase<len(levels):control['workers']={n['name']:levels[phase] for n in cfg['nodes']};began=now;baseline=None
                    elif best and not safety:control['workers']={n['name']:best['workers_per_node'] for n in cfg['nodes']}
                    write(SCHED/'control.json',control);write(SCHED/'benchmark.json',dict(trials=trials,selected=best,note='Disjoint production pairs; observed throughput, not a matched-input comparison.'))
                value=dict(utc=cluster.full.now(),state='cluster_rendering_candidates',nodes=nodes,counts=dict(counts),control=read(SCHED/'control.json'),training_ready=False)
                write(OUT/'render_summary.json',value);write(OUT/'runtime.json',dict(state=value['state'],utc=value['utc'],control=value['control'],cluster=str(SCHED),training_started=False))
                time.sleep(10)
            manifest_counts=index()
            write(OUT/'render_summary.json',dict(utc=cluster.full.now(),state='finished_with_errors' if counts['error_pairs'] else 'finished',counts=dict(counts),manifest_counts=manifest_counts,training_ready=False))
            if counts['error_pairs']:raise RuntimeError('Media errors retained; inspect render_errors')
        except BaseException:
            control=read(SCHED/'control.json');control['stop']=True;write(SCHED/'control.json',control)
            raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','worker','detach-node']);p.add_argument('--node',default='node12');a=p.parse_args()
    if a.stage=='worker':worker(a.node)
    elif a.stage=='detach-node':detach_node(a.node)
    else:main()
