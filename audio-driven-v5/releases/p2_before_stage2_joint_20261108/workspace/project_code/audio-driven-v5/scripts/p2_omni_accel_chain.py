#!/usr/bin/env python3
"""Wait for all S1 finals, benchmark four S2 arms, then train without user intervention."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,copy,fcntl,json,os,socket,statistics,subprocess,sys,threading,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_duration_execution import install_backend,Schedule
from p2_listen.contracts import read,sha,atomic_json
from p2_guided_chain import training_queue as native
from p2_legacy_qwen_queue import latest,resources
from listen_cluster import transport
import p2_omni_accel_parallel as parallel
LOSSES=native.LOSSES

def finished_parents(source):
    result={}
    for loss in LOSSES:
        c=read(source/'configs'/f'guided_{loss}_s1.json');budget=c['training']['steps']
        paths=list((source/'parallel_runs'/f'guided_{loss}_s1').glob(f'*/step_{budget:06d}.complete.json'))
        if not paths:return None
        if len(paths)!=1:raise ValueError('Ambiguous completed S1 parent: '+loss)
        marker=paths[0];path=marker.with_suffix('').with_suffix('.pt')
        result[loss]=dict(checkpoint=str(path),sha256=read(marker)['sha256'],step=budget)
    return result

def validate_parents(root,parents):
    from p2_matrix16.checkpoint import checked
    backend=install_backend()
    for loss,item in parents.items():
        saved=checked(item['checkpoint']);pc=saved['identity']['experiment'];c=read(root/'configs'/f'guided_{loss}_s2.json')
        if sha(item['checkpoint'])!=item['sha256'] or saved['step']!=pc['training']['steps'] or pc['id']!=c['parent_experiment'] or pc['stage']!=1:raise ValueError('Parent not matching completed S1')
        if saved['identity']['source']!=backend.source_identity():raise ValueError('Parent execution source changed')
        if saved['identity']['data_identity']!=backend.data_identity(c):raise ValueError('Parent data mismatch')
        for key in ('duration_sampling','loss_weights','data_config','plan_cache','teacher_cache','guided_view','text_lora'):
            if pc[key]!=c[key]:raise ValueError('Paired parent differs: '+key)
        del saved

def probe_config(root,loss,label):
    c=read(root/'configs'/f'guided_{loss}_s2.json');original=Path(c['output']);c['output']=str(root/'benchmark/runs'/label/loss)
    path=root/'benchmark/configs'/f'{label}_{loss}.json';atomic_json(path,c,immutable=True)
    atomic_json(Path(c['output'])/'preflight.json',read(original/'preflight.json'))
    return path

def probe(root,loss,label,slots,stop,port,source=None,cancel=None):
    cluster=read(root/'cluster.json');path=probe_config(root,loss,label);c=read(path);out=Path(c['output'])
    cp,step=latest(out)
    requested=out/f'step_{stop:06d}.pt'
    if requested.with_suffix('.complete.json').exists():return requested
    if cp and step>stop:raise ValueError('Requested probe checkpoint was not retained')
    if source and cp is None:
        native.migration(source,c,len(slots)*6);cp,step=latest(out)
    if cp is None and (out/'run.json').exists():(out/'run.json').rename(out/f'run.failed_start.{time.time_ns()}.json')
    parent=read(root/'parents.json')[loss]['checkpoint']
    cc,cmds=parallel.commands(cluster,slots,path,cp,parent,port,stop)
    logs=root/'benchmark/attempts'/f'{label}_{loss}_{time.time_ns()}'
    parallel.launch_checked(cc,cmds,logs,[out],cancel,stop_file=root/'STOP')
    cp,step=latest(out)
    if cp is None or step!=stop:raise RuntimeError('Probe did not reach requested step')
    return cp

def same(left,right):
    import torch
    if isinstance(left,torch.Tensor):
        return (isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape
            and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),right.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right

def compare(left,right):
    from p2_matrix16.checkpoint import checked
    a=checked(left);b=checked(right)
    return {key:same(a[key],b[key]) for key in ('model','optimizer')}

def decision(serial_means,parallel_means,equivalence,migration):
    import math
    if set(serial_means)!=set(LOSSES) or set(parallel_means)!=set(LOSSES):raise ValueError('Incomplete four-arm timing')
    if any(not math.isfinite(v) or v<=0 for v in [*serial_means.values(),*parallel_means.values()]):raise ValueError('Invalid probe timing')
    serial=sum(serial_means.values());concurrent=max(parallel_means.values())
    speedup=1-concurrent/serial
    valid=(set(equivalence)==set(LOSSES) and all(set(v)=={'model','optimizer'} and all(v.values()) for v in equivalence.values())
        and set(migration)=={'model','optimizer'} and all(migration.values()))
    return dict(policy='parallel6' if valid and speedup>=.05 else 'serial24',
        projected_time_reduction_fraction=speedup,serial_four_arm_seconds_per_step=serial,
        parallel_four_arm_seconds_per_step=concurrent,equivalence=equivalence,migration_equivalence=migration,
        serial_means=serial_means,parallel_means=parallel_means,criterion='At least 5% less projected four-arm compute time and exact model/optimizer equivalence',
        formal_probe_steps_counted=0)

def benchmark(root):
    out=root/'benchmark';out.mkdir(exist_ok=True)
    if (out/'DECISION.json').exists():return read(out/'DECISION.json')
    def status(stage,**kw):atomic_json(out/'status.json',dict(at=time.time(),stage=stage,**kw))
    serial={};concurrent={};steps=read(root/'plan.json')['probe_steps'];began=time.time()
    for i,loss in enumerate(LOSSES):
        status('serial24',loss=loss);serial[loss]=probe(root,loss,'serial24',[0,1,2,3],steps,35500+i*10)
    serial_wall=time.time()-began;status('four_groups_parallel6');began=time.time();errors=[];cancel=threading.Event();threads=[]
    def worker(i,loss):
        try:concurrent[loss]=probe(root,loss,'parallel6',[i],steps,35700+i*10,cancel=cancel)
        except Exception as exc:errors.append(dict(loss=loss,error=repr(exc)));cancel.set()
    for i,loss in enumerate(LOSSES):
        thread=threading.Thread(target=worker,args=(i,loss));threads.append(thread);thread.start()
    for thread in threads:thread.join()
    parallel_wall=time.time()-began
    if errors:raise RuntimeError('Parallel GPU probes failed: '+str(errors))
    status('checking_all_four_model_optimizer_equivalence')
    with ThreadPoolExecutor(max_workers=4) as pool:
        equivalent=dict(zip(LOSSES,pool.map(lambda loss:compare(serial[loss],concurrent[loss]),LOSSES)))
    if not all(all(v.values()) for v in equivalent.values()):
        result=dict(policy='serial24',reason='6-rank/24-rank equivalence failed',equivalence=equivalent,formal_probe_steps_counted=0)
        atomic_json(out/'DECISION.json',result,immutable=True);return result
    # Test real topology changes on the largest auxiliary branch, beyond the timed window.
    loss='av_event_feature';status('testing_6_to_12_to_24_resume')
    baseline=probe(root,loss,'serial24',[0,1,2,3],steps+2,35900)
    twelve=probe(root,loss,'migration12',[0,1],steps+1,35910,source=concurrent[loss])
    twentyfour=probe(root,loss,'migration24',[0,1,2,3],steps+2,35920,source=twelve)
    migration=compare(baseline,twentyfour);means={}
    for label in ('serial24','parallel6'):
        means[label]={}
        for loss in LOSSES:
            rows=[json.loads(line) for line in (out/'runs'/label/loss/'metrics.jsonl').read_text().splitlines()]
            selected=[m for m in rows if m['step']<=steps]
            if [m['step'] for m in selected]!=list(range(1,steps+1)):raise ValueError('Probe metric coverage mismatch')
            if not any(m['grad_norm'].get('qwen_lora',0)>0 for m in selected):raise ValueError('Missing Qwen LoRA gradients')
            means[label][loss]=statistics.mean(m['seconds'] for m in selected)
    result=decision(means['serial24'],means['parallel6'],equivalent,migration)
    result.update(serial_probe_wall_seconds=serial_wall,parallel_probe_wall_seconds=parallel_wall,
        probe_steps=steps,samples_per_arm=steps*24,scope='Guided Qwen LoRA, all four losses, duration384; no dynamic-expansion benefit assumed in projection')
    atomic_json(out/'DECISION.json',result,immutable=True);status('complete',policy=result['policy']);return result

def wait_free(cluster):
    start=time.time()
    while time.time()-start<240:
        if all(g[1]>=cluster['minimum_free_gib']*1024 for n in resources(cluster) for g in n['gpus']):return
        time.sleep(3)
    raise RuntimeError('Waiting for 24 GPUs to become free')

def release_previous(source):
    began=time.time()
    with (source/'duration_controller.lock').open('a') as previous:
        while True:
            try:fcntl.flock(previous,fcntl.LOCK_EX|fcntl.LOCK_NB);return
            except BlockingIOError:
                if time.time()-began>240:raise RuntimeError('Previous controller has not released its lock')
                time.sleep(2)

def supervise(root):
    root=Path(root);plan=read(root/'plan.json');source=Path(plan['source_root']);cluster=read(root/'cluster.json');install_backend()
    with (root/'controller.lock').open('a') as lock:
        while not (root/'STOP').exists():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:time.sleep(10);continue
            try:
                while not (root/'STOP').exists():
                    try:
                        parallel.verify(root)
                        if not (root/'parents.json').exists():
                            parents=finished_parents(source)
                            if parents is None:
                                atomic_json(root/'status.json',dict(at=time.time(),host=socket.gethostname(),pid=os.getpid(),phase='waiting_all_four_S1_finals',source_root=str(source),gpu_tasks_started=False));time.sleep(5);continue
                            # Stop the old scheduler immediately at the four-final boundary,
                            # before expensive checkpoint validation could let old S2 launch.
                            (source/'STOP').write_text('All S1 markers complete; validating handoff to '+str(root)+'\n')
                            release_previous(source)
                            validate_parents(root,parents);atomic_json(root/'parents.json',parents,immutable=True)
                        # All S1 finals are durable before stopping the old S2 scheduling path.
                        (source/'STOP').write_text('All S1 complete; handoff to '+str(root)+'\n')
                        release_previous(source)
                        if not (root/'HANDOFF.json').exists():
                            atomic_json(root/'HANDOFF.json',dict(at=time.time(),source=str(source),parents=read(root/'parents.json'),all_S1_completed=True),immutable=True)
                        if not (root/'benchmark/DECISION.json').exists():
                            wait_free(cluster);atomic_json(root/'status.json',dict(at=time.time(),phase='GPU_benchmark',host=socket.gethostname(),pid=os.getpid()))
                            try:benchmark(root)
                            except Exception as exc:
                                if (root/'STOP').exists():return
                                atomic_json(root/'benchmark/ERROR.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()))
                                atomic_json(root/'benchmark/DECISION.json',dict(policy='serial24',reason='Benchmark failed; preserve validated serial execution',error=repr(exc),formal_probe_steps_counted=0),immutable=True)
                        atomic_json(root/'status.json',dict(at=time.time(),phase='formal_S2_training',decision=read(root/'benchmark/DECISION.json'),host=socket.gethostname(),pid=os.getpid()))
                        parallel.run(root)
                        if (root/'STOP').exists():return
                        atomic_json(root/'status.json',dict(at=time.time(),phase='complete',host=socket.gethostname(),pid=os.getpid()));return
                    except Exception as exc:
                        atomic_json(root/'controller_error.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()));time.sleep(10)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

def prepare(source,root):
    install_backend();root.mkdir(parents=True,exist_ok=True)
    cluster=read(source/'cluster.json');atomic_json(root/'cluster.json',cluster,immutable=True)
    plan=dict(source_root=str(source.resolve()),probe_steps=16,policy_candidates=['serial24','parallel6'],
        switch_after='all_four_frozen_Qwen_S1_final_checkpoints',global_batch=24,steps=1249,seed=20261023,
        benchmark_failure_fallback='serial24',parallel_repeated_failure_fallback='graceful_save_then_serial24')
    atomic_json(root/'plan.json',plan,immutable=True)
    sampling=[]
    for loss in LOSSES:
        c=read(source/'configs'/f'guided_{loss}_s2.json');original=Path(c['output']);c['output']=str(root/'runs'/f'guided_{loss}_s2')
        if c['training']['steps']!=1249 or c['training']['global_batch']!=24 or c['training']['seed']!=20261023:raise ValueError('Unexpected stage budget')
        Schedule(c);sampling.append(c['duration_sampling'])
        atomic_json(root/'configs'/f'guided_{loss}_s2.json',c,immutable=True)
        atomic_json(Path(c['output'])/'preflight.json',read(original/'preflight.json'),immutable=True)
    if any(s!=sampling[0] for s in sampling):raise ValueError('Four arms have different sampling')
    frozen=read(source/'training_queue/frozen.json')
    for p in [Path(__file__),ROOT/'scripts/p2_omni_accel_parallel.py',root/'plan.json',root/'cluster.json',*list((root/'configs').glob('*.json'))]:frozen[str(p.resolve())]=sha(p)
    atomic_json(root/'FROZEN.json',frozen,immutable=True)
    atomic_json(source/'OMNI_ACCELERATION_HANDOFF.json',dict(target=str(root.resolve()),condition=plan['switch_after'],current_S1_unchanged=True))

def launch_watchers(root):
    cluster=read(root/'cluster.json');records=[]
    for node in cluster['nodes']:
        folder=root/'controllers';folder.mkdir(exist_ok=True);log=folder/(node['name']+'.log')
        code="import subprocess,sys; f=open(sys.argv[1],'a'); p=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)"
        cmd=[cluster['python'],'-c',code,str(log),'env','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',cluster['python'],str(Path(__file__).resolve()),'watch','--root',str(root)]
        for attempt in range(4):
            try:
                value=subprocess.check_output(transport(cluster,node,cmd),text=True,timeout=45);records.append(dict(node=node['name'],pid=int(value.strip().splitlines()[-1])));break
            except subprocess.SubprocessError:
                if attempt==3:raise
                time.sleep(2)
    atomic_json(root/'controllers/processes.json',records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','watch','launch']);p.add_argument('--root',required=True,type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    if a.command=='prepare':
        if a.source is None:p.error('--source required for prepare')
        prepare(a.source.resolve(),a.root.resolve())
    elif a.command=='launch':launch_watchers(a.root.resolve())
    else:supervise(a.root.resolve())
