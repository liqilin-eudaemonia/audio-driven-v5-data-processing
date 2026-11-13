#!/usr/bin/env python3
"""Durable ABBA 24-rank benchmark, automatic adopt-or-resume, isolated outputs."""
from pathlib import Path
import argparse,copy,json,os,statistics,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_duration_execution import schedule,sha,install_backend
from p2_listen.contracts import read,atomic_json
from p2_guided_chain import backend
from p2_guided_chain.training_queue import commands
from listen_cluster import launch,transport
from p2_legacy_qwen_queue import resources

def start_controllers(cluster,root,script):
    reports=[];folder=root/'controllers';folder.mkdir(parents=True,exist_ok=True)
    for node in cluster['nodes']:
        log=folder/(node['name']+'.log')
        code="import subprocess,sys; f=open(sys.argv[1],'a'); p=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)"
        cmd=[cluster['python'],'-c',code,str(log),'env','OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2',cluster['python'],str(script),'--root',str(root)]
        for attempt in range(3):
            try:
                value=subprocess.check_output(transport(cluster,node,cmd),text=True,timeout=45)
                reports.append(dict(node=node['name'],pid=int(value.strip().splitlines()[-1])));break
            except subprocess.SubprocessError:
                if attempt==2:raise
                time.sleep(2)
    atomic_json(folder/'processes.json',reports)

def wait_free(cluster,timeout=180):
    began=time.time()
    while time.time()-began<timeout:
        if all(g[1]>=135*1024 for node in resources(cluster) for g in node['gpus']):return
        time.sleep(3)
    raise RuntimeError('24 GPUs did not release after graceful checkpoint')

def equal_tensors(left,right):
    import torch
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal_tensors(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(equal_tensors(a,b) for a,b in zip(left,right))
    return left==right

def main(old,new):
    work=old/'performance/duration384';work.mkdir(parents=True,exist_ok=True)
    cluster=read(old/'cluster.json');paused=False;adopted=False;original_output=None
    def status(stage,**kw):atomic_json(work/'status.json',dict(at=time.time(),stage=stage,**kw))
    def restore():
        if original_output:(original_output/'STOP').unlink(missing_ok=True)
        (old/'STOP').unlink(missing_ok=True)
        start_controllers(cluster,old,ROOT/'scripts/p2_guided_chain_director.py')
    try:
        rows=read(old/'view/observations.json');obs_sha=sha(old/'view/observations.json')
        base=read(old/'configs/guided_av_s1.json');t=base['training'];specs={}
        for policy in ('random','duration384'):
            path=work/(policy+'.schedule.json')
            atomic_json(path,dict(observations_sha256=obs_sha,pairs=schedule(rows,t['steps'],24,t['seed'],policy)),immutable=True)
            specs[policy]=dict(policy=policy,window=384,manifest=str(path),sha256=sha(path),observations_sha256=obs_sha)
        # Build every probe before taking GPUs from the production run.
        install_backend();configs={}
        for name,policy in [('a1','random'),('b1','duration384'),('b2_resume','duration384'),('a2','random')]:
            c=copy.deepcopy(base);c['duration_sampling']=specs[policy];c['output']=str(work/name)
            path=work/(name+'.json');atomic_json(path,c,immutable=True);configs[name]=path
            pre=read(Path(base['output'])/'preflight.json');pre['source']=backend.source_identity()
            atomic_json(Path(c['output'])/'preflight.json',pre,immutable=True)
        state=read(old/'training_queue/state.json');job=state['jobs']['av']
        if job['stage']!=1 or job['status']!='running':raise RuntimeError('Unexpected original experiment; do not interrupt')
        original_output=Path(job['output']);before=job['step'];began=time.time()
        status('saving_original',step=before);(original_output/'STOP').touch();paused=True
        while True:
            state=read(original_output/'status.json');step=state['step']
            marker=original_output/f'step_{step:06d}.complete.json'
            if step>=before and marker.exists() and marker.stat().st_mtime>=began:break
            if time.time()-began>180:raise RuntimeError('Graceful checkpoint timed out')
            time.sleep(1)
        (old/'STOP').write_text('Duration-alignment ABBA benchmark; original checkpoint retained.\n')
        atomic_json(work/'ORIGINAL_SAVED.json',dict(step=step,checkpoint=str(marker.with_suffix('').with_suffix('.pt')),marker=str(marker)))
        wait_free(cluster)
        for run_index,(name,path) in enumerate(configs.items()):
            c=read(path);out=Path(c['output']);status('benchmark',run=name,world=24)
            segments=[(8,None),(16,out/'step_000008.pt')] if name=='b2_resume' else [(16,None)]
            for segment,(stop,resume) in enumerate(segments):
                cc,cmds=commands(cluster,[0,1,2,3],path,resume,None,33400+run_index*10+segment)
                for cmd in cmds:
                    index=cmd.index(str(ROOT/'scripts/p2_guided_chain_train.py'));cmd[index]=str(ROOT/'scripts/p2_duration_train.py')
                    cmd+=['--stop-after',str(stop),'--profile',str(out/'timings')]
                launch(cc,cmds,work/'attempts'/f'{name}_{segment}')
        from p2_matrix16.checkpoint import checked
        left=checked(work/'b1/step_000016.pt');right=checked(work/'b2_resume/step_000016.pt')
        equivalence={key:equal_tensors(left[key],right[key]) for key in ('model','optimizer')}
        del left,right
        timing={}
        for name in configs:
            metrics=[json.loads(line) for line in (work/name/'metrics.jsonl').read_text().splitlines()]
            if [x['step'] for x in metrics]!=list(range(1,17)):raise RuntimeError('Missing/duplicated benchmark steps')
            timing[name]=dict(mean_seconds=statistics.mean(x['seconds'] for x in metrics),total_seconds=sum(x['seconds'] for x in metrics),peak_gib=max(x['peak_reserved_gib'] for x in metrics))
        base_mean=statistics.mean(timing[n]['mean_seconds'] for n in ('a1','a2'))
        aligned_mean=statistics.mean(timing[n]['mean_seconds'] for n in ('b1','b2_resume'))
        gain=1-aligned_mean/base_mean
        decision=dict(timing=timing,baseline_mean_seconds=base_mean,aligned_mean_seconds=aligned_mean,time_reduction_fraction=gain,
                      strict_resume_bitwise_equal=equivalence,adopt=gain>=.05 and all(equivalence.values()),world=24,
                      samples_per_probe=384,formal_steps_counted=0,all_eight_use_same_schedule=True)
        atomic_json(work/'RESULT.json',decision)
        if not decision['adopt']:
            restore();paused=False;status('original_resumed',result=decision);return
        # New version: all arms start from common initialization, no random-order prefix.
        new.mkdir(parents=True,exist_ok=True)
        for name in ('cluster.json','pipeline.json'):
            atomic_json(new/name,read(old/name),immutable=True)
        for loss in ('av','av_event','av_feature','av_event_feature'):
            for stage in (1,2):
                name=f'guided_{loss}_s{stage}';c=read(old/'configs'/(name+'.json'))
                original=Path(c['output']);c['duration_sampling']=specs['duration384'];c['output']=str(new/'runs'/name)
                atomic_json(new/'configs'/(name+'.json'),c,immutable=True)
                pre=read(original/'preflight.json');pre['source']=backend.source_identity()
                atomic_json(Path(c['output'])/'preflight.json',pre,immutable=True)
        atomic_json(new/'configs/matrix.json',read(old/'configs/matrix.json'),immutable=True)
        atomic_json(new/'TRAINING_READY.json',dict(benchmark=str(work/'RESULT.json'),full_cache_audit=str(old/'CACHE_VALIDATED.json'),
                    full_data_audit=str(old/'RESTORE_AUDIT.json'),native_gpu_s0=str(old/'s0/COMPLETE.json'),samples=29975,steps=1249,
                    global_batch=24,original_prefix_archived=str(work/'ORIGINAL_SAVED.json'),source=backend.source_identity()),immutable=True)
        start_controllers(cluster,new,ROOT/'scripts/p2_duration_queue.py');adopted=True
        atomic_json(old/'DURATION_REPLACEMENT.json',dict(root=str(new),result=str(work/'RESULT.json'),original_run_archived=True))
        status('aligned_training_queued',root=str(new),result=decision)
    except BaseException as exc:
        atomic_json(work/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=time.time()))
        if paused and not adopted:
            restore();status('error_original_resumed',error=repr(exc))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True,type=Path);p.add_argument('--new',required=True,type=Path);a=p.parse_args()
    main(a.old,a.new)
