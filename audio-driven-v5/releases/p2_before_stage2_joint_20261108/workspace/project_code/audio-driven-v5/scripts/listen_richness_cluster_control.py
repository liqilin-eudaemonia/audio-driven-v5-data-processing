#!/usr/bin/env python3
"""Bounded 32-GPU benchmark and one-shot Omni -> statistics handoff."""
import argparse,concurrent.futures,hashlib,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from listen_benchmark.j_annotation import read,write,acquire
PYTHON='/private/software/conda/envs/vllm/bin/python'


def remote(work,node,body,timeout=120):
    cfg=read(work/'config.json');n=cfg['nodes'][node]
    code='WORK='+repr(str(work))+';NODE='+repr(node)+';ROOT='+repr(str(ROOT))+'\n'+body
    cmd=[sys.executable,'-'] if socket.gethostname()==n['hostname'] else ['ssh','-i',n['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/richness-%C','root@'+n['host'],'python3 -']
    for retry in range(3):
        try:
            p=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=timeout)
            if p.returncode:raise RuntimeError(p.stderr[-3000:])
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            if retry==2:raise
            time.sleep(2)


def all_nodes(work,body):
    names=list(read(work/'config.json')['nodes'])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        values=list(pool.map(lambda n:remote(work,n,body),names))
    return dict(zip(names,values))


REMOTE_COMMON='''import os,json,time,signal,subprocess
from pathlib import Path
W=Path(WORK);cfg=json.loads((W/'config.json').read_text());U=Path(cfg['upstream'])
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v));t.replace(p)
def ticks(pid):
 try:
  parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return None if parts[0] in ('Z','X') else int(parts[19])
 except FileNotFoundError:return None
def alive(s):return s is not None and s.get('start_ticks') is not None and ticks(s['pid'])==int(s['start_ticks'])
def signal_owned(s,needle,sig=signal.SIGTERM):
 if not alive(s):return False
 cmd=(Path('/proc')/str(s['pid'])/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if needle not in cmd:raise RuntimeError('Process identity differs: '+cmd)
 os.kill(s['pid'],sig);return True
def environment(gpu=None):
 e=dict(os.environ,LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn')
 if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
 return e
def spawn(cmd,log,env):
 log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
 return dict(pid=p.pid,start_ticks=ticks(p.pid),cmd=cmd,started_at=time.time(),log=str(log))
def devices():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
 return {int(x.split(',')[0]):list(map(float,x.split(',')[1:])) for x in raw.strip().splitlines()}
'''


def pause_annotation(work):
    body=REMOTE_COMMON+'''
nw=U/'nodes'/NODE;marker=nw/'STOP_SUPERVISION';tag='richness:'+str(W)
if marker.exists() and marker.read_text()!=tag:raise RuntimeError('Existing external supervision stop')
save(W/'maintenance'/f'{NODE}-before.json',dict(at=time.time(),supervisor=read(nw/'supervisor.json'),watchdog=read(nw/'watchdog.json')))
marker.write_text(tag)
sup=read(nw/'supervisor.json');watch=read(nw/'watchdog.json')
for s,role in [(watch,'watchdog'),(sup,'supervise')]:signal_owned(s,'listen_j_omni.py '+role+' --work '+str(U)+' --node '+NODE)
deadline=time.time()+50
while (alive(sup) or alive(watch)) and time.time()<deadline:time.sleep(.5)
if alive(sup) or alive(watch):raise RuntimeError('Supervisor did not stop')
sup=read(nw/'supervisor.json');stopped=[]
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 s=sup['children'].get('gpu-'+str(gpu))
 if s and signal_owned(s,'listen_j_omni.py worker --work '+str(U)+' --node '+NODE+' --gpu '+str(gpu)):stopped.append(gpu)
save(W/'maintenance'/f'{NODE}-paused.json',dict(at=time.time(),gpus=stopped))
print(json.dumps(dict(node=NODE,gpus=stopped)))
'''
    return all_nodes(work,body)


def resume_annotation(work):
    return all_nodes(work,REMOTE_COMMON+'''
nw=U/'nodes'/NODE;marker=nw/'STOP_SUPERVISION';tag='richness:'+str(W)
if marker.exists():
 if marker.read_text()!=tag:raise RuntimeError('Foreign stop marker; refusing removal')
 marker.unlink()
prior=read(nw/'watchdog.json') if (nw/'watchdog.json').exists() else None
if not alive(prior):
 child=spawn(['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/listen_j_omni.py','watchdog','--work',str(U),'--node',NODE],nw/'logs/richness-resume-watchdog.log',environment())
else:child=prior
save(W/'maintenance'/f'{NODE}-resumed.json',dict(at=time.time(),child=child))
print(json.dumps(dict(node=NODE,child=child)))
''')


def free_devices(work):return all_nodes(work,REMOTE_COMMON+"print(json.dumps(devices()))\n")


def wait_free(work,seconds=600):
    deadline=time.time()+seconds
    while True:
        states=free_devices(work)
        if all(v[0]<1024 for ns in states.values() for v in ns.values()):return states
        if time.time()>deadline:raise TimeoutError('GPUs did not become free; no other process will be killed')
        time.sleep(5)


def launch(work,phase):
    return all_nodes(work,'PHASE='+repr(phase)+'\n'+REMOTE_COMMON+'''
nodework=W/'nodes'/NODE;device=devices();children={}
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 p=nodework/(PHASE+'-gpu-'+str(gpu)+'.json');old=read(p) if p.exists() else None
 if alive(old):children[str(gpu)]=old;continue
 if device[gpu][0]>1024:continue
 cmd=['/private/software/conda/envs/vllm/bin/python',ROOT+'/scripts/listen_richness_cluster.py','worker','--work',str(W),'--node',NODE,'--gpu',str(gpu)]
 if PHASE=='profiles':cmd+=['--profile',list(cfg['profiles'])[gpu%4],'--benchmark-input',str(W/'benchmark_inputs.jsonl')]
 child=spawn(cmd,nodework/'logs'/(PHASE+'-gpu-'+str(gpu)+'.log'),environment(gpu));save(p,child);children[str(gpu)]=child
print(json.dumps(children))
''')


def stop_own(work,phase):
    return all_nodes(work,'PHASE='+repr(phase)+'\n'+REMOTE_COMMON+'''
stopped=[]
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 p=W/'nodes'/NODE/(PHASE+'-gpu-'+str(gpu)+'.json')
 if p.exists() and signal_owned(read(p),'listen_richness_cluster.py worker --work '+str(W)+' --node '+NODE+' --gpu '+str(gpu)):stopped.append(gpu)
print(json.dumps(stopped))
''')


def resource_sampler(work,node):
    from listen_benchmark.j_annotation import acquire
    lock=acquire(work/'resources'/f'{node}.lock',False)
    if lock is None:return
    def usage():return int(dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
    old=usage();t=time.time()
    try:
        with (work/'resources'/f'{node}.jsonl').open('a',buffering=1) as f:
            while not (work/'STOP_RESOURCES').exists():
                now=time.time();value=usage()
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
                cpu=(value-old)/1e6/(now-t) if now>t else 0;old=value;t=now
                f.write(json.dumps(dict(at=now,cpu_cores=cpu,loadavg=os.getloadavg(),gpus=[list(map(float,line.split(','))) for line in raw.strip().splitlines()]))+'\n');time.sleep(2)
    finally:lock.close()


def start_resources(work):
    (work/'STOP_RESOURCES').unlink(missing_ok=True)
    return all_nodes(work,REMOTE_COMMON+'''
p=W/'nodes'/NODE/'resource-child.json';old=read(p) if p.exists() else None
if alive(old):child=old
else:
 child=spawn(['python3',ROOT+'/scripts/listen_richness_cluster_control.py','resources','--work',str(W),'--node',NODE],W/'nodes'/NODE/'logs/resources.log',environment());save(p,child)
print(json.dumps(child))
''')


def workers(work):
    c=read(work/'config.json');out={}
    for n,v in c['nodes'].items():
        for g in v['gpu_ids']:
            p=work/'workers'/f'{n}-gpu-{g}.json'
            if p.exists():out[p.stem]=read(p)
    return out


def ensure_preparer(work):
    if (work/'producer_complete.json').exists():return
    path=work/'preparation-child.json'
    if path.exists():
        old=read(path);p=Path('/proc')/str(old['pid'])/'stat'
        if p.exists():
            parts=p.read_text().rsplit(')',1)[1].split()
            if parts[0] not in ('Z','X') and int(parts[19])==old['start_ticks']:return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             TOKENIZERS_PARALLELISM='false',LD_PRELOAD='/private/software/conda/envs/vllm/lib/libstdc++.so.6')
    cmd=[PYTHON,str(ROOT/'scripts/listen_richness_cluster.py'),'prepare','--work',str(work)]
    with (work/'prepare.log').open('a') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
    ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
    write(path,dict(pid=child.pid,start_ticks=ticks,cmd=cmd,started_at=time.time()))


def choose_profile(work):
    c=read(work/'config.json');summaries={};scores={}
    for label,profile in c['profiles'].items():
        ss=[]
        for n in c['nodes']:
            for g in c['nodes'][n]['gpu_ids']:
                if list(c['profiles'])[g%4]!=label:continue
                p=work/'benchmark'/f'{n}-gpu-{g}'/'summary.json'
                if p.exists():ss.append(read(p))
        valid=len(ss)==8 and all(s['receipt']['unresolved']==0 and s['receipt'].get('incomplete_generations',0)==0 for s in ss)
        summaries[label]=dict(profile=profile,valid=valid,workers=ss,
            eight_card_clips_per_second=sum(s['clips_per_second'] for s in ss),
            strict_failures=sum(s['receipt']['strict_failed'] for s in ss))
        if valid:scores[label]=summaries[label]['eight_card_clips_per_second']
    if not scores:raise RuntimeError('No complete valid eight-GPU profile')
    best=max(scores,key=scores.get);c['selected_profile']=best;write(work/'config.json',c)
    report=dict(at=time.time(),selected_profile=best,profiles=summaries,
                estimate_32_cards_clips_per_second=scores[best]*4,extrapolation_not_final_measurement=True)
    write(work/'profile_selection.json',report);return report


def benchmark(work):
    lock=acquire(work/'benchmark-control.lock',False)
    if lock is None:raise RuntimeError('Benchmark already running')
    paused=False;phase='profiles'
    def interrupted(*_):raise KeyboardInterrupt('Controlled benchmark interruption')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        assert (work/'cpu_validation.json').exists() and read(work/'cpu_validation.json')['passed']
        assert (work/'benchmark_inputs.jsonl').exists()
        ensure_preparer(work)
        # Prepare enough real production requests before taking GPUs away from
        # annotation. The CPU packer continues during profile measurements.
        while True:
            p=work/'preparation.json'
            if p.exists() and read(p)['packed_chunks']>=64:break
            ensure_preparer(work);time.sleep(10)
        paused=True;write(work/'benchmark_control.json',dict(stage='pausing_annotation',at=time.time(),pid=os.getpid()))
        write(work/'annotation_pause.json',pause_annotation(work));wait_free(work);start_resources(work)
        launched=launch(work,'profiles');write(work/'profile_launch.json',launched)
        deadline=time.time()+1200
        while True:
            ss=workers(work);finished=sum(s['pid']==launched[s['node']][str(s['gpu'])]['pid'] and s['stage'] in ('complete','error') for s in ss.values())
            write(work/'benchmark_control.json',dict(stage='profile_test',at=time.time(),pid=os.getpid(),finished=finished))
            if finished==32:break
            if time.time()>deadline:raise TimeoutError('Profile benchmark deadline')
            time.sleep(5)
        selection=choose_profile(work);wait_free(work)
        # Same verified production chunks; completed outputs will be reused later.
        qpath=work/'queues/extract/state.json';before=read(qpath)['completed_clips'] if qpath.exists() else 0
        (work/'DRAIN').unlink(missing_ok=True);(work/'WAIT_START').write_text('all engines ready before timing')
        phase='production';launched=launch(work,phase);write(work/'validation_launch.json',launched)
        deadline=time.time()+600
        while True:
            ss=workers(work)
            ready=[s for k,s in ss.items() if s['pid']==launched[s['node']][str(s['gpu'])]['pid'] and s['stage']=='ready']
            if len(ready)==32:break
            if any(s['stage']=='error' for s in ss.values()):raise RuntimeError('Production validation worker failed')
            if time.time()>deadline:raise TimeoutError('32-card load barrier')
            time.sleep(3)
        started=time.time();(work/'WAIT_START').unlink()
        write(work/'benchmark_control.json',dict(stage='32_card_production_window',at=started,pid=os.getpid()))
        deadline=started+180
        while time.time()<deadline:
            if any(s['stage']=='error' for s in workers(work).values()):raise RuntimeError('Production window failed')
            time.sleep(5)
        (work/'DRAIN').write_text('finish current reservoir and preserve every result')
        deadline=time.time()+900
        while True:
            ss=workers(work)
            if all(s['stage'] in ('complete','stopped') for s in ss.values()):break
            if any(s['stage']=='error' for s in ss.values()):raise RuntimeError('Production drain failed')
            if time.time()>deadline:raise TimeoutError('Statistics drain deadline')
            time.sleep(5)
        ended=time.time();q=read(qpath);count=q['completed_clips']-before
        if q['failed'] or count<=0:raise RuntimeError('Invalid end-to-end benchmark')
        measured=count/(ended-started)
        write(work/'benchmark_32_result.json',dict(at=ended,profile=selection['selected_profile'],started_at=started,
            ended_at=ended,completed=count,seconds=ended-started,clips_per_second=measured,
            output_reused_in_production=True,scope='32 cards, prepared input read + inference + retries + statistics persistence',
            conservative_all_candidate_hours=read(work/'config.json')['expected_upstream_clips']/measured/3600,
            workers=ss))
    except BaseException as exc:
        write(work/'benchmark_control.json',dict(stage='error',at=time.time(),pid=os.getpid(),error=repr(exc)))
        raise
    finally:
        (work/'DRAIN').write_text('maintenance ends');(work/'WAIT_START').unlink(missing_ok=True)
        if paused:
            try:stop_own(work,phase);wait_free(work,600)
            finally:write(work/'annotation_resume.json',resume_annotation(work))
        (work/'STOP_RESOURCES').write_text('finished');lock.close()
    write(work/'benchmark_control.json',dict(stage='complete',at=time.time(),pid=os.getpid(),result=str(work/'benchmark_32_result.json'),annotation_resumed=True))


def upstream_done(work):
    c=read(work/'config.json');u=Path(c['upstream']);p=u/'queues/caption/state.json'
    if not (u/'producer_complete.json').exists() or not p.exists():return False
    q=read(p)
    return not q['inflight'] and not q.get('deferred') and not q['failed'] and q['offset']>=(u/'ready.jsonl').stat().st_size


def chain(work):
    lock=acquire(work/'chain.lock',False)
    if lock is None:return
    try:
        while not upstream_done(work):
            ensure_preparer(work)
            write(work/'chain.json',dict(stage='waiting_for_annotation',at=time.time(),pid=os.getpid()));time.sleep(20)
        wait_free(work,1200)
        (work/'DRAIN').unlink(missing_ok=True);(work/'WAIT_START').unlink(missing_ok=True)
        c=read(work/'config.json')
        if c['selected_profile'] is None:raise ValueError('No tested profile selected')
        restarts={};launch(work,'production')
        while True:
            ensure_preparer(work)
            prep=read(work/'preparation.json') if (work/'preparation.json').exists() else {}
            if prep.get('source_errors'):raise RuntimeError('Input sources need recovery: '+str(prep['source_errors'][:5]))
            q=read(work/'queues/extract/state.json')
            write(work/'chain.json',dict(stage='extracting',at=time.time(),pid=os.getpid(),resolved=q['completed_clips'],failed=q['failed']))
            if (work/'producer_complete.json').exists() and not q['inflight'] and q['offset']>=(work/'ready.jsonl').stat().st_size:
                if q['failed']:raise RuntimeError('Unresolved extraction chunks: '+str(q['failed']))
                break
            # Fresh processes use the same per-chunk locks and committed output receipts.
            ss=workers(work)
            for name,s in ss.items():
                if s['stage']=='error':
                    key=(name,s['pid'])
                    if key not in restarts:
                        restarts[key]=time.time()
                        if sum(k[0]==name for k in restarts)>3:raise RuntimeError('Repeated worker failure: '+name)
            launch(work,'production');time.sleep(20)
        wait_free(work,1200)
        from listen_richness.merge_cluster import merge
        write(work/'chain.json',dict(stage='merging',at=time.time(),pid=os.getpid()));merge(work)
        write(work/'chain.json',dict(stage='complete',at=time.time(),pid=os.getpid()))
    except BaseException as exc:
        write(work/'chain.json',dict(stage='error',at=time.time(),pid=os.getpid(),error=repr(exc)))
        try:stop_own(work,'production')
        except Exception as cleanup:write(work/'chain_cleanup_error.json',dict(at=time.time(),error=repr(cleanup)))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['benchmark','chain','resources']);p.add_argument('--work',type=Path,required=True);p.add_argument('--node')
    a=p.parse_args();w=a.work.resolve()
    if a.command=='benchmark':benchmark(w)
    elif a.command=='chain':chain(w)
    else:resource_sampler(w,a.node)
