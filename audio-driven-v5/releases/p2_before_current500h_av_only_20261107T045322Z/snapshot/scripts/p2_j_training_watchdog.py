"""Two-host watchdog for the authorized AV+Event -> pure-AV 32-GPU queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=Path(__file__).resolve()
DEFAULT=ROOT/'experiments/p2_j_training_watchdog_20261101'
QUEUE=ROOT/'experiments/p2_j_av_event_then_av_20261101'
from listen_cluster import transport,launch


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:
        if default is not None:return default
        raise


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def process_info(pid):
    p=Path('/proc')/str(pid)
    try:
        raw=(p/'cmdline').read_bytes();args=[x.decode() for x in raw.split(b'\0') if x]
        stat=(p/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        birth=int(fields[19])
        return dict(pid=int(pid),args=args,start_ticks=birth,command_sha256=hashlib.sha256(raw).hexdigest(),
                    age_s=time.clock_gettime(time.CLOCK_BOOTTIME)-birth/os.sysconf('SC_CLK_TCK'))
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None


def classify_process(info,job):
    args=info['args']
    if str(ROOT/'scripts/p2_j_av_event_then_av.py') in args and 'run' in args:return 'controller'
    if str(SCRIPT) in args and 'supervise' in args:return 'controller'
    if '--guard' in args or 'torch.distributed.run' in args or '--config' not in args:return None
    for phase,entry,key in [('av_event','p2_j_duration_train.py','av_event_config'),('av_only','p2_j_av_only.py','av_config')]:
        if str(ROOT/'scripts'/entry) in args and 'train' in args:
            index=args.index('--config')
            if index+1<len(args) and Path(args[index+1]).resolve()==Path(job[key]).resolve():return phase
    return None


def probe(work):
    cfg=read(work/'config.json');job=read(Path(cfg['queue'])/'job.json')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus=[dict(index=int(x[0]),uuid=x[1].strip(),used_mib=float(x[2]),free_mib=float(x[3]),
               utilization=float(x[4]),power_w=float(x[5])) for x in csv.reader(gpu.splitlines())]
    processes=[]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                 text=True,timeout=15)
    selected={int(x) for x in apps.splitlines() if x.strip().isdigit()}
    controller=read(Path(cfg['queue'])/'process.json',{})
    if controller.get('pid'):selected.add(int(controller['pid']))
    for pid in selected:
        p=Path('/proc')/str(pid)
        info=process_info(pid)
        if info is None:continue
        role=classify_process(info,job)
        if role is None:continue
        if role!='controller':
            try:
                env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
                info.update(local_rank=int(env[b'LOCAL_RANK']),rank=int(env[b'RANK']))
            except (KeyError,FileNotFoundError,ProcessLookupError):continue
        info.pop('args');info['role']=role;processes.append(info)
    return dict(host=socket.gethostname(),at=time.time(),gpus=gpus,processes=processes)


def terminate_owned(work,targets,force=False):
    job=read(Path(read(work/'config.json')['queue'])/'job.json');killed=[]
    for old in targets:
        now=process_info(old['pid'])
        if (now is not None and now['start_ticks']==old['start_ticks'] and
                now['command_sha256']==old['command_sha256'] and classify_process(now,job) is not None):
            os.kill(now['pid'],signal.SIGKILL if force else signal.SIGTERM);killed.append(now['pid'])
    return killed


def remote(cluster,node,work,command,*extra):
    result=subprocess.run(transport(cluster,node,[cluster['python'],str(SCRIPT),command,'--work',str(work),*extra]),
                          text=True,capture_output=True,timeout=40)
    if result.returncode:raise RuntimeError(f"{node['name']}: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


def inspect_cluster(cluster,work):
    def one(node):
        try:return dict(node=node['name'],ok=True,**remote(cluster,node,work,'probe'))
        except (subprocess.SubprocessError,OSError,ValueError,RuntimeError) as exc:
            return dict(node=node['name'],ok=False,error=repr(exc),at=time.time())
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,cluster['nodes']))


def latest_checkpoint(output,cache=None):
    """Ignore incomplete writes; verify hashes and fall back if newest is corrupt."""
    cache={} if cache is None else cache
    rejected=[]
    for marker in sorted(Path(output).glob('step_*.complete.json'),reverse=True):
        try:
            step=int(marker.name.split('_')[1].split('.')[0]);meta=read(marker)
            path=marker.with_name(f'step_{step:06d}.pt');stat=path.stat()
            if meta['step']!=step:raise ValueError('step mismatch')
            key=f'{path}:{stat.st_size}:{stat.st_mtime_ns}:{meta["sha256"]}'
            if cache.get(key) is not True:
                if digest_file(path)!=meta['sha256']:raise ValueError('checksum mismatch')
                cache[key]=True
            return dict(path=str(path),step=step,sha256=meta['sha256'],rejected_newer=rejected)
        except (OSError,ValueError,KeyError) as exc:rejected.append(dict(marker=str(marker),error=repr(exc)))
    return None


def observe(job,cluster,work,cache):
    phases={}
    for phase,key in [('av_event','av_event_output'),('av_only','av_output')]:
        output=Path(job[key]);checkpoint=latest_checkpoint(output,cache)
        stat=read(output/'status.json',{})
        metrics=output/'metrics.jsonl'
        phases[phase]=dict(step=stat.get('step',0),checkpoint=checkpoint,
            metrics_mtime=metrics.stat().st_mtime if metrics.exists() else 0,
            complete=bool(checkpoint and checkpoint['step']==job['budget']))
    phase='av_only' if phases['av_event']['complete'] else 'av_event'
    return dict(at=time.time(),phase=phase,phases=phases,nodes=inspect_cluster(cluster,work),
                queue=read(QUEUE/'state.json',{}),both_complete=all(x['complete'] for x in phases.values()))


def decide(snapshot,previous,cfg,now):
    """Transient observation failures never justify killing or duplicating jobs."""
    if snapshot['both_complete']:return 'complete',previous
    if not all(n['ok'] and len(n['gpus'])==8 for n in snapshot['nodes']):return 'observation_unavailable',previous
    phase=snapshot['phase'];current=snapshot['phases'][phase]
    workers=[p for n in snapshot['nodes'] for p in n['processes'] if p['role']==phase]
    full=(len(workers)==32 and sorted(p['rank'] for p in workers)==list(range(32)) and
          all(sorted(p['local_rank'] for p in n['processes'] if p['role']==phase)==list(range(8)) for n in snapshot['nodes']))
    old=previous.get('progress',{})
    moved=(old.get('phase'),old.get('step'))!=(phase,current['step'])
    changed=now if moved else old.get('changed_at',now)
    state=dict(previous,progress=dict(phase=phase,step=current['step'],changed_at=changed))
    controllers=[p for n in snapshot['nodes'] for p in n['processes'] if p['role']=='controller']
    queue=snapshot['queue'];phase_stage='av_event_running' if phase=='av_event' else 'av_only_running'
    starting=bool(controllers) and (min(p['age_s'] for p in controllers)<cfg['startup_grace_s'] or
        queue.get('stage') in (phase_stage,'waiting_release_for_av_only') and now-queue.get('at',0)<cfg['startup_grace_s'])
    if full and now-max(changed,current['metrics_mtime'])<cfg['stall_s']:
        state['unhealthy_count']=0;return 'healthy',state
    if starting:
        state['unhealthy_count']=0;return 'starting',state
    state['unhealthy_count']=previous.get('unhealthy_count',0)+1
    if state['unhealthy_count']<2:return 'confirming_stall' if full else 'confirming_missing_workers',state
    return 'recover_stall' if full else 'recover_missing_workers',state


def recovery(work,cfg,cluster,snapshot):
    """Stop only verified members of this queue, then launch a resumable controller."""
    started=time.time();action=work/'recoveries'/str(time.time_ns())
    write(action/'request.json',dict(at=started,phase=snapshot['phase'],snapshot=snapshot))
    def stop(rows,roles,force=False):
        tasks=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for node,row in zip(cluster['nodes'],rows):
                targets=[p for p in row['processes'] if p['role'] in roles]
                if targets:tasks.append(pool.submit(remote,cluster,node,work,'terminate','--targets',json.dumps(targets),
                                                  *(['--force'] if force else [])))
            for task in tasks:task.result()
    stop(snapshot['nodes'],{'controller'})
    current=snapshot['nodes']
    for attempt in range(12):
        time.sleep(5);current=inspect_cluster(cluster,work)
        if not all(x['ok'] for x in current):
            write(action/'deferred.json',dict(at=time.time(),reason='observation unavailable after controller stop'));return
        if not any(x['processes'] for x in current):break
        if attempt==3:stop(current,{'av_event','av_only'})
        if attempt==7:stop(current,{'av_event','av_only','controller'},force=True)
    if any(x['processes'] for x in current) or any(g['free_mib']<135*1024 for x in current for g in x['gpus']):
        write(action/'deferred.json',dict(at=time.time(),reason='owned workers or GPU allocations have not released'));return
    if (work/'STOP').exists() or (QUEUE/'STOP').exists():return
    log=(action/'controller.log').open('a')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    p=subprocess.Popen([cluster['python'],'-u',str(SCRIPT),'supervise','--work',str(work)],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    info=dict(pid=p.pid,host=socket.gethostname(),at=time.time(),command='p2_j_training_watchdog.py supervise')
    write(QUEUE/'process.json',info);write(action/'launched.json',info)


def supervise(work):
    """Resume either phase from its newest complete checkpoint; preserve stage order."""
    cfg=read(work/'config.json');job=read(QUEUE/'job.json');cluster=read(QUEUE/'cluster.json')
    sys.path.insert(0,str(ROOT/'src'))
    from p2_matrix16.checkpoint import checked
    from p2_j_s1_queue import commands
    def status(stage,**extra):write(QUEUE/'state.json',dict(stage=stage,at=time.time(),watchdog=True,**extra))
    with (QUEUE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for phase,config_key,output_key,entry in [
                ('av_event','av_event_config','av_event_output','p2_j_duration_train.py'),
                ('av_only','av_config','av_output','p2_j_av_only.py')]:
                output=Path(job[output_key]);cp=latest_checkpoint(output)
                if cp and cp['step']==job['budget']:continue
                if (work/'STOP').exists() or (QUEUE/'STOP').exists() or (output/'STOP').exists():
                    status('stopped',phase=phase);return
                if digest_file(job[config_key])!=job[config_key+'_sha256']:raise ValueError('Queued config changed')
                if cp:
                    saved=checked(cp['path'])
                    if saved['step']!=cp['step'] or saved['identity']['experiment']!=read(job[config_key]):
                        raise ValueError('Checkpoint experiment differs')
                    del saved
                elif phase=='av_event':raise ValueError('No valid AV+Event checkpoint remains')
                elif (output/'run.json').exists():
                    # A crash before checkpoint0: only restart an untrained initialization.
                    if (output/'metrics.jsonl').exists() and (output/'metrics.jsonl').stat().st_size:
                        raise ValueError('Training history exists but no valid checkpoint')
                    (output/'run.json').rename(output/f'run.abandoned_initialization.{time.time_ns()}.json')
                if phase=='av_only':
                    subprocess.run([cluster['python'],str(ROOT/'scripts/p2_j_av_only.py'),'preflight',
                                    '--config',job[config_key]],check=True)
                status('av_event_running' if phase=='av_event' else 'av_only_running',
                       resume_step=cp['step'] if cp else 0,budget=job['budget'],output=str(output))
                cc=dict(cluster,master_port=30000+int(time.time())%10000)
                rows=commands(cc,job[config_key],resume=cp['path'] if cp else None)
                for row in rows:row[row.index(str(ROOT/'scripts/p2_clip_events.py'))]=str(ROOT/'scripts'/entry)
                launch(cc,rows,work/'transport'/f'{phase}_{time.time_ns()}')
                final=latest_checkpoint(output)
                if not final or final['step']!=job['budget']:status('stopped',phase=phase);return
                status('waiting_release_for_av_only' if phase=='av_event' else 'complete')
                if phase=='av_event':
                    while True:
                        if (work/'STOP').exists() or (QUEUE/'STOP').exists():return
                        nodes=inspect_cluster(cluster,work)
                        if all(n['ok'] and all(g['free_mib']>=135*1024 for g in n['gpus']) for n in nodes):break
                        time.sleep(3)
            status('complete')
        except BaseException as exc:
            status('failed',error=repr(exc),traceback=traceback.format_exc());raise


def tick(work):
    cfg=read(work/'config.json');job=read(QUEUE/'job.json');cluster=read(QUEUE/'cluster.json')
    with (work/'tick.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        previous=read(work/'state.json',{});now=time.time()
        if now-previous.get('last_poll_at',0)<cfg['poll_s']-5:return
        cache=read(work/'checkpoint_hash_cache.json',{})
        snapshot=observe(job,cluster,work,cache);now=time.time()
        paused=any(p.exists() for p in [work/'STOP',QUEUE/'STOP',Path(job['av_event_output'])/'STOP',Path(job['av_output'])/'STOP'])
        action,state=decide(snapshot,previous,cfg,now)
        if paused:action='user_stopped'
        state.update(last_poll_at=now,action=action,observer=socket.gethostname())
        snapshot.update(action=action,observer=socket.gethostname())
        write(work/'latest.json',snapshot);write(work/'state.json',state);write(work/'checkpoint_hash_cache.json',cache)
        with (work/'timeline.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=now,action=action,phase=snapshot['phase'],
                steps={k:v['step'] for k,v in snapshot['phases'].items()},
                observed_nodes=sum(n['ok'] for n in snapshot['nodes']),
                worker_counts={n['node']:len([p for p in n.get('processes',[]) if p['role'] in ('av_event','av_only')]) for n in snapshot['nodes']}))+'\n')
        hour=int((now-cfg['started_at'])//3600)
        report=work/'hourly'/f'hour_{hour:02d}.json'
        if not report.exists():write(report,snapshot)
        elif all(n['ok'] for n in snapshot['nodes']) and not all(n['ok'] for n in read(report)['nodes']):
            partial=report.with_name(report.stem+'.partial.json')
            if not partial.exists():write(partial,read(report))
            write(report,snapshot)
        if now>=cfg['handoff_at']:
            if snapshot['phase']=='av_only' and action=='healthy' and snapshot['phases']['av_only']['step']>0:
                handoff='av_only_running'
            elif not snapshot['phases']['av_event']['complete']:handoff='av_event_still_training'
            else:handoff='av_only_not_yet_healthy'
            evidence=dict(at=now,status=handoff,action=action,phase=snapshot['phase'],phases=snapshot['phases'])
            if not (work/'FOUR_HOUR_CHECK.json').exists():write(work/'FOUR_HOUR_CHECK.json',evidence)
            write(work/'handoff_latest.json',evidence)
            if handoff=='av_only_running' and not (work/'HANDOFF_VERIFIED.json').exists():write(work/'HANDOFF_VERIFIED.json',evidence)
        if now>=cfg['minimum_until'] and not (work/'TEN_HOUR_CHECK.json').exists():write(work/'TEN_HOUR_CHECK.json',snapshot)
        if action.startswith('recover_'):
            recovery(work,cfg,cluster,snapshot)
            state.update(unhealthy_count=0,last_recovery_at=time.time());write(work/'state.json',state)
        if action=='complete':write(work/'TRAINING_COMPLETE.json',snapshot)


def daemon(work):
    hostname=socket.gethostname();work.mkdir(parents=True,exist_ok=True)
    with (work/f'daemon.{hostname}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            write(work/'daemons'/f'{hostname}.json',dict(pid=os.getpid(),host=hostname,at=time.time(),process=process_info(os.getpid())))
            if (work/'STOP').exists():return
            try:tick(work)
            except Exception as exc:
                with (work/'errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.time(),host=hostname,error=repr(exc),traceback=traceback.format_exc()))+'\n')
            if (work/'TRAINING_COMPLETE.json').exists() and time.time()>=read(work/'config.json')['minimum_until']:return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['probe','terminate','tick','daemon','supervise'])
    parser.add_argument('--work',type=Path,default=DEFAULT);parser.add_argument('--targets');parser.add_argument('--force',action='store_true')
    args=parser.parse_args();work=args.work.resolve()
    if args.command=='probe':print(json.dumps(probe(work)))
    elif args.command=='terminate':print(json.dumps(terminate_owned(work,json.loads(args.targets),args.force)))
    elif args.command=='tick':tick(work)
    elif args.command=='daemon':daemon(work)
    else:supervise(work)
