#!/usr/bin/env python3
"""Bounded user-authorized handoff supervision; never kill unrelated processes."""
import argparse,concurrent.futures,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from listen_richness_cluster_control import ROOT,REMOTE_COMMON,remote,ensure_preparer,upstream_done
from listen_benchmark.j_annotation import acquire,read,write


def alive(child):
    if not child or child.get('start_ticks') is None:return False
    try:parts=(Path('/proc')/str(child['pid'])/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return False
    return parts[0] not in ('Z','X') and int(parts[19])==int(child['start_ticks'])


def mismatches(work):
    cfg=read(work/'config.json')
    return [p for p,h in cfg['code_hashes'].items() if hashlib.sha256((Path(cfg['project_root'])/p).read_bytes()).hexdigest()!=h]


def ensure_chain(work,folder):
    lock=acquire(folder/'restart.lock',False)
    if lock is None:return 'restart_locked'
    try:
        child=read(work/'chain-child.json') if (work/'chain-child.json').exists() else None
        if alive(child):return 'live'
        if (work/'complete.json').exists():return 'finished'
        if mismatches(work):return 'blocked_code_changed'
        prep=read(work/'preparation.json') if (work/'preparation.json').exists() else {}
        if prep.get('source_errors'):return 'blocked_source_errors'
        qp=work/'queues/extract/state.json'
        if qp.exists() and read(qp)['failed']:return 'blocked_failed_chunks'
        history=read(folder/'chain_restarts.json') if (folder/'chain_restarts.json').exists() else []
        if len(history)>=3:return 'restart_limit_requires_inspection'
        cmd=[sys.executable,str(ROOT/'scripts/listen_richness_cluster_control.py'),'chain','--work',str(work)]
        with (work/'chain.log').open('a') as f:
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),start_new_session=True)
        ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
        child=dict(pid=p.pid,start_ticks=ticks,started_at=time.time(),cmd=cmd)
        history.append(dict(at=time.time(),child=child));write(folder/'chain_restarts.json',history)
        write(work/'chain-child.json',child);return 'restarted'
    finally:lock.close()


def finished(work):
    if not (work/'complete.json').exists():return False
    c=read(work/'complete.json');summary=read(Path(c['summary']))
    cfg=read(work/'config.json');q=read(work/'queues/extract/state.json')
    return (c['total_indexed_samples']==cfg['expected_upstream_clips'] and
            summary['total_indexed_samples']==cfg['expected_upstream_clips'] and
            not summary['source_errors'] and not q['failed'] and not q['inflight'] and
            q['completed_clips']==summary['sample_count'])


def state_snapshot(work):
    cfg=read(work/'config.json');u=Path(cfg['upstream']);out={}
    for key,base,branch in [('annotation',u,'caption'),('statistics',work,'extract')]:
        p=base/'queues'/branch/'state.json';q=read(p) if p.exists() else {}
        out[key]={k:q.get(k) for k in ['completed','completed_clips','failed','offset']}
        out[key]['inflight']=len(q.get('inflight',{}))
    for name in ['chain','preparation']:
        p=work/(name+'.json');out[name]=read(p) if p.exists() else {}
        p=work/(name+'-child.json');out[name+'_process_alive']=alive(read(p)) if p.exists() else False
    out['code_mismatches']=mismatches(work);out['upstream_complete']=upstream_done(work)
    out['worker_states']={}
    for n,v in cfg['nodes'].items():
        for g in v['gpu_ids']:
            name=f'{n}-gpu-{g}'
            for phase,base in [('annotation',u),('statistics',work)]:
                p=base/'workers'/f'{name}.json'
                if p.exists():
                    s=read(p);out['worker_states'][phase+':'+name]={k:s.get(k) for k in ['pid','stage','heartbeat','last_progress_at','completed','completed_clips','last_error']}
    return out


RESOURCE_BODY=REMOTE_COMMON+'''
rows=[];previous=None
for i in range(10):
 now=time.time();usage=int(dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
 rows.append(dict(at=now,gpus=[list(map(float,line.split(','))) for line in raw.strip().splitlines()],cpu_cores=(usage-previous[1])/1e6/(now-previous[0]) if previous else None))
 previous=(now,usage)
 if i<9:time.sleep(5)
children={}
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 p=W/'nodes'/NODE/('production-gpu-'+str(gpu)+'.json')
 if p.exists():s=read(p);children[str(gpu)]=dict(pid=s['pid'],alive=alive(s))
p=U/'nodes'/NODE/'watchdog.json';watch=read(p) if p.exists() else None
print(json.dumps(dict(samples=rows,statistics_children=children,annotation_watchdog_alive=alive(watch))))
'''


def resource_report(work):
    nodes=list(read(work/'config.json')['nodes']);out={}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        fs={pool.submit(remote,work,n,RESOURCE_BODY,120):n for n in nodes}
        for f in concurrent.futures.as_completed(fs):
            n=fs[f]
            try:
                value=f.result();rows=value['samples'];gpus=[g for row in rows for g in row['gpus']]
                cpu=[row['cpu_cores'] for row in rows if row['cpu_cores'] is not None]
                value.update(gpu_mean=sum(g[1] for g in gpus)/len(gpus),memory_mib_mean=sum(g[2] for g in gpus)/len(gpus),cpu_cores_mean=sum(cpu)/len(cpu));out[n]=value
            except Exception as exc:out[n]=dict(error=repr(exc))
    return out


def supervise(work,folder):
    lock=acquire(folder/'monitor.lock',False)
    if lock is None:return
    cfg=read(folder/'schedule.json');stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    prior=read(folder/'state.json') if (folder/'state.json').exists() else {}
    next_check=prior.get('next_check_at',cfg['first_check_at']);next_resource=prior.get('next_resource_at',cfg['first_check_at'])
    active_seen=prior.get('statistics_started_at');last_report=None
    try:
        while not stop:
            now=time.time()
            if finished(work):write(folder/'state.json',dict(stage='complete',at=now,pid=os.getpid(),finished_at=now));break
            if now>=cfg['deadline_at']:write(folder/'state.json',dict(stage='window_expired',at=now,pid=os.getpid()));break
            if now<cfg['first_check_at']:
                write(folder/'state.json',dict(stage='waiting_first_check',at=now,pid=os.getpid(),next_check_at=next_check,next_resource_at=next_resource));time.sleep(min(15,cfg['first_check_at']-now));continue
            actions=[]
            if now>=next_check:
                result=ensure_chain(work,folder)
                if result!='live':actions.append(dict(chain=result))
                if result in ('live','restarted'):
                    ensure_preparer(work)
                snap=state_snapshot(work);phase=snap['chain'].get('stage')
                if phase in ('extracting','merging') and active_seen is None:
                    active_seen=now;next_resource=min(next_resource,now)
                if now>=next_resource:
                    resources=resource_report(work);after=state_snapshot(work);at=time.time();delta=None
                    if last_report:
                        duration=at-last_report['at'];delta=(after['statistics']['completed_clips']-last_report['state']['statistics']['completed_clips'])/duration
                    report=dict(at=at,state=after,resources=resources,actions=actions,statistics_clips_per_second_since_previous_report=delta,
                        reference=dict(gpu_mean=73.3,clips_per_second=259.4653,profile='D512',note='Compare only active extraction with sufficient queued inputs; loading, tail and CPU merge differ.'))
                    write(folder/f'report-{int(at)}.json',report);write(folder/'latest.json',report);last_report=report
                    # Extra observation five minutes after first extraction report,
                    # then the requested half-hour cadence.
                    if phase=='extracting' and not prior.get('startup_followup_done'):
                        next_resource=at+300;prior['startup_followup_done']=True
                    else:next_resource=at+cfg['interval_s']
                elif actions:write(folder/f'actions-{int(now)}.json',dict(at=now,actions=actions,state=snap))
                next_check=time.time()+60
            write(folder/'state.json',dict(stage='monitoring',at=time.time(),pid=os.getpid(),next_check_at=next_check,next_resource_at=next_resource,
                 statistics_started_at=active_seen,startup_followup_done=prior.get('startup_followup_done',False)))
            time.sleep(15)
    except BaseException as exc:
        write(folder/'state.json',dict(stage='error',at=time.time(),pid=os.getpid(),error=repr(exc)));raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--monitor',type=Path,required=True)
    a=p.parse_args();supervise(a.work.resolve(),a.monitor.resolve())
