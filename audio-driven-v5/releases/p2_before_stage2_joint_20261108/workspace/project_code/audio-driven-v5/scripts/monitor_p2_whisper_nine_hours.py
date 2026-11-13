"""User-authorized nine-hour supervision; hourly reports and local fast diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_speaker_whisper_after_annotation import alive_record,process_record
from listen_cluster import transport

QUEUE_SCRIPT=ROOT/'scripts/p2_whisper_train_after_cache.py'
SELF=Path(__file__).resolve()


def maybe(path,default=None):
    try:return read(path)
    except FileNotFoundError:return {} if default is None else default


def metrics(path):
    try:lines=Path(path).read_text().splitlines()
    except FileNotFoundError:return [],0
    rows=[];invalid=0
    for index,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:invalid+=1
    return rows,invalid


def cpu_ticks():
    values=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(values),values[3]+values[4]


def probe(config,node):
    """Run on the actual node, including fresh /proc identity checks for CUDA workers."""
    samples=[];start=cpu_ticks()
    for index in range(5):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        values=[]
        for row in csv.reader(raw.splitlines()):
            row=[v.strip() for v in row]
            values.append(dict(index=int(row[0]),gpu_util=float(row[1]),used_mib=float(row[2]),total_mib=float(row[3]),
                power_w=float(row[4]) if row[4].replace('.','',1).isdigit() else None,
                temperature_c=float(row[5]) if row[5].isdigit() else None,sm_clock_mhz=float(row[6]) if row[6].isdigit() else None))
        samples.append(dict(at=time.time(),gpus=values))
        if index<4:time.sleep(2)
    finish=cpu_ticks();total=finish[0]-start[0];idle=finish[1]-start[1]
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    own=[];other=[]
    for pid in sorted({int(x) for x in pids.splitlines() if x.strip().isdigit()}):
        p=Path('/proc')/str(pid)
        try:
            args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if (str(ROOT/'scripts/p2_j_whisper_av.py') not in args or '--config' not in args
                    or args[args.index('--config')+1]!=config['training_config']):
                other.append(pid);continue
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
            own.append(dict(pid=pid,start_ticks=fields[19],state=fields[0],rank=int(env[b'RANK']),local_rank=int(env[b'LOCAL_RANK'])))
        except (FileNotFoundError,ProcessLookupError,KeyError):continue
    return dict(ok=True,node=node,hostname=socket.gethostname(),at=time.time(),samples=samples,
        cpu_percent=100*(total-idle)/total if total else None,cpu_count=os.cpu_count(),
        memory_used_percent=100*(1-memory['MemAvailable']/memory['MemTotal']),workers=own,other_compute_pids=other)


def inspect_cluster(config,cluster,work):
    work=Path(work).resolve()
    def one(node):
        error=None
        for attempt in range(5):
            try:
                args=transport(cluster,node,[cluster['python'],str(SELF),'probe','--work',str(work),'--node',node['name']])
                p=subprocess.run(args,capture_output=True,text=True,timeout=45)
                if p.returncode==0:return json.loads(p.stdout.strip().splitlines()[-1])
                error=p.stderr[-800:]
                if 'kex_exchange_identification' not in error:break
            except (subprocess.SubprocessError,OSError,ValueError) as exc:error=repr(exc)
            time.sleep(min(2.,.5*(attempt+1)))
        return dict(ok=False,node=node['name'],at=time.time(),error=error)
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,cluster['nodes']))


def local_snapshot(config):
    out=Path(config['output']);queue=out/'queue';rows,invalid=metrics(out/'metrics.jsonl');last=rows[-1] if rows else {}
    q=maybe(queue/'state.json');controller=maybe(queue/'controller.json');guard=maybe(queue/'guard.json')
    recent=rows[-200:]
    return dict(at=time.time(),step=last.get('step',0),budget=8927,last_metrics=last,invalid_metric_lines=invalid,
        recent200_count=len(recent),recent200_mean_seconds=sum(r['seconds'] for r in recent)/len(recent) if recent else None,
        controller_started_at=controller.get('at'),
        metrics_age_s=time.time()-(out/'metrics.jsonl').stat().st_mtime if rows else None,
        controller_alive=alive_record(controller,QUEUE_SCRIPT,'run'),guard_alive=alive_record(guard,QUEUE_SCRIPT,'guard'),
        controller=controller,guard=guard,queue=q,training_complete=maybe(out/'status.json').get('complete',False),
        stop_requested=(queue/'STOP').exists() or (out/'STOP').exists(),free_disk_bytes=shutil.disk_usage(out).free)


def issues(snapshot,checkpoint_writing=False):
    result=[];last=snapshot['last_metrics']
    if snapshot['invalid_metric_lines']:result.append('invalid_metrics')
    if last and (any(not math.isfinite(last[k]) for k in ('av','av_video','av_audio','weighted_total'))
            or any(last[k]!=0 for k in ('event','feature_video','feature_audio')) or last['weighted_total']!=last['av']):
        result.append('loss_contract_violation')
    if snapshot['training_complete'] or snapshot['stop_requested']:return result
    if not snapshot['guard_alive']:result.append('guard_not_alive')
    if not snapshot['controller_alive']:result.append('controller_not_alive')
    if snapshot['queue'].get('phase') in ('needs_attention','retry_pending'):result.append('controller_needs_attention')
    starting=(snapshot['controller_alive'] and snapshot.get('controller_started_at')
        and snapshot['at']-snapshot['controller_started_at']<600
        and snapshot['at']-(snapshot['metrics_age_s'] or 0)<=snapshot['controller_started_at']
        and snapshot['queue'].get('phase') in ('preflight','waiting_gpu_release','training'))
    # A live newly-started controller is permitted to load weights/restore Adam.
    grace=600 if starting else 150
    if (snapshot['metrics_age_s'] or 0)>grace and not checkpoint_writing:
        result.append('no_recent_steps_diagnose_only')
    if snapshot.get('recent200_count',0)>=64 and snapshot['recent200_mean_seconds']>6.9:
        result.append('sustained_slow_steps_diagnose_only')
    if snapshot['free_disk_bytes']<100*2**30:result.append('low_disk_space')
    return result


def intact(config):
    out=Path(config['output']);spec=read(out/'queue/queue.json');identity=read(out/'run.json')
    if sha(spec['config'])!=spec['config_sha256'] or sha(spec['cluster'])!=spec['cluster_sha256']:return False
    return all(Path(path).exists() and sha(path)==value for path,value in identity['source'].items())


def ensure_guard(config,snapshot,work):
    """Only a verified-dead guard is recoverable here; never kill live ranks on a timeout."""
    if snapshot['guard_alive'] or snapshot['stop_requested'] or snapshot['training_complete']:return None
    if snapshot['queue'].get('phase') in ('stopped','needs_attention','complete'):return None
    if not intact(config):return dict(action='guard_restart_refused_source_mismatch')
    queue=Path(config['output'])/'queue'
    with (queue/'guard.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(action='guard_lock_owned_no_duplicate')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (queue/'guard.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,str(QUEUE_SCRIPT),'guard','--work',str(queue)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(action='verified_dead_guard_restarted',pid=child.pid,at=time.time())


def checkpoint(out,cache):
    rejected=[]
    for marker in sorted(out.glob('step_*.complete.json'),reverse=True):
        try:
            receipt=read(marker);path=marker.with_suffix('').with_suffix('.pt');stat=path.stat()
            key=f'{path}:{stat.st_size}:{stat.st_mtime_ns}:{receipt["sha256"]}'
            if key not in cache:
                if sha(path)!=receipt['sha256']:raise ValueError('SHA mismatch')
                cache[key]=True
            return dict(valid=True,path=str(path),step=receipt['step'],sha256=receipt['sha256'],bytes=stat.st_size,rejected_newer=rejected)
        except (OSError,KeyError,ValueError) as exc:rejected.append(dict(path=str(marker),error=repr(exc)))
    return dict(valid=False,rejected_newer=rejected)


def supervise(work):
    c=read(work/'schedule.json');out=Path(c['output']);cluster=read(c['cluster'])
    with (work/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic_json(work/'process.json',process_record())
        state=maybe(work/'state.json');done=state.get('completed_hours',[]);cache=state.get('checkpoint_hash_cache',{})
        last_report=maybe(work/'latest.json');last_alert=None;last_alert_at=0;next_hour_retry=0
        def emit(event,**kw):
            value=dict(event=event,at=time.time(),**kw)
            with (work/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
            print(json.dumps(value),flush=True)
        while not (work/'STOP').exists():
            now=time.time();s=local_snapshot(c)
            # Atomic checkpoint writer updates the temporary payload during legitimate pauses.
            writing=False
            for p in out.glob('step_*.pt.*'):
                try:writing=writing or now-p.stat().st_mtime<90
                except FileNotFoundError:pass
            problems=issues(s,writing);action=ensure_guard(c,s,work)
            if action:emit('action',**action)
            if problems and (problems!=last_alert or now-last_alert_at>300):
                emit('alert',issues=problems,snapshot=s);last_alert=problems;last_alert_at=now
                atomic_json(work/f'alert-{int(now)}.json',dict(local=s,nodes=inspect_cluster(c,cluster,work)))
            elif not problems and last_alert:
                emit('alert_cleared',step=s['step']);last_alert=None
            pending=[i for i in range(10) if i not in done]
            if pending and now>=c['start_at']+pending[0]*3600 and now>=next_hour_retry:
                hour=pending[0];nodes=inspect_cluster(c,cluster,work);fresh=local_snapshot(c)
                rows,_=metrics(out/'metrics.jsonl');since=last_report.get('local',{}).get('step',c['baseline_step'])
                recent=[r for r in rows if r['step']>since];last200=rows[-200:]
                ckpt=checkpoint(out,cache);source_ok=intact(c)
                elapsed=fresh['at']-last_report.get('at',c['baseline_at']);delta=fresh['step']-since
                full=all(n['ok'] and len(n['samples'][-1]['gpus'])==8 for n in nodes)
                ranks=[w['rank'] for n in nodes if n['ok'] for w in n['workers']]
                workers_ok=sorted(ranks)==list(range(32)) and all(sorted(w['local_rank'] for w in n['workers'])==list(range(8)) for n in nodes if n['ok'])
                report=dict(hour=hour,due_at=c['start_at']+hour*3600,at=fresh['at'],published_at=time.time(),local=fresh,nodes=nodes,
                    workers_32_verified=workers_ok,source_intact=source_ok,checkpoint=ckpt,
                    mean_step_seconds=sum(r['seconds'] for r in recent)/len(recent) if recent else None,
                    recent200_mean_step_seconds=sum(r['seconds'] for r in last200)/len(last200) if last200 else None,
                    net_steps_since_previous=delta,interval_seconds=elapsed,
                    wall_seconds_per_net_step=elapsed/delta if delta>0 else None,
                    rough_remaining_compute_hours=(8927-fresh['step'])*sum(r['seconds'] for r in last200)/len(last200)/3600 if last200 else None,
                    complete_observation=full)
                if full:
                    atomic_json(work/f'hour-{hour:02d}.json',report);atomic_json(work/'latest.json',report)
                    done.append(hour);last_report=report
                    emit('hourly_report',hour=hour,path=str(work/f'hour-{hour:02d}.json'),step=fresh['step'],
                        workers_32_verified=workers_ok,mean_step_seconds=report['mean_step_seconds'],source_intact=source_ok,checkpoint_step=ckpt.get('step'))
                    if not workers_ok and not fresh['training_complete']:emit('alert',issues=['worker_rank_coverage'],hour=hour)
                    if not source_ok or not ckpt['valid'] or ckpt.get('rejected_newer'):emit('alert',issues=['source_or_checkpoint_integrity'],hour=hour)
                else:
                    atomic_json(work/f'hour-{hour:02d}-incomplete-{int(now)}.json',report)
                    emit('observation_incomplete',hour=hour,path=str(work/f'hour-{hour:02d}-incomplete-{int(now)}.json'))
                    next_hour_retry=time.time()+30
            terminal=len(done)==10 and time.time()>=c['end_at']
            state=dict(stage='window_complete' if terminal else 'monitoring',at=time.time(),completed_hours=done,
                start_at=c['start_at'],end_at=c['end_at'],checkpoint_hash_cache=cache,latest_local=s)
            atomic_json(work/'state.json',state)
            if terminal:emit('window_complete',hours=done,deadline=c['end_at']);return
            time.sleep(30)
        atomic_json(work/'state.json',dict(stage='stopped',at=time.time(),completed_hours=done))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['probe','supervise']);p.add_argument('--work',type=Path,required=True);p.add_argument('--node')
    a=p.parse_args();a.work=a.work.resolve();c=read(a.work/'schedule.json')
    if a.command=='probe':print(json.dumps(probe(c,a.node)),flush=True)
    else:supervise(a.work)
