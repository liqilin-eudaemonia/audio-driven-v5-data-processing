"""Durable, cache-only handoff after the fixed-manifest annotation audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import atomic_json, digest, read, sha
from p2_j_speaker_whisper_cache import cache_commands


def queue_drained(work):
    work = Path(work)
    if not (work/'producer_complete.json').exists():
        return False
    producer=read(work/'producer_complete.json')
    if producer.get('eof') is not True:
        return False
    journal=work/'ready.jsonl';path=work/'queues/caption/state.json'
    if not journal.exists() or not path.exists():
        return False
    q=read(path)
    return (not q.get('inflight') and not q.get('deferred')
            and q.get('offset',-1) >= journal.stat().st_size)


def predecessor_ready(work, expected):
    """Generation-review flags remain visible; missing/failed processing blocks handoff."""
    work=Path(work)
    if not queue_drained(work):
        return False,dict(reason='annotation_queue_not_drained')
    detect=read(work/'queues/detect/state.json')
    if (detect.get('completed_clips') != expected or detect.get('inflight') or detect.get('deferred')
            or detect.get('failed')):
        return False,dict(reason='detection_not_complete')
    path=work/'completion_status.json'
    if not path.exists():
        return False,dict(reason='waiting_fixed_manifest_audit')
    result=read(path);counts=result.get('counts') or {}
    names={'pass','generation_failed','insufficient_evidence','processing_failed','not_processed'}
    if (set(counts)!=names or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != expected or result.get('duplicate_ids') or result.get('missing_ids')
            or counts.get('not_processed') or counts.get('processing_failed')):
        return False,dict(reason='annotation_audit_incomplete_or_processing_failed',
                          predecessor_status=result.get('status'),counts=counts)
    if result.get('status') not in ('succeeded','completed_with_failures'):
        return False,dict(reason='annotation_audit_not_terminal')
    from listen_benchmark.caption_guard import digest as caption_digest, shared_identity
    q=read(work/'queues/caption/state.json')
    queue_identity=caption_digest(dict(state=q,ready_bytes=(work/'ready.jsonl').stat().st_size))
    if (result.get('identity_version') != caption_digest(shared_identity())
            or result.get('audit_queue_identity') != queue_identity):
        return False,dict(reason='annotation_audit_stale')
    return True,dict(predecessor_status=result['status'],counts=counts,
        audit=str(path),audit_sha256=sha(path),
        note='generation_failed and insufficient_evidence remain in predecessor audit; not promoted to accepted captions')


def predecessor_resource_ready(work, expected, expected_pairs):
    """Independent frozen-speaker cache may overlap the drained caption CPU audit."""
    work=Path(work)
    if not queue_drained(work):
        return False,dict(reason='annotation_queue_not_drained')
    producer=read(work/'producer_complete.json')
    if (producer.get('published_clips') != expected or producer.get('published_sources') != expected_pairs):
        return False,dict(reason='producer_fixed_scope_mismatch')
    for branch in ('detect','caption'):
        q=read(work/'queues'/branch/'state.json')
        if (q.get('completed_clips') != expected or q.get('completed') != expected_pairs
                or q.get('inflight') or q.get('deferred') or q.get('failed')):
            return False,dict(reason=branch+'_receipts_not_complete')
    return True,dict(predecessor_status='processing_drained_cpu_audit_pending',
        predecessor_receipt_clips=expected,predecessor_receipt_pairs=expected_pairs,
        cpu_audit_pending=True,
        note='Only independent speaker feature extraction is released; no caption quality acceptance is implied')


def alive_record(record, script, command=None):
    try:
        pid=int(record['pid']);p=Path('/proc')/str(pid)
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        argv=(p/'cmdline').read_bytes().split(b'\0')
        return (stat[0]!='Z' and str(stat[19])==str(record['start_ticks'])
            and str(script).encode() in argv and (command is None or command.encode() in argv))
    except (KeyError,OSError,ValueError):
        return False


def process_record():
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=stat[19],at=time.time())


def resource_checks(cluster):
    """Observe only; never evict another GPU owner or use instantaneous utilization as free."""
    from listen_cluster import transport
    def check(node):
        code="""import json,subprocess,socket
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
a=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
print(json.dumps(dict(hostname=socket.gethostname(),gpus=[[int(v.strip()) for v in s.split(',')] for s in g.splitlines()],compute_pids=a)))
"""
        try:
            proc=subprocess.run(transport(cluster,node,[cluster['python'],'-c',code]),
                capture_output=True,text=True,timeout=35)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-1500:] or proc.stdout[-1500:])
            data=json.loads(proc.stdout.strip().splitlines()[-1])
            selected=[r for r in data['gpus'] if r[0] in node['gpus']]
            free=(len(selected)==len(node['gpus']) and all(r[1]<1024 and r[2]>=32*1024 for r in selected)
                  and not data['compute_pids'])
            return dict(node=node['name'],free=free,**data)
        except Exception as exc:
            return dict(node=node['name'],free=False,error=repr(exc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(check,cluster['nodes']))


def execute(cmd, log, work, poll=5):
    """A small local stage, stopped only through this queue's STOP marker."""
    with Path(log).open('a') as stream:
        child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if (work/'STOP').exists():
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise InterruptedError('Speaker cache queue stopped')
            time.sleep(poll)
        if child.returncode:
            raise RuntimeError('Stage failed; see '+str(log))


def run(work):
    work=Path(work);cfg=read(work/'queue.json');predecessor=Path(cfg['predecessor'])
    cluster=read(cfg['cluster']);cache=Path(cfg['cache']);statefile=work/'state.json'
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(work/'controller.json',process_record())
        state=read(statefile) if statefile.exists() else {}
        if state.get('phase') == 'complete':return
        def status(phase,**extra):
            state.update(phase=phase,heartbeat=time.time(),**extra);atomic_json(statefile,state)
            print(json.dumps(dict(phase=phase,time=time.time(),**extra)),flush=True)
        try:
            while not (work/'STOP').exists():
                ready,info=predecessor_ready(predecessor,cfg['predecessor_clips'])
                if not ready and cfg.get('independent_cache_during_cpu_audit'):
                    ready,info=predecessor_resource_ready(predecessor,cfg['predecessor_clips'],cfg['predecessor_pairs'])
                if ready:
                    marker='PREDECESSOR_RESOURCES_RELEASED.json' if info.get('cpu_audit_pending') else 'PREDECESSOR_AUDITED.json'
                    atomic_json(work/marker,info,immutable=True)
                    status('predecessor_resources_released',**info)
                    break
                qpath=predecessor/'queues/caption/state.json'
                done=read(qpath).get('completed_clips') if qpath.exists() else None
                status('waiting_predecessor',caption_receipt_clips=done,target=cfg['predecessor_clips'],**info)
                time.sleep(cfg['poll_seconds'])
            else:
                status('stopped');return
            # Source/model contracts were prepared before arming. Check them again
            # without changing the frozen list or overwriting existing results.
            if sha(cfg['config']) != cfg['config_sha256'] or sha(cfg['cluster']) != cfg['cluster_sha256']:
                raise ValueError('Queued cache config/cluster changed')
            for path, expected_sha in cfg['cache_code'].items():
                if sha(path) != expected_sha:
                    raise ValueError('Queued speaker cache implementation changed: '+path)
            if sha(predecessor/'config.json') != cfg['predecessor_config_sha256']:
                raise ValueError('Predecessor config changed while waiting')
            if sha(cfg['predecessor_manifest']) != cfg['predecessor_manifest_sha256']:
                raise ValueError('Predecessor fixed manifest changed while waiting')
            # Retire the completed annotation resource controller and its guard;
            # no GPU process is killed, and all captions/detections remain.
            control=predecessor/'cluster_control'
            control.mkdir(parents=True,exist_ok=True)
            if not (control/'STOP').exists():
                (control/'STOP').write_text('Completed fixed-manifest audit; handoff to speaker Whisper cache\n')
            while not (work/'STOP').exists():
                checks=resource_checks(cluster);atomic_json(work/'resources.json',dict(at=time.time(),nodes=checks))
                if all(r['free'] for r in checks):break
                status('waiting_gpu_release',nodes=checks);time.sleep(cfg['poll_seconds'])
            else:
                status('stopped');return
            if shutil.disk_usage(cache.parent).free < cfg['minimum_free_bytes']:
                raise ValueError('Insufficient disk space for the full FP32 Whisper cache')
            entry=str(ROOT/'scripts/p2_j_speaker_whisper_cache.py');python=cluster['python']
            execute([python,entry,'prepare','--config',cfg['config']],work/'prepare.log',work)
            attempt=int(state.get('attempts',0))+1
            if attempt>cfg['max_attempts']:
                status('needs_attention',reason='cache_launch_retry_limit');return
            commands=cache_commands(cluster,cfg['config'],batch_size=cfg['batch_size'],
                read_workers=cfg['read_workers'],cpu_threads=cfg['cpu_threads'],master_port=cfg['master_port'])
            atomic_json(work/'commands.json',dict(audio_source='speaker',commands=commands,training_started=False))
            status('extracting',attempts=attempt,samples=cfg['samples'],world=cfg['world'],audio_source='speaker')
            from listen_cluster import launch
            cancel=threading.Event();finished=threading.Event()
            launched_at=time.time()
            def watch():
                while not finished.wait(15):
                    if (work/'STOP').exists():cancel.set();return
                    phase=read(statefile);phase['heartbeat']=time.time();atomic_json(statefile,phase)
                    progress=[]
                    for rank in range(cfg['world']):
                        path=cache/'progress'/f'rank{rank:03d}.json'
                        record=read(path) if path.exists() else {}
                        current=bool(record) and path.stat().st_mtime>=launched_at
                        age=time.time()-(path.stat().st_mtime if current else launched_at)
                        limit=cfg['stall_seconds'] if current else cfg['startup_seconds']
                        if not (current and record.get('finished')) and age>limit:
                            atomic_json(work/'STALL.json',dict(at=time.time(),rank=rank,age_s=age,
                                action='Cancel only owned cache workers; retry complete receipts'))
                            cancel.set();return
                        progress.append(dict(record,rank=rank,current_attempt=current))
                    atomic_json(work/'progress.json',dict(at=time.time(),world=cfg['world'],ranks=progress,
                        samples=sum(r.get('new',0)+r.get('reused',0) for r in progress if r['current_attempt']),
                        total=cfg['samples']))
            monitor=threading.Thread(target=watch,daemon=True);monitor.start()
            try:
                launch(cluster,commands,work/f'launch_{attempt:02d}_{int(time.time())}',cancel_event=cancel)
            finally:
                finished.set();monitor.join(timeout=2)
            status('auditing_cache')
            execute([python,entry,'finalize','--config',cfg['config'],
                     '--read-workers',str(cfg.get('audit_read_workers',16))],work/'finalize.log',work)
            result=read(cache/'COMPLETE.json')
            if (result.get('success') is not True or result.get('samples') != cfg['samples']
                    or result.get('audio_source') != 'speaker'
                    or result.get('protocol_sha256') != sha(cache/'protocol.json')):
                raise ValueError('Full speaker Whisper audit did not complete')
            status('complete',samples=result['samples'],cache=str(cache),cache_complete_sha256=sha(cache/'COMPLETE.json'),
                   training_started=False,predecessor_audit_file=str(predecessor/'completion_status.json'),
                   predecessor_audit_file_present=(predecessor/'completion_status.json').exists())
        except InterruptedError:
            status('stopped')
        except Exception as exc:
            status('retry_pending',error=repr(exc));raise


def guard(work):
    work=Path(work)
    with (work/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (work/'STOP').exists():
            state=read(work/'state.json') if (work/'state.json').exists() else {}
            if state.get('phase') in ('complete','stopped','needs_attention'):return
            path=work/'controller.json';rec=read(path) if path.exists() else {}
            running=alive_record(rec,Path(__file__).resolve(),'run')
            if not running:
                cfg=read(work/'queue.json')
                if int(state.get('attempts',0))>=cfg['max_attempts']:
                    atomic_json(work/'state.json',dict(state,phase='needs_attention',heartbeat=time.time()));return
                with (work/'controller.log').open('a') as stream:
                    subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--work',str(work)],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(work/'guard.json',dict(process_record(),controller_alive=running))
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','guard','status'))
    p.add_argument('--work',required=True,type=Path)
    a=p.parse_args();w=a.work.resolve()
    if a.command=='run':run(w)
    elif a.command=='guard':guard(w)
    else:print(json.dumps(read(w/'state.json'),ensure_ascii=False,indent=2))
