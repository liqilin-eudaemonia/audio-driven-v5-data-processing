"""Pause only this evaluation; resume after another GPU job has run and all eight GPUs are idle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'scripts/p2_seamless200_nineway.py'
PYTHON=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'


def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,indent=2));q.replace(p)


def info(pid):
    try:
        root=Path('/proc')/str(pid);args=[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x]
        stat=(root/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,args=args,start_ticks=stat[19],state=stat[0])
    except (OSError,ProcessLookupError):return None


def owns(record,work,controller=False):
    if not record or '--work' not in record['args']:return False
    a=record['args'];requested=Path(a[a.index('--work')+1]).resolve()
    if requested!=work and requested.parent!=work:return False
    names={Path(x).name for x in a}
    return bool(names & ({ENTRY.name} if controller else {'p2_testsets_eval.py'}))


def tree(pid):
    result=[]
    try:children=[int(x) for x in Path('/proc',str(pid),'task',str(pid),'children').read_text().split()]
    except OSError:children=[]
    for child in children:result+=tree(child)
    r=info(pid)
    if r:result.append(r)
    return result


def signal_same(record,sig):
    now=info(record['pid'])
    if now and now['start_ticks']==record['start_ticks'] and now['state']!='Z':
        try:os.kill(record['pid'],sig)
        except ProcessLookupError:pass


def pause(work):
    work=work.resolve();folder=work/'pause_for_other_job';folder.mkdir(exist_ok=True)
    token=str(uuid.uuid4());request=dict(token=token,requested_at=time.time(),work=str(work),
        mode='observe_external_gpu_job_then_idle',idle_seconds=120,poll_seconds=30)
    save(folder/'request.json',request)
    save(work/'STOP',dict(reason='User requested resource release and automatic continuation after other job',token=token))
    stopped=[]
    # Disarm job restart before terminating only its own controller and workers.
    for name in ('watch.json','controller.json'):
        p=work/name
        if p.exists():
            r=read(p);actual=info(r['pid'])
            if owns(actual,work,True) and actual['start_ticks']==str(r['start_ticks']):
                signal_same(actual,signal.SIGTERM);stopped.append(actual)
    roots=[work,*[Path(x) for x in read(work/'groups.json')]]
    found={}
    for root in roots:
        for sub in ('progress','processes'):
            p=root/sub
            if not p.exists():continue
            for path in p.rglob('*.json'):
                try:r=read(path);pid=r.get('pid');actual=info(pid) if type(pid)is int else None
                except (OSError,ValueError):continue
                if owns(actual,work):
                    for item in tree(pid):found[item['pid']]=item
    for item in found.values():signal_same(item,signal.SIGTERM)
    stopped+=list(found.values());deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if not any((r:=info(x['pid'])) and r['start_ticks']==x['start_ticks'] and r['state']!='Z' for x in stopped):break
        time.sleep(.5)
    for item in stopped:signal_same(item,signal.SIGKILL)
    remaining=[r for x in stopped if (r:=info(x['pid'])) and r['start_ticks']==x['start_ticks'] and r['state']!='Z']
    if remaining:raise RuntimeError('Owned processes have not exited')
    save(folder/'PAUSED.json',dict(at=time.time(),token=token,stopped=stopped,
        owned_processes_remaining=0,completed_outputs_preserved=True,other_tasks_touched=False))
    save(work/'status.json',dict(stage='paused_waiting_other_job',at=time.time(),automatic_resume=True,
        resume_condition='External GPU job observed, then all8 idle for120seconds',token=token))
    print(json.dumps(dict(paused=True,stopped_processes=len(stopped),token=token)),flush=True)


def observe():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    gpus=[]
    for line in raw.splitlines():
        i,m,u=[float(x.strip()) for x in line.split(',')];gpus.append(dict(index=int(i),used_mib=m,util=u))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    pids=sorted({int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()})
    idle=len(gpus)==8 and not pids and all(g['used_mib']<1024 and g['util']<5 for g in gpus)
    busy=bool(pids) or any(g['used_mib']>=1024 or g['util']>=10 for g in gpus)
    return dict(at=time.time(),gpus=gpus,compute_pids=pids,idle=idle,external_busy=busy)


def wait(work):
    work=work.resolve();folder=work/'pause_for_other_job';r=read(folder/'request.json')
    with (folder/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(folder/'state.json') if (folder/'state.json').exists() else {}
        seen=old.get('external_job_observed',False);idle_since=None
        save(folder/'watch_process.json',dict(pid=os.getpid(),at=time.time(),host=os.uname().nodename))
        while not (folder/'CANCEL_AUTO_RESUME').exists():
            try:snapshot=observe()
            except Exception as exc:
                idle_since=None
                save(folder/'state.json',dict(stage='observation_retry',at=time.time(),error=repr(exc),external_job_observed=seen));time.sleep(30);continue
            if snapshot['external_busy']:seen=True;idle_since=None
            if seen and snapshot['idle']:
                if idle_since is None:idle_since=time.time()
            else:idle_since=None
            save(folder/'state.json',dict(stage='waiting_other_job_finish' if seen else 'waiting_other_job_start',
                external_job_observed=seen,idle_since=idle_since,token=r['token'],snapshot=snapshot))
            if idle_since is not None and time.time()-idle_since>=r['idle_seconds']:
                stop=read(work/'STOP') if (work/'STOP').exists() else {}
                if stop.get('token')!=r['token']:
                    save(folder/'state.json',dict(stage='blocked_manual_stop_changed',at=time.time()));return
                (work/'STOP').unlink()
                launched={}
                for command in ('run','watch'):
                    with (work/('queue.log' if command=='run' else 'watch.log')).open('a') as log:
                        p=subprocess.Popen([str(PYTHON),'-u',str(ENTRY),command,'--work',str(work)],
                            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    launched[command]=p.pid
                save(folder/'RESUMED.json',dict(at=time.time(),token=r['token'],external_job_observed=True,
                    all8_idle_seconds=time.time()-idle_since,processes=launched))
                return
            time.sleep(r['poll_seconds'])
        save(folder/'state.json',dict(stage='auto_resume_cancelled',at=time.time(),external_job_observed=seen))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('pause','wait'));p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();{'pause':pause,'wait':wait}[a.command](a.work)
