"""Checkpoint-gated four-host caption rollout and persistent resource reporting.

Only the explicitly configured training command is stopped. Other GPU owners
are never killed. Host watchdogs own caption recovery; CPU detection is adopted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from listen_benchmark.j_annotation import read, write

PYTHON = '/private/software/conda/envs/vllm/bin/python'
KEY = '/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'


def checkpoint_verified(path, step):
    marker = path.with_suffix('.complete.json')
    if not path.exists() or not marker.exists():
        return False
    rec = read(marker)
    if rec.get('step') != step:
        raise ValueError('Wrong checkpoint step')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != rec['sha256']:
        raise ValueError('Checkpoint checksum mismatch; training must stay running')
    return True


def remote(node, code, timeout=60):
    cmd = [PYTHON, '-c', code]
    if socket.gethostname() != node['hostname']:
        cmd = ['ssh', '-i', node.get('ssh_key', KEY), '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
               '-o', 'ControlMaster=auto', '-o', 'ControlPersist=300',
               '-o', 'ControlPath=/tmp/listen-reaction-%r-%h-%p',
               '-o', 'ServerAliveCountMax=2', '-o', 'StrictHostKeyChecking=accept-new',
               'root@' + node['host'], shlex.join(cmd)]
    for attempt in range(3):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        # Authentication never began: the remote command did not run. Retry
        # these transient handshakes, never an ambiguous observation timeout.
        if result.returncode == 255 and 'kex_exchange_identification' in result.stderr and attempt < 2:
            time.sleep(1 + attempt)
            continue
        break
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def stop_code(config):
    # Inspect /proc argv, never kill by a loose process-name match.
    return f'''
import os,signal,json,time,subprocess
from pathlib import Path
config={str(config)!r}
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20)
known={{int(s.strip()) for s in raw.splitlines() if s.strip().isdigit()}}
controller=Path({str(ROOT/'experiments/p2_j_base_h_av_event_feature_20261103/controller.pid')!r})
if controller.exists():known.add(int(controller.read_text()))
for child in list(known):
    for _ in range(5):
        try:
            fields=Path('/proc/'+str(child)+'/stat').read_text().rsplit(')',1)[1].split();parent=int(fields[1])
        except (OSError,ValueError):break
        if parent<=1:break
        known.add(parent);child=parent
def targets():
    found=[]
    for pid in known:
        if pid==os.getpid():continue
        p=Path('/proc')/str(pid)
        try:a=(p/'cmdline').read_bytes().split(b'\\0'); a=[x.decode(errors='replace') for x in a if x]
        except (OSError,ProcessLookupError):continue
        if config not in a:continue
        if any(x.endswith(('p2_j_train_observed.py','p2_j_event_feature_launch.py')) for x in a):found.append(int(p.name))
    return found
pids=targets()
for pid in pids:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
limit=time.time()+30
while targets() and time.time()<limit:time.sleep(1)
forced=targets()
for pid in forced:
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
print(json.dumps(dict(stopped=pids,forced=forced,remaining=targets())))
'''


def ensure_code(work, name, refresh_detection=False):
    return f'''
import os,sys,json,time,signal,subprocess
from pathlib import Path
sys.path[:0]=[{str(ROOT/'scripts')!r},{str(ROOT/'src')!r}]
import listen_j_reaction_vocal_omni as m
work=Path({str(work)!r});name={name!r}; nw=work/'nodes'/name
nw.mkdir(parents=True,exist_ok=True)
def get(path):
    try:return m.read(path)
    except FileNotFoundError:return {{}}
actions=[]
if {refresh_detection!r}:
    old=get(nw/'detect_supervisor.json')
    # Earlier CPU supervisors did not record start_ticks. Verify exact argv
    # before adopting their PID, rather than assuming a stale PID is ours.
    if old.get('pid') and not old.get('start_ticks'):
        try:
            argv=Path('/proc/'+str(old['pid'])+'/cmdline').read_bytes().split(b'\\0')
            if b'detect-supervise' in argv and str(work).encode() in argv and name.encode() in argv:
                old['start_ticks']=m.identity(old['pid'])
        except FileNotFoundError:pass
    if m.alive(old):
        os.kill(old['pid'],signal.SIGTERM)
        end=time.time()+25
        while m.alive(old) and time.time()<end:time.sleep(.5)
        if m.alive(old):raise RuntimeError('Old CPU supervisor still stopping; workers were not killed')
for label,command,statefile in [('detect-supervisor','detect-supervise','detect_supervisor.json'),('watchdog','watchdog','watchdog.json')]:
    rec=get(nw/statefile)
    if m.alive(rec):continue
    env=m.environment();env['VLLM_GPU_MEM']='0.78';env['J_OMNI_FER_GPU']=''
    args=[m.PYTHON,str(Path(m.__file__).resolve()),command,'--work',str(work),'--node',name]
    if command=='detect-supervise':args+=['--with-producer']
    with (nw/(label+'-rollout.log')).open('a') as log:
        p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    actions.append(dict(command=command,pid=p.pid))
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
gpus=[[int(x.strip()) for x in line.split(',')] for line in raw.splitlines()]
util_samples=[[row[2] for row in gpus]]
for _ in range(2):
    time.sleep(1)
    sample=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    util_samples.append([int(x.strip()) for x in sample.splitlines()])
workers=[]
for gpu in range(8):
    rec=get(work/'workers'/f'{{name}}-gpu-{{gpu}}.json')
    workers.append(dict(gpu=gpu,alive=m.alive(rec),state=rec))
print(json.dumps(dict(node=name,actions=actions,gpus=gpus,gpu_utilization_mean=sum(map(sum,util_samples))/sum(map(len,util_samples)),load=os.getloadavg(),workers=workers,detect=get(nw/'detect_supervisor.json'),supervisor=get(nw/'supervisor.json'))))
'''


def run(work, train, config, target, report_seconds=5400):
    control = work / 'cluster_control'
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(work / 'config.json')
    statefile = control / 'state.json'
    state = read(statefile) if statefile.exists() else dict(target_step=target, phase='waiting_checkpoint')
    if state['target_step'] != target:
        raise ValueError('Existing rollout has another checkpoint target')
    state.update(pid=os.getpid(), started_at=state.get('started_at', time.time()))
    checkpoint = train / f'step_{target:06d}.pt'
    while state['phase'] == 'waiting_checkpoint':
        if checkpoint_verified(checkpoint, target):
            state.update(phase='stopping_training', checkpoint=str(checkpoint), verified_at=time.time())
        state['heartbeat'] = time.time(); write(statefile, state)
        if state['phase'] == 'waiting_checkpoint':time.sleep(10)
    while state['phase'] == 'stopping_training':
        def stop_node(item):
            name,node=item
            try:return name, remote(node, stop_code(config), 70)
            except Exception as exc:return name,dict(error=repr(exc))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(stop_node,cfg['nodes'].items()))
        write(control / 'training_stop.json', dict(checkpoint=str(checkpoint), nodes=dict(results), at=time.time()))
        if any(rec.get('error') or rec.get('remaining') for _, rec in results):
            state.update(heartbeat=time.time(),errors=dict(results));write(statefile,state);time.sleep(10);continue
        state.update(phase='running', annotation_started_at=time.time())
        write(statefile, state)
    previous_reports = list((control / 'reports').glob('report-*.json'))
    last_report = state.get('last_report_at') or max(
        (int(p.stem.removeprefix('report-')) for p in previous_reports), default=0
    )
    refresh = not (control / 'detection_supervisors_refreshed.json').exists()
    while not (control / 'STOP').exists():
        def inspect(item):
            name, node = item
            try:return name, remote(node, ensure_code(work, name, refresh), 80)
            except Exception as exc:return name, dict(node=name, error=repr(exc))
        with ThreadPoolExecutor(max_workers=4) as pool: snapshots = dict(pool.map(inspect, cfg['nodes'].items()))
        if refresh and all('error' not in s for s in snapshots.values()):
            write(control / 'detection_supervisors_refreshed.json', {'at':time.time()}); refresh=False
        now = time.time()
        queues = {}
        for branch in ['detect','caption']:
            path = work / 'queues' / branch / 'state.json'
            if path.exists():
                q = read(path)
                queues[branch] = {k:q.get(k) for k in ['completed','completed_clips']}
                queues[branch].update(failed=len(q.get('failed',{})), inflight=len(q.get('inflight',{})), deferred=len(q.get('deferred',{})))
        rates={}
        previous=state.get('rate_baseline')
        if previous and now-previous['at']>=300:
            elapsed=now-previous['at']
            for branch,q in queues.items():
                rates[branch]=max(0,(q['completed_clips']-previous['counts'].get(branch,0))/elapsed)
        if not previous or now-previous['at']>=900:
            state['rate_baseline']=dict(at=now,counts={b:q['completed_clips'] for b,q in queues.items()})
        report = dict(at=now, nodes=snapshots, queues=queues, receipt_rates_per_s=rates,target_clips=cfg['total_retained_clips'])
        write(control / 'latest.json', report)
        with (control/'telemetry.jsonl').open('a') as history:
            history.write(json.dumps(dict(at=now,queues=queues,rates=rates,nodes={n:dict(gpu_utilization_mean=s.get('gpu_utilization_mean'),error=s.get('error')) for n,s in snapshots.items()}))+'\n')
        if now-last_report >= report_seconds:
            write(control / 'reports' / f'report-{int(now)}.json', report)
            print(json.dumps(dict(event='resource_report',at=now,queues=queues,report=str(control/'latest.json'))),flush=True)
            last_report=now
            state['last_report_at']=now
        state.update(phase='running',heartbeat=now,errors={k:v['error'] for k,v in snapshots.items() if 'error' in v})
        write(statefile,state)
        time.sleep(45)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--training-work',type=Path,required=True)
    p.add_argument('--training-config',type=Path,required=True)
    p.add_argument('--checkpoint-step',type=int,required=True)
    p.add_argument('--report-seconds',type=int,default=5400)
    a=p.parse_args()
    run(a.work.resolve(),a.training_work.resolve(),a.training_config.resolve(),a.checkpoint_step,a.report_seconds)
