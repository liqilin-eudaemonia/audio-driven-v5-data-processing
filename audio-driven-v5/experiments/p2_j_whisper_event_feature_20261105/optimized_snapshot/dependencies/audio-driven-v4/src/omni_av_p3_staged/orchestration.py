"""Owned process orchestration; no training imports or CUDA initialization."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def stopped(root):return (Path(root)/'STOP').exists()


def engineering_complete(folder, stage, phase, manifest_sha=None, world_size=None):
    """The initial gate intentionally stops at 2; resumed gate must finish at 3."""
    folder=Path(folder)
    if phase not in ('initial','resume'):raise ValueError('Invalid engineering phase')
    expected=2 if phase=='initial' else 3
    try:
        result=read(folder/'completed.json')
        if (result.get('stage')!=stage or result.get('step')!=expected
                or result.get('engineering_gate') is not True
                or result.get('stopped') is not (phase=='initial')):return False
        cp=folder/f'step_{expected:06d}.pt';marker=read(cp.with_suffix('.complete.json'))
        identity=read(folder/'run.json')['identity']
        return (marker.get('step')==expected and marker.get('sha256')==sha(cp)
                and identity.get('stage')==stage and identity.get('engineering_gate') is True
                and (world_size is None or identity.get('world_size')==world_size)
                and (manifest_sha is None or identity.get('manifest_sha256')==manifest_sha))
    except (OSError,ValueError,KeyError,TypeError):return False


def update_phase(state, **changes):
    """Keep phase start stable while the queue emits frequent polling updates."""
    now=time.time()
    if any(key in changes and changes[key]!=state.get(key) for key in ('status','stage','phase','folder')):
        state['phase_started']=now
    state.update(changes,time=now)


def notify(root,thread,kind,details):
    plan_path=Path(root)/'plan.json'
    plan=read(plan_path) if plan_path.exists() else {}
    parent=Path(plan.get('parent',root))/'plan.json'
    if parent!=plan_path and parent.exists():plan=read(parent)
    c0_steps=plan.get('scheduled_transition',{}).get('from_step',plan.get('stage_steps',{}).get('C0',2000))
    message=(f'[P3全量Connector→DiT训练监控] {kind}；目录：{root}。请先读plan.json、state.json、health.json、alerts.jsonl和当前日志确认现状。'
             '仅处理本次P3队列，不停止其他作业；等待残差P2.1最终验证完成后再用空卡。'
             f'C0只训练Connector{c0_steps}步，A额外开放DiT text-attention LoRA独立1000步；global batch72、原始720p-area、每100步保存。'
             '保留STOP与严格恢复，不自动启动B。'+json.dumps(details,ensure_ascii=False)[:1500])
    try:
        p=subprocess.run([shutil.which('codex') or 'codex','queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=40)
        receipt=dict(time=time.time(),kind=kind,returncode=p.returncode,stdout=p.stdout[-1000:],stderr=p.stderr[-1000:])
    except Exception as e:receipt=dict(time=time.time(),kind=kind,returncode=-1,error=repr(e))
    with (Path(root)/'notifications.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    return receipt['returncode']==0


def transport(cluster,node,command):
    if node.get('local'):return command
    controls=['-S',node['control_path'],'-o','ControlMaster=auto','-o','ControlPersist=14400'] if node.get('control_path') else []
    proxy=['-o','ProxyCommand='+node['proxy_command']] if node.get('proxy_command') else []
    return ['ssh',*controls,*proxy,'-i',cluster['ssh_key'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
            'root@'+node['host'],shlex.join(command)]


def probe(cluster):
    def one(node):
        command=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_worker.py','probe',
                 '--gpus',','.join(map(str,node['gpus']))]
        errors=[]
        for _ in range(3):
            try:
                p=subprocess.run(transport(cluster,node,command),capture_output=True,text=True,timeout=35)
                if p.returncode==0:return dict(node=node['name'],ok=True,**json.loads(p.stdout))
                errors.append(p.stderr[-800:])
            except Exception as e:errors.append(repr(e))
            time.sleep(1)
        return dict(node=node['name'],ok=False,errors=errors)
    with ThreadPoolExecutor(len(cluster['nodes'])) as pool:return list(pool.map(one,cluster['nodes']))


def free_slots(cluster,status):
    by={n['node']:n for n in status}
    return [(node,g['index']) for node in cluster['nodes'] for g in by.get(node['name'],{}).get('gpus',[])
            if g['index'] in node['gpus'] and g['memory_mib']<1000 and g['util']<=5 and not g['pids']]


def prerequisite(root):
    root=Path(root);done=root/'completed.json';report=root/'report.json';generation=root/'generation/completed.json'
    if not all(p.is_file() for p in (done,report,generation)):return False
    d=read(done);g=read(generation);r=read(report)
    if d.get('status')!='completed' or d.get('checkpoint_step')!=1000 or not g.get('passed'):return False
    if len(r.get('cases',[]))!=48:return False
    for mode in ('gt','omni','reference'):
        if r.get('summary',{}).get(mode,{}).get('scored')!=48:return False
    return True


def wait_idle(root,cluster,update):
    while not stopped(root):
        status=probe(cluster);write(Path(root)/'health.json',dict(time=time.time(),nodes=status))
        if len(free_slots(cluster,status))==sum(len(n['gpus']) for n in cluster['nodes']):return
        update(status='waiting_gpus');time.sleep(15)
    raise InterruptedError('Persistent STOP')


def finish_owned(tasks):
    for task in tasks:
        if task['process'].stdin and not task['process'].stdin.closed:task['process'].stdin.close()
    deadline=time.monotonic()+35
    for task in tasks:
        p=task['process']
        try:p.wait(timeout=max(.1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:p.terminate()
        task['log'].close()


def run_commands(root,cluster,commands,logs,heartbeat=None,timeout=86400):
    """Commands have explicit node/GPU ownership; guards kill only child groups."""
    logs=Path(logs);logs.mkdir(parents=True,exist_ok=True);active=[];begin=time.monotonic()
    try:
        for name,node,command in commands:
            f=(logs/(name+'.log')).open('a')
            guarded=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_worker.py','guard','--',*command]
            p=subprocess.Popen(transport(cluster,node,guarded),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
            active.append(dict(name=name,process=p,log=f))
        while any(t['process'].poll() is None for t in active):
            if stopped(root):
                # Training worker sees root/STOP and saves; give it a bounded grace.
                if heartbeat and heartbeat():
                    until=time.monotonic()+900
                    while time.monotonic()<until and any(t['process'].poll() is None for t in active):time.sleep(2)
                raise InterruptedError('User STOP; no next phase')
            failures=[(t['name'],t['process'].returncode) for t in active if t['process'].poll() not in (None,0)]
            if failures:raise RuntimeError('Owned command failed: '+repr(failures))
            if time.monotonic()-begin>timeout:raise TimeoutError('Owned phase exceeded timeout')
            time.sleep(3)
        if any(t['process'].returncode for t in active):raise RuntimeError('Owned command failed at exit')
    finally:finish_owned(active)


def run_pool(root,cluster,jobs,logs,update,timeout=10800):
    pending=list(jobs);active=[];logs=Path(logs);logs.mkdir(parents=True,exist_ok=True)
    last_probe=0;available=[]
    try:
        while pending or active:
            if stopped(root):raise InterruptedError('User STOP')
            for task in active[:]:
                rc=task['process'].poll()
                if rc is not None:
                    task['process'].stdin.close();task['log'].close();active.remove(task)
                    if rc:raise RuntimeError(f"Pool job failed: {task['name']} rc={rc}")
            if time.time()-last_probe>15:
                busy={(t['node']['name'],t['gpu']) for t in active}
                available=[(n,g) for n,g in free_slots(cluster,probe(cluster)) if (n['name'],g) not in busy];last_probe=time.time()
            while pending and available:
                node,gpu=available.pop();job=pending.pop(0)
                command=[str(gpu) if x=='{GPU}' else x for x in job['command']]
                guard=[cluster['python'],cluster['source']+'/scripts/pipeline3_staged_worker.py','guard','--',*command]
                f=(logs/(job['name']+'.log')).open('a')
                p=subprocess.Popen(transport(cluster,node,guard),stdin=subprocess.PIPE,stdout=f,stderr=subprocess.STDOUT)
                active.append(dict(name=job['name'],node=node,gpu=gpu,process=p,log=f,start=time.time()))
            if any(time.time()-t['start']>timeout for t in active):raise TimeoutError('Pool job timeout')
            update(pending=len(pending),active=[dict(name=t['name'],node=t['node']['name'],gpu=t['gpu']) for t in active])
            time.sleep(3)
    finally:finish_owned(active)
