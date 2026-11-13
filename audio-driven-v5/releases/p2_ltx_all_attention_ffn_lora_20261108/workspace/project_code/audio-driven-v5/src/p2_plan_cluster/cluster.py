"""SSH orchestration only: one independent cache shard per selected GPU."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from p2_listen.contracts import ROOT, atomic_json, digest, read, sha

DEFAULT = ROOT/'configs/p2_plan_cache_3nodes.json'
SCRIPT = ROOT/'scripts/p2_plan_cache_cluster.py'
STAGES = ('cache-text', 'cache-input-plan', 'verify')


def load(path=DEFAULT):
    c = read(path)
    if c['schema'] != 'p2-plan-cache-cluster-v1' or c['stages'] != list(STAGES):
        raise ValueError('Invalid cluster/stage protocol')
    nodes = c['nodes']
    if len(nodes) != 3 or [n['role'] for n in nodes] != ['master', 'slave', 'slave']:
        raise ValueError('Expected one master followed by two slaves')
    if len({n['host'] for n in nodes}) != 3 or len({n['node'] for n in nodes}) != 3:
        raise ValueError('Duplicate host/node')
    for n in nodes:
        if not n['gpus'] or len(set(n['gpus'])) != len(n['gpus']) or any(type(g) is not int or g < 0 for g in n['gpus']):
            raise ValueError('Invalid GPU allocation')
        if type(n['ssh_port']) is not int or not 1 <= n['ssh_port'] <= 65535:
            raise ValueError('Invalid SSH port')
    if type(c['retries']) is not int or not 0 <= c['retries'] <= 3:
        raise ValueError('Retries must be bounded (0..3)')
    for name in ('max_idle_memory_mib', 'max_idle_utilization_percent'):
        if type(c[name]) is not int or c[name] < 0:
            raise ValueError('Invalid idle threshold')
    policy=c.get('gpu_policy',{'mode':'exclusive'})
    if policy['mode'] not in ('exclusive','shared'):raise ValueError('Unknown GPU policy')
    if policy['mode']=='shared':
        if (policy['min_free_mib'] <= 0 or policy['process_limit_gib'] <= 0 or
                policy['process_limit_gib']*1024 >= policy['min_free_mib']):
            raise ValueError('Shared GPU policy requires positive memory headroom')
    for name in ('project_root', 'python', 'ssh_key', 'data_config', 'work'):
        if not Path(c[name]).is_absolute():
            raise ValueError('Shared paths must be absolute')
    if Path(c['project_root']).resolve() != ROOT:
        raise ValueError('Cluster points to a different code worktree')
    return c


def assignments(c):
    result = []
    for node in c['nodes']:
        for gpu in node['gpus']:
            result.append(dict(rank=len(result), node=node['node'], host=node['host'], gpu=gpu))
    return result


def source_identity():
    files = [SCRIPT, ROOT/'scripts/p2_plan_cache.py']
    for package in ('p2_plan_cluster', 'p2_plan_cache'):
        files.extend((ROOT/'src'/package).glob('*.py'))
    return {str(p): sha(p) for p in sorted(files)}


def ssh(c, node, args):
    return ['ssh', '-i', c['ssh_key'], '-p', str(node['ssh_port']),
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            node['ssh_user']+'@'+node['host'], shlex.join([c['python'], str(SCRIPT), *map(str, args)])]


def remote(c, node, args, timeout=90):
    for attempt in range(3):
        r = subprocess.run(ssh(c, node, args), capture_output=True, text=True, timeout=timeout)
        if r.returncode==255 and 'kex_exchange_identification' in r.stderr and attempt<2:
            time.sleep(attempt+1);continue
        break
    if r.returncode:
        raise RuntimeError(f"{node['node']} SSH command failed: {r.stderr[-2000:]} {r.stdout[-1000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def gpu_info():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    result = []
    for line in rows.splitlines():
        idx, uid, name, used, total, util, free = [v.strip() for v in line.split(',')]
        result.append(dict(index=int(idx), uuid=uid, name=name, used_mib=int(used), total_mib=int(total), utilization=int(util),free_mib=int(free)))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    for row in result:
        row['compute_process_present'] = row['uuid'] in busy
    return result


def idle(row, c):
    return (not row['compute_process_present'] and row['used_mib'] <= c['max_idle_memory_mib']
            and row['utilization'] <= c['max_idle_utilization_percent'])


def available(row,c):
    policy=c.get('gpu_policy',{'mode':'exclusive'})
    return row['free_mib']>=policy['min_free_mib'] if policy['mode']=='shared' else idle(row,c)


def probe_node(challenge, node_name):
    p = read(challenge); c = p['config']
    if source_identity() != p['source']:
        raise ValueError('Remote source differs')
    if not os.access(c['python'], os.X_OK) or not Path(c['data_config']).is_file():
        raise ValueError('Environment/data config unavailable')
    from p2_plan_cache.runtime import dependencies
    data = read(c['data_config'])
    result = dict(node=node_name, hostname=socket.gethostname(), nonce=p['nonce'],
                  code_sha256=digest(p['source']), data_config_sha256=sha(c['data_config']),
                  dependencies={part:digest(dependencies(data,part)) for part in ('text','input_plan')},
                  gpus=gpu_info(), python=sys.executable, success=True)
    atomic_json(Path(challenge).parent/(node_name+'.json'), result)
    return result


def probe(c, output):
    folder = Path(output).resolve()/('probe-'+uuid.uuid4().hex)
    folder.mkdir(parents=True)
    challenge = folder/'challenge.json'
    atomic_json(challenge, dict(config=c, source=source_identity(), nonce=uuid.uuid4().hex))
    def one(n):
        try:
            result = remote(c,n,['_probe','--challenge',challenge,'--node',n['node']])
            if read(folder/(n['node']+'.json')) != result or result['nonce'] != read(challenge)['nonce']:
                raise ValueError('Storage is not shared bidirectionally')
            gpus = {g['index']:g for g in result['gpus']}
            if set(n['gpus'])-gpus.keys():
                raise ValueError('Configured GPU missing')
            result['idle'] = all(idle(gpus[g],c) for g in n['gpus'])
            result['available'] = all(available(gpus[g],c) for g in n['gpus'])
            return result
        except Exception as exc:
            return dict(node=n['node'], success=False, error=str(exc), idle=False,available=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one,c['nodes']))
    success = all(r['success'] for r in results)
    if success:
        success = len({digest([r['dependencies'],r['data_config_sha256']]) for r in results}) == 1
    report = dict(success=success, all_idle=success and all(r['idle'] for r in results),
                  all_available=success and all(r['available'] for r in results),
                  nodes=results, output=str(folder), checked_at=time.time())
    atomic_json(folder/'report.json',report)
    return report


def worker_command(c, job, rank, stage, smoke=False):
    if stage not in STAGES:
        raise ValueError('Invalid stage')
    if smoke:
        return [c['python'], str(SCRIPT), '_smoke', '--job', str(job), '--rank', str(rank), '--stage', stage]
    cmd = [c['python'], str(ROOT/'scripts/p2_plan_cache.py'), stage, '--work', c['work'],
           '--shards', str(len(assignments(c))), '--shard-index', str(rank)]
    if stage != 'verify':
        cmd.extend(['--device','cuda:0'])  # CUDA_VISIBLE_DEVICES contains one physical GPU.
    return cmd


def check_job(job):
    m = read(Path(job)/'job.json')
    if source_identity() != m['source']:
        raise ValueError('Frozen cluster/cache source changed')
    if not m['smoke'] and sha(Path(m['config']['work'])/'protocol.json') != m['protocol_sha256']:
        raise ValueError('Prepared protocol changed')
    return m


@contextmanager
def node_lock(job, node):
    path=Path(job)/'nodes'/(node+'.lock'); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def supervise(job, node_name):
    job=Path(job); m=check_job(job); c=m['config']
    lanes=[a for a in assignments(c) if a['node']==node_name]
    if not lanes:
        raise ValueError('Unknown node')
    active={}; states={}; stopped=[False]
    def stop(*args): stopped[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def save(rank):
        atomic_json(job/'workers'/f'{rank:03d}.json',dict(states[rank],updated_at=time.time()))
    def node_status(state):
        atomic_json(job/'nodes'/(node_name+'.json'),dict(node=node_name,state=state,pid=os.getpid(),heartbeat=time.time()))
    with node_lock(job,node_name):
        try:
            if not m['smoke']:
                gpus={g['index']:g for g in gpu_info()}
                if any(not available(gpus[a['gpu']],c) for a in lanes):
                    raise RuntimeError('Selected GPU fails resource policy; no workers launched on this node')
            for a in lanes:
                states[a['rank']]=dict(**a,state='pending',stage_index=0,attempt=0)
            while True:
                if stopped[0] or (job/'STOP').exists():
                    raise RuntimeError('Cluster job stopped')
                check_job(job)
                for a in lanes:
                    rank=a['rank']; state=states[rank]
                    if rank in active:
                        process,log=active[rank]; rc=process.poll()
                        if rc is None:continue
                        log.close(); del active[rank]; state['returncode']=rc
                        if rc == 0:
                            state['stage_index']+=1; state['attempt']=0
                            state['state']='succeeded' if state['stage_index']==len(STAGES) else 'pending'
                        elif state['attempt'] <= c['retries']:
                            state['state']='pending'
                        else:state['state']='failed'
                        save(rank)
                    if state['state']=='pending':
                        stage=STAGES[state['stage_index']]
                        if not m['smoke'] and stage!='verify':
                            current={g['index']:g for g in gpu_info()}[a['gpu']]
                            if not available(current,c):
                                state.setdefault('waiting_since',time.time())
                                state['waiting_for_gpu']=True
                                if time.time()-state['waiting_since']>900:
                                    state.update(state='failed',error='GPU admission wait exceeded 900 seconds')
                                save(rank);continue
                            state.pop('waiting_since',None);state.pop('waiting_for_gpu',None)
                        state['attempt']+=1
                        logpath=job/'logs'/f'{rank:03d}.{stage}.attempt-{state["attempt"]}.log'
                        logpath.parent.mkdir(parents=True,exist_ok=True)
                        log=logpath.open('a')
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a['gpu']),OMP_NUM_THREADS='2',
                                 OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
                        if c.get('gpu_policy',{}).get('mode')=='shared':
                            env['P2_PLAN_GPU_MEMORY_GIB']=str(c['gpu_policy']['process_limit_gib'])
                        process=subprocess.Popen(worker_command(c,job,rank,stage,m['smoke']),cwd=c['project_root'],
                                                 env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        active[rank]=(process,log)
                        state.update(state='running',stage=stage,pid=process.pid,log=str(logpath));save(rank)
                node_status('running')
                if not active and all(s['state'] in ('succeeded','failed') for s in states.values()):break
                time.sleep(.5)
            success=all(s['state']=='succeeded' for s in states.values())
            node_status('succeeded' if success else 'failed')
            return dict(success=success,node=node_name)
        except Exception as exc:
            for process,log in active.values():
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
            for process,log in active.values():
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                log.close()
            for a in lanes:
                state=states.setdefault(a['rank'],dict(**a))
                if state.get('state')!='succeeded':state.update(state='failed',error=str(exc));save(a['rank'])
            node_status('failed')
            return dict(success=False,node=node_name,error=str(exc))


def detach(job, node):
    job=Path(job);m=check_job(job)
    if (job/'STOP').exists():raise ValueError('Stopped job: use a new job directory')
    # The supervisor holds this lock for its entire lifetime. Duplicate requests
    # cannot run a second set of workers; the child also acquires the same lock.
    try:
        with node_lock(job,node):pass
    except BlockingIOError:return dict(success=True,node=node,state='already_running')
    logpath=job/'logs'/(node+'.supervisor.log');logpath.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(job/'nodes'/(node+'.json'),dict(node=node,state='starting',pid=None,heartbeat=time.time()))
    with logpath.open('a') as log:
        p=subprocess.Popen([m['config']['python'],str(SCRIPT),'_node','--job',str(job),'--node',node],
                           cwd=m['config']['project_root'],stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
    return dict(success=True,node=node,state='launched',pid=p.pid,log=str(logpath))


def start(c, job, smoke=False):
    job=Path(job).resolve()
    if job == ROOT or job == Path(c['work']) or job.is_relative_to(Path(c['work'])) or job.is_relative_to(Path(c['data_config']).parent):
        raise ValueError('Use a separate job directory outside the dataset/cache work')
    # Concurrent invocations must not race while replacing launch/status records.
    with node_lock(job,'launch'):
        return _start(c,job,smoke)


def _start(c, job, smoke=False):
    job=Path(job).resolve()
    if job == ROOT or job == Path(c['work']) or (Path(c['work']).exists() and job.is_relative_to(Path(c['work']))):
        raise ValueError('Use a separate job directory')
    if (job/'STOP').exists():raise ValueError('Stopped job: use a new job directory')
    if not smoke:
        from p2_plan_cache.protocol import open_protocol
        p=open_protocol(c['work'])
        if p['config'] != read(c['data_config']):raise ValueError('Prepared work uses a different dataset config')
    m=dict(config=c,source=source_identity(),smoke=smoke,
           protocol_sha256=None if smoke else sha(Path(c['work'])/'protocol.json'),assignments=assignments(c))
    if (job/'job.json').exists() and read(job/'job.json') != m:
        raise ValueError('Job identity changed; choose a new job directory')
    if (job/'job.json').exists():
        prior=status(job)
        if prior['success']:
            return dict(success=True,job=str(job),state='already_completed')
        if any(n['state'] in ('starting','running') for n in prior['nodes']):
            raise RuntimeError('Node supervisors are still active; use status before retrying')
    report=probe(c,job/'preflight')
    if not report['success'] or (not smoke and not report.get('all_available',report['all_idle'])):
        raise RuntimeError('Cluster preflight failed or GPUs occupied; see '+report['output'])
    atomic_json(job/'job.json',m,immutable=True)
    # Only prior control JSON is reset. Cache payloads and logs remain intact.
    for folder in ('nodes','workers'):
        for path in (job/folder).glob('*.json'):path.unlink()
    def one(node):
        try:return remote(c,node,['_detach','--job',job,'--node',node['node']])
        except Exception as exc:return dict(node=node['node'],state='launch_failed',error=str(exc))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,c['nodes']))
    atomic_json(job/'launch.json',dict(results=results,at=time.time()))
    return dict(success=all(r['state']!='launch_failed' for r in results),job=str(job),nodes=results)


def status(job):
    job=Path(job);m=read(job/'job.json');rows=[];nodes=[]
    launches=read(job/'launch.json')['results'] if (job/'launch.json').exists() else []
    failed_launches={n['node']:n for n in launches if n['state']=='launch_failed'}
    for n in m['config']['nodes']:
        path=job/'nodes'/(n['node']+'.json')
        record=read(path) if path.exists() else dict(node=n['node'],state='not_started')
        if record['state']=='not_started' and n['node'] in failed_launches:
            record=failed_launches[n['node']]
        if record['state'] in ('starting','running') and time.time()-record['heartbeat']>120:
            record=dict(record,state='stale')
        nodes.append(record)
    for a in m['assignments']:
        path=job/'workers'/f'{a["rank"]:03d}.json'
        rows.append(read(path) if path.exists() else dict(**a,state='not_started'))
    success=all(r['state']=='succeeded' for r in rows) and all(n['state']=='succeeded' for n in nodes)
    failed=any(r['state']=='failed' for r in rows) or any(n['state'] in ('failed','stale','launch_failed') for n in nodes)
    result=dict(success=success,state='succeeded' if success else 'failed' if failed else 'running',
                workers=rows,nodes=nodes,finished=sum(r['state']=='succeeded' for r in rows),total=len(rows),smoke=m['smoke'])
    atomic_json(job/'status.json',result)
    return result


def smoke_worker(job, rank, stage):
    from p2_plan_cache.protocol import select
    job=Path(job);m=check_job(job)
    if not m['smoke']:raise ValueError('Not a smoke job')
    ids=[f'sample-{i:04d}' for i in range(257)]
    chosen=select(dict(sample_ids=ids,entries=dict.fromkeys(ids)),shard_index=rank,shards=len(m['assignments']))
    atomic_json(job/'smoke'/stage/f'{rank:03d}.json',dict(rank=rank,stage=stage,ids=chosen,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_used=False))
    return dict(success=True,rank=rank,samples=len(chosen))


def smoke_check(job):
    job=Path(job);m=read(job/'job.json')
    if not m['smoke']:raise ValueError('Not a smoke job')
    expected={f'sample-{i:04d}' for i in range(257)}
    for stage in STAGES:
        rows=[read(job/'smoke'/stage/f'{a["rank"]:03d}.json') for a in m['assignments']]
        ids=[sid for r in rows for sid in r['ids']]
        if len(ids)!=len(expected) or set(ids)!=expected:raise ValueError('Lost/duplicated shard coverage')
        for a,r in zip(m['assignments'],rows):
            if r['visible_devices']!=str(a['gpu']) or r['gpu_used']:raise ValueError('Wrong device assignment')
    for a in m['assignments']:
        if len({digest(read(job/'smoke'/stage/f'{a["rank"]:03d}.json')['ids']) for stage in STAGES})!=1:
            raise ValueError('Stage sample mapping differs')
    return dict(success=True,workers=len(m['assignments']),stages=len(STAGES),samples=257,gpu_used=False)
