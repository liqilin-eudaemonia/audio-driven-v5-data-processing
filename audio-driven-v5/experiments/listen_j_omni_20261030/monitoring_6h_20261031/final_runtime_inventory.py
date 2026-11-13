"""Read-only, identity-bound inventory after the authorized six-hour window."""
import concurrent.futures
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
WORK = FOLDER.parent
ROOT = WORK.parents[1]
KEY = '/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'

REMOTE = r'''
import json,socket,time
from pathlib import Path
def read(path):
    for attempt in range(3):
        try:return json.loads(Path(path).read_text())
        except (OSError,ValueError):
            if attempt==2:raise
            time.sleep(.1)
def stat(pid):
    f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(start_ticks=f[19],state=f[0],cpu_ticks=int(f[11])+int(f[12]))
def verify(record,fragment):
    s=stat(record['pid'])
    assert s['start_ticks']==record['start_ticks'] and s['state']!='Z',record
    assert fragment in Path(f"/proc/{record['pid']}/cmdline").read_bytes().decode(),record
    return dict(pid=record['pid'],**s)
w=Path(WORK);cfg=read(w/'config.json');nc=cfg['nodes'][NODE]
assert socket.gethostname()==nc['hostname']
nw=w/'nodes'/NODE
watch=verify(read(nw/'watchdog.json'),'listen_j_omni.py')
supervisor=read(nw/'supervisor.json');verify(supervisor,'listen_j_omni.py')
gpu=[]
for g in nc['gpu_ids']:
    child=supervisor['children'][f'gpu-{g}'];actual=verify(child,'listen_j_omni.py')
    h=read(w/'workers'/f'{NODE}-gpu-{g}.json')
    assert h['pid']==child['pid'] and time.time()-h['heartbeat']<60
    assert not h.get('last_error') and not h['failed_clips']
    assert h['engine_max_num_seqs']==1 and h['production_batch_size']==4 and h['adaptive_request_groups']
    assert h['preparation_profile']==cfg['preparation_profiles'][f'{NODE}-gpu-{g}']
    gpu.append(dict(gpu=g,identity=actual,stage=h['stage'],completed=h['completed_clips'],heartbeat_age=time.time()-h['heartbeat']))
qn=w/'visual_quality/nodes'/NODE
quality_watch=verify(read(qn/'watchdog.json'),'listen_j_visual_gate_cluster.py')
q=read(qn/'controller.json');quality_controller=verify(q,'listen_j_visual_gate_cluster.py')
assert q['workers']==nc['visual_quality_workers'] and q['stage']=='running'
assert time.time()-q['heartbeat']<60 and not q['failed_sources'] and not q['execution_errors_this_process']
pid=q['pid'];children=[int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
children=[p for p in children if b'spawn_main' in Path(f'/proc/{p}/cmdline').read_bytes()]
assert len(children)==q['workers'],(NODE,len(children),q['workers'])
before={p:stat(p) for p in children};time.sleep(2);after={p:stat(p) for p in children}
assert all(before[p]['start_ticks']==after[p]['start_ticks'] and after[p]['state']!='Z' for p in children)
nod=[]
if nc.get('old_work'):
    old=Path(nc['old_work']);oc=read(old/'config.json');sup=read(old/'supervisor.json')
    verify(sup,'listen_j_annotation_supervisor.py')
    assert not read(old/'queues/nod/state.json')['failed']
    for i in range(oc['nod_workers']):
        child=sup['children'][f'nod-{i}'];actual=verify(child,'listen_j_annotation_nod.py')
        h=read(old/'workers'/f'nod-{i}.json')
        assert h['pid']==child['pid'] and h['revision']>=4 and not h['errors'] and time.time()-h['heartbeat']<60
        nod.append(dict(slot=i,identity=actual,stage=h['stage'],heartbeat_age=time.time()-h['heartbeat']))
print(json.dumps(dict(at=time.time(),node=NODE,watchdog=watch,gpu=gpu,quality_watchdog=quality_watch,quality_controller=quality_controller,quality_pool=after,quality_pool_CPU_advanced=sum(after[p]['cpu_ticks']>before[p]['cpu_ticks'] for p in children),quality_heartbeat_age=time.time()-q['heartbeat'],nod=nod)))
'''


def probe(node, cfg):
    nc = cfg['nodes'][node]
    code = f'WORK={str(WORK)!r}\nNODE={node!r}\n' + REMOTE
    command = [sys.executable, '-'] if socket.gethostname() == nc['hostname'] else [
        'ssh', '-i', nc.get('ssh_key', KEY), '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=8', '-o', 'ControlMaster=auto',
        '-o', 'ControlPersist=600', '-o', 'ControlPath=/tmp/jomni-guard-%C',
        'root@' + nc['host'], 'python3 -']
    result = subprocess.run(command, input=code, text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'{node}: {result.stderr[-2000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    plan = json.loads((FOLDER/'plan.json').read_text())
    if time.time() < plan['end']:
        raise RuntimeError('Final inventory is not due before the six-hour window ends')
    cfg = json.loads((WORK/'config.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {n: pool.submit(probe, n, cfg) for n in cfg['nodes']}
        nodes = {n: f.result() for n, f in futures.items()}
    result = dict(at=time.time(),passed=True,nodes=nodes,
                  gpu_workers=sum(len(n['gpu']) for n in nodes.values()),
                  quality_pool_workers=sum(len(n['quality_pool']) for n in nodes.values()),
                  nod_workers=sum(len(n['nod']) for n in nodes.values()),
                  scope='Actual process identities, bounded child-process inventory, fresh heartbeats; no process changes or full result scan.')
    (FOLDER/'final_runtime_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'}))


if __name__ == '__main__':
    main()
