"""One bounded deployment, not a recurring monitor. No unrelated PID is signalled."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
W=D.parent
R=W.parents[1]
sys.path.insert(0,str(R/'src'))
sys.path.insert(0,str(W/'nod_cpu_share'))
from probe import remote
from listen_benchmark.j_annotation import read,write,acquire
from listen_benchmark.j_omni_tail import PROTOCOL

MARKER='+clip-lock-tail-v1'
changed=['src/listen_benchmark/j_omni.py','src/listen_benchmark/j_omni_stream.py',
         'src/listen_benchmark/j_omni_queued.py','src/listen_benchmark/j_omni_tail.py']


def call(node, code):
    for attempt in range(3):
        try:return remote(node,code,timeout=60)
        except Exception:
            if attempt==2:raise
            time.sleep(2)


def roll(node,gpus):
    code='''import json,os,time,signal,hashlib
from pathlib import Path
W=Path(WORK);R=W.parents[1];cfg=json.loads((W/'config.json').read_text());out=[]
for p in CHANGED:assert hashlib.sha256((R/p).read_bytes()).hexdigest()==cfg['code_hashes'][p],p
for gpu in GPUS:
 s=json.loads((W/'workers'/f'{NODE}-gpu-{gpu}.json').read_text());pid=s['pid']
 if MARKER in s.get('execution_revision','') and s.get('tail_packet_lifecycle')=='finish-before-source-claim':
  out.append(dict(gpu=gpu,pid=pid,action='already_updated'));continue
 p=Path('/proc')/str(pid)
 if not p.exists():out.append(dict(gpu=gpu,pid=pid,action='already_exited'));continue
 cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert 'listen_j_omni.py worker' in cmd and str(W) in cmd and '--node '+NODE in cmd and '--gpu '+str(gpu) in cmd,cmd
 ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
 os.kill(pid,signal.SIGTERM)
 out.append(dict(gpu=gpu,pid=pid,start_ticks=ticks,action='graceful_roll_requested'))
print(json.dumps(dict(node=NODE,at=time.time(),actions=out)))
'''
    return call(node, 'WORK='+repr(str(W))+';NODE='+repr(node)+';GPUS='+repr(gpus)+
                ';MARKER='+repr(MARKER)+';CHANGED='+repr(changed)+'\n'+code)


def audit_remote(node):
    code='''import json,os,time,hashlib
from pathlib import Path
W=Path(WORK);R=W.parents[1];cfg=json.loads((W/'config.json').read_text());out=[]
for p in CHANGED:assert hashlib.sha256((R/p).read_bytes()).hexdigest()==cfg['code_hashes'][p],p
for gpu in cfg['nodes'][NODE]['gpu_ids']:
 s=json.loads((W/'workers'/f'{NODE}-gpu-{gpu}.json').read_text());pid=s['pid'];p=Path('/proc')/str(pid)
 assert p.exists(),s
 cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert 'listen_j_omni.py worker' in cmd and str(W) in cmd and '--node '+NODE in cmd and '--gpu '+str(gpu) in cmd,cmd
 assert s.get('tail_protocol')=='clip-lock-tail-v1' and time.time()-s['heartbeat']<60,s
 assert s.get('tail_lock_release')=='consumer-close-after-persist' and s.get('tail_packet_lifecycle')=='finish-before-source-claim',s
 out.append(dict(gpu=gpu,pid=pid,completed_clips=s['completed_clips'],heartbeat=s['heartbeat']))
print(json.dumps(dict(node=NODE,at=time.time(),workers=out)))
'''
    return call(node,'WORK='+repr(str(W))+';NODE='+repr(node)+';CHANGED='+repr(changed)+'\n'+code)


def main():
    lock=acquire(D/'rollout.lock',False)
    if lock is None:raise RuntimeError('Rollout already running')
    cfg=read(W/'config.json');nodes=list(cfg['nodes'])
    assert cfg['tail_sharing']['enabled'] and read(D/'cpu_tests.json')['passed']
    assert not (W/'tail_sharing/ENABLED.json').exists()
    records=[]
    try:
        batches=[{'node12':[7]}]+[{n:[g for g in gs if (n,g)!=('node12',7)] for n in nodes}
                               for gs in [(0,1),(2,3),(4,5),(6,7)]]
        for batch in batches:
            batch={n:gs for n,gs in batch.items() if gs}
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                results=list(pool.map(lambda x:roll(*x),batch.items()))
            record=dict(batch=batch,actions=results,started_at=time.time(),stage='waiting')
            records.append(record);write(D/'rollout.json',records)
            deadline=time.time()+900
            while True:
                states=[read(W/'workers'/f'{n}-gpu-{g}.json') for n,gs in batch.items() for g in gs]
                errors=[(s['node'],s['gpu'],s.get('last_error')) for s in states
                        if MARKER in s.get('execution_revision','') and s.get('last_error')]
                if errors:raise RuntimeError(str(errors))
                ready=[s for s in states if MARKER in s.get('execution_revision','') and
                       s.get('tail_protocol')==PROTOCOL and s.get('tail_packet_lifecycle')=='finish-before-source-claim' and
                       s['completed_clips']>=3 and time.time()-s['heartbeat']<60]
                record.update(ready=len(ready),heartbeat=time.time());write(D/'rollout.json',records)
                if len(ready)==len(states):break
                if time.time()>deadline:raise TimeoutError('Worker restart/production deadline')
                time.sleep(5)
            record.update(stage='productive',ended_at=time.time());write(D/'rollout.json',records)
            print(json.dumps(record),flush=True)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            verified=list(pool.map(audit_remote,nodes))
        # Verify one current output per GPU with actual source/policy provenance.
        outputs=[]
        for n in nodes:
            for g in cfg['nodes'][n]['gpu_ids']:
                s=read(W/'workers'/f'{n}-gpu-{g}.json');found=None
                # The current sample may still be in inference. Search only the
                # worker's <=4 currently claimed source jobs, never result trees.
                q=read(W/'queues/caption/state.json')
                jobs=[j for j,c in q['inflight'].items() if c['worker']==f'{n}-gpu-{g}']
                for jid in jobs:
                    for row in read(W/'jobs'/f'{jid}.json')['rows']:
                        sid=row['sample_id'];p=W/'caption/results'/hashlib.sha256(sid.encode()).hexdigest()[:2]/f'{sid}.json'
                        if not p.exists():continue
                        r=read(p)
                        if r.get('node')!=n or r.get('gpu')!=g or r.get('completed_at',0)<s['started_at']:continue
                        assert r['scheduling']['protocol']==PROTOCOL and MARKER in r['execution_revision']
                        assert r.get('original_source',r['source'])==row['source']
                        assert r['pipeline_id']==cfg['pipeline_id']
                        found=dict(node=n,gpu=g,sample_id=sid,path=str(p),scheduling=r['scheduling']);break
                    if found:break
                if found is None:raise RuntimeError(f'No completed output found for {n}:{g}; retry audit')
                outputs.append(found)
        acceptance=dict(at=time.time(),protocol=PROTOCOL,remote=verified,outputs=outputs,
                        accepted=True,scope='32 productive workers, 32 source/policy checks; tail speed not yet measured')
        write(D/'deployment_acceptance.json',acceptance)
        write(W/'tail_sharing/ENABLED.json',dict(protocol=PROTOCOL,at=time.time(),acceptance=str(D/'deployment_acceptance.json')))
        write(D/'activated.json',dict(at=time.time(),protocol=PROTOCOL,policy=cfg['tail_sharing']))
        print('ALL 32 WORKERS VERIFIED; TAIL SHARING ENABLED',flush=True)
    except BaseException as exc:
        write(D/'rollout_error.json',dict(at=time.time(),error=repr(exc)))
        raise
    finally:lock.close()


if __name__=='__main__':main()
