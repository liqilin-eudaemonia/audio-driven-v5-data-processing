#!/usr/bin/env python3
"""Execution-only recovery: native single-thread frames, lossless NumPy IPC.

The original teacher protocol/model/functions/cache identity remain unchanged.
Only missing samples are dispatched; original failed block records are retained.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_j_current_teacher_cache as task
from p2_listen.contracts import read,sha,atomic_json
from p2_j_current_qwen_hidden import ensure,env,remote
MAIN=task.WORK
RETRY=MAIN/'incidents/media_timeout_retry'


def pack(x):
    import torch
    if isinstance(x,torch.Tensor):
        ensure(x.device.type=='cpu','Only CPU processor results may cross IPC')
        return dict(__teacher_cpu_tensor__=True,dtype=str(x.dtype),array=x.detach().numpy().copy())
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,list):return [pack(v) for v in x]
    return x


def unpack(x):
    import torch
    if isinstance(x,dict):
        if x.get('__teacher_cpu_tensor__') is True:
            v=torch.from_numpy(x['array']);ensure(str(v.dtype)==x['dtype'],'CPU IPC dtype changed');return v
        return {k:unpack(v) for k,v in x.items()}
    if isinstance(x,list):return [unpack(v) for v in x]
    return x


def prepare_native(item,override,size,grid):
    p=task.prepare_media(item,override,size,grid,reference=True)
    p['audit']['execution_recovery']=dict(method='native-full-25fps-single-thread+lossless-numpy-ipc',script_sha256=sha(Path(__file__)))
    return pack(p)


def extract_native(runtime,prepared):
    return task.extract_batch(runtime,[unpack(p) for p in prepared])


def worker(work,wid):
    task.IncrementReader(work);task.runner.Reader=task.IncrementReader;task.runner.prepare=prepare_native;task.runner.extract_batch=extract_native
    task.runner.worker(work,wid,batch=1,producers=1,prefetch=2)


def launch_one(c,n,gpu,wid):
    code='''import sys,subprocess,os,json,time
from pathlib import Path
root,work,script,gpu,wid,node=sys.argv[1:];f=Path(work)/'launches';f.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,script,'worker','--work',work,'--worker-id',wid];e=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
with (f/(node+'-g'+gpu+'.log')).open('a') as log:p=subprocess.Popen(cmd,cwd=root,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,worker_id=int(wid),node=node,gpu=int(gpu),time=time.time());(f/(node+'-g'+gpu+'.json')).write_text(json.dumps(d));print(json.dumps(d))'''
    return json.loads(remote(c,n,[c['python'],'-c',code,str(ROOT),str(RETRY),str(Path(__file__).resolve()),str(gpu),str(wid),n['name']]))


def recover():
    RETRY.mkdir(parents=True,exist_ok=True)
    with (MAIN/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        failures=sorted((MAIN/'failed').glob('*.json'));ensure(failures,'No failed blocks to retry')
        jobs=read(MAIN/'jobs.json');r=task.IncrementReader(MAIN);pending=[]
        for f in failures:
            d=read(f);ensure('TimeoutError' in d['error'],'Unexpected teacher failure; not an execution-only retry')
            for sid in d['sample_ids']:
                if r.load(sid,missing_ok=True) is None:pending.append(sid)
        ensure(len(pending)==len(set(pending)),'Duplicate retry sample')
        shutil.copyfile(MAIN/'protocol.json',RETRY/'protocol.json')
        for name in ['mapping.sqlite','cache']:
            f=RETRY/name
            if not f.exists():f.symlink_to(MAIN/name,target_is_directory=name=='cache')
        atomic_json(RETRY/'jobs.json',[[sid] for sid in pending],immutable=True)
        atomic_json(RETRY/'execution.json',dict(script_sha256=sha(Path(__file__)),original_protocol_sha256=sha(MAIN/'protocol.json'),pending_ids=pending,failed_jobs=[f.stem for f in failures],batch=1,producers=1,prefetch=2,decoder='existing native complete 25fps path',cpu_transport='lossless NumPy, exact dtype restored; no Torch shared-storage IPC',cache_identity_unchanged=True),immutable=True)
        c=read(MAIN/'cluster.json');targets=[]
        for i,n in enumerate(c['nodes']):
            free=json.loads(remote(c,n,[c['python'],str(ROOT/'scripts/p2_j_current_teacher_cache.py'),'free']))
            targets.extend((n,g,9000+i*100+g) for g in free)
        ensure(targets,'No free GPU for teacher retry');targets=targets[:min(len(targets),len(pending))]
        with ThreadPoolExecutor(max_workers=32) as p:launches=list(p.map(lambda x:launch_one(c,*x),targets))
        atomic_json(RETRY/'launches.json',launches)
        while True:
            done=list((RETRY/'done').glob('*.json'));failed=list((RETRY/'failed').glob('*.json'))
            atomic_json(MAIN/'chain_status.json',dict(stage='retrying_media_timeout',samples=7438-len(pending)+len(done),pending_total=len(pending),retry_completed=len(done),time=time.time(),pid=os.getpid()))
            if failed:raise RuntimeError('Conservative teacher retry failed; preserve logs')
            if len(done)==len(pending):break
            time.sleep(5)
        # Rebuild only the three original receipts after every member loads and
        # passes its native SHA/tensor/geometry checks. Preserve all failure logs.
        archive=RETRY/'original_failed';archive.mkdir(exist_ok=True)
        for f in failures:
            ids=jobs[int(f.stem)];receipts=[]
            for sid in ids:
                r.load(sid);folder=MAIN/'cache'/sid;m=read(folder/'complete.json')
                receipts.append(dict(sample_id=sid,bytes=(folder/'payload.pt').stat().st_size,payload_sha256=m['payload_sha256'],identity=m['identity']))
            atomic_json(MAIN/'done'/f.name,dict(success=True,samples=len(ids),sample_ids=ids,receipts=receipts,seconds=0,protocol_sha256=r.protocol_sha,worker='verified_execution_recovery',recovery_execution_sha256=sha(RETRY/'execution.json')))
            f.rename(archive/f.name)
        atomic_json(MAIN/'chain_status.json',dict(stage='auditing',time=time.time(),pid=os.getpid()))
        result=task.finalize(str(MAIN))
        atomic_json(RETRY/'RECOVERY_COMPLETE.json',dict(success=True,resolved_blocks=len(failures),new_samples=len(pending),original_protocol_sha256=r.protocol_sha,final_mixed_complete_sha256=sha(MAIN/'MIXED_COMPLETE.json'),time=time.time()))
        atomic_json(MAIN/'chain_status.json',dict(stage='complete',result=result,time=time.time(),pid=os.getpid()));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['recover','worker']);p.add_argument('--work',default=str(RETRY));p.add_argument('--worker-id',type=int,default=9000);a=p.parse_args()
    if a.command=='worker':worker(a.work,a.worker_id)
    else:
        try:recover()
        except BaseException:atomic_json(RETRY/'RECOVERY_ERROR.json',dict(error=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
