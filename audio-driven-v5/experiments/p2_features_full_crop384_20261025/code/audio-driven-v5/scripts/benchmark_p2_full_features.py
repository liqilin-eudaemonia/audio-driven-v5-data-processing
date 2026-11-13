"""Isolated real-media throughput trials; no raw/production outputs are overwritten."""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic


def worker(a):
    import torch
    from p2_listen.config import activate
    from listen_benchmark.resident_features import ResidentFeatures,prepare_vae
    trial=a.run/'trials'/a.trial;out=trial/f'worker-{a.rank}';out.mkdir(parents=True,exist_ok=True)
    cfg=read(a.run/'p2_config.json');cfg['output']=str(out);activate(cfg);torch.set_num_threads(a.threads)
    rows=read(a.run/'benchmark_samples.json');rows=rows[a.rank::a.replicas]
    def state(name,**kw):atomic(out/'status.json',dict(pid=os.getpid(),status=name,heartbeat=time.time(),**kw))
    def clone(row,suffix):
        x=copy.deepcopy(row);sid=x['sample_id']+'-'+suffix
        x['sample_id']=sid;x['observation']['sample_id']=sid;x['target']['sample_id']=sid;x['label']['sample_id']=sid
        atomic(out/f'labels/canonical/{sid}.json',x['label']);return x
    try:
        state('loading_models');session=ResidentFeatures(cfg,native_prefetch=a.native_prefetch,workspace_slots=a.workspace_slots,workspace_lock_dir=trial/'vae-workspaces')
        warm=clone(min(rows,key=lambda x:x['observation']['requested_duration_s']),'warm')
        session.process(warm['observation'],warm['target'],session.prepare(warm['observation'],warm['target']))
        torch.cuda.synchronize();state('ready');atomic(out/'READY.json',dict(utc=time.time()))
        while not (trial/'START.json').exists():time.sleep(.1)
        start=time.time();cpu_before=os.times();metrics=[];torch.cuda.reset_peak_memory_stats()
        if a.stress:
            import numpy as np
            row=max(rows,key=lambda x:x['observation']['requested_duration_s']);value=prepare_vae(cfg,row['observation'],row['target']);scope=read(a.run/'scope_audit.json');frames=scope['max_aligned_frames']
            value['pixels']=np.concatenate([value['pixels'],np.repeat(value['pixels'][-1:],max(0,frames-len(value['pixels'])),axis=0)])[:frames]
            value['geometry'].update(frames=frames,duration_s=frames/25,real_duration_s=scope['max_duration_s'],padding_s=frames/25-scope['max_duration_s'])
            count=round(frames/25*value['rate']);value['wave']=np.pad(value['wave'],((0,max(0,count-len(value['wave']))),(0,0)))[:count]
            for i in range(3):
                state('stress_305_frames',done=i);session.encode_vae(value)
            torch.cuda.synchronize();elapsed=time.time()-start;atomic(out/'result.json',dict(samples=3,seconds=elapsed,start=start,end=time.time(),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,resident_allocated_mib=session.resident_allocated_mib,stress_frames=frames));state('finished');return
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.prefetch) as pool:
            futures={}
            for i in range(min(a.prefetch,len(rows))):futures[i]=pool.submit(session.prepare,rows[i]['observation'],rows[i]['target'])
            for i,row in enumerate(rows):
                row=clone(row,'timed');wait=time.time();prepared=futures.pop(i).result();wait=time.time()-wait
                next_i=i+a.prefetch
                if next_i<len(rows):futures[next_i]=pool.submit(session.prepare,rows[next_i]['observation'],rows[next_i]['target'])
                state('extracting',done=i,sample_id=row['sample_id']);value=session.process(row['observation'],row['target'],prepared);del prepared
                value['prefetch_wait_s']=wait;metrics.append(value)
        torch.cuda.synchronize();elapsed=time.time()-start;cpu_after=os.times()
        result=dict(samples=len(rows),seconds=elapsed,start=start,end=time.time(),samples_per_second=len(rows)/elapsed,metrics=metrics,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            cpu_seconds=cpu_after.user+cpu_after.system-cpu_before.user-cpu_before.system,resident_allocated_mib=session.resident_allocated_mib)
        atomic(out/'result.json',result);state('finished',samples=len(rows))
    except Exception as exc:state('failed',error=repr(exc));atomic(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


def trial(a):
    root=a.run/'trials'/a.trial;root.mkdir(parents=True,exist_ok=True)
    if (root/'START.json').exists():raise ValueError('Use a new trial name')
    info=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if float(info.strip())>256:raise RuntimeError('Trial GPU is no longer empty')
    children=[];samples=[];started=time.time()
    try:
        for rank in range(a.replicas):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--run',str(a.run),'--trial',a.trial,'--mode','worker','--replicas',str(a.replicas),'--rank',str(rank),'--threads',str(a.threads),'--prefetch',str(a.prefetch)]
            if a.stress:cmd+=['--stress']
            if a.native_prefetch:cmd+=['--native-prefetch']
            if a.workspace_slots:cmd+=['--workspace-slots',str(a.workspace_slots)]
            if a.expandable:env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
            with (root/f'worker-{rank}.log').open('w') as log:child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
        while not all((root/f'worker-{i}/READY.json').exists() for i in range(a.replicas)):
            if any(x.poll() is not None for x in children):raise RuntimeError('Worker failed during warmup')
            if time.time()-started>600:raise TimeoutError('Warmup timed out')
            time.sleep(.5)
        atomic(root/'START.json',dict(utc=time.time(),replicas=a.replicas,gpu=a.gpu,threads=a.threads,prefetch=a.prefetch))
        while any(x.poll() is None for x in children):
            output=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=15)
            values=list(map(float,output.strip().split(',')));samples.append(dict(utc=time.time(),memory_mib=values[0],gpu_util=values[1],power_w=values[2]))
            atomic(root/'progress.json',dict(heartbeat=time.time(),workers=[read(root/f'worker-{i}/status.json') for i in range(a.replicas)],gpu=samples[-1]))
            if time.time()-started>1800:raise TimeoutError('Trial timed out')
            time.sleep(1)
        if any(x.returncode for x in children):raise RuntimeError('Worker failed during measurement')
        results=[read(root/f'worker-{i}/result.json') for i in range(a.replicas)]
        start=min(x['start'] for x in results);end=max(x['end'] for x in results);active=[x for x in samples if start<=x['utc']<=end]
        result=dict(trial=a.trial,replicas=a.replicas,gpu=a.gpu,threads=a.threads,prefetch=a.prefetch,native_prefetch=a.native_prefetch,expandable=a.expandable,workspace_slots=a.workspace_slots,stress_only=a.stress,samples=sum(x['samples'] for x in results),seconds=end-start,
                    samples_per_second=sum(x['samples'] for x in results)/(end-start),mean_gpu_util=sum(x['gpu_util'] for x in active)/max(1,len(active)),
                    max_device_memory_mib=max(x['memory_mib'] for x in samples),per_process_peak_reserved_mib=[x['peak_reserved_mib'] for x in results],workers=results,telemetry=samples)
        atomic(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['workers','telemetry']}),flush=True)
    except Exception as exc:
        atomic(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        for child in children:
            if child.poll() is None:child.terminate()
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--trial',required=True);p.add_argument('--mode',choices=['trial','worker'],required=True)
    p.add_argument('--gpu',type=int,default=4);p.add_argument('--replicas',type=int,default=1);p.add_argument('--rank',type=int,default=0);p.add_argument('--threads',type=int,default=4);p.add_argument('--prefetch',type=int,default=2);p.add_argument('--stress',action='store_true');p.add_argument('--native-prefetch',action='store_true');p.add_argument('--expandable',action='store_true');p.add_argument('--workspace-slots',type=int,default=0)
    a=p.parse_args();a.run=a.run.resolve();(worker if a.mode=='worker' else trial)(a)


if __name__=='__main__':main()
