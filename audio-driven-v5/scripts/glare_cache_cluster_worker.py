#!/usr/bin/env python3
"""One assigned GPU, dynamic tasks, persistent official models; no launch on import."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as cluster


def task_tracks(cfg,task):
    with cluster.ro(Path(cfg['source_plan'])/'index.sqlite') as db:
        db.execute('ATTACH DATABASE ? AS routing',(str(Path(cfg['work'])/'routing.sqlite'),))
        for tid,data in db.execute('SELECT t.track_id,t.data FROM tracks t JOIN routing.prosody_assign a '
                                   'ON a.track_id=t.track_id WHERE a.task_id=? ORDER BY t.track_id',(task['task_id'],)):
            indices=[r[0] for r in db.execute('SELECT chunk_index FROM chunks WHERE track_id=? ORDER BY chunk_index',(tid,))]
            yield json.loads(data),indices


def run_prosody(cfg,m,node,gpu,folder,queue,worker_slot=0):
    import run_glare_source_prosody_full as engine
    from glare_official_capacity_runtime import CapacityOfficialRuntime
    from glare_prosody_subset import ChunkCache
    options=cfg['prosody'];proof=Path(cfg['work'])/'nodes'/node['name']/'assets.json'
    runtime=CapacityOfficialRuntime(m['cache_contract'],gpu=gpu,max_memory_gib=options['max_memory_gib'],
        reserve_memory_gib=options['reserve_memory_gib'],asset_verifier=lambda:cluster.verify_assets(m,proof))
    cache_root=Path(cfg['benchmark']['output_root'])/'prosody' if cfg.get('benchmark') else m['cache_root']
    cache=ChunkCache(cache_root,m['cache_contract'])
    while not (Path(cfg['work'])/'STOP').exists():
        lease=queue.claim(node['name']+':gpu'+str(gpu)+':worker'+str(worker_slot))
        if lease is None:return
        with lease:
            task=lease.task
            cluster.save(folder/'active.json',dict(stage='prosody',task=task,started=time.time(),pid=os.getpid()))
            engine.tracks_for=lambda _plan:task_tracks(cfg,task)
            p=dict(chunks=task['units'],tracks=task['items'])
            state=engine.run_engine(folder,p,runtime,cache,loaders=options['loaders'],
                 batch_size=options['batch_size'],queue_size=options['queue_chunks'],retry_execution=True,
                 write_queue=cluster.execution(cfg)['prosody_write_operations'] if cluster.execution(cfg)['async_prosody'] else 0)
            if state['stage']=='stopped':return
            assert state['stage']=='complete' and state['producer_drained'],'Prosody task not drained'
            failures=sum(state['counts'].get(k,0) for k in ('input_error_exhausted','inference_error_exhausted'))
            if failures:raise cluster.TaskDataError('Prosody input/inference errors: '+str(failures)+'; retry real inputs, never replace with NULL')
            lease.finish(dict(validated=True,native_state=state,
                              policy='official normalizer; raw audit is diagnostic only'))


def configure_lia(cfg,m,node,gpu,folder,worker_slot=0):
    import run_glare_shared_cache as pipeline
    from glare_decode_execution import DecodeExecutor
    from concurrent.futures import ThreadPoolExecutor
    import torch
    old=Path(cfg['shared_lia']);pipeline.WORK=folder;pipeline.GPUS=list(range(8))
    output=Path(cfg['benchmark']['output_root'])/'nonprosody' if cfg.get('benchmark') else old
    def path_for(sid,kind):
        import hashlib
        assert Path(sid).name==sid and sid not in ('.','..')
        return output/'cache'/hashlib.sha256(sid.encode()).hexdigest()[:2]/(sid+'.'+kind)
    pipeline.path_for=path_for
    # Load original validated protocol once, then change operational limits only.
    p=cluster.read(old/'protocol.json');base_p=cluster.read(Path(m['parent'])/'protocol.json')
    caps=cluster.worker_resources(cfg,node,worker_slot)
    p.update(decode_threads_per_gpu=caps['decode_workers'],detector_services_per_gpu=caps['detector_services'],
             inflight_groups_per_gpu=caps['inflight_groups'])
    if cluster.execution(cfg)['gpu_fer']:
        from glare_gpu_reactions import service_type
        pipeline.CPUService=service_type(pipeline)
    cpu_pool=None
    if cluster.execution(cfg)['persistent_cpu_services']:
        from glare_cpu_service_pool import ServicePool
        cpu_pool=ServicePool(pipeline.CPUService,gpu,caps['detector_services'])
        pipeline.CPUService=cpu_pool.factory
    pipeline.load_protocol=lambda:dict(p)
    original_protocol=pipeline.base.load_protocol
    pipeline.base.load_protocol=lambda path:dict(base_p) if Path(path).resolve()==Path(m['parent']).resolve() else original_protocol(path)
    # Models stay alive across task bundles; no repeated checkpoint load/JIT warmup.
    original_load=pipeline.base.load_lia;model=[]
    def load_lia(protocol):
        if not model:model.append(original_load(protocol))
        return model[0]
    pipeline.base.load_lia=load_lia
    if cfg.get('benchmark'):
        # All arms recompute the same real frames; old LIA payloads must not
        # make a later arm appear faster merely through cache hits.
        pipeline.base.existing=lambda *args,**kwargs:False
    routing=Path(cfg['work'])/'routing.sqlite';current=[None]
    class RoutedConnection(sqlite3.Connection):
        def execute(self,sql,parameters=()):
            if sql=='SELECT * FROM groups WHERE worker=? ORDER BY gid':
                super().execute('ATTACH DATABASE ? AS routing',(str(routing),))
                sql='SELECT g.* FROM groups g JOIN routing.lia_assign a ON a.gid=g.gid WHERE a.task_id=? ORDER BY g.gid'
                parameters=(current[0],)
            elif sql=='SELECT * FROM jobs':
                super().execute('ATTACH DATABASE ? AS routing',(str(routing),))
                super().execute('ATTACH DATABASE ? AS grouped',(str(old/'groups.sqlite'),))
                sql=('SELECT * FROM jobs WHERE ordinal IN '
                     '(SELECT CAST(x.value AS INTEGER) FROM grouped.groups g '
                     'JOIN routing.lia_assign a ON a.gid=g.gid JOIN json_each(g.ordinals) x '
                     'WHERE a.task_id=?)')
                parameters=(current[0],)
            return super().execute(sql,parameters)
    def connect(database,*args,**kwargs):
        if str(database).split('?')[0].endswith(('/groups.sqlite','/jobs.sqlite')):kwargs['factory']=RoutedConnection
        return sqlite3.connect(database,*args,**kwargs)
    pipeline.sqlite3=SimpleNamespace(connect=connect,Row=sqlite3.Row)
    execution='baseline'
    if cfg['lia']['execution']=='fused2' and torch.cuda.get_device_capability(0)==(9,0):
        from glare_lia_fused_fp32 import install
        install(pipeline,'fused2');execution='fused2'
    cluster.save(folder/'lia_execution.json',dict(resources=caps,execution=execution,
                 reason='reuse validated SM90 FP32 path; native fallback on other GPU architecture',at=time.time()))
    decode_pool=None
    if cluster.execution(cfg)['persistent_decode']:
        from glare_persistent_decode import DecodePool
        decode_pool=DecodePool(DecodeExecutor,caps['decode_workers'])
    def execute(task):
        current[0]=task['task_id'];calls=[0]
        def executor(max_workers):
            calls[0]+=1
            if calls[0]==1:return decode_pool.lease() if decode_pool else DecodeExecutor(max_workers)
            return ThreadPoolExecutor(max_workers=max_workers)
        # Reset for every native worker invocation; its second pool is CPU reaction.
        pipeline.ThreadPoolExecutor=executor
        if cluster.execution(cfg)['async_lia']:
            from glare_async_writes import LiaWrites
            with LiaWrites(pipeline,cluster.execution(cfg)['lia_write_operations']):pipeline.worker(gpu)
        else:pipeline.worker(gpu)
        state=cluster.read(folder/'workers'/('gpu%d.json'%gpu))
        if cpu_pool:cluster.save(folder/'cpu_pool.json',cpu_pool.metrics())
        if decode_pool:cluster.save(folder/'decode_pool.json',decode_pool.metrics())
        return state
    def close():
        errors=[]
        for pool in [decode_pool,cpu_pool]:
            if pool:
                try:pool.close()
                except Exception as exc:errors.append(exc)
        if errors:raise RuntimeError('Persistent CPU pool cleanup failed') from errors[0]
    if cpu_pool or decode_pool:execute.close=close
    return execute


def _run_lia(cfg,m,node,gpu,folder,queue,worker_slot,cleanup):
    execute=None;audio_engines=[]
    while not (Path(cfg['work'])/'STOP').exists():
        lease=queue.claim(node['name']+':gpu'+str(gpu)+':worker'+str(worker_slot))
        if lease is None:return
        with lease:
            task=lease.task
            cluster.save(folder/'active.json',dict(stage='lia',task=task,started=time.time(),pid=os.getpid()))
            if execute is None:
                execute=configure_lia(cfg,m,node,gpu,folder,worker_slot)
                if hasattr(execute,'close'):cleanup.append(execute.close)
            state=execute(task)
            if state['stage']=='stopped':return
            if state['stage']=='finished_with_errors':raise cluster.TaskDataError('LIA native input failures: '+str(state['failed']))
            assert state['stage']=='finished' and state['failed']==0,'Native LIA task has runtime errors; retry from stage receipts'
            assert state['completed']+state['reused']==task['units'],'LIA task coverage mismatch'
            from glare_cluster_frozen_audio import ensure,task_jobs
            audio=ensure(cfg,m,node,gpu,folder,task_jobs(cfg,m,task),audio_engines)
            if audio.get('stopped'):return
            assert audio['validated'] and audio['total']==task['units']
            cluster.save(folder/'frozen_audio_reuse.json',audio)
            lease.finish(dict(validated=True,native_state=state,frozen_audio=audio,
                              covers=['LIA','normalized speaker audio','reaction+mask','motion statistics','Wav2Vec2 before projection','SER']))


def run_lia(cfg,m,node,gpu,folder,queue,worker_slot=0):
    cleanup=[]
    try:return _run_lia(cfg,m,node,gpu,folder,queue,worker_slot,cleanup)
    finally:
        for close in reversed(cleanup):close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=cluster.DEFAULT);ap.add_argument('--node',required=True)
    ap.add_argument('--gpu',required=True,type=int);ap.add_argument('--stage',choices=cluster.STAGES,required=True)
    ap.add_argument('--worker-slot',type=int,default=0)
    a=ap.parse_args();cfg=cluster.config(a.config);m=cluster.load_run(cfg);node=cluster.node_by_name(cfg,a.node)
    cluster.assert_local(node)
    assert a.gpu in node['gpus'] and not (a.node=='node11' and a.gpu>=4),'GPU not allocated'
    assert 0<=a.worker_slot<cluster.execution(cfg)['lia_workers_per_gpu']
    assert a.stage!='prosody' or a.worker_slot==0,'Prosody has one process per physical GPU'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu),'Unexpected CUDA visibility'
    folder=cluster.slot_dir(cfg,node,a.gpu,a.worker_slot);folder.mkdir(parents=True,exist_ok=True)
    for name,target in [('STOP',Path(cfg['work'])/'STOP'),('groups.sqlite',Path(cfg['shared_lia'])/'groups.sqlite'),
                        ('gpu7_only',Path(cfg['shared_lia'])/'gpu7_only')]:
        p=folder/name
        if not p.is_symlink():p.symlink_to(target,target_is_directory=name=='gpu7_only')
    (folder/'workers').mkdir(exist_ok=True)
    (folder/'errors').mkdir(exist_ok=True)
    proof=cluster.read(Path(cfg['work'])/'nodes'/node['name']/'preflight.json')
    cluster.validate_node_inputs(m,proof)
    cluster.verify_assets(m,Path(cfg['work'])/'nodes'/node['name']/'assets.json')
    queue=cluster.TaskQueue(cfg['work'],m['run_id'],a.stage)
    locks=Path(cfg['shared_lia'])/'cluster_slots'/node['name']
    with cluster.writer_fences(cfg,m),cluster.locked(locks/('gpu%d.lock'%a.gpu),shared=True),\
         cluster.locked(locks/('gpu%d.worker%d.lock'%(a.gpu,a.worker_slot))):
        cluster.assert_active(cfg,m)
        if (Path(cfg['work'])/'STOP').exists():return
        if a.stage=='prosody':run_prosody(cfg,m,node,a.gpu,folder,queue,a.worker_slot)
        else:run_lia(cfg,m,node,a.gpu,folder,queue,a.worker_slot)


if __name__=='__main__':main()
