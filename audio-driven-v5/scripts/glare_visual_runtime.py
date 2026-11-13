#!/usr/bin/env python3
"""Execution-only concurrency overlay; frozen numerical/cache identity unchanged."""
import argparse
from pathlib import Path
import sys
import time
import sqlite3
import threading
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_glare_shared_cache as pipeline


def enable_all_partitions(offset=0):
    """Change execution routing only; keep frozen groups and cache identities."""
    class AllGroupsConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == 'SELECT * FROM groups WHERE worker=? ORDER BY gid':
                sql, parameters = 'SELECT * FROM groups ORDER BY gid LIMIT -1 OFFSET ?', (offset,)
            return super().execute(sql, parameters)
    def connect(database, *args, **kwargs):
        # The numerical worker also reads the parent jobs index unchanged.
        if str(database).split('?')[0].endswith('/groups.sqlite'):
            kwargs['factory'] = AllGroupsConnection
        return sqlite3.connect(database, *args, **kwargs)
    pipeline.sqlite3 = SimpleNamespace(connect=connect, Row=sqlite3.Row)


def profile_cpu_stages(path):
    """Observe actual decode/detector work separately from generic wait time."""
    lock=threading.Lock()
    state={'started':time.time(),'stages':{}}
    def observe(name,elapsed):
        with lock:
            s=state['stages'].setdefault(name,dict(count=0,total_s=0.,max_s=0.,recent_s=[]))
            s.update(count=s['count']+1,total_s=s['total_s']+elapsed,max_s=max(s['max_s'],elapsed))
            s['recent_s']=(s['recent_s']+[elapsed])[-128:]
            state['heartbeat']=time.time()
            pipeline.save(path,state)
    def timed(name,fn):
        def call(*args,**kwargs):
            tick=time.monotonic()
            try:return fn(*args,**kwargs)
            finally:
                elapsed=time.monotonic()-tick
                observe(name,elapsed)
        return call
    pipeline.prepare_group=timed('prepare_group',pipeline.prepare_group)
    pipeline.CPUService.run=timed('cpu_reaction',pipeline.CPUService.run)
    return observe


def enable_process_decode(observer=None):
    from glare_decode_execution import DecodeExecutor
    threads=pipeline.ThreadPoolExecutor
    calls=[0]
    def executor(max_workers):
        calls[0]+=1
        return DecodeExecutor(max_workers,observer) if calls[0]==1 else threads(max_workers=max_workers)
    pipeline.ThreadPoolExecutor=executor


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,choices=[4,5,6,7],required=True)
    p.add_argument('--work',type=Path,default=pipeline.WORK)
    p.add_argument('--limit',type=int)
    p.add_argument('--decode-threads',type=int,default=8)
    p.add_argument('--detector-services',type=int,default=8)
    p.add_argument('--inflight-groups',type=int,default=8)
    p.add_argument('--all-partitions',action='store_true')
    p.add_argument('--benchmark-offset',type=int,default=0)
    p.add_argument('--profile-cpu-stages',action='store_true')
    p.add_argument('--process-decode',action='store_true')
    p.add_argument('--lia-execution',choices=['baseline','deferred','upload','graph','parallel2','parallel4','fast1','fast2','fused2'],default='baseline')
    a=p.parse_args()
    assert 1<=a.decode_threads<=32 and 1<=a.detector_services<=48 and 1<=a.inflight_groups<=64
    assert 0<=a.benchmark_offset and (not a.benchmark_offset or (a.limit and a.work.resolve()!=pipeline.WORK.resolve() and a.all_partitions))
    pipeline.WORK=a.work.resolve()
    if a.all_partitions:enable_all_partitions(a.benchmark_offset)
    observer=profile_cpu_stages(pipeline.WORK/'workers'/('gpu%d.cpu_stages.json'%a.gpu)) if a.profile_cpu_stages else None
    if a.process_decode:enable_process_decode(observer)
    frozen=pipeline.load_protocol
    def execution_protocol():
        value=dict(frozen())
        value.update(decode_threads_per_gpu=a.decode_threads,
                     detector_services_per_gpu=a.detector_services,
                     inflight_groups_per_gpu=a.inflight_groups,
                     shared_memory=dict(value['shared_memory'],max_groups=a.inflight_groups,
                                        aggregate_bound_gib=2*a.inflight_groups))
        return value
    pipeline.load_protocol=execution_protocol
    if a.lia_execution!='baseline':
        if a.lia_execution=='fused2':
            from glare_lia_fused_fp32 import install, sources as fused_sources
        elif a.lia_execution.startswith('fast'):
            from glare_lia_fast_parallel import install, sources as fast_sources
        elif a.lia_execution.startswith('parallel'):
            from glare_lia_parallel import install
        else:
            from glare_lia_execution import install
        install(pipeline,a.lia_execution)
    pipeline.save(pipeline.WORK/'workers'/('gpu%d.execution.json'%a.gpu),dict(
        gpu=a.gpu,all_partitions=a.all_partitions,benchmark_offset=a.benchmark_offset,process_decode=a.process_decode,decode_threads=a.decode_threads,detector_services=a.detector_services,
        inflight_groups=a.inflight_groups,shared_memory_bound_gib=2*a.inflight_groups,
        execution_source_sha256=pipeline.sha(Path(__file__)),
        decode_execution_sha256=pipeline.sha(ROOT/'scripts/glare_decode_execution.py') if a.process_decode else None,
        lia_execution=a.lia_execution,
        lia_execution_sha256=pipeline.sha(ROOT/'scripts/glare_lia_execution.py') if a.lia_execution!='baseline' else None,
        parallel_execution_sha256=pipeline.sha(ROOT/'scripts/glare_lia_parallel.py') if a.lia_execution.startswith('parallel') else None,
        fast_execution_sources=fast_sources(pipeline) if a.lia_execution.startswith('fast') else None,
        fused_execution_sources=fused_sources(pipeline) if a.lia_execution=='fused2' else None,
        numerical_source=pipeline.codes(),at=time.time()))
    pipeline.worker(a.gpu,a.limit,False)
