#!/usr/bin/env python3
"""Per-host CPU controllers sharing the immutable visual measurement queue.

Only execution ownership is distributed; measurement and policy are imported
from the frozen j_visual_gate contract. Source locks prevent double processing.
"""
import argparse,os,sys,socket,time,signal,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from listen_benchmark.j_visual_gate import feed,initialize_cpu
from listen_benchmark.j_visual_quality_schedule import guarded_measure_job as measure_job
from listen_benchmark.j_annotation import acquire,read,write,JobQueue
from listen_visual_quality.io import file_hash
def run(work,workers,node):
    sys.path.insert(0,str(ROOT/'scripts'))
    from listen_j_annotation_supervisor import identity
    work=Path(work).resolve();local=work/'nodes'/node;local.mkdir(parents=True,exist_ok=True);owner=acquire(local/'controller.lock',False)
    if owner is None:raise RuntimeError('Visual quality controller already running')
    cfg=read(work/'config.json')
    assert all(file_hash(p)==h for p,h in cfg['contract']['files'].items()),'Visual quality runtime changed'
    stop=[False]
    def stopping(*_):stop[0]=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    q=JobQueue(work,'measure');pending={};started=time.time();totals=dict(retain=0,exclude=0,review=0);errors=0
    state=dict(pid=os.getpid(),start_ticks=identity(os.getpid()),hostname=socket.gethostname(),workers=workers,started_at=started)
    state.update(heartbeat=time.time(),stage='starting',active_sources=0,completed_sources=0,completed_clips=0,failed_sources={})
    write(local/'controller.json',state)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_cpu) as pool:
        while not stop[0] or pending:
            if not stop[0]:
                feed(work)
                while len(pending)<workers:
                    claim=q.claim('visual-quality-'+socket.gethostname())
                    if claim is None:break
                    jid,handle=claim;pending[pool.submit(measure_job,str(work),jid)]=(jid,handle)
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED) if pending else ([],[])
            for future in done:
                jid,handle=pending.pop(future)
                try:
                    receipt=future.result();q.finish(jid,receipt=receipt)
                    for k in totals:totals[k]+=receipt['counts'][k]
                    errors+=receipt['execution_errors']
                except Exception as exc:
                    q.finish(jid,error=repr(exc));state['last_error']=repr(exc)
                finally:handle.close()
            qs=read(q.path);state.update(heartbeat=time.time(),stage='draining' if stop[0] else 'running',active_sources=len(pending),completed_sources=qs['completed'],completed_clips=qs['completed_clips'],failed_sources=qs['failed'],decisions_this_process=totals,execution_errors_this_process=errors)
            write(local/'controller.json',state)
            if not pending:time.sleep(1)
    state.update(stage='stopped',heartbeat=time.time());write(local/'controller.json',state);owner.close()


def watchdog(work,workers,node):
    sys.path.insert(0,str(ROOT/'scripts'))
    from listen_j_annotation_supervisor import alive,spawn,identity
    work=Path(work).resolve();local=work/'nodes'/node;local.mkdir(parents=True,exist_ok=True);owner=acquire(local/'watchdog.lock',False)
    if owner is None:raise RuntimeError('Quality watchdog already running')
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    state=dict(pid=os.getpid(),start_ticks=identity(os.getpid()),started_at=time.time(),restarts=[])
    while not stop[0] and not (work/'STOP').exists() and not (local/'STOP').exists():
        old=read(local/'controller.json') if (local/'controller.json').exists() else None
        if not alive(old) and not alive(state.get('child')):
            recent=[t for t in state['restarts'] if time.time()-t<3600]
            if len(recent)>=4:state['error']='four controller exits in one hour; inspect logs'
            else:
                c=spawn(local,'controller',[sys.executable,str(ROOT/'scripts/listen_j_visual_gate_cluster.py'),'run','--work',str(work),'--workers',str(workers),'--node',node],dict(os.environ))
                state['child']=c;state['restarts']=recent+[time.time()];state['error']=None
                # Cover the short startup gap before controller.json is published.
                for _ in range(20):
                    if (local/'controller.json').exists() and read(local/'controller.json').get('pid')==c['pid']:break
                    if not alive(c):break
                    time.sleep(.5)
        state['heartbeat']=time.time();write(local/'watchdog.json',state);time.sleep(10)
    owner.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','watchdog']);p.add_argument('--work',type=Path,required=True);p.add_argument('--node',required=True);p.add_argument('--workers',type=int,required=True);a=p.parse_args()
    assert a.node in ('node17','node8','node4','node12') and 1<=a.workers<=96
    initialize_cpu()
    (run if a.command=='run' else watchdog)(a.work,a.workers,a.node)
