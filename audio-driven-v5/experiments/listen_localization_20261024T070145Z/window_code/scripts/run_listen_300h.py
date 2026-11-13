"""Persistent data→24-GPU P→24-GPU C0 execution. No training before the full data seal."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import traceback

from listen_cluster import ROOT,DEFAULT,launch,distributed_commands
sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import config,resolve,read,write,write_once,digest,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/p3_listen_300h.json')
    p.add_argument('--cluster',type=Path,default=DEFAULT)
    p.add_argument('--existing-plan-logs',type=Path)
    a=p.parse_args();cfg=config(a.config);cluster=read(a.cluster);root=resolve(cfg['output'])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'orchestrator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(phase,**extra):
        write(root/'orchestrator_status.json',dict(pid=os.getpid(),phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             training_started=(root/'runs/P/progress.json').exists(),config=str(a.config),**extra))
        print(json.dumps(dict(phase=phase,**extra)),flush=True)
    def logdir(name):
        parent=root/'execution_logs';parent.mkdir(exist_ok=True)
        for i in range(1000):
            out=parent/f'{name}_{i:03d}'
            if not out.exists():return out
        raise RuntimeError('Too many attempts')
    def one(script,*args):
        subprocess.run([cluster['python'],'-u',str(ROOT/'scripts'/script),*map(str,args)],check=True)
    def three_cpu(script,command,workers,cancel_event=None):
        commands=[['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',cluster['python'],'-u',
          str(ROOT/'scripts'/script),command,'--config',str(a.config.resolve()),'--node',str(i),'--workers',str(workers)] for i in range(3)]
        launch(cluster,commands,logdir(command),cancel_event=cancel_event)
    def distributed(script,args,name,cancel_event=None):
        launch(cluster,distributed_commands(cluster,ROOT/'scripts'/script,list(map(str,args))),logdir(name),cancel_event=cancel_event)
    try:
        if (root/'BOUNDARY_REVIEW_HOLD.json').exists():
            status('paused_for_boundary_review',reason=read(root/'BOUNDARY_REVIEW_HOLD.json')['reason'])
            return
        write_once(root/'launch_contract.json',dict(config=cfg,cluster=cluster,stages=['P','C0'],training_unique_hours=300))
        if not (root/'data_ready.json').exists():
            if not (root/'windows.json').exists():
                status('planning')
                if a.existing_plan_logs:
                    start=time.monotonic()
                    while not (a.existing_plan_logs/'completed.json').exists():
                        if time.monotonic()-start>7200:raise TimeoutError('Initial plan did not complete within two hours')
                        time.sleep(15)
                elif not (ROOT/'data_corpora/seamless_listen_300h/selected.json').exists():
                    one('prepare_listen_300h.py','catalog')
                    three_cpu('prepare_listen_300h.py','plan',16)
                status('selecting_normalization_reserve')
                if not (ROOT/'data_corpora/seamless_listen_300h/selected.json').exists():
                    one('prepare_listen_300h.py','select','--hours','320','--dev-hours','3.3')
                for attempt in range(4):
                    status('normalizing_audio',attempt=attempt+1)
                    three_cpu('prepare_listen_300h.py','normalize',32)
                    corpus=ROOT/'data_corpora/seamless_listen_300h'
                    totals={'train':0.,'dev':0.}
                    for item in read(corpus/'selected.json')['pairs']:
                        result=read(corpus/'normalized_pairs'/(item['pair']+'.json'))['summary']
                        if result['ok']:
                            plan=read(item['path'])
                            totals[item['split']]+=sum(c['unique_training_seconds'] for c in plan['clips']
                                if c['usable'] and not c['crop_boxes']['listener'].get('wide_crop',False))
                    if totals['train']>=300*3600 and totals['dev']>=3*3600:break
                    if attempt==3:raise ValueError('Insufficient qualified hours after replacement rounds')
                    status('replacing_failed_audio_pairs',qualified_hours={k:v/3600 for k,v in totals.items()})
                    one('prepare_listen_300h.py','extend')
                status('freezing_300h_training_manifest')
                one('prepare_listen_300h.py','freeze','--config',a.config.resolve(),'--dev-hours','3')
            # Same fixed Gemma conditions as S0; validate before copying, no GT content.
            from p3_listen.features import native_contract,renderer_contract,load_text
            sys.path.insert(1,str(resolve(cfg['reuse_root'])/'src'))
            existing=ROOT/'experiments/p3_fixed_text_v2_subturns'
            load_text(existing,cfg,'cpu');load_text(existing,cfg,'cpu',negative=True)
            (root/'cache').mkdir(exist_ok=True)
            shutil.copytree(existing/'cache/text',root/'cache/text',dirs_exist_ok=True)
            write_once(root/'cache/native_contract.json',native_contract(cfg))
            status('exporting_and_caching_24gpu',cpu_workers_per_node=32)
            cancel=threading.Event()
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks=[pool.submit(three_cpu,'cache_listen_corpus.py','export',32,cancel),
                       pool.submit(distributed,'listen_stream_cache.py',['--config',a.config.resolve(),'--benchmark-batch'],'stream24',cancel)]
                try:
                    for future in as_completed(tasks):future.result()
                except BaseException:
                    cancel.set()
                    raise
            status('verifying_all_cache_shards')
            one('cache_listen_corpus.py','seal','--config',a.config.resolve())
        # Full data and production-trainer smoke checks are mandatory before formal updates.
        for path in ('cluster_checks/nccl/completed.json','cluster_checks/model_ddp/completed.json',
                     'cluster_checks/production_P/status.json','cluster_checks/production_C0/status.json'):
            if not (root/path).exists():raise ValueError('Required engineering check missing: '+path)
        if read(root/'cluster_checks/model_ddp/completed.json')['world_size']!=24:
            raise ValueError('Engineering topology differs')
        for stage in ('P','C0'):
            check=read(root/f'cluster_checks/production_{stage}/status.json')
            if check['world_size']!=24 or check['stopped'] or check['step']<2 or not check['s0']:
                raise ValueError('Incomplete production trainer engineering check: '+stage)
        for stage in ('P','C0'):
            out=root/'runs'/stage
            state=read(out/'status.json') if (out/'status.json').exists() else None
            if state and not state['stopped']:continue
            status('training_'+stage,world_size=24)
            args=['--config',a.config.resolve(),'--stage',stage,'--output',out]
            if (out/'latest.json').exists():args+=['--resume',read(out/'latest.json')['checkpoint']]
            elif stage=='C0':args+=['--checkpoint',read(root/'runs/P/status.json')['checkpoint']]
            distributed('train_listen_distributed.py',args,'train_'+stage)
            state=read(out/'status.json')
            if state['stopped']:
                status('stopped_'+stage,checkpoint=state['checkpoint']);return
        status('training_complete',checkpoint=read(root/'runs/C0/status.json')['checkpoint'],quality_evaluated=False)
    except BaseException as e:
        status('failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
