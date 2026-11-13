"""Persistent per-GPU / CPU workers for the full follow-on feature queue."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import read,atomic,lock,file_sha
from listen_benchmark.feature_pipeline import select_jobs,process_media_job


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--worker',required=True)
    p.add_argument('--role',choices=['media','features','canary'],required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--one-job',action='store_true');a=p.parse_args();run=a.run.resolve();cfg=read(run/'config.json');started=time.time();completed=0;session=None
    def status(name,**kw):
        atomic(run/f'workers/{a.worker}.json',dict(worker=a.worker,pid=os.getpid(),role=a.role,status=name,heartbeat=time.time(),started_at=started,completed_samples=completed,**kw))
    status('starting')
    pinned=read(run/'code_manifest.json')
    def verify_code():
        for path,expected in pinned.items():
            if file_sha(path)!=expected:
                status('failed',error='Pinned extraction code changed: '+path)
                raise RuntimeError('Pinned extraction code changed; versioned revalidation required')
    verify_code()
    if a.role=='canary':
        from listen_benchmark.resident_features import canary
        status('canary')
        try:result=canary(run)
        except Exception as exc:
            status('failed',error=repr(exc));atomic(run/'alerts/canary.json',dict(utc=time.time(),error=repr(exc),traceback=traceback.format_exc()[-6000:]));raise
        if (run/'alerts/canary.json').exists():atomic(run/'alerts/canary.json',dict(read(run/'alerts/canary.json'),resolved_at=time.time()))
        status('finished',result=result);return
    c=read(run/'p2_config.json')
    if a.role=='features':
        from p2_listen.config import activate
        activate(c)
        from listen_benchmark.resident_features import ResidentFeatures,prepare_vae
        import torch
        torch.set_num_threads(cfg.get('feature_cpu_threads',4))
    tries={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['prefetch']) as prefetch:
        while not (run/'STOP').exists() and not (run/f'controls/retire-{a.worker}').exists():
            if a.role=='features' and not read(run/'CANARY.json').get('passed'):status('waiting_for_canary');time.sleep(5);continue
            claimed=False
            for jid in select_jobs(run,a.role,a.rank):
                if (run/f'state/{a.role}/{jid}.json').exists():continue
                if time.time()<tries.get(jid,0):continue
                with lock(run/f'locks/worker-{a.role}-{jid}.lock') as acquired:
                    if not acquired or (run/f'state/{a.role}/{jid}.json').exists():continue
                    claimed=True
                    try:
                        verify_code()
                        if a.role=='media':
                            result=process_media_job(run,jid,a.worker,status)
                            if result:completed+=result['samples']
                        else:
                            marker=read(run/f'state/media/{jid}.json');path=run/f'manifests/shards/{jid}.json'
                            if file_sha(path)!=marker['manifest_sha256']:raise ValueError('Media manifest identity changed')
                            rows=read(path)
                            if session is None:
                                status('loading_models',job_id=jid)
                                session=ResidentFeatures(c,native_prefetch=cfg.get('prefetch_native_inputs',False),workspace_slots=cfg.get('vae_workspace_slots',0),workspace_lock_dir=run/'locks/vae-workspaces'/a.worker.rsplit('-f',1)[0])
                            def prepare(row):
                                if (run/f'cache/vae/{row["sample_id"]}/complete.json').exists():return None
                                return session.prepare(row['observation'],row['target'])
                            pending={};metrics=[]
                            for i in range(min(len(rows),cfg['prefetch'])):pending[i]=prefetch.submit(prepare,rows[i])
                            for i,row in enumerate(rows):
                                status('features',job_id=jid,sample_id=row['sample_id'])
                                t=time.time();prepared=pending.pop(i).result();wait=time.time()-t
                                nxt=i+cfg['prefetch']
                                if nxt<len(rows):pending[nxt]=prefetch.submit(prepare,rows[nxt])
                                metric=session.process(row['observation'],row['target'],prepared);del prepared
                                metric.update(cpu_prefetch_wait_s=wait,worker=a.worker,utc=time.time())
                                atomic(run/f'state/samples/{row["sample_id"]}.json',metric);metrics.append(metric);completed+=1
                                status('features',job_id=jid,last_sample=metric)
                            result=dict(done=True,job_id=jid,samples=len(rows),worker=a.worker,utc=time.time(),peak_reserved_mib=max(m['peak_reserved_mib'] for m in metrics),seconds={k:sum(m['seconds'][k] for m in metrics) for k in metrics[0]['seconds']})
                            verify_code();atomic(run/f'state/features/{jid}.json',result)
                        alert=run/f'alerts/job-{a.role}-{jid}.json'
                        if alert.exists():atomic(alert,dict(read(alert),resolved_at=time.time()))
                        status('job_complete',job_id=jid)
                        if a.one_job:return
                    except Exception as exc:
                        alert=run/f'alerts/job-{a.role}-{jid}.json';old=read(alert);attempt=old.get('attempt',0)+1
                        atomic(alert,dict(utc=time.time(),worker=a.worker,job_id=jid,stage=a.role,attempt=attempt,error=repr(exc),traceback=traceback.format_exc()[-6000:]))
                        status('failed',job_id=jid,error=repr(exc))
                        if a.role=='features' and ('out of memory' in str(exc).lower() or 'cuda' in str(exc).lower()):
                            atomic(run/f'alerts/oom-{a.worker}.json',dict(utc=time.time(),worker=a.worker,error=repr(exc)))
                            raise
                        # Invalid media/cache is visible and retryable; never mark failed work complete.
                        tries[jid]=time.time()+min(1200,30*attempt)
                    break
            if not claimed:
                if (run/'COMPLETE.json').exists():status('finished');return
                status('waiting_for_ready_job');time.sleep(5)
    status('stopped')


if __name__=='__main__':main()
