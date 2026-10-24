#!/usr/bin/env python3
"""Work-stealing benchmark replicas with process-safe per-sample claims."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True); p.add_argument('--systems', required=True)
    p.add_argument('--worker-id', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--start-offset', type=int, default=0); a=p.parse_args()
    config=json.loads(Path(a.config).read_text()); names=a.systems.split(',')
    selected=[next(s for s in config['systems'] if s['name']==name) for name in names]
    if len({(s['backend'],s['model_path']) for s in selected})!=1:
        raise ValueError('A replica can share weights only within one backend/model')
    overlay=selected[0].get('dependency_path')
    if overlay: sys.path.insert(0,overlay)
    from p2_listen.contracts import atomic_json, sha
    from listen_benchmark.assets import inspect_weights
    from listen_benchmark.backends import load_backend
    from listen_benchmark.runner import select_samples, run_system, source_identity
    root=Path(a.output); state=root/'workers'/f'{a.worker_id}.json'
    root.mkdir(parents=True,exist_ok=True)
    current=dict(worker_id=a.worker_id,pid=os.getpid(),host=os.uname().nodename,
                 visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),systems=names,
                 cluster_worker_sha256=sha(__file__),code=source_identity(),started_utc=datetime.now(timezone.utc).isoformat())
    def update(**fields):
        current.update(fields,utc=datetime.now(timezone.utc).isoformat()); atomic_json(state,current)
    update(status='preparing')
    samples=select_samples(a.manifest,None)
    base=selected[0]
    asset=({'ready':True,'sha256':sha(base['model_path'])} if base['backend']=='mediapipe' else inspect_weights(base['model_path']))
    if not asset['ready']: raise ValueError('Incomplete model assets')
    update(status='loading_model')
    try:
        backend=load_backend(base,a.device,config['generation'])
    except Exception as exc:
        update(status='initialization_failed',error=str(exc)); raise
    update(status='ready',completed_samples=0)
    ordered=selected[a.start_offset%len(selected):]+selected[:a.start_offset%len(selected)]
    jobs=[(s,row) for row in samples for s in ordered]
    completed=0
    while True:
        pending=False; claimed=False
        for system,row in jobs:
            name,sid=system['name'],row['sample_id']
            item_root=root/'items'/sid
            summary=item_root/name/'summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('complete'): continue
            pending=True
            lock_path=root/'claims'/name/(sid+'.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
            with lock_path.open('a') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: continue
                if summary.exists() and json.loads(summary.read_text()).get('complete'): continue
                claimed=True; backend.system=system
                update(status='inference',system=name,sample_id=sid)
                try:
                    report=run_system(config,name,[row],item_root,a.device,
                                      backend_factory=lambda *_:backend,asset_identity=asset)
                except Exception as exc:
                    update(status='worker_failed',error=str(exc));raise
                completed+=1
                errors=[r.get('error',{}).get('message','') for r in report['results'] if r['status']=='failed']
                update(status='ready',completed_samples=completed,last_errors=errors)
                if any('out of memory' in error.lower() for error in errors):
                    update(status='oom_stopped');return 3
                break
        if not pending:
            update(status='complete');return 0
        if not claimed: time.sleep(2)


if __name__=='__main__': raise SystemExit(main())
