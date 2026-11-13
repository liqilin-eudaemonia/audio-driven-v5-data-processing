"""Add monitored GPU7 workers to the unchanged, running DyStream queue.

Workers share the original atomic claims and exit between clips when disabled.
Reserve the final64 unclaimed clips for the original controller, so auxiliary
processes finish before its generation-complete assertion and scoring handoff.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'experiments/p2_compare_responsenet_random200_20261106/dystream'
BASE=ROOT/'compare/deployment/responsenet_dystream.py'


def save(path,value):
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def worker(work):
    spec=importlib.util.spec_from_file_location('responsenet_dy_original',BASE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.claim
    def claim(work,samples):
        config=json.loads((work/'CAPACITY_CONTROL.json').read_text())
        if not config['enabled']:return None
        claimed={p.stem for p in (work/'claims').glob('*.json')}
        complete={p.parent.name for p in (work/'generated').glob('*/complete.json')}
        if len({s['id'] for s in samples}-claimed-complete)<=config['reserve_clips']:return None
        return original(work,samples)
    module.claim=claim
    module.worker(work)


def run(work):
    import fcntl
    with (work/'capacity_supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (work/'GENERATION_COMPLETE.json').exists()
        config=json.loads((work/'CAPACITY_CONTROL.json').read_text())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1')
        processes=[];handles=[];start=time.time()
        try:
            for rank in range(config['extra_workers']):
                log=(work/f'capacity_worker_{rank:02d}.log').open('a');handles.append(log)
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker',
                    '--work',str(work)],env=env,stdout=log,stderr=subprocess.STDOUT))
            save(work/'CAPACITY_LAUNCH.json',dict(pid=os.getpid(),workers=[p.pid for p in processes],
                base_workers=4,total_workers=4+len(processes),at=start,
                original_adapter_sha256=__import__('hashlib').sha256(BASE.read_bytes()).hexdigest(),
                extra_worker_code_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
            while any(p.poll() is None for p in processes):
                codes=[p.poll() for p in processes]
                if any(c not in (None,0) for c in codes):
                    config['enabled']=False;save(work/'CAPACITY_CONTROL.json',config)
                    save(work/'CAPACITY_FAILURE.json',dict(codes=codes,at=time.time()))
                save(work/'capacity_status.json',dict(active=[p.pid for p in processes if p.poll() is None],
                    codes=codes,seconds=time.time()-start,at=time.time()))
                time.sleep(3)
            save(work/'CAPACITY_COMPLETE.json',dict(codes=[p.returncode for p in processes],seconds=time.time()-start,at=time.time()))
        finally:
            for log in handles:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker'])
    parser.add_argument('--work',type=Path,default=WORK);args=parser.parse_args()
    if args.command=='run':run(args.work.resolve())
    else:worker(args.work.resolve())
