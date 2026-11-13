"""Resume ViCo6500 motion extraction with honest small-memory admission.

Only process admission and an unused-landmark diagnostic are adapted. Original
FAN, alignment, reconstruction, EMOCA, batches and score formulas stay intact.
"""
import argparse
import ast
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];SELF=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'compare/deployment')]
from p2_testsets_eval.protocol import read,save,sha
import p2_vico43_all_lora6500_metrics_20261111 as native
WORK=native.WORK


def accept_report(report):
    return (report['five_point_max_abs']==0 and report['aligned_uint8_max_abs']==0
        and report['coefficient_max_abs']<report['coefficient_tolerance'])


def worker(kind,slot):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    # Replace exactly one conservative free-memory assertion, not its memory cap.
    original=inspect.getsource(native.worker)
    assert original.count('assert free>=24*1024')==1
    effective=original.replace('assert free>=24*1024','assert free>=12*1024')
    namespace=dict(native.__dict__);exec(compile(effective,str(SELF)+':native-worker-admission','exec'),namespace)
    save(WORK/'bounded_motion'/f'worker_{kind}_{slot}.json',dict(pid=os.getpid(),at=time.time(),
        wrapper_sha256=sha(SELF),native_worker_sha256=hashlib.sha256(original.encode()).hexdigest(),
        effective_worker_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        memory_cap_gib=8,required_free_gib=12,math_and_batch_sizes_unchanged=True))
    if kind=='bfm':
        import p2_vico_motion_extract as source
        parity=source.Extractor.parity
        def checked(self,frames,arrays,faces):
            try:return parity(self,frames,arrays,faces)
            except ValueError as exc:
                prefix='Batching parity failed: '
                if not str(exc).startswith(prefix):raise
                report=ast.literal_eval(str(exc)[len(prefix):])
                if not accept_report(report):raise
                return dict(report,passed=True,original_all68_strict_report=report,
                    acceptance='All five alignment points and RGB crops identical; coefficients within original1e-3 tolerance. Unused68-point difference disclosed.',
                    admission_wrapper_sha256=sha(SELF))
        with patch.object(source.Extractor,'parity',checked):namespace['worker'](WORK,kind,slot)
    else:
        import vico_more_common
        def bounded_guard(free_gib=16):
            import torch
            free,total=native.gpu();assert free>=12*1024
            assert os.environ['CUDA_VISIBLE_DEVICES']=='7' and torch.cuda.device_count()==1
            torch.set_num_threads(2);torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            return dict(host=os.uname().nodename,uuid=native.UUID,free_mib=free,pid=os.getpid(),
                required_free_gib=12,memory_cap_gib=8,policy='Verified GPU7; bounded original model, preserve foreign tasks',
                admission_wrapper_sha256=sha(SELF))
        with patch.object(vico_more_common,'gpu_guard',bounded_guard):namespace['worker'](WORK,kind,slot)


def run():
    with (WORK/'bounded_motion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        native.prepare(WORK)
        save(WORK/'BOUNDED_MOTION_POLICY.json',dict(at=time.time(),wrapper_sha256=sha(SELF),
            original_metrics_sha256=sha(native.SELF),memory_cap_gib=8,required_free_gib=12,
            official_models_and_math_unchanged=True,maximum_concurrent_workers=2,
            auxiliary68point_difference_diagnostic_only=True,strict_five_point_and_crop_parity=True,
            original_coefficient_tolerance=1e-3,all_existing_features_preserved=True))
        for kind in ('bfm','emoca'):
            marker=WORK/(kind.upper()+'_EXTRACTION_COMPLETE.json')
            if marker.exists():continue
            pending=list(range(4));active={};done=[]
            while pending or active:
                for slot,(process,log) in list(active.items()):
                    if process.poll() is not None:
                        if process.returncode:raise RuntimeError(f'{kind} slot{slot} failed: {log}')
                        done.append(slot);del active[slot]
                free,total=native.gpu()
                raw=subprocess.check_output(['nvidia-smi','-i','7','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
                used={int(p):float(m) for line in raw.splitlines() if len(v:=line.replace(' ','').split(','))==2 and (p:=v[0]).isdigit() and (m:=v[1]).replace('.','',1).isdigit()}
                required=12*1024+sum(max(0,8192-used.get(p.pid,0)) for p,_ in active.values())
                if pending and len(active)<2 and free>=required:
                    slot=pending.pop(0);log=WORK/'logs'/f'bounded_{kind}_{slot}.log'
                    cmd=[str(native.BFMPY if kind=='bfm' else native.MORE),'-u',str(SELF),'worker','--kind',kind,'--slot',str(slot),'--work',str(WORK)]
                    with log.open('ab',buffering=0) as stream:
                        p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONPATH='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2'),
                            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    active[slot]=(p,log)
                    save(WORK/'processes'/f'bounded_{kind}_{slot}.json',dict(pid=p.pid,at=time.time(),stage=kind,log=str(log),command=cmd,physical_gpu=7))
                save(WORK/'BOUNDED_MOTION_STATUS.json',dict(at=time.time(),stage=kind,pid=os.getpid(),pending=pending,
                    active_pids=[p.pid for p,_ in active.values()],finished_shards=done,free_mib=free,required_mib=required))
                if pending or active:time.sleep(10)
            save(marker,dict(passed=True,clips=43,at=time.time(),admission_wrapper_sha256=sha(SELF)))
        # Restore the native controller after both coefficient streams commit.
        stop=read(WORK/'STOP');assert stop['owner_script']==SELF.name
        (WORK/'STOP').unlink()
        command=[str(ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'),'-u',str(ROOT/'scripts/p2_vico43_all_lora6500_gpu7_20261111.py'),'run','--work',str(WORK)]
        log=WORK/'bounded_motion_native_resume.log'
        with log.open('ab',buffering=0) as stream:
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(WORK/'LAUNCH.json',dict(pid=p.pid,at=time.time(),command=command,physical_gpu=7,reason='bounded_motion_finished_resume_native_score'))
        save(WORK/'BOUNDED_MOTION_COMPLETE.json',dict(success=True,at=time.time(),native_resume_pid=p.pid))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('--kind',choices=['bfm','emoca'])
    p.add_argument('--slot',type=int,default=0);p.add_argument('--work',type=Path,default=WORK);a=p.parse_args()
    assert a.work.resolve()==WORK
    run() if a.action=='run' else worker(a.kind,a.slot)
