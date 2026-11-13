"""CPU separation only: strict independent pilots, all86 streams, audit, exit."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import parallel

WORK=Path(__file__).resolve().parent


def status(stage,**extra):
    parallel.save(WORK/'RUN_STATUS.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert len(os.sched_getaffinity(0))>=64
    parallel.helper().prepare()
    protocol=parallel.prepare()
    parallel.save(WORK/'RUN_PROTOCOL.json',dict(implementation_sha256=parallel.sha(__file__),
        audit_implementation_sha256=parallel.sha(WORK/'finish_separation.py'),
        parallel_protocol_sha256=parallel.sha(WORK/'PARALLEL_PROTOCOL.json'),
        finish='All43pairs86streams; independent artifact audit; exit without AF generation/scoring',
        canonical_device='All CPU,FP32,8threads per worker; no GPU or legacy waveform reuse'))
    with (WORK/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[]
        try:
            for name in ['pilot_a','pilot_b','pilot_c','pilot_d']:
                with (WORK/(name+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'execute','--name',name],
                        cwd=parallel.ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
            status('running_four_CPU_pilots',child_pids=[c.pid for c in children])
            codes=[c.wait() for c in children]
            parallel.save(WORK/'PILOT_EXIT.json',dict(pids=[c.pid for c in children],returncodes=codes,finished_unix=time.time()))
            assert codes==[0,0,0,0],codes
            parallel.parity_check()
            assert parallel.read(WORK/'PARALLEL_PARITY.json')['passed']
            # This is a device rounding diagnostic on one source clip only;
            # the actual gate is independent CPU-process reproducibility.
            import numpy as np
            uid=parallel.read(WORK/'PROTOCOL.json')['canary_ids'][0]
            here=WORK/'execution/pilot_a/separated'/uid/'speaker'
            old=WORK.parent/'p2_avatarforcing_table4_audio_preparation_cpu_frontend_20261110_v5d/execution/pilot_a/separated'/uid/'speaker'
            a=np.load(here/'estimate.npy');b=np.load(old/'estimate.npy')
            parallel.save(WORK/'CPU_GPU_CANARY_COMPARISON.json',dict(id=uid,role='speaker',
                mouth_ROI_exact=bool(np.array_equal(np.load(here/'mouthroi/speaker1.npz')['data'],np.load(old/'mouthroi/speaker1.npz')['data'])),
                waveform_max_abs=float(np.abs(a-b).max()),waveform_correlation=float(np.corrcoef(a,b)[0,1]),
                interpretation='Single-clip backend diagnostic; not proof of all-stream GPU byte parity or separation quality'))
            with (WORK/'manager.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'manage'],
                    cwd=parallel.ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            status('running_full86_CPU',manager_pid=child.pid)
            code=child.wait()
            parallel.save(WORK/'MANAGER_EXIT.json',dict(pid=child.pid,returncode=code,finished_unix=time.time()))
            assert code==0,code
            assert parallel.read(WORK/'FULL_SEPARATION_COMPLETE.json')['passed']
            status('auditing_full86')
            assert parallel.sha(WORK/'finish_separation.py')==parallel.read(WORK/'RUN_PROTOCOL.json')['audit_implementation_sha256']
            with (WORK/'final_audit.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(WORK/'finish_separation.py'),'audit'],
                    cwd=parallel.ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert parallel.read(WORK/'COMPLETED_AND_STOPPED.json')['passed']
            status('completed_and_stopped',pairs=43,streams=86,AF_jobs_launched=False)
        except BaseException as error:
            status('failed',error=repr(error))
            for child in children:
                if child.poll() is None:child.terminate()
            raise


if __name__=='__main__':main()
