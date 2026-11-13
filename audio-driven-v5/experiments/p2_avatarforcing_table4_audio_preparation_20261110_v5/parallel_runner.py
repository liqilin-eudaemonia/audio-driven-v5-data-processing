"""Run strict one-video parity before starting independent-video parallel work."""
import fcntl
import os
from pathlib import Path
import sys
import subprocess
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    with (WORK/'parallel_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert row[0].strip()=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5';free=int(row[1])
                if free>=32768:break
                save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='waiting_gpu_capacity',pid=os.getpid(),free_mib=free,updated_unix=time.time()));time.sleep(20)
            with (WORK/'parallel_parity.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'execute','--name','parity'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='running_parity',pid=os.getpid(),child_pid=child.pid,updated_unix=time.time()))
                code=child.wait();save(WORK/'PARALLEL_PARITY_EXIT.json',dict(pid=child.pid,returncode=code,finished_unix=time.time()));assert code==0,code
            with (WORK/'parallel_parity_check.log').open('a') as log:subprocess.run([sys.executable,str(WORK/'parallel.py'),'parity-check'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(WORK/'PARALLEL_PARITY.json')['passed']
            with (WORK/'parallel_manager.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'manage'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='running_full43',pid=os.getpid(),child_pid=child.pid,updated_unix=time.time()))
                code=child.wait();save(WORK/'PARALLEL_MANAGER_EXIT.json',dict(pid=child.pid,returncode=code,finished_unix=time.time()));assert code==0,code
            assert read(WORK/'FULL_SEPARATION_COMPLETE.json')['passed']
            save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='complete',pid=os.getpid(),pairs=43,streams=86,updated_unix=time.time()))
        except Exception as error:
            save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='failed',pid=os.getpid(),error=repr(error),updated_unix=time.time()));raise


if __name__=='__main__':main()
