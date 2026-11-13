"""Require four concurrent native-process pilots to agree before full43."""
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    with (WORK/'parallel_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[]
        try:
            while True:
                row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert row[0].strip()=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5';free,total=int(row[1]),int(row[2]);required=max(32768,16384+4*(math.ceil(total*.05)+1024))
                if free>=required:break
                save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='waiting_gpu_capacity',pid=os.getpid(),free_mib=free,required_mib=required,updated_unix=time.time()));time.sleep(20)
            for name in ['pilot_a','pilot_b','pilot_c','pilot_d']:
                with (WORK/(name+'.log')).open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'execute','--name',name],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
            save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='running_concurrent_pilots',pid=os.getpid(),child_pids=[c.pid for c in children],updated_unix=time.time()))
            codes=[c.wait() for c in children];save(WORK/'PILOT_EXIT.json',dict(pids=[c.pid for c in children],returncodes=codes,finished_unix=time.time()));assert codes==[0,0,0,0],codes
            with (WORK/'parity_check.log').open('a') as log:subprocess.run([sys.executable,str(WORK/'parallel.py'),'parity-check'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert read(WORK/'PARALLEL_PARITY.json')['passed']
            with (WORK/'parallel_manager.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(WORK/'parallel.py'),'manage'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='running_full43',pid=os.getpid(),child_pid=child.pid,updated_unix=time.time()));code=child.wait()
            save(WORK/'MANAGER_EXIT.json',dict(pid=child.pid,returncode=code,finished_unix=time.time()));assert code==0,code
            assert read(WORK/'FULL_SEPARATION_COMPLETE.json')['passed']
            save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='complete',pid=os.getpid(),pairs=43,streams=86,updated_unix=time.time()))
        except Exception as error:
            for child in children:
                if child.poll() is None:child.terminate()
            save(WORK/'PARALLEL_RUNNER_STATUS.json',dict(stage='failed',pid=os.getpid(),error=repr(error),updated_unix=time.time()));raise


if __name__=='__main__':main()
