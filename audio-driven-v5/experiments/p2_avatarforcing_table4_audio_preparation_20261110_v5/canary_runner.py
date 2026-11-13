"""Capacity-aware, one-shot canary; no automatic retry after a failed GPU run."""
import os
from pathlib import Path
import sys
import subprocess
import time
import fcntl

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save


def main():
    with (WORK/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
        assert read(WORK.parent/'p2_avatarforcing_table4_canonical_gt_20261110_v4/STATUS.json')['stage']=='complete'
        assert read(WORK.parent/'p2_avatarforcing_table4_canonical_gt_20261110_v4/FINAL_AUDIT.json')['passed_execution_audit']
        while True:
            row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert row[0].strip()=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5';free=int(row[1])
            if free>=32768:break
            save(WORK/'RUNNER_STATUS.json',dict(stage='waiting_gpu_capacity',pid=os.getpid(),free_mib=free,required_mib=32768,updated_unix=time.time()));time.sleep(20)
        with (WORK/'canary.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(WORK/'separate.py'),'canary'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            save(WORK/'RUNNER_STATUS.json',dict(stage='running_canary',pid=os.getpid(),child_pid=child.pid,updated_unix=time.time()))
            code=child.wait()
        save(WORK/'CANARY_RUN_EXIT.json',dict(child_pid=child.pid,returncode=code,finished_unix=time.time()))
        if code!=0:
            save(WORK/'RUNNER_STATUS.json',dict(stage='failed',pid=os.getpid(),child_pid=child.pid,exit_code=code,updated_unix=time.time()));raise RuntimeError(f'Canary exited{code}')
        assert read(WORK/'CANARY_COMPLETE.json')['passed']
        save(WORK/'RUNNER_STATUS.json',dict(stage='canary_complete',pid=os.getpid(),child_pid=child.pid,updated_unix=time.time()))


if __name__=='__main__':main()
