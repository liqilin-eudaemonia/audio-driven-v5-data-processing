"""Queue published-panel GPU diagnostics after the full cohort has exited GPU."""
import fcntl
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
    out=WORK/'published_motion_control_v2'
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**kw):save(out/'RUNNER_STATUS.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
        try:
            subprocess.run([sys.executable,str(WORK/'published_motion_compare.py'),'prepare'],cwd=ROOT,check=True)
            while not (WORK/'FINAL_AUDIT.json').exists():
                current=read(WORK/'NATIVE_PIPELINE_STATUS.json');assert current['phase']!='failed',current
                if current['phase']=='execution_complete':
                    pid=read(WORK/'AUDIT_WATCH_LAUNCH.json')['pid']
                else:pid=current['pid']
                proc=Path('/proc')/str(pid)/'stat'
                assert proc.exists() and proc.read_text().split()[2]!='Z','Main pipeline/audit exited without final audit'
                status('waiting_for_full_cohort_audit',verified_live_pid=pid)
                time.sleep(20)
            assert read(WORK/'FINAL_AUDIT.json')['passed_execution_audit']
            while True:
                row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
                assert row[0].strip()=='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
                free=int(row[1])
                if free>=32768:break
                status('waiting_gpu_capacity',free_mib=free,required_mib=32768)
                time.sleep(20)
            status('running_published_extraction')
            with (out/'run.log').open('a') as log:subprocess.run([sys.executable,'-u',str(WORK/'published_motion_control_v2.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(WORK/'published_motion_compare.py'),'report'],cwd=ROOT,check=True)
            status('complete',comparison=str(out/'COMPARISON.json'))
        except Exception as error:
            status('failed',error=repr(error));raise


if __name__=='__main__':main()
