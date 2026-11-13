"""Serial GPU handoff from existing full cohort and published-panel controls."""
import os
import fcntl
from pathlib import Path
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[1]
PARENT=WORK.parent/'p2_avatarforcing_table4_reference_mean_20261110_v3'
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save


def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split()[2]!='Z'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    with (WORK/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,**extra):save(WORK/'STATUS.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
        try:
            while not (PARENT/'published_motion_control_v2/COMPARISON.json').exists():
                r=read(PARENT/'published_motion_control_v2/RUNNER_STATUS.json')
                assert r['stage']!='failed',r
                assert alive(r['pid']),'Published-control runner exited without report'
                status('waiting_prior_control',verified_live_pid=r['pid'],prior_stage=r['stage'])
                time.sleep(20)
            # The comparison is CPU-only and produced after its GPU child exited.
            assert read(PARENT/'FINAL_AUDIT.json')['passed_execution_audit']
            status('extracting_canonical86')
            with (WORK/'extract.log').open('a') as log:subprocess.run([sys.executable,'-u',str(WORK/'canonical_eval.py'),'extract'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
            status('scoring')
            with (WORK/'score.log').open('a') as log:subprocess.run([sys.executable,'-u',str(WORK/'canonical_eval.py'),'score'],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
            audit=read(WORK/'FINAL_AUDIT.json');status('complete',matched_primary=audit['matched_cells_primary'],matched_DIM=audit['matched_cells_DIM'],goal_complete=False)
        except Exception as error:status('failed',error=repr(error));raise


if __name__=='__main__':main()
