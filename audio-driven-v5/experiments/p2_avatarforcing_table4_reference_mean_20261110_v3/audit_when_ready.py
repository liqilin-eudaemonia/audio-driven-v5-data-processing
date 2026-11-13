"""Run the independent CPU audit only after actual scoring finishes."""
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from p2_avatarforcing_table4_generate import read,save,sha


def main():
    marker = WORK/'OFFICIAL_EXECUTION_COMPLETE.json'
    while not marker.exists():
        status = read(WORK/'NATIVE_PIPELINE_STATUS.json')
        assert status['phase'] != 'failed', status
        process = Path('/proc')/str(status['pid'])
        assert process.exists() and (process/'stat').read_text().split()[2] != 'Z', 'Pipeline ended before scoring completed'
        save(WORK/'AUDIT_WATCH_STATUS.json',dict(stage='waiting_for_score',pid=os.getpid(),
            pipeline_pid=status['pid'],checked_live_unix=time.time(),audit_script_sha256=sha(WORK/'final_audit.py')))
        time.sleep(20)
    with (WORK/'final_audit.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(WORK/'final_audit.py')],cwd=ROOT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(WORK/'FINAL_AUDIT.json')
    save(WORK/'AUDIT_WATCH_STATUS.json',dict(stage='audit_complete',pid=os.getpid(),
        matched_cells=result['matched_cells'],goal_complete_proven=result['goal_complete_proven']))


if __name__=='__main__':main()
