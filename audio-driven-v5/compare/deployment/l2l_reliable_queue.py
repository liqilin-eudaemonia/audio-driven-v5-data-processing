"""CPU-only wait for GPU7, then launch the already-authorized frozen run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_l2l_reliable_retrain_20261107'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'


def save(name,value):
    value=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**value)
    path=WORK/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def watch():
    lock=(WORK/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    idle=None
    while True:
        if (WORK/'STOP').exists():
            save('QUEUE_STATUS.json',dict(action='queue_stopped_by_marker'));return
        raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        uuid,memory=[x.strip() for x in raw.split(',')];assert uuid==UUID
        used=int(memory)
        if used<128:
            if idle is None:idle=time.monotonic()
            if time.monotonic()-idle>=60:
                command=[str(ROOT/'compare/.envs/l2l/bin/python'),str(ROOT/'compare/deployment/l2l_reliable_launch.py')]
                result=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
                if result.returncode==0:
                    save('QUEUE_STATUS.json',dict(action='training_launched',launch=json.loads(result.stdout)));return
                if 'GPU7 not idle' not in result.stderr:
                    save('QUEUE_FAILED.json',dict(action='launch_failed',error=result.stderr));return
                idle=None
        else:idle=None
        save('QUEUE_STATUS.json',dict(action='waiting_for_gpu7',memory_used_mib=used,
              idle_seconds=0 if idle is None else int(time.monotonic()-idle),required_idle_seconds=60,other_jobs_untouched=True))
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    if args.watch:watch()
    else:
        path=WORK/'QUEUE_LAUNCH.json'
        if path.exists():
            old=json.loads(path.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
            assert not proc.exists() or b'l2l_reliable_queue.py' not in proc.read_bytes(),'Queue already active'
        with (WORK/'queue.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save('QUEUE_LAUNCH.json',dict(pid=proc.pid,host='172.29.189.119',gpu=7));print(proc.pid)
