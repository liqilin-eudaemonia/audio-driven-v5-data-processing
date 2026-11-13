"""Keep only the authorized distributed CPU cache-audit coordinator alive."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,atomic_json


def main(work):
    work=Path(work)
    with (work/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        starts=0
        while not (work/'STOP').exists():
            record=read(work/'armed.json');command=record['command']
            config=Path(command[command.index('--config')+1]);cache=Path(read(config)['whisper_cache'])
            if (cache/'COMPLETE.json').exists() and read(cache/'COMPLETE.json').get('success'):return
            try:
                args=Path('/proc')/str(record['pid'])/'cmdline';argv=args.read_bytes().split(b'\0')
                alive=(str(ROOT/'scripts/p2_whisper_cache_audit_cluster.py').encode() in argv
                       and str(work.resolve()).encode() in argv and b'run' in argv)
            except OSError:alive=False
            if not alive:
                if starts>=3:
                    atomic_json(work/'watch.json',dict(phase='needs_attention',at=time.time()));return
                with (work/'controller.log').open('a') as log:
                    p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(work/'armed.json',dict(record,pid=p.pid,at=time.time()));starts+=1
            atomic_json(work/'watch.json',dict(at=time.time(),coordinator_alive=alive,restarts=starts))
            time.sleep(20)


if __name__=='__main__':main(sys.argv[1])
