"""Keep the authorized checkpoint/annotation controller alive across failures."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import sys


def main(control):
    control=Path(control)
    with (control/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (control/'STOP').exists():
            running=False
            pidfile=control/'controller.pid'
            if pidfile.exists():
                try:
                    pid=int(pidfile.read_text())
                    argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
                    running=stat[0]!='Z' and any(a.endswith(b'/listen_reaction_cluster_control.py') for a in argv)
                except (OSError,ValueError):pass
            if not running:
                cmd=json.loads((control/'launch.json').read_text())
                with (control/'controller.log').open('a') as log:
                    p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                pidfile.write_text(str(p.pid))
                with (control/'guard-restarts.jsonl').open('a') as f:
                    f.write(json.dumps({'at':time.time(),'pid':p.pid})+'\n')
            (control/'guard-heartbeat.json').write_text(json.dumps({'pid':os.getpid(),'at':time.time(),'controller_alive':running}))
            time.sleep(30)


if __name__=='__main__':main(sys.argv[1])
