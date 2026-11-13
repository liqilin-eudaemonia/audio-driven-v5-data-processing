"""Hold the legacy supervisor lock while the cross-host coordinator owns work."""
import argparse,fcntl,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_corpora/seamless_listen_v8_qc4_full'
SCHED=OUT/'cluster'
def process_start(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except FileNotFoundError:return None
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--coordinator',type=int,required=True);a=p.parse_args()
    identity=process_start(a.coordinator);assert identity is not None
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (SCHED/'legacy_entry_guard.json').write_text(json.dumps(dict(pid=os.getpid(),coordinator=a.coordinator,coordinator_start=identity,legacy_supervisor_lock_held=True)))
        while True:
            launch=json.loads((SCHED/'launch.json').read_text());current=launch['pid']
            identity=process_start(current)
            if identity is not None:
                command=Path(f'/proc/{current}/cmdline').read_bytes()
                if b'run_listen_v8_cluster.py' in command and b'supervise' in command:
                    time.sleep(10);continue
            # Unknown or stale remote state remains locked for inspection.
            nodes=[json.loads(p.read_text()) for p in (SCHED/'nodes').glob('*.json')]
            if len(nodes)==3 and all(n['state']=='finished' for n in nodes):break
            time.sleep(10)
