"""Yield only registered workers of this evaluation; never signal foreign processes."""
from runtime import *
import signal

def defer_owned(r,reason):
    d=Path(r['directory']);path=d/f"exit_{r['attempt']:03d}.json"
    if path.exists():return read(path)
    running=d/'running.json'
    if not running.exists():return dict(returncode=0,planned_defer=True,reason='Not started')
    rr=read(running);pid=rr['pid'];sup=r['supervisor_pid'];proc=Path(f'/proc/{pid}/cmdline');sp=Path(f'/proc/{sup}/cmdline')
    if proc.exists() and proc.read_bytes():
        assert (str(WORK)+'/').encode() in proc.read_bytes(),('Refuse to signal unowned worker',pid)
        assert sp.exists() and (str(WORK)+'/').encode() in sp.read_bytes(),('Refuse to signal unowned supervisor',sup)
        assert os.getpgid(pid)==os.getpgid(sup)==sup
        os.killpg(sup,signal.SIGTERM)
    result=dict(returncode=0,planned_defer=True,reason=reason,ended=time.time(),worker_pid=pid,gpu=r['gpu'])
    if not path.exists():save(path,result)
    with (WORK/'RESOURCE_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(tag=r['tag'],**result))+'\n')
    return read(path)


def refresh_idle_clocks(active,leases,cooldown,empty_since,resources,policy,now):
    """A fresh GPU lease must not inherit time from before a foreground/idle gap."""
    for gpu in policy['allowed_gpus']:
        key=str(gpu)
        same=any(r['gpu']==gpu and r['estimate_gib']>0 for r in active.values())
        if same:
            empty_since.pop(key,None)
            continue
        empty_since.setdefault(key,now)
        external=resources.get(key,{}).get('external',[])
        if external or now-empty_since[key]>=policy['release_seconds']:
            leases.pop(key,None)
        elif key in leases and now-leases[key]>=policy['lease_seconds']:
            leases.pop(key,None)
            cooldown[key]=max(cooldown.get(key,0),empty_since[key]+policy['release_seconds'])
