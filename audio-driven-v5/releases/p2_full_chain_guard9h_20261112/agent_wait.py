"""Await the exact live watchdog; surface new audits and actionable alerts."""
from pathlib import Path
import importlib.util, json, os, sys, time
G=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('chain_guard_wait',G/'watch.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
watcher=M.read(G/'CONTROLLER.json');seen=set()
M.atomic_json(G/'AGENT_WAITER.json',dict(pid=os.getpid(),watcher=watcher,at=time.time()))
while True:
    if not M.live(watcher) and not (G/'WINDOW_FINISHED.json').exists():
        raise RuntimeError('Confirmed watchdog exit before deadline; inspect and restore supervision')
    heartbeat=M.read(G/'HEARTBEAT.json')
    if time.time()-heartbeat['at']>600:raise RuntimeError('Watchdog heartbeat stale; inspect live handle, do not restart training blindly')
    for f in [*G.glob('hour*.json'),*G.glob('*ALERT*.json'),G/'WINDOW_FINISHED.json']:
        if not f.exists():continue
        key=(f.name,f.stat().st_mtime)
        if key in seen:continue
        seen.add(key)
        value=M.read(f)
        if f.name.startswith('hour'):
            value=dict(health={k:v for k,v in value['health'].items() if k not in ('last_rows','owner')},
                       observed=value['cluster']['observed'],healthy28=value['cluster']['healthy28'])
        print(json.dumps(dict(file=f.name,at=time.time(),value=value)),flush=True)
    M.atomic_json(G/'VERIFIED_WAIT.json',dict(at=time.time(),watcher=watcher,watcher_live=M.live(watcher),
                   heartbeat_age=time.time()-heartbeat['at'],arm=heartbeat['arm'],step=heartbeat['step']))
    if (G/'WINDOW_FINISHED.json').exists():break
    time.sleep(60)
