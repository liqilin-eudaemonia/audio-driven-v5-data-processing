"""One-time controlled restart of the existing cache launchers only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
WORK=ROOT/'experiments/p2_j_train500h_cache_20261101'
OUT=WORK/'revisions/video_gap_v5'
mode,node,rank=sys.argv[1:]
path=WORK/'launchers'/(node+'.json')
old=json.loads(path.read_text())
if mode=='stop':
    (OUT/(node+'_before.json')).write_text(json.dumps(old,indent=2))
    parent=psutil.Process(old['pid'])
    command=parent.cmdline()
    assert 'node' in command and node in command and any('p2_j_cluster.py' in arg for arg in command),command
    children=parent.children(recursive=True)
    # Pause the parent before signaling workers, preventing a GPU guard race.
    parent.suspend()
    for process in children:
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    gone,alive=psutil.wait_procs(children,timeout=12)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    parent.kill()
    result=dict(node=node,old_launcher=old['pid'],children=len(children),forced=len(alive),time=time.time())
    (OUT/(node+'_stopped.json')).write_text(json.dumps(result,indent=2))
elif mode=='start':
    if psutil.pid_exists(old['pid']):
        assert psutil.Process(old['pid']).status()==psutil.STATUS_ZOMBIE,'Old launcher still running'
    command=[sys.executable,str(ROOT/'scripts/p2_j_cluster.py'),'node','--work',str(WORK),
        '--node',node,'--rank',rank,'--phase','features','--prefetch','8','--media-processes','8']
    with (WORK/'logs'/(node+'-launcher.log')).open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(node=node,pid=p.pid,time=time.time(),command=command)
    (OUT/(node+'_restarted.json')).write_text(json.dumps(result,indent=2))
else:raise ValueError(mode)
print(json.dumps(result),flush=True)
