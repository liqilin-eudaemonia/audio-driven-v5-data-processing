"""One card per node per wave, with live guardian evidence before the next wave."""
import json
from pathlib import Path
import subprocess
import time

W=Path(__file__).resolve().parents[1]
ROOT=W.parents[1]
def read(path):return json.loads(path.read_text())
assert read(W/'validation/pts_tail_recovery_audit.json')['passed']
waves=[ [('node17',7),('node8',7),('node4',0)] ]
waves += [[('node17',g),('node8',g),('node4',g+1)] for g in range(6)]
waves += [[('node17',6),('node8',6)]]
history=[]
for index,wave in enumerate(waves):
    assert not read(W/'queues/caption/state.json')['failed'],'Inspect failed sources before continuing rollout'
    for node,gpu in wave:
        subprocess.run(['python',str(W/'performance_check/roll_pts_tail.py'),'--node',node,'--gpus',str(gpu)],check=True,cwd=ROOT)
    deadline=time.time()+420
    while time.time()<deadline:
        state=read(W/'monitoring_10h/state.json')
        proc=Path('/proc')/str(state['pid'])/'stat'
        assert proc.exists() and proc.read_text().split()[21]==state['start_ticks']
        assert time.time()-state['heartbeat']<180
        latest=read(W/'monitoring_10h/latest.json');ready=[]
        for node,gpu in wave:
            h=read(W/'workers'/f'{node}-gpu-{gpu}.json')
            d=latest['nodes'][node]
            assert not d.get('blocked_gpus') and not d.get('supervisor_error')
            observed=next((x for x in d.get('workers',[]) if x['gpu']==gpu),{})
            if ('pts-tail-validation-v1' in h.get('execution_revision','') and h['completed_clips']>=5
                    and time.time()-h['heartbeat']<45 and not h.get('last_error')
                    and observed.get('pid')==h['pid'] and observed.get('alive') and observed.get('matching_heartbeat')
                    and time.time()-latest['at']<60):
                ready.append((node,gpu,h['pid'],h['completed_clips']))
        if len(ready)==len(wave):
            print('WAVE_READY',index,ready,flush=True)
            history.append({'wave':index,'at':time.time(),'workers':ready})
            break
        time.sleep(10)
    else:raise RuntimeError(f'Wave {index} not ready; inspect live workers without blind restart')
out={'at':time.time(),'waves':history,'passed':True}
(W/'revisions/pts_tail_validation/cluster-rollout.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL_24_UPDATED_AND_PRODUCTIVE',flush=True)
