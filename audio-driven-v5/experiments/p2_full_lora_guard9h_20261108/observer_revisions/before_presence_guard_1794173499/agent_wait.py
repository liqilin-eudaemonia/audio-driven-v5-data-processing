"""Bounded read-only wait on verified-live observer, supervisor and training progress."""
from pathlib import Path
import sys,time,json
WORK=Path(__file__).resolve().parent;sys.path.insert(0,str(WORK));import common as C
p=C.read(WORK/'protocol.json');until=min(time.time()+50,p['deadline']);initial_reports=len(list((WORK/'reports').glob('hour_*.json')));reason='timer'
while True:
 state=C.read(WORK/'status.json');g=C.read(WORK/'GUARDIAN.json');s=C.read(WORK/'SUPERVISOR.json')
 live=C.alive(g);superlive=C.alive(s);latest=sorted(C.OUT.glob('step_*.complete.json'))
 pending=latest[-1].name.replace('.complete.json','.json')if latest else None
 if not live or not superlive:reason='observer_process_check';break
 if state['flags']or(WORK/'NEEDS_AGENT.json').exists():reason='needs_review';break
 if not state.get('saving')and state['speed'].get('ratio')and state['speed']['ratio']>1.35:
  shared_path=WORK/'SHARED_COMPUTE.json';shared=C.read(shared_path)if shared_path.exists()else None
  known=shared and time.time()-shared['at']<90 and state.get('last_cluster_healthy31')and state['metrics_age_seconds']<60
  if not known:
   from guardian import shared_compute
   shared=shared_compute(C.probe(),state)
   if shared:C.atomic_json(shared_path,shared)
   else:reason='needs_review';break
 if time.time()-state['at']>90:reason='observation_delayed';break
 if pending and not(WORK/'checkpoints'/pending).exists():reason='new_complete_checkpoint';break
 if len(list((WORK/'reports').glob('hour_*.json')))>initial_reports:reason='new_hour_report';break
 if time.time()>=until:break
 time.sleep(min(10,max(.01,until-time.time())))
v=dict(classification='verified_wait',at=time.time(),reason=reason,step=state['step'],guardian_alive=live,supervisor_alive=superlive,heartbeat_age=time.time()-state['at'],flags=state['flags'],speed_ratio=state['speed'].get('ratio'),remaining_seconds=max(0,p['deadline']-time.time()),needs_agent=(WORK/'NEEDS_AGENT.json').exists(),reports=len(list((WORK/'reports').glob('hour_*.json'))),incidents=len(list(WORK.glob('incident_*'))))
C.atomic_json(WORK/'LATEST_AGENT_AUDIT.json',v)
with(WORK/'agent_checks.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v),flush=True)
