"""Bounded live-process wait; a timeout never stops or restarts any task."""
from pathlib import Path
import sys,time,json
WORK=Path(__file__).resolve().parent;sys.path.insert(0,str(WORK));import swap as S
end=time.time()+50;initial=S.read(WORK/'STATE.json')['stage'];reason='timer';obs=S.observation_module()
while time.time()<end:
 controller=S.read(WORK/'CONTROLLER.json');gate=S.read(WORK/'GATE_CONTROLLER.json')
 if not S.alive(controller):reason='controller_exited';break
 full=S.read(S.path('full')/'status.json')['step'];stage=S.read(WORK/'STATE.json')['stage']
 if not S.alive(gate)and not(WORK/'GATE_STOP_REQUESTED.json').exists():reason='gate_exited_without_request';break
 if list(WORK.glob('ERROR_*.json')):reason='needs_review';break
 if full<6000 and obs.health()['flags']:reason='parent_training_needs_review';break
 if stage!=initial:reason='stage_changed';break
 if(WORK/'SWAP_ACCEPTED.json').exists():reason='swap_accepted';break
 time.sleep(min(5,max(.01,end-time.time())))
v=dict(at=time.time(),reason=reason,controller_alive=S.alive(S.read(WORK/'CONTROLLER.json')),gate_alive=S.alive(S.read(WORK/'GATE_CONTROLLER.json')),stage=S.read(WORK/'STATE.json')['stage'],full_parent_step=S.read(S.path('full')/'status.json')['step'],noevent_parent_step=S.read(S.path('noevent')/'status.json')['step'])
S.write(WORK/'LATEST_AGENT_CHECK.json',v)
with(WORK/'agent_checks.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v),flush=True)
