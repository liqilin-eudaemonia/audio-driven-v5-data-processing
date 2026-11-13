from pathlib import Path
import json,os,signal,time,subprocess
W=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx6500_latest_sft_dual200_20261107')
WORKER='/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_ltx6500_latest_sft_dual200_20261107.py'
CONTROLLER='/private/liqi_workspace/project_code/audio-driven-v5/scripts/p2_ltx6500_overlap_20261107.py'
(W/'STOP').write_text(json.dumps(dict(owner='user-prioritize-av-only31',at=time.time(),preserve_completed=True))+'\n')
def args(pid):
 try:
  p=Path('/proc',str(pid));s=(p/'stat').read_text().split();return s,(p/'cmdline').read_bytes().split(b'\0')
 except OSError:return [],[]
record=json.loads((W/'overlap/CONTROLLER.json').read_text());s,a=args(record['pid'])
stopped=[]
if s and s[2]!='Z'and s[21]==record['start_ticks']and CONTROLLER.encode()in a and str(W).encode()in a:
 os.kill(record['pid'],signal.SIGTERM);stopped.append(dict(pid=record['pid'],start_ticks=s[21],kind='controller'))
candidates={json.loads(p.read_text())['pid']for p in (W/'overlap/processes').rglob('*.json')}
for pid in candidates:
 p=Path('/proc',str(pid))
 s,a=args(pid)
 if s and s[2]!='Z'and WORKER.encode()in a and str(W).encode()in a and b'worker'in a:
  os.kill(int(p.name),signal.SIGTERM);stopped.append(dict(pid=int(p.name),start_ticks=s[21],kind='worker'))
for _ in range(100):
 live=[]
 for v in stopped:
  s,a=args(v['pid'])
  if s and s[2]!='Z'and s[21]==v['start_ticks']:live.append(v)
 if not live:break
 time.sleep(.1)
for v in live:
 s,a=args(v['pid'])
 if s and s[2]!='Z'and s[21]==v['start_ticks']and (WORKER.encode()in a or CONTROLLER.encode()in a):os.kill(v['pid'],signal.SIGKILL)
result=dict(at=time.time(),user_authorized=True,stopped=stopped,completed_outputs_preserved=True,stop_marker=str(W/'STOP'))
(W/'USER_PAUSED_FOR_AV31.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
