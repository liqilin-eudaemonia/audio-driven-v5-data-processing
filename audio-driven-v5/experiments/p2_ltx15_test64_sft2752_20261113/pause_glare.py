import json,os,signal,time
from pathlib import Path
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5')
w=ROOT/'experiments/glare10618_seamless_test6624_20261113';out=ROOT/'experiments/p2_ltx15_test64_sft2752_20261113'
launch=json.loads((w/'CONTROLLER_LAUNCH.json').read_text());pid=launch['pid'];proc=Path('/proc')/str(pid)
reason=dict(at=time.time(),owner=out.name,reason='User requests pause GLARE GPU4 and prioritize15 LTX checkpoints on fixed64',resume_automatically=False)
(w/'STOP').write_text(json.dumps(reason)+'\n')
before=json.loads((w/'status.json').read_text())
cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else ''
stopped=[]
if cmd:
 assert 'eval_glare10618_seamless_test_20261113.py' in cmd and os.getpgid(pid)==pid
 members=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:
    if os.getpgid(int(p.name))==pid:members.append(int(p.name))
   except ProcessLookupError:pass
 os.killpg(pid,signal.SIGTERM);stopped=members
 time.sleep(3)
 live=[]
 for child in members:
  p=Path('/proc')/str(child)
  try:
   if p.joinpath('stat').read_text().split(') ',1)[1][0]!='Z':live.append(child)
  except FileNotFoundError:pass
 for child in live:
  if os.getpgid(child)==pid:os.kill(child,signal.SIGKILL)
result=dict(**reason,controller_pid=pid,controller_command=cmd,stopped_pids=stopped,prior_status=before,outputs_preserved=True)
(w/'PAUSED_FOR_LTX15.json').write_text(json.dumps(result,indent=2)+'\n');(out/'GLARE_PAUSED.json').write_text(json.dumps(result,indent=2)+'\n')
(w/'status.json').write_text(json.dumps(dict(before,stage='paused_for_ltx15_test64',paused_at=time.time()),indent=2)+'\n')
print(json.dumps(dict(pid=pid,stopped=len(stopped),prior_status=before)))
