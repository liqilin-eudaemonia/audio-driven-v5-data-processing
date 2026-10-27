"""User authorized stopping this identified two-GPU inference job on node 3."""
import os,signal,json,time,socket
from pathlib import Path
assert socket.gethostname()=='dedicated-developjob-lq-dev-machine-3-zhw7q'
expected={2583647:('442938900','test-wf27'),2583648:('442938906','LTX-infer-f27'),2584088:('442940947','LTX-infer-f27')}
records=[]
def inspect(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,birth=fields[19],state=fields[0],pgid=os.getpgid(pid),cwd=str((p/'cwd').readlink()),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
for pid,(birth,command) in expected.items():
 r=inspect(pid)
 if r is None:continue
 assert r['birth']==birth and r['argv']==command and r['cwd']=='/private/codes/learning/LTX-infer-LiHR/training-f27-code',r
 records.append(r)
# Stop the job's own supervisor first so it cannot respawn the GPU workers.
if any(r['pid']==2583647 for r in records):os.kill(2583647,signal.SIGTERM)
leader=inspect(2583648)
if leader:
 assert leader['birth']==expected[2583648][0] and leader['pgid']==2583648
 os.killpg(2583648,signal.SIGTERM)
time.sleep(10)
remaining=[]
for pid,(birth,command) in expected.items():
 try:r=inspect(pid)
 except FileNotFoundError:continue
 if r and r['birth']==birth and r['state']!='Z':
  os.kill(pid,signal.SIGKILL);remaining.append(pid)
report=dict(host=socket.gethostname(),user_authorized=True,stopped=records,required_sigkill=remaining,time=time.time())
output=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_preflight_fix_20261026/authorized_inference_stop.json')
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
