from pathlib import Path
import json,subprocess,shlex,concurrent.futures,sys,time
root=Path('/private/liqi_workspace/project_code/audio-driven-v5');folder=Path(__file__).parent
previous=json.loads((folder.parent/'scale24/launch.json').read_text())['workers']
selected=[int(x) for x in sys.argv[1:]]
assert selected and len(set(selected))==len(selected) and all(0<=x<24 for x in selected)
key='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
code='''import sys,json,os,signal,time,subprocess
from pathlib import Path
row=json.loads(sys.argv[1]);pid=row['pid'];folder=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_matrix16_teacher50h_20261027/scale48')
if (folder/f"launch-{row['rank']:02d}.json").exists():raise RuntimeError('Already repartitioned; inspect rather than duplicate')
try: cmd=open(f'/proc/{pid}/cmdline','rb').read().decode().replace('\\x00',' ')
except FileNotFoundError:cmd=''
if cmd:
 if 'scale24/worker.py' not in cmd:raise RuntimeError('Wrong previous PID')
 os.kill(pid,signal.SIGTERM)
for _ in range(60):
 try: state=open(f'/proc/{pid}/stat').read().split(') ')[1].split()[0]
 except FileNotFoundError: state='gone'
 if state in ('gone','Z'):break
 time.sleep(.5)
else:raise RuntimeError('Previous worker has not stopped')
workers=[]
for rank in (row['rank'],row['rank']+24):
 log=folder/f'worker-{rank:02d}.log'
 with log.open('ab',buffering=0) as stream:
  proc=subprocess.Popen(['/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python',str(folder/'worker.py'),str(rank),str(row['gpu'])],cwd=folder.parents[2],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 workers.append(dict(node=row['node'],host=row['host'],gpu=row['gpu'],rank=rank,pid=proc.pid,log=str(log)))
result=dict(replaced=row,stopped_state=state,workers=workers,at=time.time())
(folder/f"launch-{row['rank']:02d}.json").write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
'''
def run(row):
 command=shlex.join(['python','-c',code,json.dumps(row)])
 p=subprocess.run(['ssh','-i',key,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+row['host'],command],capture_output=True,text=True,timeout=55)
 if p.returncode:raise RuntimeError(str(row['rank'])+p.stderr)
 return json.loads(p.stdout)
# Repartitioning old rank i as i/48 and (i+24)/48 covers precisely i/24.
for i in selected:assert set(range(29975)[i::24])==set(range(29975)[i::48])|set(range(29975)[i+24::48])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,[r for r in previous if r['rank'] in selected]))
print(json.dumps(dict(replaced=selected,launched=sum(len(x['workers']) for x in results))))
