"""One-time continuation after in-flight workers finish; never kills productive work."""
import fcntl,json,os,subprocess,time
from pathlib import Path
work=Path(__file__).resolve().parent;root=work.parents[1]
old=json.loads((work/'launch.json').read_text())['pid']
while True:
 process=Path(f'/proc/{old}/stat')
 alive=process.exists() and process.read_text().split(') ',1)[1].split()[0]!='Z'
 handles=[];busy=False
 for path in (work/'locks').glob('cache_*.lock'):
  f=path.open('a');handles.append(f)
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:busy=True;break
 for f in handles:f.close()
 if not alive and not busy:break
 time.sleep(10)
if (work/'STOP').exists():raise SystemExit('Stopped by user')
if json.loads((work/'status.json').read_text())['stage']!='failed':raise SystemExit('Original queue did not fail; no restart needed')
cmd=[str(root.parent/'Omni-LTX-stage2/.venv/bin/python'),'-u',str(root/'scripts/p2_testsets_eval.py'),'queue','--work',str(work)]
p=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=(work/'queue.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'})
(work/'launch.json').write_text(json.dumps(dict(pid=p.pid,command=cmd,at=time.time(),resumed_after='endpoint_and_Gemma_projection_workspace_fix'),indent=2))
print('Resumed',p.pid,flush=True)
