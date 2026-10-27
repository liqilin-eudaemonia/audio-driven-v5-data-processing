"""Finite GPU experiment subprocess with status, logs and device telemetry."""
import json,os,subprocess,sys,time,signal,socket
from pathlib import Path
specpath=Path(sys.argv[1]);spec=json.loads(specpath.read_text());out=specpath.parent
status=out/'status.json'
def save(value):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(status)
env=dict(os.environ,**spec.get('env',{}));start=time.time()
with (out/'run.log').open('w') as log,(out/'gpu.jsonl').open('w') as telemetry:
 proc=subprocess.Popen(spec['argv'],cwd=spec['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 info=dict(host=socket.gethostname(),wrapper_pid=os.getpid(),pid=proc.pid,started=start,argv=spec['argv'])
 save(dict(info,state='running'))
 while proc.poll() is None:
  if (out/'STOP').exists() or time.time()-start>spec.get('timeout',5400):
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   break
  query=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  telemetry.write(json.dumps(dict(time=time.time(),gpu=query.stdout.strip()))+'\n');telemetry.flush()
  time.sleep(2)
 save(dict(info,state='completed' if proc.returncode==0 else 'failed',returncode=proc.returncode,finished=time.time(),seconds=time.time()-start))
