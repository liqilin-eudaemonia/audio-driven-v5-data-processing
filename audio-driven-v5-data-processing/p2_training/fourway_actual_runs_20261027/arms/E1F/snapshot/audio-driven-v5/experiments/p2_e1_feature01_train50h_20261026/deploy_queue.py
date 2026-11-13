"""Deploy three detached CPU watchdogs; never directly launches a GPU job."""
from queue_common import *
files=[OUT/n for n in ('queue_common.py','queue_node.py','queue_watchdog.py','control.py','release.json','queue_known_hosts')]
write(OUT/'queue-release.json',dict(files={str(p):sha(p) for p in files},created_at=now()))
from queue_watchdog import verify_pins
verify_pins('E0');verify_pins('E1')
assert read(OUT/'queue_lock_validation.json')['status']=='passed'
assert '22 passed' in (OUT/'queue_tests.log').read_text()
def launch(rank):
 cmd=[str(PYTHON),str(OUT/'queue_watchdog.py'),'--node-rank',str(rank),'--detach']
 if rank:cmd=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+HOSTS[rank],shlex.join(cmd)]
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 if result.returncode:raise RuntimeError(result.stderr+result.stdout)
 return json.loads(result.stdout)
with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(launch,range(3)))
write(OUT/'queue_deployment.json',dict(status='deployed',watchdogs=results,at=now()))
print(json.dumps(results,indent=2))
