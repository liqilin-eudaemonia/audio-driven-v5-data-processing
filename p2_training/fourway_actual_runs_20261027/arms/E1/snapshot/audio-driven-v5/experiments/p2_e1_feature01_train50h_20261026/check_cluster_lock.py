from queue_common import *
import uuid
folder=OUT/'queue_lock_checks'/uuid.uuid4().hex;folder.mkdir(parents=True)
children=[]
for rank in range(3):
 cmd=[str(PYTHON),str(OUT/'queue_lock_probe.py'),str(folder),str(rank)]
 if rank:cmd=['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=10','root@'+HOSTS[rank],shlex.join(cmd)]
 children.append(subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,close_fds=True))
deadline=time.monotonic()+30
while not all((folder/f'ready{r}').exists() for r in range(3)):
 if time.monotonic()>deadline:raise TimeoutError('Three-host ready barrier failed')
 time.sleep(.1)
(folder/'go').touch()
for child in children:
 _,error=child.communicate(timeout=15)
 if child.returncode:raise RuntimeError(error.decode())
results=[read(folder/f'node{r}.json') for r in range(3)]
assert sum(r['leader'] for r in results)==1,results
write(OUT/'queue_lock_validation.json',dict(status='passed',exactly_one_leader=True,nodes=results,checked_at=now()))
print(json.dumps(results))
