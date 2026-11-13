import subprocess,time,json
from pathlib import Path
W=Path(__file__).resolve().parent;key='/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519'
reports=[]
def command(host):return ['ssh','-i',key,'-o','BatchMode=yes','-o','ConnectTimeout=8','root@'+host,'python3 -u -']
for holder,contender in [('172.29.15.91','172.29.225.51'),('172.29.225.51','172.29.15.91')]:
 lock=str(W/'validation/node4-flock.lock')
 p=subprocess.Popen(command(holder),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 p.stdin.write('import fcntl,time\nf=open('+repr(lock)+',"a+")\nfcntl.flock(f,fcntl.LOCK_EX)\nprint("HELD",flush=True)\ntime.sleep(6)\n');p.stdin.close()
 assert p.stdout.readline().strip()=='HELD'
 code='import fcntl\nf=open('+repr(lock)+',"a+")\ntry:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n print("BAD_ACQUIRED")\nexcept BlockingIOError:print("BLOCKED_OK")\n'
 c=subprocess.run(command(contender),input=code,text=True,capture_output=True,timeout=15);assert c.stdout.strip()=='BLOCKED_OK',c
 p.wait(timeout=15);assert p.returncode==0
 reports.append(dict(holder=holder,contender=contender,result='BLOCKED_OK'))
(W/'validation/node4_lock.json').write_text(json.dumps(dict(passed=True,at=time.time(),checks=reports),indent=2)+'\n')
print('node4 shared queue exclusion passed both directions')
