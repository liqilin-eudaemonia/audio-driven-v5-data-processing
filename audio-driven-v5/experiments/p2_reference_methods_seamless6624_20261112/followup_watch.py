"""CPU-only follow-up supervisor: Seamless6624 completes before ResponseNet586."""
from runtime import *

def alive(pid,script):
    try:
        proc=Path(f'/proc/{pid}')
        return proc.joinpath('stat').read_text().split(') ')[1][0]!='Z' and str(script).encode() in proc.joinpath('cmdline').read_bytes()
    except FileNotFoundError:return False

def marker_ok(path,scope):
    if not Path(path).exists():return False
    r=read(path);return r.get('passed') is True and r.get('scope')==scope

def main():
    with (WORK/'followup_watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states={};children={};plans=WORK/'FOLLOWUPS.json'
        prior=WORK/'FOLLOWUPS_STATUS.json'
        if prior.exists():states=read(prior)['entries']
        while True:
            done=True
            for job in read(plans)['entries']:
                key=job['id'];target=Path(job['workdir']);script=target/'monitor.py';state=states.get(key,{})
                assert target.parent==WORK.parent and read(target/'RUN_SCOPE.json')['scope']==job['scope']
                if marker_ok(target/'COMPLETE.json',job['scope']):
                    states[key]=dict(stage='complete',at=time.time(),workdir=str(target));continue
                done=False
                if not marker_ok(job['wait_for'],job['wait_scope']):
                    states[key]=dict(stage='waiting_for_predecessor',wait_for=job['wait_for'],workdir=str(target),at=time.time());continue
                assert read(target/'PREFLIGHT.json')['passed'] and read(target/'REUSE_COMPLETE.json')['passed']
                pidfile=target/'MONITOR_PROCESS.json'
                if pidfile.exists():
                    pid=read(pidfile)['pid']
                    if alive(pid,script):states[key]=dict(stage='monitor_running',pid=pid,workdir=str(target),at=time.time());continue
                if state.get('pid') and alive(state['pid'],script):continue
                if time.time()-state.get('started',0)<60:continue
                logs=WORK/'followups'/key;logs.mkdir(parents=True,exist_ok=True);attempt=len(list(logs.glob('monitor_*.log')));log=logs/f'monitor_{attempt:03d}.log'
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                child=subprocess.Popen([job['python'],str(script)],cwd=ROOT,env=env,stdout=open(log,'w'),stderr=subprocess.STDOUT,start_new_session=True);children[key]=child
                states[key]=dict(stage='monitor_running',pid=child.pid,attempt=attempt,log=str(log),workdir=str(target),started=time.time(),at=time.time())
                print('Started follow-up',key,child.pid,flush=True)
            save(prior,dict(at=time.time(),order='Seamless6624 -> ResponseNet586',cpu_only=True,entries=states))
            if done:return
            time.sleep(20)
if __name__=='__main__':main()
