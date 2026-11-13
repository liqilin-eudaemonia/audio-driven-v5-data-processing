"""Detached Monitor for authorized node11 GPU4/GPU7 and the resumable evaluator."""
from runtime import *

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False

def launch_controller():
    d=WORK/'monitor';d.mkdir(exist_ok=True);attempt=len(list(d.glob('controller_*.log')));log=d/f'controller_{attempt:03d}.log'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    p=subprocess.Popen([sys.executable,str(WORK/'controller.py')],cwd=ROOT,env=env,stdout=open(log,'w'),stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(pid=p.pid,attempt=attempt,log=str(log),started=time.time());save(d/'controller.json',r);return r,p

def main():
    with (WORK/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(WORK/'REUSE_COMPLETE.json')['passed'];assert read(WORK/'PREFLIGHT.json')['passed'];controller=None;child=None;failures=0;idle_since={4:None,7:None}
        old=WORK/'monitor/controller.json'
        if old.exists():
            r=read(old)
            if alive(r['pid']) and str(WORK/'controller.py').encode() in Path(f"/proc/{r['pid']}/cmdline").read_bytes():controller=r
        while True:
            now=time.time();gpus={}
            try:
                lines=subprocess.check_output(['nvidia-smi','-i','4,7','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines()
                for line in lines:
                    index,uuid,total,free,util=[x.strip() for x in line.split(',')];gpu=int(index);assert GPU_UUIDS[gpu]==uuid
                    gpus[index]=dict(uuid=uuid,total_mib=int(total),free_mib=int(free),utilization_pct=int(util))
                assert set(gpus)=={'4','7'}
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines()
                foreign={4:[],7:[]};owned={4:[],7:[]};by_uuid={v:k for k,v in GPU_UUIDS.items()}
                for line in apps:
                    parts=[v.strip() for v in line.split(',')]
                    if len(parts)!=3 or parts[0] not in by_uuid:continue
                    gpu=by_uuid[parts[0]];pid=int(parts[1]);proc=Path(f'/proc/{pid}/cmdline')
                    command=proc.read_bytes() if proc.exists() else b''
                    row=dict(pid=pid,memory_mib=int(parts[2]) if parts[2].isdigit() else None)
                    (owned if (str(WORK)+'/').encode() in command else foreign)[gpu].append(row)
                policy=read(WORK/'RESOURCE_POLICY.json')
                for gpu in GPU_UUIDS:
                    if foreign[gpu]:idle_since[gpu]=None
                    elif idle_since[gpu] is None:idle_since[gpu]=now
                    gpus[str(gpu)].update(external=foreign[gpu],owned=owned[gpu],external_idle_since=idle_since[gpu],eligible_idle=not foreign[gpu] and now-(idle_since[gpu] or now)>=policy['external_idle_seconds'])
                value=dict(at=now,gpus=gpus,controller=controller,policy=policy,resource_policy_sha256=sha(WORK/'RESOURCE_POLICY.json'),controller_restarts=failures)
                save(WORK/'MONITOR.json',value)
                with (WORK/'monitor_resources.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
            except Exception as e:
                save(WORK/'MONITOR_ERROR.json',dict(at=now,error=repr(e)));time.sleep(20);continue
            if (WORK/'COMPLETE.json').exists():return
            if controller is None or not alive(controller['pid']):
                if controller is not None:
                    code=child.poll() if child is not None else None;save(WORK/'monitor'/f"controller_exit_{controller['attempt']:03d}.json",dict(returncode=code,at=now));failures+=1
                if controller is None or now-controller['started']>60:controller,child=launch_controller()
            time.sleep(20)
if __name__=='__main__':main()
