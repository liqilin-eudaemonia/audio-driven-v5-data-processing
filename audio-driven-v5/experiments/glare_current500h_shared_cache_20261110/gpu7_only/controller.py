"""GPU7-only execution with the original complete queue and cache identities."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
ROOT=WORK.parents[1]
spec=importlib.util.spec_from_file_location('cache_execution_guards',WORK/'optimization_execution_v2/controller.py')
guards=importlib.util.module_from_spec(spec);spec.loader.exec_module(guards)
save,read=guards.save,guards.read
DEFAULT_PROFILE=dict(decode_threads=8,detector_services=8,inflight_groups=8,process_decode=False,lia_execution='baseline')


def desired_profile():
    path=HERE/'visual_profile.json'
    value=read(path) if path.exists() else dict(DEFAULT_PROFILE)
    value={k:value.get(k,default) for k,default in DEFAULT_PROFILE.items()}
    assert 1<=value['decode_threads']<=32 and 1<=value['detector_services']<=48 and 1<=value['inflight_groups']<=64
    assert isinstance(value['process_decode'],bool)
    assert value['lia_execution'] in ['baseline','deferred','upload','graph','parallel2','parallel4','fast1','fast2','fused2']
    return value


def current_profile(sig):
    argv=sig['argv'];value={}
    for key in ['decode_threads','detector_services','inflight_groups']:
        flag='--'+key.replace('_','-');value[key]=int(argv[argv.index(flag)+1]) if flag in argv else 8
    value['process_decode']='--process-decode' in argv
    value['lia_execution']=argv[argv.index('--lia-execution')+1] if '--lia-execution' in argv else 'baseline'
    return value


def live_worker(record):
    sig=guards.signature(record['pid'])
    if sig is None:return None
    # A finished/exited worker's PID can be reused during a multi-day run.
    # That unrelated process is neither our live worker nor a signal target.
    if sig['start_ticks']!=record['start_ticks']:return None
    # Linux can clear cmdline/environ during process exit between the stat
    # and cmdline reads. Wait for it to disappear; never signal that state.
    if sig['argv'] and sig['cuda'] is not None:assert guards.owned(sig,record['kind'],record['gpu'])
    return sig


def signal_worker(record):
    sig=live_worker(record)
    if sig is None or not sig['argv'] or sig['cuda'] is None:return
    try:os.kill(record['pid'],signal.SIGTERM)
    except ProcessLookupError:pass


def finished_worker(record,state):
    return bool(record and state.get('pid')==record['pid'] and state.get('stage')=='finished'
                and state.get('completed',0)+state.get('reused',0)==284218 and not state.get('failed'))


def run():
    # The prior controller owns this same lock until all its workers have drained.
    lock=(WORK/'run_all.lock').open('a')
    deadline=time.time()+900
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.time()>deadline:raise RuntimeError('Previous controller did not finish draining')
            time.sleep(2)
    before=read(HERE/'before.json')
    for kind,group in before['worker_groups'].items():
        for gpu,state in group.items():
            current=guards.signature(state['process_pid'])
            assert not guards.owned(current,kind,int(gpu)),('Old cache worker still active',current)
    floor={kind:before[kind+'_completed'] for kind in ['visual','audio']}
    # Include atomic commits made while the old controller was draining.
    if (HERE/'verified_floor.json').exists():floor=read(HERE/'verified_floor.json')
    else:
        for kind in floor:
            folder=WORK/'workers' if kind=='visual' else WORK/'frozen_audio/workers'
            counts=[]
            for gpu in [4,5,6,7]:
                s=read(folder/('gpu%d.json'%gpu))
                counts.append(s.get('completed',0)+s.get('reused',0))
            floor[kind]=max(floor[kind],sum(counts))
        save(HERE/'verified_floor.json',floor)
    records=read(HERE/'registry.json') if (HERE/'registry.json').exists() else {}
    children={};streams=[];stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    def launch(kind,failed_restart=False):
        folder=WORK/'workers' if kind=='visual' else WORK/'frozen_audio/workers'
        script=ROOT/'scripts'/('glare_visual_runtime.py' if kind=='visual' else 'glare_audio_execution_v2.py')
        args=[str(guards.pipeline.GPU_PY),str(script),'--gpu','7','--all-partitions']
        profile=desired_profile() if kind=='visual' else None
        if kind=='visual':
            args+=['--decode-threads',str(profile['decode_threads']),'--detector-services',str(profile['detector_services']),
                   '--inflight-groups',str(profile['inflight_groups']),'--profile-cpu-stages']
            if profile['process_decode']:args+=['--process-decode']
            args+=['--lia-execution',profile['lia_execution']]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(str(x) for x in [
                guards.pipeline.PARENT/'runtime_deps',ROOT/'compare/.envs/glare-deps',ROOT/'src',ROOT/'scripts']))
        log=(folder/'gpu7_all_partitions.log').open('a');streams.append(log)
        proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children[kind]=proc
        for _ in range(40):
            sig=guards.signature(proc.pid)
            if guards.owned(sig,kind,7):break
            time.sleep(.05)
        assert guards.owned(sig,kind,7)
        records[kind]=dict(kind=kind,gpu=7,pid=proc.pid,start_ticks=sig['start_ticks'],
            attempts=records.get(kind,{}).get('attempts',0)+1,profile=profile,
            failures=records.get(kind,{}).get('failures',0)+int(failed_restart))
        save(HERE/'registry.json',records)
    for kind in ['visual','audio']:
        record=records.get(kind)
        folder=WORK/'workers' if kind=='visual' else WORK/'frozen_audio/workers'
        path=folder/'gpu7.json';state=read(path) if path.exists() else {}
        if finished_worker(record,state):continue
        sig=live_worker(record) if record else None
        if sig:
            assert guards.owned(sig,kind,7)
            if kind=='visual':record.setdefault('profile',current_profile(sig))
        else:
            launch(kind)
    while True:
        groups={};done={}
        for kind,r in list(records.items()):
            folder=WORK/'workers' if kind=='visual' else WORK/'frozen_audio/workers'
            path=folder/'gpu7.json';s=read(path) if path.exists() else {}
            if s.get('pid')!=r['pid']:s=dict(stage='initializing')
            sig=None if finished_worker(r,s) else live_worker(r)
            alive=sig is not None
            if kind=='visual' and alive and r.get('profile')!=desired_profile() and not stopping[0] and not (WORK/'STOP').exists():
                if not r.get('draining'):
                    signal_worker(r);r['draining']=True;save(HERE/'registry.json',records)
            observed=s.get('completed',0)+s.get('reused',0)
            floor[kind]=max(floor[kind],observed)
            done[kind]=s.get('stage')=='finished' and observed==284218 and not s.get('failed')
            if not alive and not done[kind] and (r.get('failures',0)<3 or r.get('draining')) and not stopping[0] and not (WORK/'STOP').exists():
                launch(kind,failed_restart=not bool(r.get('draining')));r=records[kind];alive=True;s=dict(stage='initializing')
            if alive and (stopping[0] or (WORK/'STOP').exists()):signal_worker(r)
            groups[kind]={'7':dict(s,process_pid=r['pid'],alive=alive,attempt=r['attempts'],
                cumulative_completed=floor[kind],all_partitions=True,execution_profile=r.get('profile'),reconfiguring=r.get('draining',False))}
        value=dict(stage='running',controller_pid=os.getpid(),samples=284218,gpus=[7],
            worker_groups=groups,heartbeat=time.time(),prosody='deferred_by_user',
            full_glare_training_admitted=False,execution_profile='gpu7-only-all-partitions-v2',
            visual_completed=floor['visual'],audio_completed=floor['audio'],
            visual_failed=groups['visual']['7'].get('failed',0),audio_failed=groups['audio']['7'].get('failed',0))
        prosody=WORK/'prosody/summary.json'
        if prosody.exists():
            value['prosody_status']=read(prosody)
            value['prosody']=value['prosody_status']['stage']
        if stopping[0] or (WORK/'STOP').exists():value['stage']='stopping'
        alive=any(g['7']['alive'] for g in groups.values())
        if not alive:value['stage']='nonprosody_complete' if all(done.values()) else 'needs_error_review'
        save(WORK/'status_all.json',value)
        save(HERE/'verified_floor.json',floor)
        save(HERE/'registry.json',records)
        if not alive:return
        time.sleep(5)


if __name__=='__main__':run()
