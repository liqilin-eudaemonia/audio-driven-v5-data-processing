"""Adopt only the current cache workers; replace execution, keep frozen data.

The prior controller is replaced alone. Existing visual workers remain alive
while audio drains, then visual concurrency is promoted one GPU at a time.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_glare_shared_cache as pipeline
import glare_frozen_audio_cache as audio
save,read=pipeline.save,pipeline.read
HERE=Path(__file__).resolve().parent


def signature(pid):
    try:
        path=Path('/proc')/str(pid)
        rest=(path/'stat').read_text().rsplit(')',1)[1].split()
        if rest[0]=='Z':return None
        return dict(pid=pid,start_ticks=rest[19],argv=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x],
                    cuda=next((x.split(b'=',1)[1].decode() for x in (path/'environ').read_bytes().split(b'\0') if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None))
    except (FileNotFoundError,ProcessLookupError):return None


def owned(sig,kind,gpu):
    if not sig or sig['cuda']!=str(gpu):return False
    allowed={'visual':['run_glare_shared_cache.py','glare_visual_runtime.py'],
             'audio':['glare_frozen_audio_cache.py','glare_audio_execution_v2.py']}[kind]
    argv=sig['argv']
    if len(argv)<2 or argv[1] not in [str(ROOT/'scripts'/name) for name in allowed] or '--gpu' not in argv:return False
    index=argv.index('--gpu')
    return index+1<len(argv) and argv[index+1]==str(gpu)


def signal_owned(record,sig):
    current=signature(record['pid'])
    if current is None:return
    assert owned(current,record['kind'],record['gpu']) and current['start_ticks']==record['start_ticks'],'Refuse signaling an unowned/reused PID'
    os.kill(record['pid'],sig)


def capture():
    HERE.mkdir(parents=True,exist_ok=True)
    snapshot=read(WORK/'status_all.json')
    controller=signature(snapshot['controller_pid'])
    assert controller and str(WORK/'run_all.py') in controller['argv'],'Unexpected prior controller'
    records={}
    for kind,group in snapshot['worker_groups'].items():
        for text,state in group.items():
            gpu=int(text);sig=signature(state['process_pid'])
            assert owned(sig,kind,gpu) or (sig is None and state.get('stage')=='finished' and not state.get('failed')),(kind,gpu,sig)
            records[kind+':'+text]=dict(kind=kind,gpu=gpu,pid=state['process_pid'],start_ticks=sig['start_ticks'] if sig else None,
                floor=state.get('completed',0)+state.get('reused',0),phase='legacy' if sig else 'finished_legacy',attempts=0)
    config=dict(controller=controller,records=records,snapshot=snapshot,
                profile=dict(decode_threads=8,detector_services=8,inflight_groups=8),at=time.time())
    save(HERE/'ADOPTION.json',config)
    return config


def run():
    config=read(HERE/'ADOPTION.json')
    lock=(WORK/'run_all.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=read(HERE/'registry.json') if (HERE/'registry.json').exists() else config['records']
    children={};streams=[];stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    for r in records.values():
        current=signature(r['pid'])
        if current is None and r['phase'] not in ('draining','v2'):
            folder=WORK/'workers' if r['kind']=='visual' else audio.WORK/'workers'
            state=read(folder/('gpu%d.json'%r['gpu']))
            assert state.get('pid')==r['pid'] and state.get('stage')=='finished' and not state.get('failed')
            r['phase']='finished_legacy'
        elif current is not None:assert owned(current,r['kind'],r['gpu'])
    def launch(key):
        r=records[key];kind,gpu=r['kind'],r['gpu']
        folder=WORK/'workers' if kind=='visual' else audio.WORK/'workers'
        script=ROOT/'scripts'/('glare_visual_runtime.py' if kind=='visual' else 'glare_audio_execution_v2.py')
        args=['--gpu',str(gpu)]
        if kind=='visual':args+=['--decode-threads','8','--detector-services','8','--inflight-groups','8']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(str(x) for x in [
                pipeline.PARENT/'runtime_deps',ROOT/'compare/.envs/glare-deps',ROOT/'src',ROOT/'scripts']))
        stream=(folder/('gpu%d_execution_v2.log'%gpu)).open('a');streams.append(stream)
        proc=subprocess.Popen([str(pipeline.GPU_PY),str(script),*args],cwd=ROOT,env=env,
                               stdout=stream,stderr=subprocess.STDOUT)
        children[key]=proc
        s=signature(proc.pid)
        for _ in range(40):
            if owned(s,kind,gpu):break
            time.sleep(.05);s=signature(proc.pid)
        assert owned(s,kind,gpu)
        r.update(pid=proc.pid,start_ticks=s['start_ticks'],phase='v2',attempts=r['attempts']+1)
        save(HERE/'registry.json',records)
    # Drain only the four audio producers, leaving visual production running.
    for key,r in records.items():
        if r['kind']=='audio' and r['phase']=='legacy':
            signal_owned(r,signal.SIGTERM);r['phase']='draining'
    promoted=read(HERE/'promotion.json')['promoted'] if (HERE/'promotion.json').exists() else []
    while True:
        groups={'visual':{},'audio':{}}
        for key,r in records.items():
            folder=WORK/'workers' if r['kind']=='visual' else audio.WORK/'workers'
            path=folder/('gpu%d.json'%r['gpu']);state=read(path) if path.exists() else {}
            if state.get('pid')!=r['pid']:state={'stage':'initializing'}
            actual=signature(r['pid']);alive=actual is not None and actual['start_ticks']==r['start_ticks']
            if not alive:
                if r['phase']=='draining' and not stopping[0] and not (WORK/'STOP').exists():
                    r['floor']=max(r['floor'],state.get('completed',0)+state.get('reused',0))
                    launch(key);state={'stage':'initializing'};alive=True
                elif r['phase']=='v2' and state.get('stage')!='finished' and r['attempts']<3 and not stopping[0] and not (WORK/'STOP').exists():
                    launch(key);state={'stage':'initializing'};alive=True
            observed=state.get('completed',0)+state.get('reused',0)
            groups[r['kind']][str(r['gpu'])]=dict(state,process_pid=r['pid'],execution_phase=r['phase'],
                legacy_verified_floor=r['floor'],cumulative_completed=max(r['floor'],observed),
                alive=alive,attempt=r['attempts'])
        # Promote one visual GPU only after the previous one produced8new pairs.
        audio_ready=all(records['audio:'+str(g)]['phase']=='finished_legacy' or
                       (records['audio:'+str(g)]['phase']=='v2' and groups['audio'][str(g)].get('completed',0)>0) for g in [4,5,6,7])
        previous_ready=not promoted or (records['visual:'+str(promoted[-1])]['phase']=='v2' and
                                         groups['visual'][str(promoted[-1])].get('completed',0)>=8 and
                                         groups['visual'][str(promoted[-1])].get('failed',0)==0)
        if audio_ready and previous_ready and len(promoted)<4 and not stopping[0] and not (WORK/'STOP').exists():
            gpu=[4,5,6,7][len(promoted)];r=records['visual:'+str(gpu)]
            state=groups['visual'][str(gpu)]
            r['floor']=max(r['floor'],state['cumulative_completed'])
            signal_owned(r,signal.SIGTERM);r['phase']='draining';promoted.append(gpu)
            save(HERE/'promotion.json',dict(promoted=promoted,at=time.time()))
        value=dict(stage='running',controller_pid=os.getpid(),samples=284218,gpus=[4,5,6,7],
            worker_groups=groups,heartbeat=time.time(),prosody='deferred_by_user',
            full_glare_training_admitted=False,execution_profile='async-audio+visual8-staggered-v2',
            visual_promoted_gpus=promoted)
        for kind in groups:
            value[kind+'_completed']=sum(s['cumulative_completed'] for s in groups[kind].values())
            value[kind+'_failed']=sum(s.get('failed',0) for s in groups[kind].values())
        if stopping[0] or (WORK/'STOP').exists():
            for r in records.values():signal_owned(r,signal.SIGTERM)
            value['stage']='stopping'
        if not any(s['alive'] for group in groups.values() for s in group.values()):
            # Final completeness uses this run's verified counts, not floors.
            actual_totals={kind:sum(s.get('completed',0)+s.get('reused',0) for s in groups[kind].values()) for kind in groups}
            value['stage']='nonprosody_complete' if all(n==284218 for n in actual_totals.values()) and not value['visual_failed'] and not value['audio_failed'] else 'needs_error_review'
            save(WORK/'status_all.json',value);return
        save(WORK/'status_all.json',value)
        save(HERE/'registry.json',records)
        time.sleep(5)


def activate():
    for path in [WORK/'optimization_audio_v2/RESULT.json',WORK/'optimization_audio_v2/ORIGINAL_CACHE_PARITY.json',WORK/'optimization_cpu_v2/RESULT.json']:
        assert read(path)['passed'],path
    config=capture()
    # Replace only our supervisor. It has no exit hook which kills the workers.
    controller=config['controller'];current=signature(controller['pid'])
    assert current==controller and str(WORK/'run_all.py') in current['argv']
    os.kill(controller['pid'],signal.SIGKILL)
    stream=(HERE/'controller.log').open('a')
    proc=subprocess.Popen(['python',str(Path(__file__).resolve()),'run'],cwd=ROOT,
                           stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    save(HERE/'LAUNCHED.json',dict(pid=proc.pid,old_controller_pid=controller['pid'],at=time.time()))
    print(json.dumps(dict(controller_pid=proc.pid)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['capture','activate','run']);a=ap.parse_args()
    activate() if a.action=='activate' else run() if a.action=='run' else capture()
