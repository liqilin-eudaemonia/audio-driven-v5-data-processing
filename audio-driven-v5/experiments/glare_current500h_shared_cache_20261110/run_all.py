"""Own eight workers only:4visual/shared-decode +4frozen-audio on GPU4-7."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_glare_shared_cache as pipeline
import glare_frozen_audio_cache as audio
save,read=pipeline.save,pipeline.read


def main():
    lock=(WORK/'run_all.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=pipeline.load_protocol()
    for name in ['SHARED_DECODE_ACCEPTED.json','CPU_TESTS.json','CANARY_ACCEPTED.json','RESUME_ACCEPTED.json','PRODUCTION_INPUT_REPLAY.json']:
        assert read(WORK/name)['passed'],name
    for name in ['CANARY_ACCEPTED.json','RESUME_ACCEPTED.json']:
        assert read(audio.WORK/name)['passed'],name
    assert read(pipeline.PARENT/'status.json')['stage']=='stopped','Old four workers must finish before replacement'
    stopping=[False];signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    children,attempts,streams={}, {}, []
    keys=[(kind,gpu) for kind in ['visual','audio'] for gpu in [4,5,6,7]]
    def launch(key):
        kind,gpu=key
        script=ROOT/'scripts'/('run_glare_shared_cache.py' if kind=='visual' else 'glare_frozen_audio_cache.py')
        folder=WORK/'workers' if kind=='visual' else audio.WORK/'workers'
        folder.mkdir(parents=True,exist_ok=True)
        stream=(folder/('gpu%d.log'%gpu)).open('a');streams.append(stream)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=':'.join(str(x) for x in [
                pipeline.PARENT/'runtime_deps',ROOT/'compare/.envs/glare-deps',ROOT/'src',ROOT/'scripts']))
        children[key]=subprocess.Popen([str(pipeline.GPU_PY),str(script),'worker','--gpu',str(gpu)],
            cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        attempts[key]=attempts.get(key,0)+1
    for key in keys:launch(key)
    started=time.time()
    while True:
        groups={'visual':{},'audio':{}}
        for key,child in list(children.items()):
            kind,gpu=key
            folder=WORK/'workers' if kind=='visual' else audio.WORK/'workers'
            path=folder/('gpu%d.json'%gpu)
            state=read(path) if path.exists() else {}
            if state.get('pid')!=child.pid:state={'stage':'initializing'}
            groups[kind][str(gpu)]=dict(state,process_pid=child.pid,returncode=child.poll(),attempt=attempts[key])
            if child.poll() is not None and state.get('stage')!='finished' and attempts[key]<3 and not (WORK/'STOP').exists() and not stopping[0]:
                launch(key)
        value=dict(stage='running',controller_pid=os.getpid(),samples=p['samples'],gpus=[4,5,6,7],
            worker_groups=groups,started=started,heartbeat=time.time(),prosody='deferred_by_user',
            full_glare_training_admitted=False)
        for kind in groups:
            value[kind+'_completed']=sum(s.get('completed',0)+s.get('reused',0) for s in groups[kind].values())
            value[kind+'_failed']=sum(s.get('failed',0) for s in groups[kind].values())
        if stopping[0] or (WORK/'STOP').exists():
            for child in children.values():
                if child.poll() is None:child.terminate()
            value['stage']='stopping'
        if all(c.poll() is not None for c in children.values()):
            value['stage']='nonprosody_complete' if all(value[k+'_completed']==p['samples'] and not value[k+'_failed'] for k in groups) else 'needs_error_review'
        save(WORK/'status_all.json',value)
        if value['stage'] in ['nonprosody_complete','needs_error_review']:
            save(WORK/'NONPROSODY_STAGE_RESULT.json',value)
            return
        time.sleep(10)


if __name__=='__main__':main()
