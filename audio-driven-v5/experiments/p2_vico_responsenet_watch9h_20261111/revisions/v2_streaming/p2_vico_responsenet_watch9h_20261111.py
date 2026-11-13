"""Hourly evidence and bounded recovery of exited evaluation controllers only.

No process discovery scans, no foreign-process signals, no live-worker resets.
The interactive operator reviews stalls/slowdowns from the hourly evidence.
"""
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_testsets_eval.protocol import read,save,sha,PYTHON

WORK=ROOT/'experiments/p2_vico_responsenet_watch9h_20261111'
RN=ROOT/'experiments/p2_responsenet_full586_all_lora6000_6500_20261111'
VICO=ROOT/'experiments/p2_vico43_all_lora6500_sft2752_20261111'
UUID='GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
MODES={
 'vico':dict(work=VICO,script=ROOT/'scripts/p2_vico43_all_lora6500_gpu7_20261111.py',args=['run','--work',str(VICO)],runs=[VICO]),
 'responsenet':dict(work=RN,script=ROOT/'scripts/p2_responsenet_full586_all_lora_20261111.py',args=['controller'],
                    runs=[RN/'conditioning',RN/'step_006000',RN/'step_006500'])}


def identity(pid,needles):
    p=Path('/proc')/str(pid)
    try:
        stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        cmd=p.joinpath('cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
        return dict(pid=pid,live=stat[0]!='Z' and all(x in cmd for x in needles),state=stat[0],start_ticks=stat[19],cmd=cmd)
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,live=False)


def controller(mode):
    m=MODES[mode];work=m['work']
    f=work/'CONTROLLER.json' if mode=='responsenet' and (work/'CONTROLLER.json').exists() else work/'LAUNCH.json'
    return identity(read(f)['pid'],[m['script'].name]) if f.exists() else dict(live=False,pid=None)


def live_children(mode):
    result=[]
    for work in MODES[mode]['runs']:
        for f in (work/'processes').glob('*.json'):
            r=read(f)
            # Recorded exact PIDs/commands only; never enumerate /proc.
            cmd=r.get('command',[])
            needles=[str(x) for x in cmd[1:]] if cmd else [str(ROOT/'scripts'),str(work)]
            i=identity(r['pid'],needles)
            if i['live']:result.append(i)
    return result


def counts():
    result={}
    for label,w,name,dataset in [('vico6500',VICO,'all_lora6500_sft2752','vico'),
        ('responsenet6000',RN/'step_006000','all_lora6000_sft2752','responsenet'),
        ('responsenet6500',RN/'step_006500','all_lora6500_sft2752','responsenet')]:
        z=list((w/'latents'/name/dataset).glob('*/metadata.json'))
        result[label]=dict(latents=len(z),videos=len(list((w/'generated'/name/dataset).glob('*/complete.json'))),
            last_latent_at=max((f.stat().st_mtime for f in z),default=None),complete=(w/'COMPLETE.json').exists())
        base=ROOT/'experiments/p2_vico43_all_lora6000_sft2752_20261111' if dataset=='vico' else RN/'step_006000'
        baseline=base/'latents/all_lora6000_sft2752'/dataset
        ratios=[]
        for f in sorted(z,key=lambda f:f.stat().st_mtime)[-20:]:
            ref=baseline/f.parent.name/'metadata.json'
            if label=='responsenet6000' or not ref.exists():continue
            m=read(f);b=read(ref)
            if m['geometry']==b['geometry']:ratios.append(m['seconds']/b['seconds'])
        result[label]['same_sample_runtime_ratios']=ratios
        result[label]['same_sample_ratio_median']=statistics.median(ratios) if ratios else None
        result[label]['sustained_slowdown']=len(ratios)>=20 and all(statistics.median(v)>1.5 for v in (ratios[:10],ratios[-10:])) and len(z)<(43 if dataset=='vico' else 586)
    result['conditions']=dict(qwen=len(list((RN/'conditioning/conditions/responsenet').glob('*/qwen/complete.json'))),
        whisper=len(list((RN/'conditioning/whisper/responsenet').glob('*/complete.json'))))
    return result


def recover(mode,config,state):
    m=MODES[mode];work=m['work']
    if time.time()>=config['end_at'] or (WORK/'STOP').exists() or any((w/'STOP').exists() for w in [work,*m['runs']]):return
    if (work/'COMPLETE.json').exists() or controller(mode)['live'] or live_children(mode):return
    previous=state.setdefault('repairs',{}).get(mode,[])
    recent=[x for x in previous if time.time()-x['at']<3600]
    if len(recent)>=3 or (recent and time.time()-recent[-1]['at']<300):return
    # Verify frozen execution files before any recovery; code defects need review.
    for w in m['runs']:
        if not (w/'protocol.json').exists():continue
        for f,digest in read(w/'protocol.json').get('execution_code',{}).items():
            if sha(f)!=digest:raise ValueError('Changed evaluation source, repair requires review: '+f)
    with (work/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
    if controller(mode)['live'] or live_children(mode):return
    command=[str(PYTHON),'-u',str(m['script']),*m['args']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    log=work/f'watch_recovery_{time.time_ns()}.log'
    with log.open('ab',buffering=0) as stream:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    action=dict(pid=p.pid,at=time.time(),mode=mode,command=command,log=str(log),physical_gpus=[7],reason='controller_exited_and_no_recorded_live_children')
    save(work/'LAUNCH.json',action);previous.append(action);state['repairs'][mode]=previous
    save(WORK/'repairs'/f'{time.time_ns()}.json',action)


def main():
    WORK.mkdir(exist_ok=True)
    with (WORK/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=read(WORK/'CONFIG.json');assert c['end_at']-c['start_at']==9*3600
        assert c['physical_gpus']==[7] and c['script_sha256']==sha(__file__)
        save(WORK/'CONTROLLER.json',dict(pid=os.getpid(),at=time.time(),end_at=c['end_at']))
        state=read(WORK/'STATE.json') if (WORK/'STATE.json').exists() else {}
        while time.time()<c['end_at'] and not (WORK/'STOP').exists():
            try:
                for mode in MODES:recover(mode,c,state)
                hour=min(9,int((time.time()-c['start_at'])//3600))
                if hour not in state.get('audited_hours',[]):
                    report=dict(at=time.time(),hour=hour,counts=counts(),controllers={m:controller(m) for m in MODES},
                        status={m:read(v['work']/'status.json') for m,v in MODES.items()})
                    raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.used,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=20)
                    assert raw.split(',')[0].strip()==UUID
                    report['gpu_csv']=raw.strip()
                    save(WORK/'hourly'/f'{hour:02d}.json',report)
                    state.setdefault('audited_hours',[]).append(hour)
                state.update(at=time.time(),end_at=c['end_at']);save(WORK/'STATE.json',state)
            except Exception as exc:
                save(WORK/'errors'/f'{time.time_ns()}.json',dict(at=time.time(),error=repr(exc),traceback=traceback.format_exc()))
            time.sleep(min(60,max(0,c['end_at']-time.time())))
        save(WORK/'WINDOW_FINISHED.json',dict(at=time.time(),deadline=c['end_at'],stopped=(WORK/'STOP').exists(),
            evaluations_left_running=True,counts=counts()))


if __name__=='__main__':main()
