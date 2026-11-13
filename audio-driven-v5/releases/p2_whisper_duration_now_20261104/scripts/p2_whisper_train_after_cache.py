"""After verified speaker cache, validate and train Qwen H + Whisper AV on 32 GPUs."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_speaker_whisper_after_annotation import alive_record,process_record,resource_checks,execute


def cache_ready(cache,expected):
    cache=Path(cache)
    if not (cache/'COMPLETE.json').exists():return False
    record=read(cache/'COMPLETE.json');protocol=read(cache/'protocol.json')
    return (record.get('success') is True and record.get('samples')==expected==protocol.get('samples')
        and record.get('audio_source')=='speaker' and record.get('protocol_sha256')==sha(cache/'protocol.json'))


def verify_initial(output,expected_step=2):
    rows=[json.loads(line) for line in (Path(output)/'metrics.jsonl').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r['step']<=expected_step]
    if [r['step'] for r in rows]!=list(range(1,expected_step+1)):
        raise ValueError('Initial training steps incomplete')
    for row in rows:
        if any(not math.isfinite(row[k]) for k in ('av','av_video','av_audio','weighted_total')):
            raise ValueError('Nonfinite initial AV losses')
        if row['weighted_total']!=row['av'] or any(row[k]!=0 for k in ('event','feature_video','feature_audio')):
            raise ValueError('Only AV loss is permitted')
    for group in ('connector','text_lora','whisper'):
        if not any(math.isfinite(r['grad_norm'][group]) and r['grad_norm'][group]>0 for r in rows):
            raise ValueError('Missing gradient '+group)
    probes=[read(p) for p in (Path(output)/'memory_probe').glob('rank*.json')]
    if len(probes)!=32 or {r['rank'] for r in probes}!=set(range(32)):
        raise ValueError('All 32 GPUs must pass the maximum-duration sample check')
    return dict(success=True,steps=expected_step,maximum_duration_s=max(r['duration_s'] for r in probes),
        maximum_probe_reserved_gib=max(r['peak_reserved_gib'] for r in probes),
        mean_initial_step_s=sum(r['seconds'] for r in rows)/len(rows),pure_av=True,world=32)


def run(work):
    work=Path(work);spec=read(work/'queue.json');out=Path(spec['output']);cache=Path(spec['cache'])
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(work/'controller.json',process_record())
        state=read(work/'state.json') if (work/'state.json').exists() else {}
        def status(phase,**kw):
            state.update(phase=phase,heartbeat=time.time(),**kw);atomic_json(work/'state.json',state)
            print(json.dumps(dict(phase=phase,at=time.time(),**kw)),flush=True)
        try:
            while not (work/'STOP').exists():
                if cache_ready(cache,spec['samples']):break
                status('waiting_cache',cache=str(cache),samples=spec['samples']);time.sleep(20)
            else:status('stopped');return
            if sha(spec['config'])!=spec['config_sha256'] or sha(spec['cluster'])!=spec['cluster_sha256']:
                raise ValueError('Queued training configuration changed')
            cluster=read(spec['cluster']);python=cluster['python'];entry=str(ROOT/'scripts/p2_j_whisper_av.py')
            status('preflight')
            execute([python,entry,'preflight','--config',spec['config'],'--processes','32'],work/'preflight.log',work)
            while not (work/'STOP').exists():
                rows=resource_checks(cluster);atomic_json(work/'resources.json',dict(at=time.time(),nodes=rows))
                if all(r['free'] for r in rows):break
                status('waiting_gpu_release');time.sleep(15)
            else:status('stopped');return
            from p2_j_whisper_av import commands
            from listen_cluster import launch
            def distributed(resume=None,stop_after=None,label='train'):
                cancel=threading.Event();finished=threading.Event();began=time.time()
                initial_mtime=(out/'metrics.jsonl').stat().st_mtime if (out/'metrics.jsonl').exists() else 0
                def monitor():
                    while not finished.wait(15):
                        if (work/'STOP').exists() or (out/'STOP').exists():cancel.set();return
                        mtime=(out/'metrics.jsonl').stat().st_mtime if (out/'metrics.jsonl').exists() else 0
                        age=time.time()-(mtime if mtime>initial_mtime else began)
                        if age>(1200 if mtime>initial_mtime else 2400):
                            atomic_json(work/'STALL.json',dict(at=time.time(),stage=label,age_s=age));cancel.set();return
                        s=read(work/'state.json');s['heartbeat']=time.time();atomic_json(work/'state.json',s)
                thread=threading.Thread(target=monitor,daemon=True);thread.start()
                try:launch(cluster,commands(cluster,spec['config'],resume=resume,stop_after=stop_after),
                    work/(label+'_'+str(time.time_ns())),cancel_event=cancel)
                finally:finished.set();thread.join(timeout=2)
            from p2_whisper_av.checkpoint import checked
            resume=None
            if (out/'run.json').exists():
                for marker in sorted(out.glob('step_*.complete.json'),reverse=True):
                    path=marker.with_suffix('').with_suffix('.pt')
                    try:
                        saved=checked(path)
                        if saved['identity']['world']!=32 or saved['identity']['experiment']!=read(spec['config']):continue
                        resume=path;break
                    except (ValueError,FileNotFoundError):continue
                if resume is None:raise ValueError('No complete checkpoint; explicit recovery required')
            if not (out/'GPU_VALIDATION.json').exists():
                status('initial_32gpu_training',validation_steps=2)
                distributed(resume=resume,stop_after=2,label='initial2')
                verify=verify_initial(out,2);path=out/'step_000002.pt';saved=checked(path)
                if saved['step']!=2 or saved['identity']['world']!=32:raise ValueError('Initial checkpoint mismatch')
                atomic_json(out/'GPU_VALIDATION.json',verify);resume=path
            if (work/'STOP').exists() or (out/'STOP').exists():status('stopped');return
            status('training',budget=8927,resume=str(resume))
            distributed(resume=resume,label='continue_epoch')
            final=read(out/'status.json')
            if not final.get('complete') or final.get('step')!=8927:raise ValueError('Training budget incomplete')
            status('complete',step=8927)
        except InterruptedError:status('stopped')
        except Exception as exc:status('retry_pending',error=repr(exc));raise


def guard(work):
    work=Path(work)
    with (work/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        restarts=0
        while not (work/'STOP').exists():
            state=read(work/'state.json') if (work/'state.json').exists() else {}
            if state.get('phase') in ('complete','stopped','needs_attention'):return
            record=read(work/'controller.json') if (work/'controller.json').exists() else {}
            alive=alive_record(record,Path(__file__).resolve(),'run')
            if not alive:
                if restarts>=5:
                    atomic_json(work/'state.json',dict(state,phase='needs_attention',heartbeat=time.time()));return
                with (work/'controller.log').open('a') as stream:
                    subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--work',str(work)],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                restarts+=1
            atomic_json(work/'guard.json',dict(process_record(),controller_alive=alive,restarts=restarts))
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('run','guard'))
    p.add_argument('--work',required=True,type=Path);a=p.parse_args()
    (run if a.command=='run' else guard)(a.work.resolve())
