"""Run Event+Whisper then Feature+Whisper on the same 31 authorized GPUs.

Waits for existing tasks; never evicts them or uses node8 GPU7. Source files,
cache identities and each arm's own full checkpoints stay independently pinned.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from p2_listen.contracts import atomic_json, read, sha

WORK = ROOT/'experiments/p2_whisper_single_aux31_20261106'


def verify_row(row, arm):
    for key in ('av', 'event', 'feature_video', 'feature_audio', 'weighted_total', 'seconds'):
        if not math.isfinite(row[key]):
            raise ValueError('Nonfinite training field: '+key)
    expected = row['av']+.05*row['event']+.025*(row['feature_video']+row['feature_audio'])
    if abs(expected-row['weighted_total']) > 1e-8:
        raise ValueError('Objective coefficients differ')
    disabled = ('feature_video', 'feature_audio') if arm == 'event' else ('event',)
    if any(row[k] != 0 for k in disabled):
        raise ValueError('Disabled objective is active')
    groups = {'connector', 'text_lora', 'whisper', 'events' if arm == 'event' else 'features'}
    if set(row['grad_norm']) != groups | {'main'}:
        raise ValueError('Unexpected parameter groups')
    if any(not math.isfinite(row['grad_norm'][k]) or row['grad_norm'][k] <= 0 for k in groups):
        raise ValueError('Missing/nonfinite gradient')


def recent_rows(path):
    if not path.exists():
        return []
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-1000000))
        result = []
        for line in f.read().splitlines()[-120:]:
            try:
                result.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                pass  # Ignore only a concurrent partial line.
    return result


def resume_point(out):
    markers = sorted(out.glob('step_*.complete.json'))
    if not markers:
        if (out/'run.json').exists():
            raise RuntimeError('Registered run has no complete initial checkpoint; inspect startup before restarting')
        return None
    marker = markers[-1]
    path = marker.with_suffix('').with_suffix('.pt')
    value = read(marker)
    if value['step'] != int(path.stem.split('_')[-1]) or sha(path) != value['sha256']:
        raise ValueError('Latest complete checkpoint failed checksum/step check')
    return path


def final_check(out, c):
    status = read(out/'status.json')
    if not status.get('complete') or status['step'] != c['training']['steps']:
        return False
    from p2_whisper_ablation.checkpoint import checked
    path = out/f"step_{status['step']:06d}.pt"
    value = checked(path)
    if value['identity'] != read(out/'run.json') or len(value['rank_rng']) != 31 or not value['optimizer']['state']:
        raise ValueError('Final checkpoint identity/optimizer/RNG incomplete')
    atomic_json(out/'TRAINING_COMPLETE.json', dict(success=True,step=value['step'],
        checkpoint=str(path),sha256=read(path.with_suffix('.complete.json'))['sha256'],
        model_finite=True,optimizer_present=True,rank_rng_count=31,at=time.time()))
    return True


def run(work=WORK):
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_ablation import config
    from p2_whisper_ablation.data import data_identity
    from p2_seamless200_whisper_eval import idle_snapshot
    # The CLI module and source package share a name; load its file explicitly.
    import importlib.util
    spec=importlib.util.spec_from_file_location('ablation_cli',ROOT/'scripts/p2_whisper_ablation.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    from listen_cluster import launch
    work.mkdir(parents=True,exist_ok=True)
    with (work/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from p2_speaker_whisper_after_annotation import process_record
        atomic_json(work/'controller.json',process_record())
        def state(stage,**kw):
            value=dict(stage=stage,at=time.time(),**kw)
            atomic_json(work/'state.json',value)
            with (work/'states.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
            print(json.dumps(value),flush=True)
        for arm in ('event','feature'):
            c=config.load(cli.config_path(arm));out=Path(c['output']);cluster=read(out/'cluster.json')
            atomic_json(out/'config.json',c,immutable=True)
            config.validate_baseline(c,read(c['baseline_config']))
            if (out/'status.json').exists() and final_check(out,c):
                state('arm_already_complete',arm=arm);continue
            for attempt in range(1,6):
                while True:
                    if (work/'STOP').exists() or (out/'STOP').exists():
                        state('stopped',arm=arm);return
                    snapshot=idle_snapshot(cluster);atomic_json(work/'resources.json',snapshot)
                    if snapshot['all_idle']:break
                    state('waiting_selected31_gpus',arm=arm);time.sleep(15)
                proof=read(out/'preflight.json')
                if proof['source']!=config.source_identity() or proof['data_identity']!=data_identity(c):
                    raise ValueError('Pinned source/cache proof changed')
                resume=resume_point(out)
                logs=out/'queue'/f'attempt_{attempt}_{time.time_ns()}'
                cancel=threading.Event();done=threading.Event();started=time.time()
                state('training31',arm=arm,attempt=attempt,logs=str(logs),resume=str(resume) if resume else None)
                def monitor():
                    resource_at=0
                    while not done.wait(20):
                        try:
                            if (work/'STOP').exists():(out/'STOP').touch()
                            path=out/'metrics.jsonl'
                            stamp=max(started,path.stat().st_mtime) if path.exists() else started
                            if time.time()-stamp>(900 if path.exists() else 1800):
                                raise TimeoutError('No training progress within startup/step timeout')
                            rows=recent_rows(path)
                            if rows and path.stat().st_mtime>=started:
                                row=rows[-1];verify_row(row,arm)
                                seconds=[r['seconds'] for r in rows if r['step']>=5]
                                speed=dict(arm=arm,step=row['step'],steps=8000,
                                    window=len(seconds),mean_step_s=statistics.mean(seconds) if seconds else None,
                                    input_wait_s=row['input_wait_seconds'],peak_reserved_gib=row['peak_reserved_gib'],at=time.time())
                                if seconds:speed['remaining_hours']=(8000-row['step'])*speed['mean_step_s']/3600
                                atomic_json(work/'speed.json',speed)
                                if row['step']>=5 and not (out/'STARTUP_ACCEPTANCE.json').exists():
                                    atomic_json(out/'STARTUP_ACCEPTANCE.json',dict(success=True,world=31,arm=arm,
                                        four_groups_receive_gradients=True,disabled_loss_zero=True,objective_correct=True,
                                        reserved_gpu='node8:7',sample=row,at=time.time()))
                            if time.time()-resource_at>120:
                                # SSH observation errors never cancel healthy distributed workers.
                                try:atomic_json(work/'resources.json',idle_snapshot(cluster))
                                except Exception as exc:atomic_json(work/'RESOURCE_OBSERVATION_ERROR.json',dict(error=repr(exc),at=time.time()))
                                resource_at=time.time()
                        except Exception as exc:
                            atomic_json(work/'WATCH_ERROR.json',dict(arm=arm,error=repr(exc),at=time.time()))
                            cancel.set();return
                watcher=threading.Thread(target=monitor,daemon=True)
                try:
                    with (work/'launch.lock').open('a') as launch_lock:
                        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        watcher.start()
                        launch(cluster,cli.commands(c,resume,logs/'ranks',cli.config_path(arm)),logs,cancel_event=cancel)
                except Exception as exc:
                    state('retry_pending',arm=arm,attempt=attempt,error=repr(exc))
                    if attempt==5:raise
                    time.sleep(15);continue
                finally:
                    done.set()
                    if watcher.ident is not None:watcher.join(timeout=45)
                if final_check(out,c):
                    state('arm_complete',arm=arm,step=8000);break
                if (out/'STOP').exists() or (work/'STOP').exists():
                    state('stopped',arm=arm);return
                raise RuntimeError('Workers exited without full training completion')
        state('complete',arms=['event','feature'],steps_each=8000)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=WORK)
    a=p.parse_args()
    try:run(a.work)
    except Exception as exc:
        a.work.mkdir(parents=True,exist_ok=True)
        atomic_json(a.work/'FAILED.json',dict(error=repr(exc),at=time.time()))
        raise
