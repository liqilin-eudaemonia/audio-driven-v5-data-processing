"""Checkpoint, benchmark on 32 GPUs, then automatically promote or safely resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,sha,atomic_json
from p2_speaker_whisper_after_annotation import alive_record,process_record,resource_checks

WORK=ROOT/'experiments/p2_j_whisper_execution_tune_20261104'
PARENT=ROOT/'experiments/p2_j_qwen_whisper_av_duration_now_20261104'
QUEUE_SCRIPT=ROOT/'scripts/p2_whisper_train_after_cache.py'
HELPER=ROOT/'src/p2_whisper_av/migration.py'


def launch_guard(queue):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (queue/'guard.log').open('a') as stream:
        subprocess.Popen(['/private/software/conda/envs/vllm/bin/python',str(QUEUE_SCRIPT),'guard','--work',str(queue)],cwd=ROOT,
            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def restore_parent(helper_sha):
    if sha(HELPER)!=helper_sha:raise ValueError('Administrative helper changed concurrently; cannot overwrite it')
    if sha(WORK/'migration_before.py')!=read(PARENT/'run.json')['source'][str(HELPER)]:
        raise ValueError('Rollback helper does not match the original training identity')
    HELPER.write_bytes((WORK/'migration_before.py').read_bytes())
    for file in (PARENT/'STOP',PARENT/'queue/STOP'):file.unlink(missing_ok=True)
    atomic_json(PARENT/'queue/state.json',dict(phase='resume_after_performance_probe',at=time.time()))
    launch_guard(PARENT/'queue')


def promote(path):
    import torch
    torch.set_num_threads(2)
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from p2_whisper_av.checkpoint import checked
    from p2_whisper_av.config import validate,validate_baseline,source_identity
    from p2_whisper_av.execution import settings
    from p2_whisper_av.data import data_identity
    from p2_whisper_av.migration import validate_transition,validate_sources
    from p2_listen.cache import atomic_torch
    from p2_j_duration_execution import Schedule
    from p2_j_duration_benchmark import equal_tensors
    report=read(WORK/'RESULT.json');selected=report['selected'];saved=checked(path)
    if not report['adopt'] or saved['identity']!=read(PARENT/'run.json'):raise ValueError('Not a valid tuning handoff')
    base=saved['identity']['experiment'];out=ROOT/'experiments/p2_j_qwen_whisper_av_tuned_20261104';out.mkdir(exist_ok=True)
    if (out/'run.json').exists():raise ValueError('Tuned run already exists; do not overwrite')
    c=copy.deepcopy(base);c['output']=str(out);c['training']['gradient_checkpointing']=selected['policy']
    c['whisper_training_execution']['gradient_bucket_mb']=selected['bucket_mb']
    marker=read(path.with_suffix('.complete.json'))
    c['whisper_execution_migration']=dict(kind='execution-tune-v1',parent_run=str(PARENT),parent_checkpoint=str(path),
        parent_sha256=marker['sha256'],prefix_steps=saved['step'],benchmark_report=str(WORK/'RESULT.json'),benchmark_report_sha256=sha(WORK/'RESULT.json'))
    validate_transition(base,c,saved['step']);validate(c);validate_baseline(c,read(c['baseline_config']))
    if Schedule(c).pairs!=Schedule(base).pairs or data_identity(c)!=saved['identity']['data_identity']:
        raise ValueError('Tuning changed data or sample/noise order')
    sources=source_identity();changes=validate_sources(saved['identity']['source'],sources,ROOT)
    identity=copy.deepcopy(saved['identity']);identity.update(experiment=c,source=sources,execution=settings(c))
    target=out/path.name;atomic_torch(target,dict(saved,identity=identity))
    atomic_json(target.with_suffix('.complete.json'),dict(step=saved['step'],sha256=sha(target)))
    verify=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
    if any(not equal_tensors(saved[k],verify[k]) for k in ('model','optimizer','rank_rng')):
        raise ValueError('Tuning changed model/optimizer/RNG checkpoint tensors')
    config=out/'config.json';atomic_json(config,c,immutable=True);atomic_json(out/'run.json',identity,immutable=True)
    (out/'metrics.jsonl').write_text(''.join(line+'\n' for line in (PARENT/'metrics.jsonl').read_text().splitlines()
        if json.loads(line)['step']<=saved['step']))
    atomic_json(out/'GPU_VALIDATION.json',dict(read(PARENT/'GPU_VALIDATION.json'),performance_probe=str(WORK/'RESULT.json')))
    atomic_json(out/'status.json',dict(step=saved['step'],budget=8927,complete=False))
    atomic_json(out/'migration.json',dict(source=str(path),source_sha256=marker['sha256'],step=saved['step'],source_changes=changes,
        model_optimizer_rng_bitwise_unchanged=True,sample_schedule_identical=True,data_identity_identical=True,
        benchmark_report=str(WORK/'RESULT.json'),benchmark_report_sha256=sha(WORK/'RESULT.json')),immutable=True)
    queue=out/'queue';queue.mkdir(exist_ok=True);spec=read(PARENT/'queue/queue.json')
    spec.update(output=str(out),config=str(config),config_sha256=sha(config));atomic_json(queue/'queue.json',spec,immutable=True)
    atomic_json(queue/'state.json',dict(phase='prepared_execution_tuning',resume_step=saved['step']))
    for parent in (PARENT,ROOT/'experiments/p2_j_qwen_whisper_av_20261104'):
        atomic_json(parent/'CURRENT_RUN.json',dict(output=str(out),config=str(config),resume_step=saved['step']))
    launch_guard(queue)
    return out


def run():
    from p2_j_whisper_av import commands
    from listen_cluster import launch
    WORK.mkdir(exist_ok=True);atomic_json(WORK/'driver.json',process_record());helper_sha=sha(HELPER)
    def status(stage,**kw):atomic_json(WORK/'state.json',dict(stage=stage,at=time.time(),**kw));print(stage,flush=True)
    record=read(PARENT/'queue/controller.json');paused=False;was_stopped=False;promoted=False
    try:
        alive=alive_record(record,QUEUE_SCRIPT,'run')
        if not alive and not (PARENT/'queue/STOP').exists():raise RuntimeError('Parent controller not alive or safely stopped')
        before=read(PARENT/'status.json')['step']
        if alive:os.kill(record['pid'],signal.SIGSTOP);paused=True
        (PARENT/'queue/STOP').write_text('Authorized matched performance comparison.\n');(PARENT/'STOP').write_text('Save current complete step for performance comparison.\n');was_stopped=True
        status('saving_current_formal_checkpoint',at_least_step=before);began=time.time()
        while True:
            step=read(PARENT/'status.json')['step'];path=PARENT/f'step_{step:06d}.pt'
            if step>=before and path.with_suffix('.complete.json').exists():break
            if time.time()-began>600:raise TimeoutError('Graceful checkpoint exceeded 10 minutes')
            time.sleep(3)
        time.sleep(3)
        if paused:os.kill(record['pid'],signal.SIGCONT);paused=False
        cluster=read(ROOT/'configs/p2_j_qwen_whisper_av_train32_cluster_20261104.json');began=time.time()
        while True:
            try:free=all(x['free'] for x in resource_checks(cluster))
            except (OSError,subprocess.SubprocessError):free=False
            if free:break
            if time.time()-began>240:raise TimeoutError('Own workers did not release GPUs')
            time.sleep(5)
        status('matched_32gpu_benchmark',parent_step=step)
        cluster['master_port']=29847
        cmds=commands(cluster,PARENT/'config.json',resume=path)
        for cmd in cmds:
            cmd[cmd.index(str(ROOT/'scripts/p2_j_whisper_av.py'))]=str(ROOT/'scripts/p2_whisper_execution_benchmark.py')
            cmd.extend(['--work',str(WORK)])
        atomic_json(WORK/'commands.json',dict(cluster=cluster,commands=cmds))
        cancel=threading.Event();done=threading.Event();started=time.time()
        def watch():
            while not done.wait(15):
                progress=WORK/'progress.json';age=time.time()-(progress.stat().st_mtime if progress.exists() else started)
                if (WORK/'STOP').exists() or age>(300 if progress.exists() else 900):cancel.set();return
        threading.Thread(target=watch,daemon=True).start()
        logs=WORK/'attempts'/str(time.time_ns());atomic_json(WORK/'CURRENT_ATTEMPT.json',dict(logs=str(logs)))
        try:launch(cluster,cmds,logs,cancel_event=cancel)
        finally:done.set()
        report=read(WORK/'RESULT.json')
        if report['adopt']:
            status('promoting_verified_execution',selected=report['selected']['name']);out=promote(path);promoted=True
            status('training_queued',output=str(out),parent_step=step,selected=report['selected']['name'])
        else:
            restore_parent(helper_sha);was_stopped=False;status('original_training_resumed',reason='No safe exact candidate at least5percent faster')
    except Exception as exc:
        status('failed',error=repr(exc));raise
    finally:
        if paused and alive_record(record,QUEUE_SCRIPT,'run'):os.kill(record['pid'],signal.SIGCONT)
        if was_stopped and not promoted:
            restore_parent(helper_sha);status('fallback_original_training_resumed')


if __name__=='__main__':
    import fcntl
    WORK.mkdir(exist_ok=True)
    with (WORK/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/'RESULT.json').exists():
            raise SystemExit('Benchmark already complete; do not rerun or stop another training run. Use the recorded result.')
        run()
