#!/usr/bin/env python3
"""Freeze and rerun seven real-data trainers, then restart each from disk."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from baseline_training.data import WORK, read, save, sha

METHODS = ['glare', 'dystream', 'avatarforcing', 'l2l', 'dim', 'vico', 'listenformer']


def freeze(work):
    dest = work / 'code_snapshot'
    assert not dest.exists(), 'Never overwrite accepted frozen sources; use a new work directory'
    shutil.copytree(ROOT / 'src/baseline_training', dest / 'src/baseline_training',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (dest / 'scripts').mkdir(parents=True)
    shutil.copyfile(ROOT / 'scripts/p2_baseline_train.py', dest / 'scripts/p2_baseline_train.py')
    save(work / 'CODE_FROZEN.json', dict(project_root=str(ROOT),
         files={str(p.relative_to(work)): sha(p) for p in dest.rglob('*.py')}))


def environment(method):
    env = dict(os.environ, BASELINE_PROJECT_ROOT=str(ROOT), CUDA_VISIBLE_DEVICES='7',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if method == 'glare':
        python, overlay = 'avatarforcing', 'glare-deps'
    elif method in ['vico', 'listenformer']:
        python, overlay = 'dim', 'listenformer-deps'
    else:
        python, overlay = method, None
    paths = [str(ROOT / 'compare/.envs' / overlay)] if overlay else []
    env['PYTHONPATH'] = os.pathsep.join(paths + [str(ROOT / 'src'), str(ROOT / 'scripts')])
    return ROOT / 'compare/.envs' / python / 'bin/python', env


def run(work, methods, steps):
    frozen = read(work / 'CODE_FROZEN.json')
    assert all(sha(work / p) == h for p, h in frozen['files'].items())
    script = work / 'code_snapshot/scripts/p2_baseline_train.py'
    reports = {}
    for method in methods:
        python, env = environment(method)
        status = dict(method=method, stage='native_updates_and_replay', passed_methods=list(reports), started=time.time())
        save(work / 'RUN_STATUS.json', status)
        command = [str(python), '-u', str(script), 'validate', '--method', method, '--work', str(work), '--steps', str(steps)]
        print('START', method, flush=True)
        log = work / ('frozen_' + method + '.log')
        with log.open('w') as f:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f'{method} failed; read {log}'
        accepted = read(work / 'runs' / method / 'ACCEPTED.json')
        stages = accepted.get('stages', [accepted])
        assert all(x['passed'] and all(x['resume_replay'].values()) for x in stages)
        last = stages[-1]
        # A brand-new Python process reconstructs the original full network and
        # frozen encoders, loads step3 and reproduces step4 from disk.
        status['stage'] = 'new_process_resume'
        save(work / 'RUN_STATUS.json', status)
        command[3] = 'train'
        command.extend(['--resume', last['replay_checkpoint']])
        log = work / ('cold_resume_' + method + '.log')
        with log.open('w') as f:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f'{method} cold resume failed; read {log}'
        resumed = read(work / 'runs' / method / 'ACCEPTED.json')
        actual = resumed.get('stages', [resumed])[-1]
        keys = ['final_parameters_sha256', 'final_optimizer_sha256'] if method in ['glare', 'vico', 'listenformer'] else ['final_mutable_state_sha256']
        checks = {k: last[k] == actual[k] for k in keys}
        assert all(checks.values()), (method, checks)
        uninterrupted_sha = last['checkpoint_sha256']
        # torch.save container bytes can differ after reconstruction even when
        # every model/optimizer tensor is exact. Bind the delivered file hash
        # to the final file on disk, retaining the uninterrupted-run hash too.
        last['uninterrupted_checkpoint_sha256'] = uninterrupted_sha
        last['checkpoint_sha256'] = actual['checkpoint_sha256']
        save(work / 'runs' / method / 'COLD_RESUME.json', dict(passed=True, checks=checks,
             replay_checkpoint=last['replay_checkpoint'], new_python_process=True, total_step=steps,
             uninterrupted_checkpoint_sha256=uninterrupted_sha, resumed_checkpoint_sha256=actual['checkpoint_sha256']))
        # Preserve complete acceptance evidence rather than the one-step resume
        # report replacing the original multi-stage / replay record.
        save(work / 'runs' / method / 'ACCEPTED.json', accepted)
        reports[method] = dict(accepted=accepted, cold_resume=checks)
        print('PASSED', method, 'native replay and new-process resume', flush=True)
        save(work / 'SUITE_PROGRESS.json', reports)
    save(work / 'SUITE_ACCEPTED.json', dict(passed=True, methods=reports, steps_per_stage=steps,
         synthetic_input=False, full500h_training_completed=False, source_freeze=sha(work / 'CODE_FROZEN.json')))
    save(work / 'RUN_STATUS.json', dict(stage='complete', passed_methods=list(reports), completed=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'run'])
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--steps', type=int, default=4)
    a = p.parse_args()
    work = a.work.resolve()
    freeze(work) if a.stage == 'freeze' else run(work, a.methods, a.steps)
