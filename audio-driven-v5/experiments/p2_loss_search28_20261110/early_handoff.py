"""User-requested immediate handoff after5800, preserving the actual saved step."""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from p2_listen.contracts import read, atomic_json
from p2_loss_search28.config import WORK, source_identity
from p2_loss_search28 import resources as resource_checks
original_idle = resource_checks.idle


def idle(cluster):
    # Only retry read-only resource observation. Never restart a worker based
    # on an SSH observation failure; launches keep their own process handles.
    for attempt in range(8):
        try:
            return original_idle(cluster)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            if isinstance(exc, subprocess.CalledProcessError) and exc.returncode != 255:
                raise
            atomic_json(WORK / 'RESOURCE_OBSERVATION_RETRY.json',
                        dict(at=time.time(), attempt=attempt + 1, error=str(exc), readonly=True))
            if attempt == 7: raise
            time.sleep(3)


resource_checks.idle = idle

OLD = ROOT / 'experiments/p2_ltx_all_lora_noevent28_batch28_20261109'


def main():
    request = read(WORK / 'EARLY_HANDOFF_REQUEST.json')
    if request['requested_save_step'] != 5800 or request['observed_step'] < 5800:
        raise ValueError('Unexpected early-handoff request')
    if not (OLD / 'STOP').exists():
        raise ValueError('Old training must have a user-requested STOP')
    if read(WORK / 'PREFLIGHT_ACCEPTED.json')['source'] != source_identity():
        raise ValueError('Training code changed since preflight')
    if not read(WORK / 'TESTS_ACCEPTED.json')['success']:
        raise ValueError('Search tests not accepted')
    with (WORK / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        atomic_json(WORK / 'EARLY_HANDOFF_CONTROLLER.json',
                    dict(pid=os.getpid(), start_ticks=ticks, at=time.time()))
        cluster = read(OLD / 'cluster.json')
        while not (WORK / 'STOP').exists():
            step = read(OLD / 'status.json')['step']
            if step < request['observed_step']:
                raise ValueError('Predecessor step regressed')
            marker = OLD / f'step_{step:06d}.complete.json'
            atomic_json(WORK / 'HANDOFF_STATE.json',
                        dict(stage='waiting_for_early_save_and_release', requested_step=5800,
                             actual_step=step, marker_exists=marker.exists(), at=time.time()))
            if marker.exists():
                resources = idle(cluster)
                if resources['idle']:
                    from p2_whisper_av.bootstrap import activate_ltx
                    activate_ltx()
                    from p2_ltx_all_lora_noevent.checkpoint import checked
                    saved = checked(marker.with_suffix('').with_suffix('.pt'))
                    if saved['step'] != step or len(saved['rank_rng']) != 28:
                        raise ValueError('Invalid predecessor checkpoint step/RNG')
                    if saved['identity']['experiment'] != read(OLD / 'config.json'):
                        raise ValueError('Predecessor checkpoint config mismatch')
                    del saved
                    atomic_json(WORK / 'HANDOFF_ACCEPTED.json',
                        dict(success=True, step=step, requested_step=5800,
                             source=str(OLD / f'step_{step:06d}.pt'), world=28,
                             resources=resources, at=time.time(), early_handoff=True))
                    from p2_loss_search28.queue import run
                    run()
                    return
            time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(WORK / 'EARLY_HANDOFF_FAILURE.json',
                    dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
