"""Train two interaction candidates, evaluate both seeds, compare four cells."""
from pathlib import Path
import argparse
import copy
import fcntl
import importlib.util
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extension import WORK, ROOT, ORIGINAL, LEGACY, CONTROL, CANDIDATES
from extension import activate, make, cluster_for, source_identity
from p2_listen.contracts import read, sha, atomic_json as save
from p2_testsets_eval.protocol import PYTHON, partition

NAMES = ('event020', 'event100', *CANDIDATES)
SEEDS = (137, 238)


def folder(name, seed):
    return WORK / 'evaluations' / f'{name}_500_seed{seed}'


def original_driver():
    spec = importlib.util.spec_from_file_location('completed_event4_driver', CONTROL / 'driver.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seal_score(work):
    original_driver().seal_score(work)


def report():
    module = original_driver()
    module.NAMES = NAMES
    module.folder = folder
    module.WORK = WORK
    module.report()


def prepare():
    from p2_combined_whisper_eval import link
    activate()
    protocol = dict(candidates={n: list(v) for n, v in CANDIDATES.items()},
                    stages=[500], world=28, global_batch=28,
                    controls=['event020', 'event100'], seeds=list(SEEDS), samples=64,
                    original_schedule_sha256=sha(ORIGINAL / 'schedule.json'),
                    initialization='same_pretrained_and_seed_as_original_search')
    save(WORK / 'protocol.json', protocol, immutable=True)
    link(ORIGINAL / 'runs/baseline', WORK / 'runs/baseline')
    for name in CANDIDATES:
        out = WORK / 'runs' / name
        (out / 'rank_logs').mkdir(parents=True, exist_ok=True)
        save(out / 'config.json', make(name), immutable=True)
        save(out / 'cluster.json', cluster_for(name), immutable=True)
    for name in NAMES[:2]:
        for seed in SEEDS:
            link(CONTROL / 'evaluations' / f'{name}_500_seed{seed}', folder(name, seed))
    save(WORK / 'PREPARED.json', dict(success=True, at=time.time(), source=source_identity()))


def prepare_evaluation():
    from p2_combined_whisper_eval import link
    import eval_runtime as runtime
    activate()
    cluster = cluster_for(next(iter(CANDIDATES)))
    for name in CANDIDATES:
        first = folder(name, 137)
        first.mkdir(parents=True, exist_ok=True)
        checkpoint = WORK / 'runs' / name / 'step_000500.pt'
        save(first / 'REQUEST.json', dict(checkpoint=str(checkpoint), samples=64), immutable=True)
        protocol = runtime.prepare(first)
        for seed in SEEDS:
            out = folder(name, seed)
            out.mkdir(parents=True, exist_ok=True)
            if seed != 137:
                for directory in ('export', 'conditions', 'whisper', 'negative'):
                    link(first / directory, out / directory)
                for filename in ('REQUEST.json', 'DATA_CONFIG.json', 'shared_items.json'):
                    save(out / filename, read(first / filename), immutable=True)
            config = copy.deepcopy(protocol)
            config.pop('scoring_device', None)
            config.update(seed=seed, workers=28, cluster=str(out / 'cluster.json'),
                          metrics_cluster=str(out / 'cluster.json'),
                          worker_cli=str(WORK / 'evaluate28.py'),
                          shared_items_manifest=str(out / 'shared_items.json'))
            save(out / 'cluster.json', cluster)
            for rank, items in enumerate(partition(read(first / 'shared_items.json'), 28)):
                save(out / 'shards' / f'{rank:02d}.json', items)
            save(out / 'protocol.json', config)
    sources = source_identity()
    sources.update(runtime.code_identity())
    save(WORK / 'PLAN.json', dict(source=sources, cluster=cluster, candidates=list(NAMES),
                                 seeds=list(SEEDS), samples=64, outputs=512,
                                 reused_outputs=256, new_outputs=256, at=time.time()))
    save(WORK / 'ROLLING_PLAN.json', dict(source=sources, at=time.time()))


def run():
    activate()
    from p2_loss_search28 import queue
    from p2_loss_search28.resources import idle
    from stage_driver import command
    with (WORK / 'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        save(WORK / 'EXPERIMENT_CONTROLLER.json', dict(pid=os.getpid(), start_ticks=ticks, at=time.time()))
        # Use the original allocation lock throughout this new experiment.
        with (ORIGINAL / 'queue.lock').open('a') as allocation:
            fcntl.flock(allocation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue.run()
            if (WORK / 'STOP').exists():
                return
            prepare_evaluation()
            snapshot = idle(cluster_for(next(iter(CANDIDATES))))
            if not snapshot['idle']:
                raise RuntimeError('Training allocation not yet released')
            save(WORK / 'TRAINING_PAUSED_ACCEPTED.json', dict(success=True, at=time.time(),
                 candidates=list(CANDIDATES), completed_step=500, resources=snapshot))
            save(WORK / 'STATE.json', dict(stage='dual_seed_evaluation', at=time.time()))
            command([str(PYTHON), '-u', str(WORK / 'rolling_driver.py'), 'run'],
                    WORK / 'evaluation.log', [WORK / 'STOP'],
                    cancellation_marker=WORK / 'STOP')
            if not (WORK / 'COMPARISON.json').exists():
                raise RuntimeError('Four-cell comparison missing')
            save(WORK / 'STATE.json', dict(stage='experiment_complete', at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'run'])
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            activate()
            from p2_loss_search28.queue import preflight
            preflight()
        else:
            globals()[args.action]()
    except Exception as exc:
        save(WORK / 'FAILURE.json', dict(at=time.time(), error=repr(exc), traceback=traceback.format_exc()))
        raise
