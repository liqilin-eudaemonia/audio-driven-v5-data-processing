"""Evaluate the first interaction on node11 GPU7 while the second trains."""
from pathlib import Path
import argparse
import concurrent.futures
import copy
import fcntl
import importlib.util
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extension import activate, WORK, ROOT, ORIGINAL, LEGACY, CONTROL, CANDIDATES
activate()
from p2_listen.contracts import read, sha, atomic_json as save
from p2_testsets_eval.protocol import PYTHON, partition
from listen_cluster import transport
from stage_driver import command, verify_scoring
import driver as D
import eval_runtime as E
import gpu7_score as G

FIRST = 'event020_feature010'
SECOND = 'event100_feature010'
RUN = WORK / 'gpu7_first'
HANDOFF = 'Continue active second candidate to500, then hand evaluation control to gpu7_first.py\n'


def previous_resume():
    spec = importlib.util.spec_from_file_location('historical_gpu7_resume', CONTROL / 'gpu7_resume.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_first():
    from p2_combined_whisper_eval import link
    first = D.folder(FIRST, 137)
    first.mkdir(parents=True, exist_ok=True)
    save(first / 'REQUEST.json', dict(checkpoint=str(WORK / 'runs' / FIRST / 'step_000500.pt'), samples=64), immutable=True)
    p = G.configure(first)
    second = D.folder(FIRST, 238)
    if not (second / 'protocol.json').exists():
        second.mkdir(parents=True, exist_ok=True)
        for directory in ('export', 'conditions', 'whisper', 'negative'):
            link(first / directory, second / directory)
        for filename in ('REQUEST.json', 'DATA_CONFIG.json', 'shared_items.json', 'cluster.json'):
            save(second / filename, read(first / filename), immutable=True)
        q = copy.deepcopy(p)
        q.update(seed=238, cluster=str(second / 'cluster.json'), metrics_cluster=str(second / 'cluster.json'),
                 shared_items_manifest=str(second / 'shared_items.json'))
        save(second / 'protocol.json', q, immutable=True)
        for rank, items in enumerate(partition(read(first / 'shared_items.json'), 4)):
            save(second / 'shards' / f'{rank:02d}.json', items, immutable=True)


def remote(work, stage):
    if stage != 'events':
        raise ValueError(stage)
    cluster = read(work / 'cluster.json')
    node = cluster['nodes'][0]
    guard = ROOT.parent / 'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
    argv = [str(PYTHON), str(guard), '--guard', 'env', 'CUDA_VISIBLE_DEVICES=7',
            'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1', str(PYTHON), '-u',
            str(Path(__file__)), 'events', '--work', str(work)]
    command(transport(cluster, node, argv), work / 'gpu7_events.log',
            [work / 'STOP', RUN / 'STOP'], stdin_guard=True)


def inference(work):
    resume = previous_resume()
    def physical_policy(policy):
        resume.validate_gpu()
        return dict(policy, physical_gpu=7, node11_gpu_uuid=resume.UUID,
                    gpu_memory_utilization=.80, cpu_offload_gb=0)
    resume.runtime_policy = physical_policy
    resume.adapted_infer()(work / 'events')


def events(work):
    G.validate_device(read(work / 'protocol.json'))
    from p2_event_presence.contracts import rows, save_rows
    from p2_event_presence.media import bind
    p = read(work / 'protocol.json')
    arm = p['arms'][0]['name']
    destination = work / 'events'
    if not (destination / 'contract.json').exists():
        records = [r for rank in range(4) for r in rows(work / 'metric_manifests' / arm / 'seamless' / f'{rank:02d}.jsonl')]
        if len(records) != 64 or len({r['id'] for r in records}) != 64:
            raise ValueError('Incomplete GPU7 event population')
        manifest = work / 'event_pairs_manifest.jsonl'
        save_rows(manifest, records)
        policy = read(CONTROL / 'evaluations/baseline_500_seed137/events/contract.json')['policy']
        save(work / 'event_detector_config.json', policy, immutable=True)
        bind([manifest], destination, work / 'event_detector_config.json', method=arm)
    resume = previous_resume()
    resume.prepare_work(work)
    policy = read(destination / 'contract.json')['policy']
    py = policy['inference_python']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', VLLM_PORT='27000', MASTER_PORT='27030',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(Path(py).parent.parent / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    with (destination / 'gpu7_inference.log').open('ab', buffering=0) as log:
        subprocess.run([py, '-u', str(Path(__file__)), 'infer', '--work', str(work)],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    expected = read(CONTROL / 'evaluations/baseline_500_seed137/events/detector_contract.json')['identity_hash']
    if read(destination / 'detector_contract.json')['identity_hash'] != expected:
        raise ValueError('Detector identity differs from the controls')
    with (destination / 'gpu7_score.log').open('ab', buffering=0) as log:
        result = subprocess.run([str(PYTHON), str(ROOT / 'scripts/p2_event_presence.py'), 'score', '--work', str(destination)],
                                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'src')),
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 2):
        raise RuntimeError('Native event scoring failed')
    summary = read(destination / 'summary.json')
    save(work / 'EVENTS_COMPLETE.json', dict(summary_sha256=sha(destination / 'summary.json'), status=summary['status'],
         detector_based=True, human_calibrated=False, pilot_only=True))
    save(destination / 'PARALLEL_EXECUTION_ACCEPTED.json', dict(success=True, workers=1,
         host=G.HOST, physical_gpu=7, uuid=G.UUID, gpu_memory_utilization=.80, cpu_offload_gb=0,
         entrypoint_sha256=sha(Path(__file__)), detector_identity=expected))


def first_queue():
    prepare_first()
    G.__file__ = str(Path(__file__))
    G.remote = remote
    for seed in D.SEEDS:
        work = D.folder(FIRST, seed)
        save(RUN / 'STATUS.json', dict(stage='gpu7_generation_and_scoring', candidate=FIRST, seed=seed, at=time.time()))
        G.score_one(work)
    save(RUN / 'FIRST_COMPLETE.json', dict(success=True, seeds=list(D.SEEDS), at=time.time()))


def second_queue():
    from p2_loss_search28.resources import idle
    while not (WORK / 'runs' / SECOND / 'STAGE_500.json').exists():
        if (RUN / 'STOP').exists():
            return
        time.sleep(10)
    with (ORIGINAL / 'queue.lock').open('a') as allocation:
        fcntl.flock(allocation, fcntl.LOCK_EX)
        if (WORK / 'STOP').read_text() != HANDOFF:
            raise RuntimeError('New user STOP supersedes the evaluation handoff')
        (WORK / 'STOP').rename(RUN / 'TRAINING_HANDOFF_STOP')
        save(RUN / 'SECOND_STATUS.json', dict(stage='prepare_second_evaluation', at=time.time()))
        # The first GPU7 contracts remain bound to four logical metric shards.
        D.CANDIDATES = {SECOND: CANDIDATES[SECOND]}
        D.prepare_evaluation()
        import rolling_driver
        for seed in D.SEEDS:
            work = D.folder(SECOND, seed)
            if (work / 'SCORING_COMPLETE.json').exists():
                verify_scoring(work)
                continue
            if not idle(read(work / 'cluster.json'))['idle']:
                raise RuntimeError('28-card evaluation allocation busy')
            save(RUN / 'SECOND_STATUS.json', dict(stage='second_generation_and_scoring', seed=seed, at=time.time()))
            command([str(PYTHON), str(WORK / 'evaluate28.py'), 'queue', '--work', str(work)],
                    work / 'eval28.log', [RUN / 'STOP'], cancellation_marker=work / 'STOP')
            rolling_driver.score_events(work)
        save(RUN / 'SECOND_COMPLETE.json', dict(success=True, at=time.time()))


def run():
    RUN.mkdir(exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(RUN / 'CONTROLLER.json', dict(pid=os.getpid(), at=time.time(),
             start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(first_queue), pool.submit(second_queue)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    save(RUN / 'FAILURE.json', dict(at=time.time(), error=repr(exc), traceback=traceback.format_exc()))
                    (RUN / 'STOP').write_text('Stop owned evaluation after a failed stage\n')
                    for seed in D.SEEDS:
                        (D.folder(FIRST, seed) / 'STOP').parent.mkdir(parents=True, exist_ok=True)
                        (D.folder(FIRST, seed) / 'STOP').write_text('Owned evaluation failed\n')
                    raise
        if (RUN / 'STOP').exists():
            return
        D.report()
        save(RUN / 'COMPLETE.json', dict(success=True, at=time.time(), comparison=str(WORK / 'COMPARISON.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'worker', 'events', 'infer'])
    parser.add_argument('--work', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    try:
        if args.action == 'run': run()
        elif args.action == 'worker': G.worker(args.work.resolve(), args.stage, args.slot)
        elif args.action == 'events': events(args.work.resolve())
        else: inference(args.work.resolve())
    except Exception as exc:
        save(RUN / 'FAILURE.json', dict(at=time.time(), error=repr(exc), traceback=traceback.format_exc()))
        raise
