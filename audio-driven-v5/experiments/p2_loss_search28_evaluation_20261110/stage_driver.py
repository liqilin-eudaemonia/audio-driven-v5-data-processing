"""Evaluate a completed training stage on the same released 28-GPU allocation.

Owns the training queue lock during generation/scoring. Never infers a worker's
death from an old timestamp, kills foreign jobs, or advances using training loss.
This driver runs one stage; promotion is a separately evidenced decision.
"""
from pathlib import Path
import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(WORK)]
from p2_listen.contracts import read, sha, atomic_json
from p2_loss_search28.config import WORK as TRAIN, CANDIDATES
from p2_loss_search28.resources import idle
from listen_cluster import transport


def stage_candidates(target):
    if target == 500:
        return list(CANDIDATES)
    v = read(TRAIN / f'promotion_{target}.json')
    names = v['candidates']
    if v.get('target') != target or v.get('basis') != 'generated_av_quality':
        raise ValueError('Invalid training promotion')
    if not names or len(set(names)) != len(names) or any(n not in CANDIDATES for n in names):
        raise ValueError('Invalid stage candidates')
    # Keeping the control also establishes new reference labels when expanding
    # from 64 to 200 clips. Do not silently train an unselected control here.
    if 'baseline' not in names:
        raise ValueError('Stage needs the baseline control for matched scoring')
    return ['baseline', *[n for n in names if n != 'baseline']]


def checkpoint_receipts_ready(target, names):
    for name in names:
        out = TRAIN / 'runs' / name
        receipt = out / f'STAGE_{target}.json'
        marker = out / f'step_{target:06d}.complete.json'
        if not receipt.exists() or not marker.exists():
            return False
        r, m = read(receipt), read(marker)
        if r.get('success') is not True or r['step'] != target or r['sha256'] != m['sha256']:
            raise ValueError('Training receipt does not match checkpoint')
    return True


def verify_scoring(work):
    """Validate a completed score without rewriting any frozen evidence."""
    import eval_runtime as runtime
    from p2_event_presence.media import verify as verify_media
    from p2_event_presence.contracts import digest
    work=Path(work);receipt=read(work/'SCORING_COMPLETE.json');p=read(work/'protocol.json')
    request=read(work/'REQUEST.json');checkpoint=Path(request['checkpoint'])
    arm=p['arms'][0];name=p['checkpoint_identity']['experiment']['loss_search']['candidate']
    if receipt.get('success')is not True or p['execution_code']!=runtime.code_identity():
        raise ValueError('Scoring receipt/source not accepted')
    if checkpoint.parent.name!=name or checkpoint.name!=f"step_{arm['step']:06d}.pt" or p['total_samples']!=request['samples']:
        raise ValueError('Scoring request/candidate/step mismatch')
    if receipt['checkpoint_sha256']!=read(checkpoint.with_suffix('.complete.json'))['sha256']or receipt['checkpoint_sha256']!=arm['original_sha256']:
        raise ValueError('Scored checkpoint changed')
    folder='metrics/'+arm['name']+'/seamless/'
    required={'FINAL_AUDIT.json','RESULTS.json','REFERENCE_EVENTS_ACCEPTED.json',
        'event_pairs_fixed_reference.jsonl','fixed_reference_event_summary.json',folder+'pairs.jsonl',folder+'audio_pairs.jsonl'}
    if not required<=set(receipt['evidence']):raise ValueError('Incomplete scoring receipt')
    for path,expected in receipt['evidence'].items():
        if sha(work/path)!=expected:raise ValueError('Scoring evidence changed: '+path)
    for sid,expected in read(work/'REFERENCE_EVENTS_ACCEPTED.json')['reference_sha256_by_id'].items():
        if sha(WORK/'reference_events'/(digest(sid)+'.json'))!=expected:raise ValueError('Frozen reference labels changed')
    verify_media(work/'events')  # Includes hashes of generated own video/WAV.
    return receipt


def command(argv, log, stop_paths, stdin_guard=False, cancellation_marker=None):
    """Close only our guarded remote child's stdin on cancellation."""
    with log.open('ab', buffering=0) as stream:
        p = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if stdin_guard else subprocess.DEVNULL)
        try:
            while p.poll() is None:
                if any(f.exists() for f in stop_paths):
                    if p.stdin:
                        p.stdin.close()
                    else:
                        # evaluate28 handles its work/STOP cooperatively; do not
                        # terminate its Python controller before worker cleanup.
                        if cancellation_marker is None:
                            raise ValueError('Local cooperative command needs a STOP marker')
                        cancellation_marker.touch(exist_ok=True)
                        break
                    break
                time.sleep(2)
            if any(f.exists() for f in stop_paths) and not stdin_guard:
                return p.wait()
            rc = p.wait()
            if rc:
                raise RuntimeError(f'Command exited {rc}: {log}')
            return rc
        finally:
            if p.stdin and not p.stdin.closed:
                p.stdin.close()


def evaluate(target):
    if (WORK/'GPU7_DISPATCH.json').exists():
        from gpu7_score import evaluate as score_on_gpu7
        return score_on_gpu7(target)
    import eval_runtime as runtime
    from reference_events import freeze
    names = stage_candidates(target)
    stopped = lambda: (WORK / 'STOP').exists() or (TRAIN / 'STOP').exists()
    state = WORK / f'STAGE_{target}_STATUS.json'
    with (WORK / 'stage_driver.lock').open('a') as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        atomic_json(WORK / 'STAGE_CONTROLLER.json', dict(pid=os.getpid(), start_ticks=ticks,
                                                        target=target, at=time.time()))
        while not checkpoint_receipts_ready(target, names):
            if stopped(): return
            atomic_json(state, dict(stage='waiting_for_training_checkpoints', target=target,
                                    candidates=names, at=time.time()))
            time.sleep(30)
        with (TRAIN / 'queue.lock').open('a') as train_lock:
            while True:
                if stopped(): return
                try:
                    fcntl.flock(train_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(5)
            cluster = read(TRAIN / 'runs/baseline/cluster.json')
            for name in names:
                if stopped(): return
                work = WORK / 'evaluations' / f'{name}_{target}'
                expected_request = dict(checkpoint=str(TRAIN / 'runs' / name / f'step_{target:06d}.pt'),
                                        samples=64 if target == 500 else 200)
                if read(work / 'REQUEST.json') != expected_request:
                    raise ValueError('Evaluation request differs from selected stage/candidate')
                if (work / 'STOP').exists():
                    raise RuntimeError('Explicit evaluation STOP retained: ' + str(work))
                if(work/'SCORING_COMPLETE.json').exists():
                    verify_scoring(work)
                    # In particular, do not call runtime.final again: its new
                    # timestamp would invalidate prior immutable decisions.
                    continue
                # A busy card is not evidence that its process died. Wait and
                # preserve all other tasks, especially node11GPU4-7.
                while True:
                    if stopped(): return
                    snap = idle(cluster)
                    atomic_json(WORK / 'RESOURCES_LATEST.json', snap)
                    if snap['idle']: break
                    time.sleep(30)
                atomic_json(state, dict(stage='generation_and_scoring', target=target,
                                        candidate=name, at=time.time()))
                # Strict checkpoint/source/export validation also covers resume.
                runtime.prepare(work)
                if not (work / 'FINAL_AUDIT.json').exists():
                    cmd = [cluster['python'], '-u', str(WORK / 'evaluate28.py'),
                           'queue', '--work', str(work)]
                    command(cmd, work / 'stage_generation.log',
                            [work / 'STOP', WORK / 'STOP', TRAIN / 'STOP'],
                            cancellation_marker=work / 'STOP')
                    if stopped() or (work / 'STOP').exists(): return
                runtime.final(work)
                if stopped(): return
                # All generation/decode/metric workers must have released their
                # GPUs before the independent detector takes node12GPU7.
                while not idle(cluster)['idle']:
                    if stopped(): return
                    time.sleep(5)
                if not (work / 'EVENTS_COMPLETE.json').exists():
                    guard = ROOT.parent / 'audio-driven-v4-p3-autoselect-20261013/scripts/pipeline3_cluster.py'
                    cmd = ['env', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1', cluster['python'],
                           str(guard), '--guard', cluster['python'], '-u', str(WORK / 'run_events.py'),
                           '--work', str(work)]
                    command(transport(cluster, cluster['nodes'][0], cmd),
                            work / 'stage_events.log', [WORK / 'STOP', TRAIN / 'STOP'], True)
                    if stopped(): return
                event_done = read(work / 'EVENTS_COMPLETE.json')
                if event_done['summary_sha256'] != sha(work / 'events/summary.json'):
                    raise ValueError('Event summary changed after detector completion')
                freeze(work)
                arm = read(work / 'protocol.json')['arms'][0]['name']
                metric_folder = 'metrics/' + arm + '/seamless/'
                atomic_json(work / 'SCORING_COMPLETE.json', dict(success=True, at=time.time(),
                    checkpoint_sha256=read(work / 'export/RECEIPT.json')['original_sha256'],
                    evidence={f: sha(work / f) for f in ['FINAL_AUDIT.json', 'RESULTS.json',
                        'REFERENCE_EVENTS_ACCEPTED.json', 'event_pairs_fixed_reference.jsonl',
                        'fixed_reference_event_summary.json', metric_folder+'pairs.jsonl',
                        metric_folder+'audio_pairs.jsonl']},
                    partial_event_coverage=event_done['status'] != 'complete',
                    detector_human_calibration_pending=True))
            atomic_json(state, dict(stage='generation_review_ready', target=target,
                                    candidates=names, at=time.time(), promoted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, choices=(500, 1500, 3000, 6000), required=True)
    args = parser.parse_args()
    try:
        evaluate(args.target)
    except Exception as exc:
        atomic_json(WORK / f'STAGE_{args.target}_FAILURE.json',
                    dict(error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
        raise
