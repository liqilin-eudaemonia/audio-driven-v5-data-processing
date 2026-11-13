"""Concurrent independent videos with unchanged official SPECTRE operations."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
AUDIT = WORK / 'official_tracker_audit'
STATE = AUDIT / 'workers'
sys.path[:0] = [str(WORK), str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import read, save, sha, guard


def tasks():
    result = []
    for sample in read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')['samples']:
        uid = sample['id']
        for role, path, receipt in [
            ('speaker', WORK / 'inputs' / uid / 'speaker/cropped25.mkv', WORK / 'inputs' / uid / 'speaker/complete.json'),
            ('gt', WORK / 'inputs' / uid / 'listener/cropped25.mkv', WORK / 'inputs' / uid / 'listener/complete.json'),
            ('avatarforcing', WORK / 'native_generated' / uid / 'generated.mp4', WORK / 'native_generated' / uid / 'complete.json')]:
            output = WORK / 'official_features' / role / (uid+'.npz')
            result.append(dict(id=uid, role=role, video=str(path), source_receipt=str(receipt),
                               frames=read(WORK / 'native_generated' / uid / 'complete.json')['frames'], output=str(output)))
    return result


def prepare():
    STATE.mkdir(parents=True, exist_ok=True)
    plan = read(WORK / 'OFFICIAL_PARALLEL_CANDIDATE.json')
    assert plan['allocator_fraction_per_worker'] == .025
    assert plan['workers_during_generation'] == 1 and plan['workers_after_generation'] <= 4
    p = dict(schema='same-official-spectre-parallel-v1', script_sha256=sha(__file__),
             source_helper_sha256=sha(WORK / 'official_spectre.py'),
             feature_protocol_sha256=sha(WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json'),
             plan_sha256=sha(WORK / 'OFFICIAL_PARALLEL_CANDIDATE.json'),
             unchanged='Same source-video bytes, frontend weights, JPEG, crop,46valid-frame chunks and FP32 flags',
             atomicity='One advisory lock per independent video; receipt only after atomic feature commit',
             output_protocol='Reuse original official-feature contract; record executor SHA separately',
             tasks=129, frontend_warmup_calls=4, warmup_reason='Original serial first-call rounding differs at frame0; warmed repeat exactly matches sealed features')
    path = WORK / 'OFFICIAL_PARALLEL_PROTOCOL.json'
    if path.exists():
        assert read(path) == p
    else:
        save(path, p)
    return p


def check_locks():
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / 'filesystem_lock_check.lock'
    code = ('import fcntl,sys; f=open(sys.argv[1],"a"); '
            'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print("acquired")')
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
        assert blocked.returncode != 0 and 'BlockingIOError' in blocked.stderr
    acquired = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
    assert acquired.returncode == 0 and acquired.stdout.strip() == 'acquired'
    save(AUDIT / 'FILESYSTEM_LOCK_CHECK.json', dict(passed=True, host=os.uname().nodename,
         held_lock_blocks_independent_process=True, released_lock_can_be_acquired=True))


def load_models():
    import cv2
    import torch
    import official_spectre as source
    guard(.025, 0)
    cv2.setNumThreads(1)
    # The imported helper emits one common weight-audit file. Serialize only
    # this initialization; video processing runs independently afterward.
    with (STATE / 'initialization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = source.Encoder('cuda')
        tracker, get_landmarks = source.load_frontend()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert read(AUDIT / 'JPEG_PARITY.json')['passed']
    # The source's traced FAN has a one-frame cold-execution rounding change.
    # First canonical stream is already sealed; prime the same inference-only
    # graph for resumed independent streams, retaining frozen weights and ops.
    first = read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')['samples'][0]['id']
    cap = cv2.VideoCapture(str(WORK / 'inputs' / first / 'speaker/cropped25.mkv'))
    okay, image = cap.read()
    cap.release()
    assert okay
    for _ in range(4):
        boxes = tracker.face_detector(image, rgb=False)
        tracker.landmark_detector(image, boxes, rgb=False)
    assert not tracker.face_detector.net.training and not tracker.landmark_detector.net.training
    return source, model, tracker, get_landmarks


def parity():
    import numpy as np
    import torch
    prepare()
    source, model, tracker, get_landmarks = load_models()
    uid = 'ClNUxPsUByg_000210_000221.0'
    receipt = read(WORK / 'official_features/gt' / (uid+'.json'))
    assert sha(receipt['path']) == receipt['sha256']
    with np.load(receipt['path']) as values:
        expected_motion, expected_landmarks = values['motion'].copy(), values['landmarks'].copy()
    frames, _ = source.decode25(receipt['video'], receipt['frames'])
    images, landmarks, audit = source.official_crop(frames, tracker, get_landmarks)
    motion = model.chunks(images, batch=46)
    motion_error = float(np.abs(motion-expected_motion).max())
    landmark_error = float(np.abs(landmarks-expected_landmarks).max())
    assert motion_error <= 1e-6 and landmark_error <= 1e-6, (motion_error, landmark_error)
    assert audit == receipt['face_audit']
    result = dict(passed=True, id=uid, role='gt', frames=receipt['frames'],
        source_sha256=receipt['video_sha256'], reference_features_sha256=receipt['sha256'],
        motion_max_abs=motion_error, landmarks_max_abs=landmark_error,
        exact_motion_bytes=motion.tobytes()==expected_motion.tobytes(),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        fraction=.025, executor_protocol_sha256=sha(WORK / 'OFFICIAL_PARALLEL_PROTOCOL.json'))
    save(AUDIT / 'PARALLEL_PARITY.json', result)
    print('Parallel worker source parity', result, flush=True)


def checked_receipt(task):
    output = Path(task['output'])
    receipt = output.with_suffix('.json')
    if not receipt.exists():
        return None
    r = read(receipt)
    assert r['protocol_sha256'] == sha(WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json')
    assert r['frames'] == task['frames'] and r['id'] == task['id'] and r['role'] == task['role']
    assert sha(output) == r['sha256']
    return r


def worker():
    import numpy as np
    import torch
    prepare()
    assert read(AUDIT / 'PARALLEL_PARITY.json')['passed']
    source, model, tracker, get_landmarks = load_models()
    verified = set()
    all_tasks = tasks()
    while True:
        claimed = False
        for index, task in enumerate(all_tasks):
            if index in verified:
                continue
            if checked_receipt(task) is not None:
                verified.add(index)
                continue
            if not Path(task['source_receipt']).exists():
                continue
            lock_path = STATE / ('task_'+task['role']+'_'+task['id']+'.lock')
            with lock_path.open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if checked_receipt(task) is not None:
                    verified.add(index)
                    continue
                started = time.time()
                record = read(task['source_receipt'])
                expected = record['sha256'] if task['role']=='avatarforcing' else next(d for q,d in record['files'].items() if Path(q).resolve()==Path(task['video']).resolve())
                assert sha(task['video']) == expected
                frames, grid = source.decode25(task['video'], task['frames'])
                images, landmarks, face_audit = source.official_crop(frames, tracker, get_landmarks)
                motion = model.chunks(images, batch=46)
                assert motion.shape == (task['frames'], 56) and np.isfinite(motion).all()
                output = Path(task['output'])
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_name(output.name+f'.tmp.{os.getpid()}')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, motion=motion, landmarks=landmarks, fps=25,
                                        frame_indices=np.arange(len(motion)))
                temp.replace(output)
                r = {key: task[key] for key in ['id', 'role', 'video', 'frames']}
                r.update(path=str(output), video_sha256=expected, sha256=sha(output),
                    protocol_sha256=sha(WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json'), grid=grid,
                    face_audit=face_audit, seconds=time.time()-started,
                    executor_protocol_sha256=sha(WORK / 'OFFICIAL_PARALLEL_PROTOCOL.json'),
                    worker_pid=os.getpid(), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                save(output.with_suffix('.json'), r)
                verified.add(index)
                save(STATE / f'worker_{os.getpid()}.json', dict(pid=os.getpid(),
                    last_id=task['id'], last_role=task['role'], completed_known=len(verified),
                    last_seconds=r['seconds'], peak_allocated_mib=r['peak_allocated_mib'],
                    updated_unix=time.time()))
                print('worker', os.getpid(), task['role'], task['id'], round(r['seconds'], 2), flush=True)
                del frames, images, landmarks, motion
                claimed = True
                break
        if len(verified) == len(all_tasks):
            return
        if not claimed:
            time.sleep(2)


def gpu_state():
    row = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert row[0].strip() == 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    used = {}
    for line in text.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields)==3 and fields[0]==row[0].strip() and fields[1].isdigit() and fields[2].isdigit():
            used[int(fields[1])] = int(fields[2])
    return int(row[1]), int(row[2]), used


def manage():
    prepare()
    assert read(AUDIT / 'PARALLEL_PARITY.json')['passed']
    assert read(AUDIT / 'FILESYSTEM_LOCK_CHECK.json')['passed']
    plan = read(WORK / 'OFFICIAL_PARALLEL_CANDIDATE.json')
    assert read(WORK / 'PREPROCESS_COMPLETE.json')['roles'] == 86
    children = []
    all_tasks = tasks()
    with (STATE / 'manager.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                for child in children:
                    code = child.poll()
                    if code not in [None, 0]:
                        raise RuntimeError(f'Worker {child.pid} exited with {code}; inspect its own log')
                generation_done = (WORK / 'NATIVE_GENERATION_COMPLETE.json').exists()
                complete = sum(Path(t['output']).with_suffix('.json').exists() for t in all_tasks)
                if complete == len(all_tasks) and generation_done:
                    records = [checked_receipt(task) for task in all_tasks]
                    assert all(records)
                    for r in records:
                        assert sha(r['video']) == r['video_sha256']
                    save(WORK / 'OFFICIAL_EXTRACTION_COMPLETE.json', dict(passed=True,
                        streams=129, records=records, protocol_sha256=sha(WORK / 'OFFICIAL_SPECTRE_PROTOCOL.json'),
                        executor_protocol_sha256=sha(WORK / 'OFFICIAL_PARALLEL_PROTOCOL.json')))
                    while any(child.poll() is None for child in children):
                        save(WORK / 'OFFICIAL_PARALLEL_STATUS.json', dict(stage='workers_exiting',
                            pid=os.getpid(), completed=129, total=129,
                            live_worker_pids=[c.pid for c in children if c.poll() is None],
                            updated_unix=time.time()))
                        time.sleep(2)
                    save(WORK / 'OFFICIAL_PARALLEL_STATUS.json', dict(stage='complete',
                        pid=os.getpid(), completed=129, total=129, live_worker_pids=[],
                        updated_unix=time.time()))
                    return
                live = [child for child in children if child.poll() is None]
                desired = plan['workers_after_generation'] if generation_done else plan['workers_during_generation']
                free, total, used = gpu_state()
                budget = math.ceil(total*plan['allocator_fraction_per_worker'])+plan['extra_context_mib_per_worker']
                # Reserve each existing worker's possible future allocator use,
                # plus the generator's remaining budget while it is running.
                reserve = plan['reserve_mib'] + sum(max(0, budget-used.get(c.pid, 0)) for c in live)
                if not generation_done:
                    current = read(WORK / 'NATIVE_GENERATION_RUNTIME.json')
                    gen_budget = math.ceil(total*.15)+1024
                    reserve += max(0, gen_budget-used.get(current['pid'], 0))
                required = reserve+budget
                if len(live) < desired and free >= required:
                    log_path = STATE / f'worker_launch_{len(children)}.log'
                    with log_path.open('a') as log:
                        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker'],
                            cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
                    live.append(child)
                save(WORK / 'OFFICIAL_PARALLEL_STATUS.json', dict(stage='parallel_official_spectre',
                    pid=os.getpid(), live_worker_pids=[c.pid for c in live], completed=complete, total=129,
                    generation_complete=generation_done, desired_workers=desired,
                    free_mib=free, required_to_add_worker_mib=required, updated_unix=time.time()))
                time.sleep(5)
        except BaseException:
            for child in children:
                if child.poll() is None:
                    child.terminate()  # Only this manager's own children.
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check-locks', 'prepare', 'parity', 'worker', 'manage'])
    args = parser.parse_args()
    {'check-locks': check_locks, 'prepare': prepare, 'parity': parity,
     'worker': worker, 'manage': manage}[args.action]()
