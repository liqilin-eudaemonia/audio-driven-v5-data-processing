#!/usr/bin/env python3
"""Current500h non-prosody pipeline: shared decode, LIA, audio, reactions, stats.

Uses only node11 GPU4-7; existing LIA payloads are verified and referenced. CPU
reaction workers retain the production runtime, sampling, encoding and gates.
Wav2Vec2/SER stay online: their outputs depend on the final training window.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import prepare_glare_nonprosody_cache as base
from glare_shared_decode import decode_group, media_key

WORK = ROOT / 'experiments/glare_current500h_shared_cache_20261110'
PARENT = base.DEFAULT
MP_PY = ROOT / 'experiments/listen_j_omni_20261030/nod_cpu_share/runtime_env/bin/python'
GPU_PY = ROOT / 'compare/.envs/avatarforcing/bin/python'
GPUS = (4, 5, 6, 7)
VERSION = 'glare-current500h-shared-nonprosody-v2'
read, save, sha, canonical = base.read, base.save, base.sha, base.canonical


def codes():
    paths = [Path(__file__), ROOT / 'scripts/glare_shared_decode.py',
             ROOT / 'scripts/glare_shared_reactions.py', ROOT / 'scripts/listen_j_annotation_nod.py',
             ROOT / 'scripts/listen_j_corrected_reaction.py',
             ROOT / 'scripts/check_glare_expression_gate.py', ROOT / 'scripts/check_glare_headshake_direction.py']
    return {str(p): sha(p) for p in paths}


def load_protocol():
    p = read(WORK / 'protocol.json')
    assert p['schema'] == VERSION and p['code'] == codes(), 'Frozen shared pipeline changed'
    assert p['parent_protocol_sha256'] == sha(PARENT / 'protocol.json')
    assert p['groups_sha256'] == sha(WORK / 'groups.sqlite'), 'Frozen group index changed'
    return p


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = base.load_protocol(PARENT)
    if (WORK / 'protocol.json').exists():
        load_protocol()
        print('ALREADY_PREPARED', flush=True)
        return
    tmp = WORK / 'groups.sqlite.tmp'
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute('CREATE TABLE pairs(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE,media_key TEXT,start REAL,end REAL,duration REAL)')
    old_db = sqlite3.connect((PARENT / 'jobs.sqlite').as_uri() + '?mode=ro', uri=True)
    old_db.row_factory = sqlite3.Row
    jobs = {r['sid']: dict(r) for r in old_db.execute('SELECT * FROM jobs')}
    old_db.close()
    hashed = hashlib.sha256()
    with Path(old['selection']).open('rb') as stream:
        for i, raw in enumerate(stream):
            hashed.update(raw)
            row = json.loads(raw)
            job = jobs[row['sample_id']]
            assert hashlib.sha256(raw).hexdigest() == job['row_sha']
            assert canonical(row['source']) == job['source_sha']
            assert row['split'] in ('train', 'dev') and row['source_split'] == row['split']
            source = row['source']
            db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?)',
                       (job['ordinal'], row['sample_id'], media_key(source),
                        source['source_start_s'], source['source_end_s'], row['duration_s']))
            if (i + 1) % 20000 == 0:
                print(json.dumps(dict(indexed=i + 1)), flush=True)
    assert hashed.hexdigest() == old['selection_sha256'] and len(jobs) == 284218
    db.execute('CREATE TABLE groups(gid INTEGER PRIMARY KEY,worker INTEGER,ordinals TEXT,pairs INTEGER,seconds REAL)')
    current, last_key, end, seconds = [], None, 0., 0.
    groups, shared, worker_pairs = 0, 0, [0] * 4
    def emit():
        nonlocal groups, shared
        if not current:
            return
        worker = int.from_bytes(hashlib.sha256(last_key.encode()).digest()[:4], 'big') % 4
        db.execute('INSERT INTO groups VALUES(?,?,?,?,?)', (groups, worker, json.dumps(current), len(current), seconds))
        groups += 1
        shared += len(current) - 1
        worker_pairs[worker] += len(current)
    for ordinal, sid, key, start, stop, duration in db.execute('SELECT * FROM pairs ORDER BY media_key,start,ordinal'):
        if current and (key != last_key or start - end > 1. or seconds + duration > 16. or len(current) >= 2):
            emit()
            current, seconds = [], 0.
        current.append(ordinal)
        end = max(end, stop) if key == last_key else stop
        last_key = key
        seconds += duration
    emit()
    db.execute('CREATE INDEX grouped_workers ON groups(worker,gid)')
    db.commit()
    assert db.execute('SELECT SUM(pairs) FROM groups').fetchone()[0] == 284218
    db.close()
    tmp.replace(WORK / 'groups.sqlite')
    config = read(ROOT / 'experiments/listen_j_reaction_vocal_omni_20261103/config.json')
    protocol = dict(schema=VERSION, samples=284218, hours=old['hours'],
        parent_protocol_sha256=sha(PARENT / 'protocol.json'), parent=str(PARENT),
        selection_sha256=old['selection_sha256'], groups=groups,
        original_opens_avoided_if_all_pairs_need_decode=shared,
        worker_pairs=worker_pairs, groups_sha256=sha(WORK / 'groups.sqlite'), code=codes(),
        gpus=list(GPUS), frame_batch=4, decode_threads_per_gpu=4,
        detector_services_per_gpu=4, inflight_groups_per_gpu=4,
        group_limits=dict(pairs=2, summed_seconds=16, gap_seconds=1),
        shared_memory=dict(root='/dev/shm', max_groups=16, max_group_gib=2, aggregate_bound_gib=32),
        production_reaction_config=config['reaction_input'],
        reaction_classes=['smiling','visual_laughter','brow_lowering','brows_raised','head_nodding','head_shaking'],
        reaction_policy='production nod/shake/brow retained; missing smile/laughter measured with existing native supplement defaults; independent channels',
        stages=['lia','speaker_audio','reaction','motion_statistics'],
        prosody='deferred_by_user', wav2vec_ser='online_on_final_training_window; cached normalized audio ready',
        full_glare_training_admitted=False, created_at=time.time())
    save(WORK / 'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def path_for(sid, kind):
    assert Path(sid).name == sid
    folder = WORK / 'cache' / hashlib.sha256(sid.encode()).hexdigest()[:2]
    return folder / (sid + '.' + kind)


def ident(job, p):
    return dict(schema=VERSION, sample_id=job['sid'], row_sha256=job['row_sha'],
                selection_sha256=p['selection_sha256'], code_sha256=canonical(p['code']))


def stage_ok(job, kind, p):
    receipt = path_for(job['sid'], kind + '.json')
    if not receipt.exists():
        return False
    r = read(receipt)
    assert r['identity'] == ident(job, p), 'Stage identity changed'
    assert sha(r['payload']) == r['payload_sha256'], 'Stage payload corrupt'
    return True


def finish_stage(job, kind, payload, p, **extra):
    save(path_for(job['sid'], kind + '.json'), dict(identity=ident(job, p), payload=str(payload),
         payload_sha256=sha(payload), completed_at=time.time(), **extra))


def complete(job, p):
    receipt = path_for(job['sid'], 'complete.json')
    if not receipt.exists():
        return False
    r = read(receipt)
    assert r['identity'] == ident(job, p)
    return all(stage_ok(job, k, p) for k in ['lia', 'audio', 'reaction', 'statistics'])


def save_tensor(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, tmp)
    back = torch.load(tmp, map_location='cpu', weights_only=True)
    assert set(back) == set(value)
    assert all(torch.equal(back[k], v) for k, v in value.items())
    tmp.replace(path)


class CPUService:
    def __init__(self, gpu, slot):
        folder = WORK / 'workers'
        folder.mkdir(exist_ok=True)
        self.log = (folder / ('cpu_gpu%d_%d.log' % (gpu, slot))).open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'scripts'))
        self.process = subprocess.Popen([str(MP_PY), str(ROOT / 'scripts/glare_shared_reactions.py'), 'server'],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            start_new_session=True)

    def run(self, folder):
        output = Path(folder) / 'reaction.npz'
        self.process.stdin.write(json.dumps(dict(folder=str(folder), output=str(output))) + '\n')
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 1200)[0]:
            raise TimeoutError('Own CPU detector exceeded1200s; inspect its log')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Own CPU detector exited; inspect its log')
        result = json.loads(line)
        assert result['ok'], result.get('error')
        return output, result['result']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)


def prepare_group(group, jobs, old, tmp_root):
    started = time.time()
    rows, by_sid = [], {}
    from p2_j_current_qwen_hidden import item_from_row
    for ordinal in json.loads(group['ordinals']):
        job = jobs[ordinal]
        row = base.load_row(job, old)
        item_from_row(row)
        rows.append(row)
        by_sid[row['sample_id']] = job
    directory = Path(tempfile.mkdtemp(prefix='group-%08d-' % group['gid'], dir=tmp_root))
    try:
        try:
            decoded = decode_group(rows, directory)
            fallback = None
        except Exception as exc:
            # Unsupported source timing is explicit; don't substitute other frames.
            raise RuntimeError('Shared decode failed for group%d: %r' % (group['gid'], exc)) from exc
        for item in decoded['items']:
            folder = Path(item['folder'])
            row = next(r for r in rows if r['sample_id'] == item['sample_id'])
            try:
                ev = base.evidence(row)
            except Exception as exc:
                ev = dict(status='evidence_validation_failed', error=repr(exc), training_labels_admitted=False)
            save(folder / 'production_evidence.json', ev)
            item['job'] = by_sid[item['sample_id']]
        decoded.update(directory=str(directory), prepare_s=time.time() - started)
        return decoded
    except BaseException:
        shutil.rmtree(directory)
        raise


def worker(gpu, limit=None, canary=False):
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    from scipy.signal import resample_poly
    from p2_j_cache.data import read_speaker
    from p2_j_current_qwen_hidden import item_from_row
    assert gpu in GPUS and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    p, old = load_protocol(), base.load_protocol(PARENT)
    lock = WORK / 'workers' / ('gpu%d.lock' % gpu)
    lock.parent.mkdir(exist_ok=True)
    handle = lock.open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 16 * 1024**3
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = base.load_lia(old)
    direction = model.dec.direction.weight.detach().cpu()
    from float.utils import compute_alpha_stats
    processor = Wav2Vec2FeatureExtractor.from_pretrained(ROOT / 'compare/AvatarForcing/pretrained_dir/wav2vec2-base-960h', local_files_only=True)
    db = sqlite3.connect((PARENT / 'jobs.sqlite').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    jobs = {r['ordinal']: dict(r) for r in db.execute('SELECT * FROM jobs')}
    by_sid = {r['sid']: r for r in jobs.values()}
    db.close()
    db = sqlite3.connect((WORK / 'groups.sqlite').as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    groups = [dict(g) for g in db.execute('SELECT * FROM groups WHERE worker=? ORDER BY gid', (GPUS.index(gpu),))]
    if canary:
        diagnostic = read(ROOT / 'experiments/baseline_training_realdata_20261109/DATA_PROTOCOL.json')
        ids = diagnostic['diagnostic_train'] + diagnostic['diagnostic_validation']
        groups = [dict(gid=-i-1, ordinals=json.dumps([by_sid[sid]['ordinal']]), pairs=1) for i, sid in enumerate(ids)]
    db.close()
    if limit:
        groups = groups[:limit]
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    state = dict(gpu=gpu, pid=os.getpid(), stage='running', total_pairs=sum(g['pairs'] for g in groups),
                 completed=0, reused=0, failed=0, lia_reused=0, original_opens=0,
                 groups_with_shared_decode=0, started=time.time(), decode_wait_s=0., encode_s=0.,
                 shared_bytes_peak=0, prosody='deferred_by_user')
    last_pulse = [0.]
    def pulse():
        if state['stage'] == 'running' and time.time() - last_pulse[0] < 2.:
            return
        state.update(heartbeat=time.time(), elapsed_s=time.time()-state['started'],
                     peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
        save(WORK / 'workers' / ('gpu%d.json' % gpu), state)
        last_pulse[0] = time.time()
    thread_local = threading.local()
    clients, client_lock = [], threading.Lock()
    def reactions(folder):
        if not hasattr(thread_local, 'client'):
            with client_lock:
                client = CPUService(gpu, len(clients))
                clients.append(client)
            thread_local.client = client
        return thread_local.client.run(folder)
    tmp_root = Path('/dev/shm') / ('glare-shared-%d-%d' % (gpu, os.getpid()))
    tmp_root.mkdir()
    pending_decode, pending_reaction, held = {}, {}, {}
    next_group = 0
    def error(job, exc):
        state['failed'] += 1
        save(WORK / 'errors' / (job['sid'] + '.json'), dict(sample_id=job['sid'], error=repr(exc), traceback=traceback.format_exc()))
    pulse()
    try:
        with ThreadPoolExecutor(max_workers=p['decode_threads_per_gpu']) as decoders, ThreadPoolExecutor(max_workers=p['detector_services_per_gpu']) as detectors:
            while next_group < len(groups) or pending_decode or pending_reaction:
                if stopped[0] or (WORK / 'STOP').exists():
                    break
                while next_group < len(groups) and len(pending_decode) + len(held) < p['inflight_groups_per_gpu']:
                    group = groups[next_group]
                    next_group += 1
                    selected = [jobs[o] for o in json.loads(group['ordinals'])]
                    if all(complete(j, p) for j in selected):
                        state['reused'] += len(selected)
                        continue
                    future = decoders.submit(prepare_group, group, jobs, old, tmp_root)
                    pending_decode[future] = group
                ready_decode = [f for f in pending_decode if f.done()]
                for future in ready_decode:
                    group = pending_decode.pop(future)
                    key = None
                    try:
                        result = future.result()
                        state['original_opens'] += result['original_opens']
                        state['groups_with_shared_decode'] += len(result['items']) > 1
                        key = result['directory']
                        held[key] = len(result['items'])
                        state['shared_bytes_peak'] = max(state['shared_bytes_peak'], sum(
                            f.stat().st_size for f in tmp_root.glob('*/*/*.npy')))
                        for item in result['items']:
                            job, folder = item['job'], Path(item['folder'])
                            if complete(job, p):
                                state['reused'] += 1
                                held[key] -= 1
                                continue
                            row = read(folder / 'row.json')
                            lia_payload = path_for(job['sid'], 'lia.pt')
                            if stage_ok(job, 'lia', p):
                                lia_payload = Path(read(path_for(job['sid'], 'lia.json'))['payload'])
                            elif base.existing(PARENT, job, old):
                                lia_payload, _ = base.paths(PARENT, job['sid'])
                                finish_stage(job, 'lia', lia_payload, p, reused_parent=True)
                                state['lia_reused'] += 1
                            else:
                                images = np.load(folder / 'lia_rgb.npy', mmap_mode='r')
                                tick = time.time()
                                motion = base.encode(model, images, batch=4)
                                state['encode_s'] += time.time() - tick
                                save_tensor(lia_payload, {'motion': motion})
                                finish_stage(job, 'lia', lia_payload, p, reused_parent=False, pts=item['pts'])
                            if not stage_ok(job, 'statistics', p):
                                motion = torch.load(lia_payload, map_location='cpu', weights_only=True)['motion']
                                stats = compute_alpha_stats(motion[1:] @ direction)
                                stats_path = path_for(job['sid'], 'statistics.pt')
                                save_tensor(stats_path, stats)
                                finish_stage(job, 'statistics', stats_path, p, reference='training target frames1..N-1; inference must not use target statistics')
                            if not stage_ok(job, 'audio', p):
                                audio, rate = read_speaker(item_from_row(row))
                                audio = audio.mean(axis=1)
                                audio = resample_poly(audio, 16000 // math.gcd(rate,16000), rate // math.gcd(rate,16000)).astype(np.float32)
                                exact = len(audio)
                                count = item['frames'] * 640
                                audio = np.pad(audio, (0,max(0,count-exact)))[:count]
                                normalized = np.asarray(processor(audio,sampling_rate=16000).input_values[0],np.float32)
                                audio_path = path_for(job['sid'], 'audio.pt')
                                save_tensor(audio_path, {'audio':torch.from_numpy(normalized)})
                                finish_stage(job,'audio',audio_path,p,sample_rate=16000,original_samples=exact,
                                    grid_samples=count,tail_padding=max(0,count-exact),tail_trim=max(0,exact-count),
                                    frozen_encoders='Wav2Vec2/SER online on final training window')
                            if stage_ok(job,'reaction',p):
                                save(path_for(job['sid'],'complete.json'),dict(identity=ident(job,p),prosody='deferred',full_glare_training_admitted=False))
                                state['completed'] += 1
                                held[key] -= 1
                            else:
                                rf = detectors.submit(reactions, folder)
                                pending_reaction[rf] = (job, folder, key)
                        if held[key] == 0:
                            shutil.rmtree(key)
                            del held[key]
                    except Exception as exc:
                        scheduled = {v[0]['sid'] for v in pending_reaction.values()}
                        for ordinal in json.loads(group['ordinals']):
                            job = jobs[ordinal]
                            if job['sid'] not in scheduled and not complete(job,p):
                                error(job,exc)
                        # A failed audio/statistics/encode stage must not leave
                        # an unscheduled item retaining a shared-memory group.
                        if key is not None and key in held:
                            held[key] = sum(k == key for _,_,k in pending_reaction.values())
                            if held[key] == 0:
                                shutil.rmtree(key)
                                del held[key]
                for future in [f for f in pending_reaction if f.done()]:
                    job, folder, key = pending_reaction.pop(future)
                    try:
                        output, result = future.result()
                        target = path_for(job['sid'],'reaction.npz')
                        target.parent.mkdir(parents=True,exist_ok=True)
                        tmp = target.with_name(target.name+'.tmp.'+str(os.getpid()))
                        shutil.copyfile(output,tmp)
                        values = np.load(tmp)
                        assert values['reaction'].shape == values['reaction_valid'].shape == (math.ceil(job['duration']*25-1e-7),6)
                        assert values['reaction_valid'].dtype == bool and np.isfinite(values['reaction']).all()
                        tmp.replace(target)
                        finish_stage(job,'reaction',target,p,measurement=result)
                        save(path_for(job['sid'],'complete.json'),dict(identity=ident(job,p),prosody='deferred',full_glare_training_admitted=False))
                        failed_path = WORK/'errors'/(job['sid']+'.json')
                        if failed_path.exists():
                            failed_path.unlink()
                        state['completed'] += 1
                    except Exception as exc:
                        error(job,exc)
                    finally:
                        held[key] -= 1
                        if held[key] == 0:
                            shutil.rmtree(key)
                            del held[key]
                pulse()
                if not ready_decode:
                    tick=time.time()
                    wait([*pending_decode,*pending_reaction],timeout=.2,return_when=FIRST_COMPLETED)
                    state['decode_wait_s'] += time.time()-tick
        state['stage'] = 'stopped' if stopped[0] or (WORK/'STOP').exists() else ('finished' if not state['failed'] else 'finished_with_errors')
        pulse()
    finally:
        for client in clients:
            client.close()
        shutil.rmtree(tmp_root,ignore_errors=True)


def controller():
    p=load_protocol()
    for name in ['SHARED_DECODE_ACCEPTED.json','CPU_TESTS.json','CANARY_ACCEPTED.json','RESUME_ACCEPTED.json']:
        assert read(WORK/name)['passed'],name
    lock=(WORK/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children,attempts,logs={}, {}, []
    def launch(gpu):
        attempts[gpu]=attempts.get(gpu,0)+1
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH=':'.join(str(x) for x in [PARENT/'runtime_deps',ROOT/'compare/.envs/glare-deps',ROOT/'src',ROOT/'scripts']))
        stream=(WORK/'workers'/('gpu%d.log'%gpu)).open('a');logs.append(stream)
        children[gpu]=subprocess.Popen([str(GPU_PY),str(Path(__file__).resolve()),'worker','--gpu',str(gpu)],
                       cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    (WORK/'workers').mkdir(exist_ok=True)
    for gpu in GPUS:launch(gpu)
    while True:
        statuses={}
        for gpu,child in list(children.items()):
            file=WORK/'workers'/('gpu%d.json'%gpu)
            state=read(file) if file.exists() else {}
            if state.get('pid')!=child.pid:state=dict(stage='initializing')
            statuses[str(gpu)]=dict(state,process_pid=child.pid,returncode=child.poll(),attempt=attempts[gpu])
            if child.poll() is not None and state.get('stage')!='finished' and attempts[gpu]<3 and not (WORK/'STOP').exists():
                launch(gpu)
        stopped=(WORK/'STOP').exists()
        ended=all(c.poll() is not None for c in children.values())
        value=dict(stage='running',controller_pid=os.getpid(),samples=p['samples'],workers=statuses,
                   completed=sum(s.get('completed',0)+s.get('reused',0) for s in statuses.values()),
                   failed=sum(s.get('failed',0) for s in statuses.values()),heartbeat=time.time(),
                   prosody='deferred_by_user',full_glare_training_admitted=False)
        if stopped:
            for child in children.values():
                if child.poll() is None:child.terminate()
            value['stage']='stopping'
        if ended:
            value['stage']='nonprosody_complete' if value['completed']==p['samples'] and not value['failed'] else 'needs_error_review'
        save(WORK/'status.json',value)
        if ended:return
        time.sleep(10)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','worker','controller'])
    ap.add_argument('--gpu',type=int,choices=GPUS)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--canary',action='store_true')
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='controller':controller()
    else:worker(args.gpu,args.limit,args.canary)
