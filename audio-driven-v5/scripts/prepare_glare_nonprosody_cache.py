#!/usr/bin/env python3
"""Independent current500h LIA cache; production evidence pointers, no prosody.

Only final listener RGB is decoded, in memory. No speaker video, WAV, or RGB
arrays are exported. Each immutable receipt commits a verified FP32 payload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
DEFAULT = ROOT / 'experiments/glare_current500h_nonprosody_cache_20261110'
SELECTION = ROOT / 'experiments/listen_j_upright_av500h_train_dev_v3_20261105/selection'
INPUT = ROOT / 'experiments/p2_j_train_dev500h_qwen_hidden_20261105'
MODEL = ROOT / 'compare/AvatarForcing/pretrained_dir/flow_transformer.pth'
GPUS = (4, 5, 6, 7)
VERSION = 'glare-current500h-lia-fp32-production-evidence-v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def paths(work, sid):
    assert Path(sid).name == sid and sid not in ('.', '..')
    folder = Path(work) / 'lia' / hashlib.sha256(sid.encode()).hexdigest()[:2]
    return folder / (sid + '.pt'), folder / (sid + '.json')


def source_code():
    files = [Path(__file__), ROOT / 'src/p2_j_cache/data.py',
             ROOT / 'scripts/p2_j_current_qwen_hidden.py',
             ROOT / 'src/listen_benchmark/frame_crop.py',
             ROOT / 'src/listen_visual_quality/measurement.py',
             ROOT / 'src/listen_benchmark/caption_guard.py']
    files += sorted((ROOT / 'related/glare/float/models').glob('*.py'))
    return {str(p): sha(p) for p in files}


def prepare(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (work / 'protocol.json').exists():
        protocol = load_protocol(work)
        assert sha(work / 'jobs.sqlite') == protocol['jobs_sha256']
        print(json.dumps({'already_prepared': True, 'samples': protocol['samples']}))
        return
    summary = read(SELECTION / 'SUMMARY.json')
    audit = read(SELECTION / 'INDEPENDENT_FINAL_AUDIT.json')
    prior = read(INPUT / 'protocol.json')
    assert summary['samples'] == 284218 and summary['test_excluded'] and audit['passed']
    assert prior['selection_sha256'] == summary['files']['selected.jsonl']['sha256']
    assert sha(INPUT / 'inputs.sqlite') == prior['index_sha256']
    assert sha(SELECTION / 'index.sqlite') == summary['files']['index.sqlite']['sha256']
    tmp = work / 'jobs.sqlite.tmp'
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute('CREATE TABLE jobs(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE,offset INTEGER,row_sha TEXT,split TEXT,duration REAL,caption_sha TEXT,source_sha TEXT,worker INTEGER)')
    src = sqlite3.connect((INPUT / 'inputs.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    src.execute('ATTACH DATABASE ? AS selected', (str(SELECTION / 'index.sqlite'),))
    rows = src.execute('SELECT i.sid,i.offset,i.row_sha,i.split,i.duration,i.caption_sha,s.source_sha256,s.offset,s.caption_sha256,s.split FROM samples i JOIN selected.samples s ON i.sid=s.sid ORDER BY i.rowid')
    members = set()
    counts = [0] * 4
    seconds = [0.] * 4
    splits = {}
    for ordinal, row in enumerate(rows):
        sid, offset, row_sha, split, duration, caption, source, selected_offset, selected_caption, selected_split = row
        assert split in ('train', 'dev') and split == selected_split
        assert offset == selected_offset and caption == selected_caption
        worker = int.from_bytes(hashlib.sha256(sid.encode()).digest()[:4], 'big') % 4
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)', (ordinal, sid, offset, row_sha, split, duration, caption, source, worker))
        members.add(sid)
        counts[worker] += 1
        seconds[worker] += duration
        splits[split] = splits.get(split, 0) + 1
    assert members == set(prior['sample_ids']) and len(members) == 284218
    assert splits == {'train': 281060, 'dev': 3158}
    assert abs(sum(seconds) / 3600 - summary['hours']) < 1e-7
    db.execute('CREATE INDEX worker_jobs ON jobs(worker,ordinal)')
    db.commit()
    db.close()
    src.close()
    tmp.replace(work / 'jobs.sqlite')
    accepted = read(ROOT / 'experiments/baseline_training_realdata_20261109/runs/glare/protocol.json')
    expected = accepted['implementation']['assets_sha256'][str(MODEL)]
    assert sha(MODEL) == expected
    protocol = dict(schema=VERSION, samples=len(members), hours=summary['hours'],
                    selection=str(SELECTION / 'selected.jsonl'), selection_sha256=prior['selection_sha256'],
                    jobs_sha256=sha(work / 'jobs.sqlite'), model=str(MODEL), model_sha256=expected,
                    code=source_code(), gpus=list(GPUS), worker_counts=counts,
                    worker_hours=[x / 3600 for x in seconds], fps=25, size=512,
                    dtype='float32', frame_batch=4, prep_threads=2, prefetch=2,
                    numerics=dict(cudnn_allow_tf32=True, matmul_allow_tf32=False,
                                  cudnn_benchmark=False, deterministic=True),
                    production_labels='reuse existing production configuration; physical observation names preserved',
                    stages=['lia', 'production_evidence_index'], prosody='deferred_by_user',
                    full_glare_training_admitted=False, raw_media_export=False,
                    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    save(work / 'protocol.json', protocol)
    print(json.dumps({'prepared': True, 'samples': len(members), 'worker_counts': counts, 'worker_hours': protocol['worker_hours']}), flush=True)


def load_protocol(work):
    protocol = read(Path(work) / 'protocol.json')
    assert protocol['schema'] == VERSION and protocol['code'] == source_code(), 'Frozen source changed'
    assert protocol['gpus'] == list(GPUS) and protocol['prosody'] == 'deferred_by_user'
    return protocol


def load_row(job, protocol):
    with Path(protocol['selection']).open('rb') as stream:
        stream.seek(job['offset'])
        raw = stream.readline()
    assert hashlib.sha256(raw).hexdigest() == job['row_sha'], 'Selected row changed'
    row = json.loads(raw)
    assert row['sample_id'] == job['sid'] and row['caption_file_sha256'] == job['caption_sha']
    assert row['split'] == row['source_split'] == job['split']
    # The published selection uses listen_richness.common's compact JSON hash,
    # whereas p2_listen.contracts.digest deliberately uses spaced JSON.
    assert canonical(row['source']) == job['source_sha'], 'Selected source changed'
    assert abs(row['duration_s'] - job['duration']) < 1e-7
    return row


def evidence(row):
    """Index authentic existing evidence, without relabeling it or filling zeros."""
    caption_path = Path(row['caption_file'])
    raw = caption_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['caption_file_sha256']
    cap = json.loads(raw)
    assert cap['sample_id'] == row['sample_id'] and cap['source'] == row['source']
    ev = cap.get('input_evidence', {})
    nod, reaction = ev.get('nod_supplement'), ev.get('reaction_supplement')
    if not reaction or not reaction.get('detector_file'):
        return dict(status='missing_production_evidence', training_labels_admitted=False)
    path = Path(reaction['detector_file'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reaction['detector_sha256'], 'Production detector changed'
    record = json.loads(raw)
    from listen_benchmark.caption_guard import reaction_cache_problems
    assert not reaction_cache_problems(record), 'Unqualified production detector configuration'
    assert record['sample_id'] == row['sample_id'] and record['fps'] == 25
    assert abs(record['duration_s'] - row['duration_s']) < 1e-6
    if record['source'] != row['source']:
        return dict(status='source_differs_requires_review', detector_file=str(path),
                    detector_sha256=reaction['detector_sha256'], training_labels_admitted=False)
    assert record['reaction_cues'] == reaction['cues']
    frames = record['frames']
    assert len(record['pose']['detected']) == frames
    if record['status'] == 'ok':
        assert len(record['scores']) == frames
        assert nod and nod['events'] == [dict(id=x['event_id'], start=x['start'], end=x['end']) for x in record['events']]
    return dict(status='indexed', detector_file=str(path), detector_sha256=reaction['detector_sha256'],
                frames=frames, fps=25, detector_status=record['status'],
                fer_status=record.get('fer_status'),
                supplied_observations=sorted({x['observation'] for x in record['reaction_cues']}),
                retains_nod_scores_and_face_mask=True,
                training_labels_admitted=False,
                reason='Evidence index only; six-channel target assembly is a separate stage')


def identity(job, protocol):
    return dict(schema=VERSION, sample_id=job['sid'], row_sha256=job['row_sha'],
                selection_sha256=protocol['selection_sha256'], model_sha256=protocol['model_sha256'],
                code_sha256=canonical(protocol['code']), fps=25, dtype='float32')


def existing(work, job, protocol):
    payload, receipt = paths(work, job['sid'])
    if not receipt.exists():
        return False
    value = read(receipt)
    assert value['identity'] == identity(job, protocol), 'Existing LIA identity changed'
    assert payload.exists() and sha(payload) == value['payload_sha256'], 'Existing LIA corrupt'
    assert value['frames'] == math.ceil(job['duration'] * 25 - 1e-7)
    return True


def decode(job, protocol):
    row = load_row(job, protocol)
    # Reuse the verified speaker/listener and final-caption contract checks.
    from p2_j_current_qwen_hidden import item_from_row
    from p2_j_cache.data import decode_frames
    item = item_from_row(row)
    try:
        indexed = evidence(row)
    except Exception as exc:
        # LIA does not depend on reaction extraction. Preserve the separate
        # evidence failure explicitly, without failing or admitting its labels.
        indexed = dict(status='evidence_validation_failed', error=repr(exc),
                       training_labels_admitted=False)
    frames, pts = decode_frames(item['source'], fps=25)
    assert frames.shape == (math.ceil(job['duration'] * 25 - 1e-7), 512, 512, 3)
    return row, frames, pts, indexed


def load_lia(protocol):
    import torch
    sys.path.insert(0, str(ROOT / 'related/glare'))
    from float.models.generator import Generator
    assert sha(MODEL) == protocol['model_sha256']
    state = torch.load(MODEL, map_location='cpu', weights_only=True)
    state = {k.removeprefix('motion_autoencoder.'): v for k, v in state.items() if k.startswith('motion_autoencoder.')}
    model = Generator(size=512, style_dim=512, motion_dim=20)
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    return model.eval().requires_grad_(False).cuda()


def encode(model, frames, batch=4):
    import numpy as np
    import torch
    result = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch):
            x = torch.from_numpy(np.array(frames[start:start + batch])).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1
            appearance, _, _ = model.enc(x, input_target=None)
            result.append(model.dec.direction(model.enc.fc(appearance)).cpu())
    motion = torch.cat(result)
    assert motion.dtype == torch.float32 and motion.shape == (len(frames), 512)
    assert torch.isfinite(motion).all()
    return motion


def commit(work, job, protocol, motion, pts, indexed):
    import torch
    payload, receipt = paths(work, job['sid'])
    payload.parent.mkdir(parents=True, exist_ok=True)
    tmp = payload.with_name(payload.name + '.tmp.' + str(os.getpid()))
    torch.save({'motion': motion}, tmp)
    back = torch.load(tmp, map_location='cpu', weights_only=True)['motion']
    assert torch.equal(back, motion)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    tmp.replace(payload)
    save(receipt, dict(identity=identity(job, protocol), payload_sha256=sha(payload),
                       frames=len(motion), shape=list(motion.shape), source_frame_pts_s=pts,
                       production_evidence=indexed, prosody='deferred',
                       full_glare_training_admitted=False, completed_at=time.time()))


def worker(work, gpu, limit=None, ids=None):
    import torch
    import numpy as np
    work = Path(work)
    assert gpu in GPUS and os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    protocol = load_protocol(work)
    lock_path = work / 'workers' / ('gpu' + str(gpu) + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert torch.cuda.mem_get_info()[0] >= 16 * 1024**3, 'Require 16GiB free; preserve other GPU jobs'
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the original feature export's cuDNN TF32 policy. Disabling it here
    # changes the frozen LIA representation despite retaining FP32 tensors.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    db = sqlite3.connect((work / 'jobs.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    jobs = [dict(x) for x in db.execute('SELECT * FROM jobs WHERE worker=? ORDER BY ordinal', (GPUS.index(gpu),))]
    if ids:
        jobs = [dict(db.execute('SELECT * FROM jobs WHERE sid=?', (sid,)).fetchone()) for sid in ids]
    if limit:
        jobs = jobs[:limit]
    db.close()
    model = load_lia(protocol)
    direction = model.dec.direction.weight.detach().cpu()
    dpath = work / ('direction_gpu' + str(gpu) + '.pt')
    if dpath.exists():
        assert torch.equal(torch.load(dpath, weights_only=True, map_location='cpu'), direction)
    else:
        temp = dpath.with_suffix('.tmp.' + str(os.getpid()))
        torch.save(direction, temp)
        temp.replace(dpath)
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    started = time.time()
    done = resumed = failed = frames_done = 0
    status_path = work / 'workers' / ('gpu' + str(gpu) + '.json')
    def status(stage, sid=None):
        save(status_path, dict(stage=stage, gpu=gpu, pid=os.getpid(), total=len(jobs),
                               completed=done, reused=resumed, failed=failed,
                               frames=frames_done, current=sid, started=started,
                               elapsed_s=time.time() - started, heartbeat=time.time(),
                               peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3))
    status('running')
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = []
        next_job = 0
        while next_job < len(jobs) or pending:
            if stopping[0] or (work / 'STOP').exists():
                for _, future in pending:
                    future.cancel()
                status('stopped')
                return
            while next_job < len(jobs) and len(pending) < 2:
                job = jobs[next_job]
                next_job += 1
                if existing(work, job, protocol):
                    resumed += 1
                    continue
                pending.append((job, executor.submit(decode, job, protocol)))
            if not pending:
                continue
            job, future = pending.pop(0)
            status('processing', job['sid'])
            try:
                row, pixels, pts, indexed = future.result()
                motion = encode(model, pixels)
                commit(work, job, protocol, motion, pts, indexed)
                error = work / 'errors' / (job['sid'] + '.json')
                if error.exists():
                    error.unlink()
                done += 1
                frames_done += len(motion)
                del pixels, motion, row
            except Exception as exc:
                failed += 1
                save(work / 'errors' / (job['sid'] + '.json'),
                     dict(sample_id=job['sid'], identity=identity(job, protocol),
                          error=repr(exc), traceback=traceback.format_exc(), at=time.time()))
            status('running')
            print(json.dumps(dict(gpu=gpu, completed=done, reused=resumed, failed=failed,
                                  sample_id=job['sid'], elapsed_s=time.time() - started)), flush=True)
    status('finished' if not failed else 'finished_with_errors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker'])
    parser.add_argument('--work', type=Path, default=DEFAULT)
    parser.add_argument('--gpu', type=int, choices=GPUS)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.work)
    else:
        assert args.gpu is not None
        worker(args.work, args.gpu, args.limit, args.ids)


if __name__ == '__main__':
    main()
