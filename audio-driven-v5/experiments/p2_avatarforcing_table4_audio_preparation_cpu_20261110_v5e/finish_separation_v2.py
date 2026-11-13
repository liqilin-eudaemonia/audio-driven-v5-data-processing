"""Independently audit the full separation artifact, then exit without AF work."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def state(pid):
    p = Path('/proc')/str(pid)/'stat'
    if not p.exists():
        return None
    return p.read_text().rsplit(')', 1)[1].split()[0]


def metrics(mix, estimate):
    import numpy as np
    x = mix.astype(np.float64)
    y = estimate.astype(np.float64)
    rms_x = float(np.sqrt(np.mean(x*x)))
    rms_y = float(np.sqrt(np.mean(y*y)))
    centered_x = x-x.mean()
    centered_y = y-y.mean()
    nx = float(np.linalg.norm(centered_x))
    ny = float(np.linalg.norm(centered_y))
    dot = float(centered_x@centered_y)
    alpha = dot/(ny*ny) if ny*ny > 1e-15 else None
    return dict(mix_rms=rms_x, estimate_rms=rms_y,
        gain_db=20*float(np.log10(max(rms_y,1e-15)/max(rms_x,1e-15))),
        centered_correlation=dot/(nx*ny) if nx*ny > 1e-15 else None,
        scale_invariant_residual_rms_ratio=float(np.linalg.norm(centered_x-alpha*centered_y)/nx) if alpha is not None and nx>0 else None,
        raw_abs_peak=float(np.max(np.abs(estimate))),
        samples_over_pcm16_range=int(np.count_nonzero(np.abs(estimate)>1)), samples=len(mix))


def audit():
    import numpy as np
    import soundfile as sf
    p = read(WORK/'PROTOCOL.json')
    full = read(WORK/'FULL_SEPARATION_COMPLETE.json')
    assert full['passed'] and full['pairs']==43 and full['streams']==86
    assert full['parent_protocol_sha256']==digest(WORK/'PROTOCOL.json')
    assert full['parallel_protocol_sha256']==digest(WORK/'PARALLEL_PROTOCOL.json')
    assert full['parity_sha256']==digest(WORK/'PARALLEL_PARITY.json')
    assert read(WORK/'PARALLEL_PARITY.json')['passed']
    for path, expected in p['source_and_weights'].items():
        assert digest(path)==expected, path
    scheduler = read(WORK/'PARALLEL_PROTOCOL.json')
    assert digest(WORK/'parallel.py')==scheduler['script_sha256']
    assert digest(WORK/'separate.py')==scheduler['helper_sha256']
    assert scheduler['GPU_used'] is False and scheduler['workers']==8
    manager = read(WORK/'MANAGER_EXIT.json')
    assert manager['returncode']==0 and state(manager['pid']) in [None, 'Z'], 'Manager must exit before independent final audit'
    live_state = read(WORK/'PARALLEL_STATUS.json')
    assert live_state['stage']=='complete'
    metadata_path = Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv')
    metadata = {r['uuid']:r for r in csv.DictReader(metadata_path.open()) if r['data_split']=='test'}
    assert set(metadata)=={s['id'] for s in p['samples']}
    generation = read(WORK.parent/'p2_avatarforcing_table4_reference_mean_20261110_v3/NATIVE_GENERATION_PROTOCOL.json')
    old_samples = {s['id']:s for s in generation['samples']}
    records = {(r['id'],r['role']):r for r in full['records']}
    assert len(records)==86
    rows = []
    for sample in p['samples']:
        uid = sample['id']
        mix, sr = sf.read(sample['mix'], dtype='float32')
        assert digest(sample['mix'])==sample['mix_sha256']
        assert sr==16000 and mix.shape==(sample['samples'],) and np.isfinite(mix).all()
        for role in sample['roles']:
            name = role['role']
            old_key = 'speaker_video' if name=='speaker' else 'gt_video'
            # by_split files are role aliases named .speaker/.listener;
            # metadata stores the identity of their actual video target.
            assert Path(old_samples[uid][old_key]).resolve().stem==metadata[uid][name]
            entry = records[uid,name]
            assert digest(entry['receipt'])==entry['receipt_sha256']
            receipt = read(entry['receipt'])
            assert receipt['id']==uid and receipt['role']==name
            assert receipt['video']==role['video'] and receipt['video_sha256']==role['video_sha256']
            assert digest(role['video'])==role['video_sha256']
            assert receipt['mix_sha256']==sample['mix_sha256']
            folder = Path(entry['receipt']).parent
            shard = folder.parents[2]
            assert digest(shard/'PROTOCOL.json')==receipt['protocol_sha256']
            runtime_path = shard/'runtime_history'/(receipt['runtime_sha256']+'.json')
            assert digest(runtime_path)==receipt['runtime_sha256']
            runtime = read(runtime_path)
            assert state(runtime['pid']) in [None,'Z'], 'Separation worker must exit'
            assert runtime['tracking_device']=='cpu' and runtime['global_deterministic']
            assert runtime['cublas_workspace']==':4096:8'
            device = runtime['device']
            assert device['device']=='cpu' and device['GPU_used'] is False
            assert device['threads']==8 and device['MKLDNN_deterministic']
            assert receipt['execution_device']=='cpu' and receipt['cuda_initialized'] is False
            for path, expected in receipt['files'].items():
                assert digest(path)==expected, path
            raw = np.load(folder/'estimate.npy', allow_pickle=False)
            wav, rate = sf.read(folder/'target.wav', dtype='float32')
            assert rate==16000 and raw.shape==wav.shape==mix.shape
            assert raw.dtype==np.float32 and np.isfinite(raw).all() and np.isfinite(wav).all()
            info = sf.info(folder/'target.wav')
            assert info.channels==1 and info.subtype==receipt['subtype']=='FLOAT'
            serialization_error = float(np.abs(raw-wav).max())
            assert serialization_error<=1e-6, serialization_error
            for key, waveform in [('raw_audio_metrics',raw),('saved_audio_metrics',wav)]:
                values = metrics(mix,waveform)
                for field,value in values.items():
                    expected = receipt[key][field]
                    assert (expected is None and value is None) or (expected is not None and value is not None and abs(expected-value)<1e-7), (uid,name,key,field,expected,value)
            with np.load(folder/'mouthroi/speaker1.npz',allow_pickle=False) as data:
                roi = data['data']
            frames = round(sample['samples']/16000*25)
            assert roi.shape==(frames,96,96) and np.isfinite(roi).all()
            assert receipt['roi_frames']==receipt['source_video_frames']==frames
            with np.load(folder/'landmark/speaker1.npz',allow_pickle=True) as data:
                landmarks = data['data']
            assert len(landmarks)==frames
            missing = sum(points is None for points in landmarks)
            assert missing < frames, 'No valid original landmarks'
            row = dict(id=uid, role=name, official_person_id=metadata[uid][name],
                target_wav=entry['target_wav'], target_wav_sha256=digest(entry['target_wav']),
                raw_npy=str(folder/'estimate.npy'), raw_npy_sha256=digest(folder/'estimate.npy'),
                receipt=entry['receipt'], receipt_sha256=entry['receipt_sha256'],
                seconds=sample['samples']/16000, roi_frames=frames,
                source_missing_landmarks=missing, roi_standard_deviation=float(np.std(roi,dtype=np.float64)),
                float_WAV_max_abs_error=serialization_error, execution_device='cpu', peak_allocated_mib=receipt['peak_allocated_mib'],
                **metrics(mix,wav))
            rows.append(row)
    assert len(rows)==86 and len({(r['id'],r['role']) for r in rows})==86
    result = dict(passed=True, pairs=43, streams=86, sample_rate=16000, output_dtype='float32 WAV / FLOAT subtype',
        metadata_sha256=digest(metadata_path), protocol_sha256=digest(WORK/'PROTOCOL.json'),
        source_complete_sha256=digest(WORK/'FULL_SEPARATION_COMPLETE.json'),
        scheduler_protocol_sha256=digest(WORK/'PARALLEL_PROTOCOL.json'),
        audit_implementation_sha256=digest(__file__), completed_unix=time.time(), records=rows,
        all_separation_workers_terminal=True, GPU_used=False, no_AF_generation_or_scoring_launched=True,
        limitation='No isolated target-voice ground truth or manually verified separation quality; source/weights, target mappings, execution reproducibility and artifact integrity checked. This is not a Table4 score or verified author-private audio recipe.')
    save(WORK/'SEPARATION_INDEPENDENT_AUDIT.json', result)
    with (WORK/'separated_audio_manifest.jsonl').open('w') as out:
        for row in rows:
            out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    lines = ['# IIANet音源分离完成与停止记录','',
        '43对／86路已完成：每对保留speaker和listener两条16kHz单声道FLOAT WAV及原始float32数组。整批在CPU上统一执行，不混用前序GPU部分输出。',
        '已独立核验官方test43身份映射、输入及权重哈希、全部采样数、嘴部帧数、有限数值、WAV序列化误差和处理进程退出。',
        '按用户要求，本轮到音源分离为止；没有启动新的AF生成或评分。', '',
        '[完整清单](separated_audio_manifest.jsonl) · [独立验收](SEPARATION_INDEPENDENT_AUDIT.json) · [处理协议](PROTOCOL.json)', '',
        '执行一致性和文件验收通过不代表已证明分离质量等同于作者，也不代表复现Table4；没有独立干净人声真值，不能给出有监督分离质量结论。', '',
        '| 角色 | 数量 | 总时长（秒） | 最大WAV误差 | 峰值分配显存（MiB） |',
        '|---|---:|---:|---:|---:|']
    for role in ['speaker','listener']:
        part = [r for r in rows if r['role']==role]
        lines.append(f"| {role} | {len(part)} | {sum(r['seconds'] for r in part):.4f} | {max(r['float_WAV_max_abs_error'] for r in part):.9g} | {max(r['peak_allocated_mib'] for r in part):.1f} |")
    (WORK/'SEPARATION_COMPLETE.md').write_text('\n'.join(lines)+'\n')
    save(WORK/'COMPLETED_AND_STOPPED.json', dict(passed=True, stopped_at='audio separation',
        user_instruction='把这个音源分离做好就停下来', pairs=43, streams=86,
        audit_sha256=digest(WORK/'SEPARATION_INDEPENDENT_AUDIT.json'),
        manifest_sha256=digest(WORK/'separated_audio_manifest.jsonl'),
        no_followup_AF_jobs=True, completed_unix=time.time()))
    print('43 pairs / 86 audio streams audited; stop after separation', flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['audit'])
    args=parser.parse_args()
    try:
        audit()
        save(WORK/'FINISH_STATUS.json',dict(stage='completed_and_stopped',pid=os.getpid(),audit_revision=2,updated_unix=time.time()))
    except BaseException as error:
        save(WORK/'FINISH_STATUS.json',dict(stage='failed',pid=os.getpid(),error=repr(error),updated_unix=time.time()))
        raise
