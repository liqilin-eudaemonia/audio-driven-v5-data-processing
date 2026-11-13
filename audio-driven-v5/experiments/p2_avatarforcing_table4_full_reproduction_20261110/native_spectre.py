"""Separate native-view SPECTRE outputs; sealed earlier runs stay read-only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import read, save, sha, guard
from p2_spectre_coefficients import Encoder, crop_frames, decode25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'all'])
    args = parser.parse_args()
    import cv2
    import numpy as np
    import torch
    import face_alignment
    guard(.08, 32768)
    cv2.setNumThreads(1)
    model = Encoder('cuda')  # Reuses the sealed, hash-checked checkpoint only.
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', flip_input=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = read(WORK / 'NATIVE_GENERATION_PROTOCOL.json')
    samples = p['samples']
    if args.action == 'probe':
        samples = [s for s in samples if s['id'] in ['ClNUxPsUByg_000210_000221.0', 'Ew2z_sYABE0_000063_000083.0']]
    else:
        assert read(WORK / 'NATIVE_GENERATION_COMPLETE.json')['clips'] == 43
    tasks = []
    for s in samples:
        uid = s['id']
        generated = WORK / 'native_generated' / uid
        assert sha(generated / 'generated.mp4') == read(generated / 'complete.json')['sha256']
        for role, video in [('speaker', WORK / 'inputs' / uid / 'speaker/cropped25.mkv'),
                            ('gt', WORK / 'inputs' / uid / 'listener/cropped25.mkv'),
                            ('avatarforcing', generated / 'generated.mp4')]:
            tasks.append(dict(id=uid, role=role, video=str(video), video_sha256=sha(video),
                              frames=round(s['duration']*25)))
    # Keep one invariant extraction contract regardless of probe/all task count.
    contract = dict(schema='af-native-original-view-spectre-v1', encoder=model.info,
        source_sha256=sha(ROOT / 'scripts/p2_spectre_coefficients.py'), script_sha256=sha(__file__),
        generation_protocol_sha256=sha(WORK / 'NATIVE_GENERATION_PROTOCOL.json'),
        temporal='2frame edge replication and context around each32 valid frames',
        landmark_backend='face_alignment1.4.1 SFD/2DFAN first-face; not SPECTRE tracking fork',
        precision='FP32; benchmarkFalse deterministicTrue TF32False', fps=25, batch=32,
        feature_order='pose6 then exp50', full_declared_interval=True, source_time_offset=0)
    path = WORK / 'NATIVE_SPECTRE_PROTOCOL.json'
    if path.exists():
        assert read(path) == contract
    else:
        save(path, contract)
    records = []
    for task in tasks:
        folder = WORK / 'native_features' / task['role']
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / (task['id'] + '.npz')
        receipt = output.with_suffix('.json')
        if receipt.exists():
            r = read(receipt)
            assert r['protocol_sha256'] == sha(path)
            assert r['video_sha256'] == task['video_sha256'] and sha(output) == r['sha256']
            records.append(r)
            continue
        frames, grid = decode25(task['video'], task['frames'])
        images, audit = crop_frames(frames, detector)
        if not (WORK / 'NATIVE_SPECTRE_PARITY.json').exists():
            save(WORK / 'NATIVE_SPECTRE_PARITY.json', model.parity(images[:8]))
        values = model.chunks(images)
        assert values.shape == (task['frames'], 56) and np.isfinite(values).all()
        np.savez_compressed(output, motion=values, fps=25, frame_indices=np.arange(len(values)))
        r = dict(**task, path=str(output), sha256=sha(output), protocol_sha256=sha(path),
                 grid=grid, face_audit=audit)
        save(receipt, r)
        records.append(r)
        save(WORK / 'NATIVE_SPECTRE_STATUS.json', dict(stage='spectre', action=args.action,
            completed=len(list((WORK / 'native_features').glob('*/*.json'))), total=129,
            last_id=task['id'], last_role=task['role'], pid=os.getpid()))
        print('SPECTRE native', len(records), '/', len(tasks), task['role'], task['id'], flush=True)
        del frames, images, values
    save(WORK / ('NATIVE_SPECTRE_PROBE.json' if args.action == 'probe' else 'NATIVE_EXTRACTION_COMPLETE.json'),
         dict(passed=True, streams=len(records), records=records, protocol_sha256=sha(path)))


if __name__ == '__main__':
    main()
