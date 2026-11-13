"""Diagnose original serial numerical repeatability; never changes scoring."""
from pathlib import Path
import sys
import time

AUDIT = Path(__file__).resolve().parent
WORK = AUDIT.parent
ROOT = WORK.parents[1]
sys.path[:0] = [str(WORK), str(ROOT / 'scripts'), str(ROOT / 'compare/deployment')]
from p2_avatarforcing_table4_generate import guard, read, save, sha


def main():
    import cv2
    import numpy as np
    import torch
    import official_spectre as source
    guard(.08, 0)
    cv2.setNumThreads(1)
    model = source.Encoder('cuda')
    tracker, get_landmarks = source.load_frontend()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    uid = 'ClNUxPsUByg_000210_000221.0'
    receipt = read(WORK / 'official_features/gt' / (uid+'.json'))
    frames, _ = source.decode25(receipt['video'], receipt['frames'])
    with np.load(receipt['path']) as data:
        ref_motion, ref_landmarks = data['motion'].copy(), data['landmarks'].copy()
    rows = []
    previous = None
    for repeat in range(2):
        started = time.time()
        images, landmarks, audit = source.official_crop(frames, tracker, get_landmarks)
        motion = model.chunks(images, batch=46)
        output = AUDIT / f'numeric_repeat_{repeat}.npz'
        np.savez_compressed(output, motion=motion, landmarks=landmarks)
        error_m = np.max(np.abs(motion-ref_motion), axis=1)
        error_l = np.max(np.abs(landmarks-ref_landmarks), axis=(1, 2))
        row = dict(repeat=repeat, motion_max_abs=float(error_m.max()),
            landmarks_max_abs=float(error_l.max()),
            motion_different_frames=np.flatnonzero(error_m>0).tolist(),
            landmarks_different_frames=np.flatnonzero(error_l>0).tolist(),
            per_frame_motion_max_abs=error_m.tolist(), per_frame_landmarks_max_abs=error_l.tolist(),
            jpeg_identical=audit['jpeg_sequence_sha256']==receipt['face_audit']['jpeg_sequence_sha256'],
            output_sha256=sha(output), seconds=time.time()-started,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        if previous is not None:
            row['same_process_motion_max_abs']=float(np.abs(motion-previous[0]).max())
            row['same_process_landmarks_max_abs']=float(np.abs(landmarks-previous[1]).max())
        previous=(motion, landmarks)
        rows.append(row)
        print({k:v for k,v in row.items() if not isinstance(v, list)}, flush=True)
        save(AUDIT / 'NUMERIC_REPEAT.json', dict(rows=rows,
            comparison='Same original serial frontend and8% allocator; independent process vs sealed GT, then same-process repeat',
            implementation_sha256=sha(__file__), no_AF_table_metrics_computed=True))


if __name__ == '__main__':
    main()
