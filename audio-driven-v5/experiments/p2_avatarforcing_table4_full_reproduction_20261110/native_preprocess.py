"""Restore original-view ViCo inputs with the official AF crop implementation."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REPO = ROOT / 'compare/AvatarForcing'
sys.path.insert(0, str(ROOT / 'scripts'))
from p2_avatarforcing_table4_generate import extract_methods, read, save, sha


class TraceDetector:
    def __init__(self, detector):
        self.detector = detector
        self.rows = []

    def detect_from_image(self, image):
        boxes = self.detector.detect_from_image(image)
        self.rows.append([list(map(float, box)) for box in boxes])
        return boxes


def source_rectangle(rows, height, width):
    """Mirror the official integer arithmetic, retaining its observable trace."""
    import numpy as np
    mult = 360. / height
    measurements = []
    for row in rows:
        x1, y1, x2, y2, _ = next(b for b in row if b[4] > .95)
        x1, y1, x2, y2 = [int(v / mult) for v in [x1, y1, x2, y2]]
        measurements.append([int((y2-y1)/2), int((x2-x1)/2), int((x1+x2)/2), int((y1+y2)/2)])
    bsy, bsx, mx, my = np.mean(measurements, axis=0)
    bs = int(max(bsy, bsx) * 2)
    x1, y1, x2, y2 = max(mx-bs, 0), max(my-bs, 0), min(mx+bs, width), min(my+bs, height)
    bsx, bsy = x2-x1, y2-y1
    mx, my = int(x1 + bsx // 2), int(y1 + bsy // 2)
    bs = int(min(bsx, bsy) // 2)
    return [mx-bs, my-bs, mx+bs, my+bs]


def prepare_role(uid, role, original, duration, detector):
    import cv2
    import numpy as np
    dest = WORK / 'inputs' / uid / role
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        result = read(dest / 'complete.json')
        for path, digest in result['files'].items():
            assert sha(path) == digest, path
        return result
    started = time.time()
    raw, cropped = dest / 'raw25', dest / 'cropped25'
    raw.mkdir(exist_ok=True)
    cropped.mkdir(exist_ok=True)
    # Explicit frame-rate conversion: the public helper's --fps argument is
    # unused during ffmpeg extraction. No empirically chosen temporal offset.
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1',
                    '-i', original['path'], '-t', str(duration), '-vf', 'fps=25',
                    '-frames:v', str(round(duration*25)), '-qscale:v', '0',
                    '-threads', '1', str(raw / '%05d.jpg')], check=True)
    paths = sorted(raw.glob('*.jpg'))
    assert len(paths) == round(duration * 25), (uid, role, len(paths), duration)
    traced = TraceDetector(detector.face_detector)
    crop_cls = extract_methods(REPO / 'preprocess_user_video.py', 'UserVideoProcessor',
        ['img_loader', 'dectect_and_crop_face_from_frames'], dict(cv2=cv2, np=np, Path=Path, os=os))
    cropper = crop_cls()
    cropper.fa = SimpleNamespace(face_detector=traced)
    cropper.opt = SimpleNamespace(input_size=512)
    cropper.dectect_and_crop_face_from_frames(str(raw), str(cropped), pad_ratio=1.0)
    first = cv2.imread(str(paths[0]))
    height, width = first.shape[:2]
    rectangle = source_rectangle(traced.rows, height, width)
    save(dest / 'detection_trace.json', dict(boxes=traced.rows, source_rectangle=rectangle,
        source_size=[width, height], fps=25, frames=len(paths)))
    # Preserve the decoded/cropped GT pixels without a further lossy codec.
    video = dest / 'cropped25.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1',
                    '-framerate', '25', '-i', str(cropped / '%05d.jpg'),
                    '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-threads', '1', str(video)], check=True)
    files = [video, dest / 'detection_trace.json', *sorted(cropped.glob('*.jpg'))]
    if role == 'listener':
        # The generation reference uses a single source frame, once. It is not
        # recropped from the mean-cropped GT video and uses no future imagery.
        cap = cv2.VideoCapture(original['path'])
        ok, frame = cap.read()
        cap.release()
        assert ok
        cv2.imwrite(str(dest / 'first_original.png'), frame)
        ref_cls = extract_methods(REPO / 'inference.py', 'DataProcessor',
            ['default_img_loader', 'preprocess_face'], dict(cv2=cv2, np=np))
        ref = ref_cls()
        ref.fa, ref.opt = detector, SimpleNamespace(input_size=512)
        rgb = ref.preprocess_face(str(dest / 'first_original.png'), pad_ratio=1.0)
        cv2.imwrite(str(dest / 'reference.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        files += [dest / 'first_original.png', dest / 'reference.png']
    result = dict(id=uid, role=role, original=original, duration=duration,
                  fps=25, frames=len(paths), source_rectangle=rectangle,
                  elapsed_seconds=time.time()-started, official_mean_crop=True,
                  source_code_sha256=sha(REPO / 'preprocess_user_video.py'),
                  files={str(p): sha(p) for p in files})
    save(dest / 'complete.json', result)
    print(uid, role, rectangle, 'frames', len(paths), 'seconds', round(time.time()-started, 2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['probe', 'all'])
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    gpu = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=uuid,memory.free,memory.total',
                                   '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert gpu[0].strip() == 'GPU-4d5ee808-e829-c824-c506-cad80ff133f5'
    required = math.ceil(int(gpu[2])*.08)+16384
    assert int(gpu[1]) >= required, (gpu, required)
    import cv2
    import torch
    import face_alignment
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.08)
    torch.backends.cudnn.deterministic = True
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')
    manifest = read(WORK / 'ORIGINAL_TEST43_MANIFEST.json')['records']
    sources = {(r['id'], r['role']): r for r in manifest}
    samples = read(WORK.parent / 'p2_avatarforcing_table4_recheck_20261110/protocol.json')['samples']
    if args.action == 'probe':
        samples = [s for s in samples if s['id'] in ['ClNUxPsUByg_000210_000221.0', 'Ew2z_sYABE0_000063_000083.0']]
    records = []
    for sample in samples:
        for role in (['speaker'] if args.action == 'probe' else ['speaker', 'listener']):
            original = sources[sample['id'], role]
            assert sha(original['path']) == original['sha256']
            records.append(prepare_role(sample['id'], role, original, sample['duration'], detector))
            save(WORK / 'PREPROCESS_STATUS.json', dict(action=args.action, completed_this_invocation=len(records),
                total_this_invocation=len(samples)*(1 if args.action == 'probe' else 2),
                last_id=sample['id'], last_role=role, pid=os.getpid(),
                torch=torch.__version__, face_alignment=face_alignment.__version__,
                benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    save(WORK / ('CROP_PROBE.json' if args.action == 'probe' else 'PREPROCESS_COMPLETE.json'),
         dict(passed=True, roles=len(records), records=records, script_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
