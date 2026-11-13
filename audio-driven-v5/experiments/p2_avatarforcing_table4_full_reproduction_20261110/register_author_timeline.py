"""Match real speaker pixels, using sequential decode to avoid seek ambiguity."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
DIAG = WORK.parent / 'p2_avatarforcing_table4_diagnosis_20261110'


def descriptor(frame):
    # Exclude the published title and panel edges. This is image registration,
    # independent of generated motion, coefficient scoring, and table targets.
    gray = cv2.cvtColor(frame[100:440, 80:440], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), .7).astype(np.float32)
    return (gray - gray.mean()) / max(float(gray.std()), 1)


def read_all(path, rect=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rect is not None:
            x1, y1, x2, y2 = rect
            frame = cv2.resize(frame[y1:y2, x1:x2], (512, 512), interpolation=cv2.INTER_AREA)
        frames.append(descriptor(frame))
    cap.release()
    return fps, np.stack(frames)


def main():
    cv2.setNumThreads(2)
    manifest = json.loads((WORK / 'ORIGINAL_TEST43_MANIFEST.json').read_text())['records']
    results = []
    cases = [
        ('03_listening_01', 'ClNUxPsUByg_000210_000221.0', [116, 91, 820, 795]),
        ('03_listening_02', 'Ew2z_sYABE0_000063_000083.0', [296, 241, 650, 595]),
    ]
    for base, uid, rect in cases:
        record = next(r for r in manifest if r['id'] == uid and r['role'] == 'speaker')
        sfps, source = read_all(record['path'], rect)
        pfps, published = read_all(DIAG / 'published_supplement' / (base + '_speaker.mp4'))
        probes = []
        for p in range(0, len(published), 5):
            expected = p * sfps / pfps
            indices = np.arange(max(0, round(expected) - 7), min(len(source), round(expected) + 8))
            errors = np.mean(np.abs(source[indices] - published[p]), axis=(1, 2))
            order = np.argsort(errors)
            best = int(indices[order[0]])
            probes.append(dict(published_index=p, best_source_index=best,
                               expected_source_index=expected,
                               offset=best - expected,
                               minimum_error=float(errors[order[0]]),
                               second_best_error=float(errors[order[1]]),
                               candidates=[dict(index=int(i), error=float(e)) for i, e in zip(indices, errors)]))
        result = dict(id=uid, rect=rect, source_fps=sfps, published_fps=pfps,
                      source_frames=len(source), published_frames=len(published), probes=probes,
                      median_offset=float(np.median([p['offset'] for p in probes])),
                      median_error=float(np.median([p['minimum_error'] for p in probes])))
        results.append(result)
        print(uid, 'offset', result['median_offset'], 'error', result['median_error'], flush=True)
    output = dict(method='Sequential decode; normalized blurred central grayscale pixel matching; source-image comparison only',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=results)
    (WORK / 'AUTHOR_TIMELINE_REGISTRATION.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
