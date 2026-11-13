"""Check numerical parity before considering batched face detection/FAN."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from p2_avatarforcing_table4_generate import WORK, guard, read, save
from p2_spectre_coefficients import decode25


def batch_landmarks(frames, detector, batch=8):
    import numpy as np
    import torch
    from face_alignment.utils import crop, get_preds_fromhm
    answers = []
    counts = []
    assert not detector.flip_input
    with torch.inference_mode():
        for start in range(0, len(frames), batch):
            images = frames[start:start + batch]
            # Keep the detector's exact per-image proposal/NMS path. Its public
            # batch API changes the proposal set; that candidate failed parity.
            boxes = [detector.face_detector.detect_from_image(image.copy()) for image in images]
            crops, centres, scales, active = [], [], [], []
            local = [None] * len(images)
            counts.extend(len(box) for box in boxes)
            for idx, (image, faces) in enumerate(zip(images, boxes)):
                if not len(faces):
                    continue
                d = faces[0]
                centre = torch.tensor([d[2] - (d[2] - d[0]) / 2., d[3] - (d[3] - d[1]) / 2.])
                centre[1] = centre[1] - (d[3] - d[1]) * .12
                scale = (d[2] - d[0] + d[3] - d[1]) / detector.face_detector.reference_scale
                crops.append(crop(image, centre, scale).transpose(2, 0, 1))
                centres.append(centre)
                scales.append(scale)
                active.append(idx)
            if crops:
                inp = torch.from_numpy(np.stack(crops)).float().to(detector.device) / 255.
                out = detector.face_alignment_net(inp).detach().cpu().numpy()
                for j, idx in enumerate(active):
                    _, pts, _ = get_preds_fromhm(out[j:j + 1], centres[j].numpy(), scales[j])
                    local[idx] = pts.reshape(68, 2)
            answers.extend(local)
    return answers, counts


def main():
    import cv2
    import numpy as np
    import torch
    import face_alignment
    device = guard(.04, 32768)
    cv2.setNumThreads(1)
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda', flip_input=False)
    samples = read(WORK / 'protocol.json')['samples']
    s = samples[0]
    count = read(WORK / 'generated' / s['id'] / 'complete.json')['frames']
    frames, _ = decode25(s['gt_video'], count)
    frames = frames[np.linspace(0, len(frames) - 1, 16).astype(int)]
    detector.get_landmarks_from_image(frames[0])
    torch.cuda.synchronize()
    started = time.monotonic()
    original = [detector.get_landmarks_from_image(frame) for frame in frames]
    torch.cuda.synchronize()
    serial = time.monotonic() - started
    started = time.monotonic()
    batched, counts = batch_landmarks(frames, detector)
    torch.cuda.synchronize()
    batch_time = time.monotonic() - started
    errors = []
    same_counts = True
    for old, new, count in zip(original, batched, counts):
        old_count = len(old) if old is not None else 0
        same_counts &= old_count == count
        if old_count:
            errors.append(float(np.abs(old[0] - new).max()))
        else:
            assert new is None
    result = dict(device=device, frames=len(frames), serial_s=serial, batched_s=batch_time,
                  batching='FAN only; original per-image SFD proposals',
                  speedup=serial / batch_time, same_face_counts=bool(same_counts),
                  max_abs_landmark_delta=max(errors), exact_coordinates=all(e == 0 for e in errors),
                  peak_memory_gib=torch.cuda.max_memory_reserved() / 1024**3,
                  candidate_only=True, production_changed=False)
    save(WORK / 'LANDMARK_BATCH_BENCHMARK.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
