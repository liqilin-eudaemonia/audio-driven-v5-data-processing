"""Recrop the 400 listener clips with the Seamless final crop.

The box is listen_local_recovery.preview_crop (the full-run head-shoulder
proposal), then listen_visual_quality.rules.propose_crop, which is the
quality-bounded-crop-v1 adjustment used as the final training crop.
These videos have no Seamless keypoints, so the 68-point face envelope is
taken from MediaPipe on the listener only.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, "/private/liqi_workspace/project_code/audio-driven-v5/src")
from listen_visual_quality.rules import propose_crop

POLICY = json.loads(Path("/private/liqi_workspace/project_code/audio-driven-v5/configs/listen_visual_quality_v1.json").read_text())
MODEL = "/private/ckpts/mediapipe/face_landmarker.task"
OUT = Path("/public/0_DATA/2_Avatar/Speakervid/realtalk_fullscan/full_front/listener_seamless")
IDX68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107,
    336, 296, 334, 293, 301,
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 82, 13, 312, 308, 317, 14, 87,
]


def preview_crop(points, fps, width, height, start, end):
    """Same geometry as listen_local_recovery.preview_crop."""
    k = points[int(start * fps):int(math.ceil(end * fps)), 23:91]
    valid = ((k[:, :, 2] > .4) & np.isfinite(k).all(axis=2)).sum(axis=1) >= 45
    if not valid.any():
        return dict(status="unmeasured")
    xy = np.where((k[valid, :, 2] > .4)[:, :, None], k[valid, :, :2], np.nan)
    low, high = np.nanpercentile(xy, [3, 97], axis=1)
    fw, fh = np.median(high - low, axis=0)
    left, top = np.percentile(low, 1, axis=0)
    right, chin = np.percentile(high, 99, axis=0)
    head_top = top - .55 * fh
    desired = max(fh / .42, right - left + .5 * fw, (chin - head_top) / .85)
    side = min(float(min(width, height)), float(desired))
    x = float(np.clip(np.median((low[:, 0] + high[:, 0]) / 2) - side / 2, 0, width - side))
    y = float(np.clip(head_top - .06 * side, 0, height - side))
    return dict(status="proposal_only", x=x, y=y, size=side,
                face_width=float(fw), face_height=float(fh),
                fits_estimated_head=bool(x <= left and x + side >= right and y <= head_top and y + side >= chin),
                valid_fraction=float(valid.mean()), head_top=float(head_top),
                boxes=[[float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])] for lo, hi in zip(low, high)])


def hist_score(img, box, ref):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    patch = img[max(0, y1 + int(.15 * bh)):max(0, y2 - int(.15 * bh)), max(0, x1 + int(.2 * bw)):max(0, x2 - int(.2 * bw))]
    rh, rw = ref.shape[:2]
    target = ref[int(.2 * rh):int(.8 * rh), int(.2 * rw):int(.8 * rw)]
    if patch.size == 0 or target.size == 0:
        return -1.0

    def sig(image):
        hsv = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    return float(cv2.compareHist(sig(patch), sig(target), cv2.HISTCMP_CORREL))


def even_box(x, y, size, width, height):
    size = int(size) // 2 * 2
    x = min(max(0, int(x)), width - size) // 2 * 2
    y = min(max(0, int(y)), height - size) // 2 * 2
    if x + size > width:
        x = max(0, (width - size) // 2 * 2)
    if y + size > height:
        y = max(0, (height - size) // 2 * 2)
    return x, y, size


def crop_one(row, landmarker):
    cap = cv2.VideoCapture(row["source_mp4"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    lcap = cv2.VideoCapture(row["listener_mp4"])
    lcap.set(cv2.CAP_PROP_POS_MSEC, max(0, row["file_dur"] * 500))
    ok, ref = lcap.read()
    lcap.release()
    if not ok or not frames:
        return dict(id=row["id"], status="no_frame")
    step = max(1, int(round(fps / 5)))
    picked = frames[::step]

    def faces_of(fr):
        rgb = np.ascontiguousarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        found = []
        for lm in res.face_landmarks:
            xs = [p.x * width for p in lm]
            ys = [p.y * height for p in lm]
            box = (min(xs), min(ys), max(xs), max(ys))
            found.append((hist_score(fr, box, ref), (box[0] + box[2]) / 2 / width, lm))
        return found

    mid = faces_of(picked[len(picked) // 2])
    if not mid:
        return dict(id=row["id"], status="no_face")
    anchor = max(mid, key=lambda item: item[0])
    if anchor[0] < 0.4:
        return dict(id=row["id"], status="listener_not_matched", match_score=round(anchor[0], 3))
    pts, scores = [], []
    for fr in picked:
        choice = None
        for sc, cx, lm in faces_of(fr):
            if abs(cx - anchor[1]) > 0.18:
                continue
            if choice is None or abs(cx - anchor[1]) < abs(choice[0] - anchor[1]):
                choice = (cx, sc, lm)
        rowp = np.zeros((91, 3), np.float32)
        if choice is not None:
            scores.append(choice[1])
            for j, ix in enumerate(IDX68):
                p = choice[2][ix]
                rowp[23 + j] = (p.x * width, p.y * height, 1.0)
        pts.append(rowp)
    arr = np.stack(pts)
    prop = preview_crop(arr, 5.0, width, height, 0, len(pts) / 5.0)
    if prop["status"] != "proposal_only":
        return dict(id=row["id"], status=prop["status"])
    x, y, size = [math.floor(prop[k] + .5) for k in ("x", "y", "size")]
    raw = [dict(detected=True, bbox_native=b) for b in prop["boxes"]]
    base = dict(x=x, y=y, width=size, height=size)
    adjusted = propose_crop(raw, dict(video_transform=dict(crop=base, source_size=[width, height])), POLICY)
    final = adjusted.get("new", base)
    x, y, size = even_box(final["x"], final["y"], final["width"], width, height)
    if size < 64:
        return dict(id=row["id"], status="tiny_crop")
    out = OUT / f"{row['id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-nostdin", "-v", "error", "-i", row["source_mp4"],
        "-vf", f"crop={size}:{size}:{x}:{y},scale=512:512:flags=bicubic",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ], check=True)
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)
    ], text=True))
    return dict(id=row["id"], status="ok", path=str(out), file_dur=round(dur, 3),
                crop_x=x, crop_y=y, crop_size=size, fits_estimated_head=prop["fits_estimated_head"],
                valid_fraction=round(prop["valid_fraction"], 3),
                crop_adjust=adjusted.get("status"), match_score=round(float(np.median(scores)), 3) if scores else None)


def worker(rows):
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2)
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    done = []
    for n, row in enumerate(rows, 1):
        try:
            rec = crop_one(row, landmarker)
        except Exception as exc:
            rec = dict(id=row["id"], status="error", error=str(exc)[:300])
        done.append(rec)
        if n % 5 == 0:
            print(f"shard {rows[0]['id'][:12]} {n}/{len(rows)}", flush=True)
    landmarker.close()
    return done


def main():
    rows = [json.loads(line) for line in Path(sys.argv[1]).open()]
    parts = int(sys.argv[2])
    index = int(sys.argv[3])
    mine = rows[index::parts]
    out = Path(sys.argv[4])
    recs = worker(mine)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("DONE", index, len(recs), flush=True)


if __name__ == "__main__":
    main()
