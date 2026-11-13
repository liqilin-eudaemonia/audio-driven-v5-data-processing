"""Reconstruct identity, expression and pose on the 512 ViCo test faces.

Each frame is aligned to 224, which is what Deep3DFaceRecon reads. Saved
transform_params keep the 512 canvas. Placement numbers are not compared with
the official 256 crop loss and are not sent to PIRender.

Workers are pinned to the requested GPUs. Several processes share each GPU so
the small landmark and reconstruction networks do not leave the card idle.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECON_ROOT = REPO / "evaluation/vico_challenge_baseline/Deep3DFaceRecon_pytorch"
INPUT_DIR = Path("/public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/faces_512")
OUTPUT_DIR = Path("/public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/faces_512_recon")
CHECKPOINT = Path("/public/0_DATA/2_Avatar/ViCo/epoch_20.pth")
RECON_BATCH = 64


def align_frame(image, landmarks, lm3d):
    """Same placement as align_img. Newer NumPy rejects that function's last line."""
    from util.preprocess import POS, extract_5p, resize_n_crop_img

    width, height = image.size
    points = landmarks if landmarks.shape[0] == 5 else extract_5p(landmarks)
    translation, scale = POS(points.transpose(), lm3d.transpose())
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    scale = np.float64(102.0 / float(np.asarray(scale).reshape(-1)[0]))
    aligned, _, _ = resize_n_crop_img(image, landmarks, translation, scale, target_size=224.0)
    params = np.array([width, height, scale, translation[0], translation[1]], dtype=np.float32)
    return params, aligned


def landmarks_for_frames(detector, frames):
    """One face per frame, the same call the baseline landmark extractor uses."""
    found = []
    for frame in frames:
        detected = detector.get_landmarks_from_image(frame)
        found.append(None if not detected else detected[0].astype(np.float32))
    return found


def reconstruct_video(path, detector, net, lm3d, out_path, device):
    import cv2
    import torch
    from PIL import Image
    from scipy.io import savemat

    capture = cv2.VideoCapture(str(path))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError("no frames")

    found = landmarks_for_frames(detector, frames)
    images, params = [], []
    missed = 0
    width = height = None
    for frame, landmarks in zip(frames, found):
        image = Image.fromarray(frame)
        width, height = image.size
        if landmarks is None:
            missed += 1
            landmarks = (lm3d[:, :2] + 1) / 2.0
            landmarks = np.concatenate([landmarks[:, :1] * width, landmarks[:, 1:2] * height], 1)
        else:
            landmarks = landmarks.copy()
            landmarks[:, -1] = height - 1 - landmarks[:, -1]
        transform, aligned = align_frame(image, landmarks, lm3d)
        images.append(np.asarray(aligned, dtype=np.float32) / 255.0)
        params.append(transform)

    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    pieces = []
    with torch.no_grad():
        for start in range(0, batch.shape[0], RECON_BATCH):
            pieces.append(net(batch[start:start + RECON_BATCH].to(device)).cpu().numpy())
    coeff = np.concatenate(pieces, 0)
    transform = np.stack(params)
    if coeff.shape != (len(frames), 257) or transform.shape != (len(frames), 5):
        raise RuntimeError(f"unexpected shapes {coeff.shape} {transform.shape}")
    if not np.all(transform[:, 0] == width) or not np.all(transform[:, 1] == height):
        raise RuntimeError("transform canvas does not match the input frame")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(out_path, {"coeff": coeff, "transform_params": transform})
    return {"video": path.name, "frames": len(frames), "missed": missed, "width": int(width), "height": int(height)}


def claim(pending, inflight):
    for path in sorted(pending.iterdir()):
        if not path.is_file():
            continue
        target = inflight / path.name
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        return target
    return None


def run_worker(gpu, slot, output_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(RECON_ROOT))
    import torch
    import face_alignment
    from models.networks import ReconNetWrapper
    from util.load_mats import load_lm3d

    device = "cuda"
    lm3d = load_lm3d(str(RECON_ROOT / "BFM"))
    net = ReconNetWrapper("resnet50", use_last_fc=False, init_path=None)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net.load_state_dict(state["net_recon"])
    net.to(device).eval()
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=device, flip_input=False
    )
    pending = output_dir / "queue" / "pending"
    inflight = output_dir / "queue" / "inflight"
    done = output_dir / "queue" / "done"
    idle = 0
    while idle < 5:
        ticket = claim(pending, inflight)
        if ticket is None:
            idle += 1
            time.sleep(0.2)
            continue
        idle = 0
        video = Path(ticket.read_text().strip())
        destination = output_dir / f"{video.stem}.mat"
        started = time.time()
        try:
            row = reconstruct_video(video, detector, net, lm3d, destination, device)
            row.update(seconds=round(time.time() - started, 2), gpu=gpu, slot=slot)
            print(f"gpu {gpu}.{slot} ok {video.name} frames {row['frames']} missed {row['missed']} {row['seconds']}s", flush=True)
            (done / f"{video.stem}.json").write_text(json.dumps(row) + "\n")
            ticket.unlink(missing_ok=True)
        except Exception as error:
            print(f"gpu {gpu}.{slot} FAIL {video.name} {error}", flush=True)
            failed = output_dir / "queue" / "failed"
            failed.mkdir(parents=True, exist_ok=True)
            ticket.rename(failed / ticket.name)
            raise


def videos_in(directory, only):
    videos = sorted(path for path in directory.glob("*.mp4") if path.is_file())
    if only:
        videos = [video for video in videos if only in video.name]
    return videos


def gpu_table(gpus):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = {}
    for line in text.splitlines():
        index, memory, util = (int(part.strip()) for part in line.split(","))
        if index in gpus:
            rows[index] = (memory, util)
    return rows


def assert_gpus_free(gpus):
    rows = gpu_table(gpus)
    busy = {index: memory for index, (memory, _) in rows.items() if memory > 2000}
    if busy:
        raise SystemExit(f"requested GPUs are busy: {busy}")


def launch(gpus, workers_per_gpu, videos, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = output_dir / "queue" / "pending"
    inflight = output_dir / "queue" / "inflight"
    done = output_dir / "queue" / "done"
    for path in (pending, inflight, done):
        path.mkdir(parents=True, exist_ok=True)
        for old in path.iterdir():
            if old.is_file():
                old.unlink()
    for video in videos:
        mat = output_dir / f"{video.stem}.mat"
        if mat.is_file():
            continue
        (pending / video.stem).write_text(str(video) + "\n")
    queued = len(list(pending.iterdir()))
    print(f"queued {queued} of {len(videos)} on gpus {gpus} x{workers_per_gpu}", flush=True)
    if queued == 0:
        return
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu in gpus:
        for slot in range(workers_per_gpu):
            log_file = open(log_dir / f"gpu{gpu}-{slot}.log", "w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(gpu), "--slot", str(slot), "--output", str(output_dir)],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            processes.append((gpu, slot, process, log_file))
    while any(process.poll() is None for _, _, process, _ in processes):
        time.sleep(20)
        rows = gpu_table(gpus)
        summary = " ".join(f"{index}:{memory}MiB/{util}%" for index, (memory, util) in sorted(rows.items()))
        left = len(list(pending.iterdir()))
        flying = len(list(inflight.iterdir()))
        print(f"pending {left} inflight {flying} {summary}", flush=True)
    failed = []
    for gpu, slot, process, log_file in processes:
        code = process.wait()
        log_file.close()
        if code != 0:
            failed.append(f"{gpu}.{slot}:{code}")
    if failed:
        raise SystemExit("workers failed: " + ", ".join(failed))


def collect(output_dir, videos):
    from scipy.io import loadmat

    rows = []
    for video in videos:
        path = output_dir / f"{video.stem}.mat"
        if not path.is_file():
            rows.append({"video": video.name, "error": "missing mat"})
            continue
        mat = loadmat(path)
        coeff = mat["coeff"]
        transform = mat["transform_params"]
        canvas_ok = bool(coeff.shape[1] == 257 and np.all(transform[:, 0] == 512) and np.all(transform[:, 1] == 512) and coeff.shape[0] == transform.shape[0])
        rows.append({"video": video.name, "frames": int(coeff.shape[0]), "canvas_512": canvas_ok})
    summary = {
        "input": str(INPUT_DIR),
        "output": str(output_dir),
        "checkpoint": str(CHECKPOINT),
        "videos": len(rows),
        "canvas_512": sum(1 for row in rows if row.get("canvas_512")),
        "frames": sum(row.get("frames", 0) for row in rows),
        "files": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    bad = [row["video"] for row in rows if not row.get("canvas_512")]
    print(f"videos {summary['videos']} canvas_512 {summary['canvas_512']} frames {summary['frames']} bad {len(bad)}", flush=True)
    if bad:
        raise SystemExit("bad mats: " + ", ".join(bad))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.worker is not None:
        run_worker(args.worker, args.slot, args.output)
        return
    gpus = [int(item) for item in args.gpus.split(",") if item != ""]
    assert_gpus_free(gpus)
    videos = videos_in(INPUT_DIR, args.only)
    launch(gpus, args.workers_per_gpu, videos, args.output)
    collect(args.output, videos)


if __name__ == "__main__":
    main()
