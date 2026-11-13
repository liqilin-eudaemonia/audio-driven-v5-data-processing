"""Extract aligned 3DMM using original ViCo network and alignment functions.

GPU scope: authorized node8 physical GPU7 only. No renderer is needed to
obtain network coefficients. The batched FAN path is checked against the
installed upstream single-image implementation before production extraction.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/p2_vico256_combined8000_20261106"
WORK = ROOT / "experiments/p2_vico_motion8000_20261106"
RECON = ROOT / "evaluation/vico_challenge_baseline/Deep3DFaceRecon_pytorch"
ARM = "base_event_feature_whisper_8000"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n");temp.replace(path)


def probe(path):
    x = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)]))["streams"][0]
    n, d = map(int, x["r_frame_rate"].split("/"))
    return dict(width=x["width"], height=x["height"], fps=n/d, frames=int(x["nb_frames"]),
                duration_s=float(x["duration"]))


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    if (work/"protocol.json").exists():
        return json.loads((work/"protocol.json").read_text())
    parent = json.loads((BASE/"protocol.json").read_text())
    pre = json.loads((BASE/"INPUT_PREFLIGHT.json").read_text())
    gt_dir = Path(parent["datasets"]["vico"]["gt_dir"])
    samples = []
    for item in pre["samples_audit"]:
        uid = item["id"]
        paths = dict(speaker=gt_dir/(uid+".speaker.mp4"), gt=gt_dir/(uid+".listener.mp4"),
                     prediction=BASE/"scoring"/ARM/(uid+".listener.mp4"))
        streams = {}
        for role, path in paths.items():
            p = probe(path)
            if (p["width"], p["height"], p["fps"], p["frames"]) != (256, 256, 30, item["original_gt"]["frames"]):
                raise ValueError(f"Not on exact 256/30 GT frame grid: {uid}, {role}: {p}")
            streams[role] = dict(path=str(path), sha256=sha(path), **p)
        samples.append(dict(id=uid, streams=streams))
    assets = [Path("/public/0_DATA/2_Avatar/ViCo/epoch_20.pth"),
              RECON/"BFM/BFM_model_front.mat", RECON/"BFM/similarity_Lm3D_all.mat",
              Path("/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip"),
              Path("/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth")]
    source_files = [RECON/p for p in ("models/networks.py", "models/bfm.py", "util/preprocess.py",
                                      "util/load_mats.py", "face_recon_videos.py", "extract_kp_videos.py")]
    own_files = [Path(__file__), ROOT/"scripts/p2_vico_motion_metrics.py",
                 ROOT/"tests/test_p2_vico_motion_metrics.py"]
    protocol = dict(schema="vico-motion-public-definitions-v1", created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        author_scoring_code=False, strict_customlistener_table_reproduction=False,
        user_confirmed="raw coefficients; TLCC -2..+2s at frame step; 4s windows every 1s; equal clip weighting",
        parent_generation=str(BASE), parent_files={p: sha(BASE/p) for p in ("RESULTS.json", "COMPLETE.json", "FINAL_AUDIT.json", "protocol.json")},
        source_dataset="ViCo paper test43; excludes ood and training", samples=samples,
        extractor=dict(code="https://github.com/dc3ea9f/vico_challenge_baseline", network="resnet50; use_last_fc=False",
            checkpoint="epoch_20.pth", strict_load=True, coefficient_order="motion: exp[64],angle[3],trans[3]; complete raw coeff[257] retained",
            landmark="face-alignment 1.3.5; TWO_D; SFD; flip_input=False; first detected face",
            alignment="official align_img: y=H-1-y, target_size=224, rescale_factor=102; bicubic PIL",
            missing_landmark="previous valid landmarks forward-fill; initial missing: official lm3d 5-point fallback; all missing clip rejected",
            renderer="omitted: coefficient output is the ReconNetWrapper forward pass before rendering",
            precision="FP32; no autocast, no TF32; deterministic cuDNN; inference mode; eval mode", normalization="none; raw native coefficients; rotation radians; no degree conversion",
            identity="GT and prediction independently extracted with identical weights/preprocessing; no fitting prediction to GT",
            batch=dict(detector=32, fan=1, fan_submission_threads=1, reconstruction=128)),
        timing=dict(fps=30, width=256, height=256, alignment="exact equal frame counts; origin zero; no lag fitting or dynamic time warping",
            generation="existing native 512x512/25FPS -> antialiased 256 -> nearest-time 30FPS score copies",
            limitation="25->30 nearest-time repeats frames and affects adjacent-frame and correlation metrics; native25 output retained in parent"),
        metric_parameters=dict(max_lag=60, window=120, stride=30),
        metric_definitions=dict(FD="mean absolute coordinate difference per clip; exp/angle/trans separately",
            RTLCC="mean absolute difference of per-coordinate, per-lag Pearson curves of (speaker,GT) and (speaker,pred); positive lag listener later; overlap recentered per lag",
            RWTLCC="RTLCC in complete 120-frame windows at starts 0,30,...; no padding; incomplete trailing window omitted; average windows then clips",
            correlation_zero_variance="denominator <=1e-12 -> correlation0; undefined count reported",
            FID_fm="per-clip Gaussian Frechet squared distance; raw per-frame motion features; unbiased sample covariance ddof1; PSD symmetric eigensolver",
            FID_delta_fm="same distance on consecutive frame differences, never across clip boundaries; no division by dt",
            SND="FID_fm+FID_delta_fm on raw scale; before any display multipliers",
            V_D="mean coordinate temporal population variance ddof0, then mean clips; report GT reference as well",
            aggregation="equal clips, equal windows within each clip; exp64 and pose6 separately",
            table_scale="FD and FID_delta_fm multiplied100; RTLCC/RWTLCC/SND/V_D unchanged; raw values retained"),
        limitations=["Not verified against unavailable CustomListener author scoring code/checkpoint/normalization",
                    "Correlation lags, strides, channel aggregation, constant handling and boundaries are explicit local choices approved by user",
                    "Rendered RGB re-extraction can differ from directly scoring a coefficient-generating model's native outputs",
                    "SND/variance/FID aggregation follow public definitions; exact author implementations not available"],
        references=dict(customlistener="https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_CustomListener_Text-guided_Responsive_Interaction_for_User-friendly_Listening_Head_Generation_CVPR_2024_paper.pdf",
            snd="https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_Talking_Head_Generation_with_Probabilistic_Audio-to-Visual_Diffusion_Priors_ICCV_2023_paper.pdf",
            vico="https://github.com/dc3ea9f/vico_challenge_baseline"),
        assets={str(p): sha(p) for p in assets}, source_sha256={str(p): sha(p) for p in source_files+own_files})
    save(work/"protocol.json", protocol)
    return protocol


def load_frames(path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path));frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:cap.release()
    return np.stack(frames)


class Extractor:
    def __init__(self):
        import cv2
        import torch
        import face_alignment
        sys.path.insert(0, str(RECON))
        from models.networks import define_net_recon
        from util.load_mats import load_lm3d
        from util.preprocess import align_img
        torch.set_num_threads(4);cv2.setNumThreads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.torch, self.align_img = torch, align_img
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda", flip_input=False)
        self.net = define_net_recon("resnet50", use_last_fc=False, init_path=None)
        ck = torch.load("/public/0_DATA/2_Avatar/ViCo/epoch_20.pth", map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["net_recon"], strict=True);self.net.eval().cuda()
        self.lm3d = load_lm3d(str(RECON/"BFM"))

    def detect(self, frames):
        import numpy as np
        output = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), 32):
                tensor = self.torch.from_numpy(frames[start:start+32].transpose(0, 3, 1, 2).copy()).float().cuda()
                output.extend(self.fa.face_detector.detect_from_batch(tensor))
        # face-alignment's SFD helper enables benchmark internally.
        self.torch.backends.cudnn.benchmark = False
        return output

    def landmarks(self, frames, faces):
        import numpy as np
        detected = np.zeros(len(frames), dtype=bool)
        face_counts = np.array([len(x) for x in faces], dtype=np.int32)
        landmarks = np.full((len(frames), 68, 2), -1., dtype=np.float32)
        indices = [i for i, boxes in enumerate(faces) if len(boxes)]
        # Keep the official sequential, no_grad FAN batch1 path. Both larger
        # FAN batches and concurrent calls failed real-video repeatability.
        # Independent processes parallelize clips instead of sharing the model.
        if not indices:raise ValueError("All frames missing face; refusing unsupported metrics")
        for i in indices:
            landmarks[i] = self.fa.get_landmarks_from_image(frames[i], detected_faces=faces[i][:1])[0]
            detected[i] = True
        previous = None
        for i in range(len(landmarks)):
            if detected[i]:previous = landmarks[i].copy()
            elif previous is not None:landmarks[i] = previous
        if not detected.any():raise ValueError("All frames missing face; refusing unsupported metrics")
        return landmarks, detected, face_counts

    def align(self, frames, landmarks):
        import numpy as np
        from PIL import Image
        def one(pair):
            frame, lm = pair;lm = lm.copy()
            h, w = frame.shape[:2]
            if np.mean(lm) == -1:
                lm = (self.lm3d[:, :2]+1)/2.
                lm = np.concatenate([lm[:, :1]*w, lm[:, 1:2]*h], 1)
            else:lm[:, -1] = h-1-lm[:, -1]
            transform, img, _, _ = self.align_img(Image.fromarray(frame), lm, self.lm3d)
            return np.array(img), np.asarray(transform, dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(one, zip(frames, landmarks)))
        return np.stack([x[0] for x in rows]), np.stack([x[1] for x in rows])

    def reconstruct(self, aligned):
        import numpy as np
        rows = []
        with self.torch.inference_mode():
            for start in range(0, len(aligned), 128):
                # Match upstream uint8 /255 -> tensor float32 (division in numpy float64).
                x = self.torch.tensor(np.array(aligned[start:start+128])/255., dtype=self.torch.float32)
                x = x.permute(0, 3, 1, 2).contiguous().cuda()
                rows.append(self.net(x).cpu().numpy())
        return np.concatenate(rows)

    def extract(self, frames):
        import numpy as np
        t = time.time();faces = self.detect(frames);td = time.time()
        lm, detected, counts = self.landmarks(frames, faces);tl = time.time()
        aligned, transform = self.align(frames, lm);ta = time.time()
        coeff = self.reconstruct(aligned);tr = time.time()
        motion = np.concatenate((coeff[:, 80:144], coeff[:, 224:227], coeff[:, 254:257]), 1)
        return dict(coeff=coeff, motion=motion, landmarks=lm, detected=detected, face_counts=counts,
                    transform_params=transform), dict(detect=td-t, landmarks=tl-td, align=ta-tl, recon=tr-ta, total=tr-t), faces

    def parity(self, frames, arrays, faces):
        """Compare to upstream API and network batch1, retaining numerical evidence."""
        import numpy as np
        valid = np.where(arrays["detected"])[0]
        picks = valid[np.linspace(0, len(valid)-1, min(8, len(valid))).astype(int)]
        lm_diffs, alignment_diffs, five_point_diffs = [], [], []
        from util.preprocess import extract_5p
        for i in picks:
            upstream = self.fa.get_landmarks_from_image(frames[i], detected_faces=faces[i])[0]
            lm_diffs.append(float(np.abs(upstream-arrays["landmarks"][i]).max()))
            five_point_diffs.append(float(np.abs(extract_5p(upstream)-extract_5p(arrays["landmarks"][i])).max()))
            a, _ = self.align(frames[i:i+1], upstream[None])
            b, _ = self.align(frames[i:i+1], arrays["landmarks"][i:i+1])
            alignment_diffs.append(int(np.abs(a.astype(int)-b.astype(int)).max()))
        # Compare original alignment+forward per image with batched coefficients.
        upstream_coeff = []
        with self.torch.inference_mode():
            for i in picks:
                lm = self.fa.get_landmarks_from_image(frames[i], detected_faces=faces[i])[0].astype(np.float32)
                aligned, _ = self.align(frames[i:i+1], lm[None])
                upstream_coeff.append(self.reconstruct(aligned)[0])
        coeff_diff = float(np.abs(np.stack(upstream_coeff)-arrays["coeff"][picks]).max())
        # Alignment depends on five aggregated facial landmarks, not every
        # jaw/eyebrow point. Demand pixel-identical aligned network inputs.
        passed = max(lm_diffs) == 0 and max(alignment_diffs) == 0 and coeff_diff < 1e-3
        report = dict(passed=passed, checked_frames=picks.tolist(), landmark_max_abs=max(lm_diffs),
                      five_point_max_abs=max(five_point_diffs), aligned_uint8_max_abs=max(alignment_diffs),
                      coefficient_max_abs=coeff_diff, coefficient_tolerance=1e-3,
                      comparison="upstream single-image FAN with same SFD boxes and original alignment; batch1 recon vs batched FP32")
        if not passed:raise ValueError(f"Batching parity failed: {report}")
        return report


def run(work, smoke=False, shard=0, shards=1):
    import numpy as np
    import fcntl
    suffix = "" if shards == 1 else f"_{shard:02d}"
    with (work/f"run{suffix}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(work)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":raise ValueError("Must expose physical GPU7 only")
        if subprocess.check_output(["hostname"], text=True).strip() not in ("node8", "zz-dev-1"):
            # Actual hostnames can be provider-generated; require the authorized
            # node8 GPU UUID as the definitive device identity below.
            pass
        gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", "7", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        if gpu_uuid != "GPU-d5cb2cac-0f7a-e5f8-1870-c2efcae7aedc":raise ValueError("Wrong authorized GPU")
        for path, expected in protocol["source_sha256"].items():
            if sha(path) != expected:raise ValueError(f"Frozen code changed: {path}")
        ext = Extractor()
        save(work/f"device{suffix}.json", dict(gpu_uuid=gpu_uuid, physical_gpu=7, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
             hostname=subprocess.check_output(["hostname"], text=True).strip(), pid=os.getpid(), torch=ext.torch.__version__))
        jobs = [(sample["id"], role, info) for sample in protocol["samples"] for role, info in sample["streams"].items()]
        if smoke:jobs = jobs[:3]
        else:jobs = jobs[shard::shards]
        start = time.time();reports = []
        def read(job):
            uid, role, info = job
            if sha(info["path"]) != info["sha256"]:raise ValueError("Input hash changed")
            frames = load_frames(info["path"])
            if len(frames) != info["frames"]:raise ValueError("Decoded frame count mismatch")
            if smoke:frames = frames[:64]
            return frames
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read, jobs[0])
            for index, job in enumerate(jobs):
                uid, role, info = job;frames = pending.result()
                if index+1 < len(jobs):pending = pool.submit(read, jobs[index+1])
                dest = work/("smoke_features" if smoke else "features")/role/(uid+".npz")
                receipt = dest.with_suffix(".json")
                if receipt.exists():
                    old = json.loads(receipt.read_text())
                    if old["input_sha256"] == info["sha256"] and old["feature_sha256"] == sha(dest):
                        reports.append(old);continue
                    raise ValueError(f"Invalid feature cache: {dest}")
                save(work/f"status{suffix}.json", dict(stage="smoke" if smoke else "extracting", completed=index, total=len(jobs),
                    id=uid, role=role, elapsed_s=time.time()-start, at=time.time()))
                arrays, timing, faces = ext.extract(frames)
                if smoke or index == 0:
                    parity = ext.parity(frames, arrays, faces)
                    save(work/(f"BATCH_PARITY_smoke_{role}.json" if smoke else f"BATCH_PARITY{suffix}.json"), parity)
                    if not (work/"BATCH_PARITY.json").exists():save(work/"BATCH_PARITY.json", parity)
                if not all(np.isfinite(arrays[key]).all() for key in ("coeff", "motion", "landmarks", "transform_params")):
                    raise ValueError("Nonfinite extracted features")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.with_suffix(".npz.tmp").open("wb") as f:np.savez_compressed(f, **arrays)
                dest.with_suffix(".npz.tmp").replace(dest)
                report = dict(id=uid, role=role, frames=len(frames), feature=str(dest), feature_sha256=sha(dest),
                    input_sha256=info["sha256"], protocol_sha256=sha(work/"protocol.json"), missing_face_frames=int((~arrays["detected"]).sum()),
                    multi_face_frames=int((arrays["face_counts"]>1).sum()), seconds=timing)
                save(receipt, report);reports.append(report)
                print(json.dumps(dict(completed=index+1, total=len(jobs), **report)), flush=True)
        save(work/("SMOKE_COMPLETE.json" if smoke else f"EXTRACTION_COMPLETE{suffix}.json"), dict(streams=len(reports),
             frames=sum(r["frames"] for r in reports), missing_face_frames=sum(r["missing_face_frames"] for r in reports),
             seconds=time.time()-start, reports=reports))
        save(work/f"status{suffix}.json", dict(stage="smoke_complete" if smoke else "extraction_complete", completed=len(jobs), total=len(jobs), at=time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--prepare", action="store_true");p.add_argument("--smoke", action="store_true")
    p.add_argument("--shard", type=int, default=0);p.add_argument("--shards", type=int, default=1)
    args = p.parse_args();args.work.mkdir(parents=True, exist_ok=True)
    try:
        if args.prepare:prepare(args.work)
        else:run(args.work, args.smoke, args.shard, args.shards)
    except Exception as e:
        import traceback
        suffix = "" if args.shards == 1 else f"_{args.shard:02d}"
        save(args.work/f"FAILURE{suffix}.json", dict(error=str(e), traceback=traceback.format_exc(), at=time.time()))
        save(args.work/f"status{suffix}.json", dict(stage="failed", error=str(e), at=time.time()))
        raise
