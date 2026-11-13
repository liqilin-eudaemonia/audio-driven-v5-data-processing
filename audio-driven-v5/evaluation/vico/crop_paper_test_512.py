"""Crop ViCo paper-test originals to 512x512 with the current face-crop rules.

The square comes from scripts/listen_turns_v8.py:full_turn_crop. The 2026-10-30
bounded adjustment in src/listen_visual_quality/rules.py:propose_crop may then
move or grow that square by at most 5 percent. Pixels are resized with the same
PIL bicubic used by src/listen_labeling/media.py:crop_resize.

ViCo has no Seamless whole-body keypoints. Face points come from the project's
MediaPipe face landmarker, written into the 68 face slots (indices 23:91) that
full_turn_crop reads. Official 256 crops are not touched.
"""
import csv
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from listen_visual_quality.rules import propose_crop

def _load_full_turn_crop():
    """Execute the current full_turn_crop without importing the turn-script dependencies."""
    source = (ROOT / "scripts/listen_turns_v8.py").read_text()
    start = source.index("def full_turn_crop(")
    end = len(source)
    namespace = {"math": math, "np": np}
    exec(source[start:end], namespace)
    return namespace["full_turn_crop"]

full_turn_crop = _load_full_turn_crop()

MODEL = ROOT / "related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task"
POLICY = json.loads((ROOT / "configs/listen_visual_quality_v1.json").read_text())
ORIGINALS = ROOT / "data_previews/vico_paper_test/original"
META = Path("/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv")
OUT = Path("/public/0_DATA/2_Avatar/ViCo/dataset/by_split/test/faces_512")

# iBUG-68 order, MediaPipe face-mesh indices. Jaw, brows, nose, eyes, lips.
MESH_68 = [
    127, 34, 139, 143, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 195, 4,
    98, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]


def landmarks(video):
    import mediapipe as mp
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
    frames = []
    stack = []
    with av.open(str(video)) as container, mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)
        width, height = stream.codec_context.width, stream.codec_context.height
        for index, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            stamp = int(round(index * 1000 / fps))
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), stamp)
            point = np.zeros((91, 3), np.float32)
            box = None
            if result.face_landmarks:
                mesh = result.face_landmarks[0]
                chosen = np.array([[mesh[i].x * width, mesh[i].y * height, 1.0] for i in MESH_68], np.float32)
                point[23:91] = chosen
                box = [float(chosen[:, 0].min()), float(chosen[:, 1].min()),
                       float(chosen[:, 0].max()), float(chosen[:, 1].max())]
            frames.append(box)
            stack.append(point)
    return dict(width=width, height=height, fps=fps, boxes=frames,
                arrays={"boxes_and_keypoints:keypoints": np.stack(stack)})


def square_for(record):
    duration = len(record["boxes"]) / record["fps"]
    base = full_turn_crop(record, 0, duration)
    raw = [{"detected": box is not None, "bbox_native": box} for box in record["boxes"]]
    source = {"video_transform": {
        "crop": {"x": base["x"], "y": base["y"], "width": base["size"], "height": base["size"]},
        "source_size": [record["width"], record["height"]],
    }}
    proposal = propose_crop(raw, source, POLICY)
    chosen = base
    applied = "full_turn_crop"
    new = proposal.get("new") or {}
    if (proposal.get("status") in ("small_expansion_proposal", "translation_proposal", "unchanged")
            and all(isinstance(new.get(k), int) for k in ("x", "y", "width", "height"))
            and new["width"] == new["height"]
            and new["width"] <= base["size"] * POLICY["crop_max_side_growth"] + 1e-9):
        chosen = dict(base, x=new["x"], y=new["y"], size=new["width"])
        applied = proposal["status"]
    transform = dict(
        crop=dict(x=int(chosen["x"]), y=int(chosen["y"]), width=int(chosen["size"]), height=int(chosen["size"])),
        source_size=[record["width"], record["height"]],
        output_size=[512, 512],
        resample="bicubic",
    )
    return transform, dict(base_crop=base, bounded_status=proposal.get("status"), applied=applied,
                           wide_crop=bool(base.get("wide_crop")), valid_fraction=base.get("valid_fraction"))


def resize_512(image, transform):
    """Same integer crop and PIL bicubic as listen_labeling.media.crop_resize."""
    crop = transform["crop"]
    if list(image.size) != transform["source_size"]:
        raise ValueError("Decoded source size mismatch")
    box = (crop["x"], crop["y"], crop["x"] + crop["width"], crop["y"] + crop["height"])
    return image.crop(box).resize((512, 512), resample=Image.Resampling.BICUBIC)


def write_video(src, dest, transform, fps):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp, av.open(str(dest), "w") as out:
        stream = inp.streams.video[0]
        stream.thread_type = "AUTO"
        writer = out.add_stream("libx264", rate=fps)
        writer.width = 512
        writer.height = 512
        writer.pix_fmt = "yuv420p"
        writer.options = {"crf": "16", "preset": "veryfast"}
        for frame in inp.decode(stream):
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            cropped = resize_512(image, transform)
            video = av.VideoFrame.from_image(cropped)
            for packet in writer.encode(video):
                out.mux(packet)
        for packet in writer.encode():
            out.mux(packet)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in csv.DictReader(META.open()) if row["data_split"] == "test"]
    stems = []
    for row in rows:
        stems.append(row["listener"])
        stems.append(row["speaker"])
    stems = list(dict.fromkeys(stems))
    manifest = []
    for stem in stems:
        src = ORIGINALS / f"{stem}.mp4"
        if not src.exists():
            raise FileNotFoundError(src)
        dest = OUT / f"{stem}.mp4"
        try:
            record = landmarks(src)
            transform, detail = square_for(record)
            write_video(src, dest, transform, Fraction_fps(record["fps"]))
            item = dict(stem=stem, source=str(src), output=str(dest), frames=len(record["boxes"]),
                        fps=record["fps"], transform=transform, detail=detail, status="ok")
        except (ValueError, RuntimeError) as exc:
            item = dict(stem=stem, source=str(src), output=str(dest), status="failed", error=str(exc))
        manifest.append(item)
        print(stem, item.get("status"), item.get("transform", {}).get("crop"), item.get("error", ""), flush=True)
    by_uuid = []
    done = {item["stem"]: item for item in manifest}
    for row in rows:
        by_uuid.append(dict(uuid=row["uuid"],
                            listener=done[row["listener"]].get("output"),
                            speaker=done[row["speaker"]].get("output"),
                            listener_status=done[row["listener"]]["status"],
                            speaker_status=done[row["speaker"]]["status"],
                            listener_crop=done[row["listener"]].get("transform"),
                            speaker_crop=done[row["speaker"]].get("transform")))
    (OUT / "crops.json").write_text(json.dumps(
        dict(revision="full_turn_crop+quality-bounded-crop-v1-20261030", output_size=[512, 512],
             resample="bicubic", clips=by_uuid, files=manifest), ensure_ascii=False))
    print("wrote", len(manifest), "videos", len(by_uuid), "clips")


def Fraction_fps(fps):
    from fractions import Fraction
    return Fraction(fps).limit_denominator(1000)


if __name__ == "__main__":
    main()
