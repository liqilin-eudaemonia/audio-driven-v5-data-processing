"""Read the RealTalk test package without changing source assets or admitting training data."""
import base64
import io
from pathlib import Path
import av
from PIL import Image, ImageDraw
from p2_listen.contracts import atomic_json, positive, read, safe_id, sha
from .media import frames, prepare_row, validate_transform


def package_samples(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        import json
        samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        value = read(path)
        samples = value["samples"] if isinstance(value, dict) else value
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a nonempty RealTalk samples array")
    ids = [safe_id(r["sample_id"]) for r in samples]
    if len(set(ids)) != len(ids) or len({r["pair_id"] for r in samples}) != len(samples):
        raise ValueError("Duplicate sample/pair IDs")
    return samples


def adapt(sample, base, manifest_hash):
    """Always use original_video + original-pixel crop, never listener_video."""
    if sample["crop"] is None:
        return None
    if sample["listener"] not in ("p0", "p1") or sample["speaker"] not in ("p0", "p1") or sample["listener"] == sample["speaker"]:
        raise ValueError("Invalid speaker/listener role declarations")
    duration = positive(sample["duration_seconds"], "duration")
    fps = positive(sample["fps"], "fps")
    start, end, count = (sample[k] for k in ("start_frame", "end_frame", "frame_count"))
    if any(type(v) is not int for v in (start, end, count)) or start < 0 or end - start != count or count < 1:
        raise ValueError("Invalid frame interval")
    if (abs(count / fps - duration) > 1e-6 or abs(sample["start_seconds"] - start / fps) > 1e-6 or
        abs(sample["end_seconds"] - end / fps) > 1e-6):
        raise ValueError("Source frame/time interval mismatch")
    transform = validate_transform(dict(crop={k: sample["crop"][k] for k in ("x", "y", "width", "height")},
                                        source_size=[sample["source_width"], sample["source_height"]],
                                        output_size=[512, 512], resample="bicubic"))
    row = dict(sample_id=safe_id("realtalk_" + sample["pair_id"]), experiment_split="unassigned",
               target_video=sample["original_video"], target_audio=sample["audio"], duration_s=duration,
               time_origin="clip_start_seconds", video_role="listener", audio_role="mixed", audio_isolated=False,
               training_admitted=False, video_transform=transform,
               source_reference=dict(format="realtalk-random-test-package", manifest_sha256=manifest_hash,
                                     package_sample_id=sample["sample_id"], pair_id=sample["pair_id"],
                                     video_id=sample["video_id"], speaker=sample["speaker"], listener=sample["listener"],
                                     start_frame=start, end_frame=end, fps=fps, crop=sample["crop"],
                                     flags=sample.get("flags", []), source_training_admitted=sample.get("training_admitted")))
    for key in ("metadata", "frame_annotations"):
        p = (Path(base) / sample[key]).resolve()
        row["source_reference"][key] = dict(path=str(p), sha256=sha(p))
    return prepare_row(row, base, mode="visual")


def preview(row, content, meta, root):
    """Show source crop and exactly one JPEG that would be sent to the model."""
    index = len(meta["frames"]) // 2
    record = meta["frames"][index]
    images = [x["image_url"]["url"] for x in content if x["type"] == "image_url"]
    jpeg = base64.b64decode(images[index].split(",", 1)[1])
    sid = row["source_reference"]["package_sample_id"]
    folder = root / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sid + "_model_512.jpg")
    target.write_bytes(jpeg)
    with av.open(row["target_video"]) as container:
        original = None
        for frame in container.decode(video=0):
            if abs(float(frame.time) - record["time_s"]) < 1e-6:
                original = frame.to_image()
                break
    if original is None:
        raise ValueError("Cannot locate sampled preview PTS")
    c = row["video_transform"]["crop"]
    ImageDraw.Draw(original).rectangle((c["x"], c["y"], c["x"] + c["width"] - 1, c["y"] + c["height"] - 1), outline="red", width=4)
    original.thumbnail((768, 512))
    sheet = Image.new("RGB", (1280, 552), "#202020")
    sheet.paste(original, (0, 40))
    sheet.paste(Image.open(io.BytesIO(jpeg)), (768, 40))
    ImageDraw.Draw(sheet).text((10, 10), f"sample {sid} | listener={row['source_reference']['listener']} | t={record['time_s']:.3f}s | source crop -> 512x512", fill="white")
    comparison = folder / (sid + "_comparison.jpg")
    sheet.save(comparison, quality=90)
    return dict(sample_id=row["sample_id"], package_sample_id=sid, time_s=record["time_s"],
                model_frame=str(target), comparison=str(comparison), model_jpeg_sha256=record["jpeg_sha256"])


def prepare_package(samples_path, output, media_cfg, preview_ids=()):
    path = Path(samples_path).resolve()
    root = Path(output).resolve()
    samples = package_samples(path)
    requested = set(preview_ids)
    if requested - {r["sample_id"] for r in samples}:
        raise ValueError("Preview IDs not in package")
    report = dict(schema="realtalk-annotation-512-v1", source_manifest=str(path), source_sha256=sha(path),
                  total=len(samples), ready=0, missing_crop=0, invalid=0, items=[], previews=[],
                  network_called=False, training_admitted=False, audio_role="mixed", model_input_size=[512, 512])
    manifest = []
    for sample in samples:
        item = dict(package_sample_id=sample["sample_id"], pair_id=sample["pair_id"], flags=sample.get("flags", []),
                    duration_s=sample["duration_seconds"])
        if sample["crop"] is None:
            item.update(status="needs_crop", reason="crop=null; no whole-frame fallback")
            report["missing_crop"] += 1
        else:
            try:
                row = adapt(sample, path.parent, report["source_sha256"])
                content, meta = frames(row["target_video"], row["duration_s"], media_cfg, row["video_transform"])
                if meta["decoded_frames"] != sample["frame_count"]:
                    raise ValueError("Decoded frame count differs from package")
                if any(abs(r["time_s"] - r["clip_frame"] / sample["fps"]) > 1e-6 for r in meta["frames"]):
                    raise ValueError("Clip PTS differ from package frame timeline")
                item.update(status="visual_ready", sample_id=row["sample_id"], video_transform=row["video_transform"],
                            sampled_frames=len(meta["frames"]), all_frame_sizes_512=all(r["size"] == [512, 512] for r in meta["frames"]))
                if sample["sample_id"] in requested:
                    report["previews"].append(preview(row, content, meta, root))
                manifest.append(row)
                report["ready"] += 1
            except (ValueError, KeyError, OSError) as exc:
                item.update(status="invalid", error=str(exc))
                report["invalid"] += 1
        report["items"].append(item)
    manifest_path = root / "visual_manifest.json"
    report["visual_manifest"] = str(manifest_path)
    atomic_json(manifest_path, manifest, immutable=True)
    atomic_json(root / "preparation.json", report, immutable=True)
    return dict(report=str(root / "preparation.json"), manifest=str(manifest_path), ready=report["ready"],
                missing_crop=report["missing_crop"], failed=report["invalid"], model_called=False)
