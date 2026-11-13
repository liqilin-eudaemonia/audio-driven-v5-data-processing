"""Adapt portable Seamless clips with independent WAVs and native-pixel crop proposals."""
import math
from pathlib import Path
import soundfile as sf
from p2_listen.contracts import atomic_json, positive, read, rows, safe_id, sha
from .media import frames, prepare_row, validate_transform, waveform
from .realtalk import preview


def integer_crop(proposal, width, height):
    """Round each native-pixel square parameter half up; retain proposal separately."""
    if not isinstance(proposal, dict) or proposal.get("status") != "proposal_only":
        raise ValueError("No measured crop proposal")
    values = [proposal[k] for k in ("x", "y", "size")]
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in values) or values[2] <= 0:
        raise ValueError("Invalid crop proposal")
    if values[0] + values[2] > width + 1e-6 or values[1] + values[2] > height + 1e-6:
        raise ValueError("Floating crop outside source bounds")
    x, y, size = [math.floor(v + .5) for v in values]
    return validate_transform(dict(crop=dict(x=x, y=y, width=size, height=size), source_size=[width, height],
                                   output_size=[512, 512], resample="bicubic"))


def adapt(item, base, manifest_hash):
    safe_id(item["sample_id"])
    duration = positive(item["duration_s"], "duration")
    roles = item["roles"]
    if roles["listener"]["person_id"] == roles["speaker"]["person_id"]:
        raise ValueError("Speaker/listener identities must differ")
    listener = roles["listener"]
    checks = listener["media_checks"]["clip"]
    transform = integer_crop(item["crop_proposals"]["listener"], checks["width"], checks["height"])
    references = {}
    audio_support = {}
    for role in ("speaker", "listener"):
        data = roles[role]
        references[role] = dict(person_id=data["person_id"], source_index=data["source_index"], media_checks=data["media_checks"])
        for key in ("clip_video", "audio_wav", "context_video"):
            p = (Path(base) / data[key]).resolve()
            references[role][key] = dict(path=str(p), sha256=sha(p))
        info = sf.info(references[role]["audio_wav"]["path"])
        expected = data["media_checks"]["clip"]
        if (info.channels != 1 or info.samplerate != expected["sample_rate"] or
            info.frames != expected["pcm_source_end_sample"] - expected["pcm_source_start_sample"] or
            abs(info.frames / info.samplerate - duration) > 1 / info.samplerate + 1e-6):
            raise ValueError("Independent role WAV does not match exported sample support")
        audio_support[role] = dict(frames=info.frames, sample_rate=info.samplerate, duration_s=info.frames / info.samplerate)
    # Waveform support is authoritative; retain source times for traceability only.
    row = dict(sample_id=item["sample_id"], experiment_split=item["split"],
               target_video=listener["clip_video"], target_audio=listener["audio_wav"],
               duration_s=audio_support["listener"]["duration_s"], time_origin="clip_start_seconds",
               video_role="listener", audio_role="listener", audio_isolated=True, training_admitted=False,
               video_transform=transform, source_reference=dict(
                   format="seamless-annotation-test-package", manifest_sha256=manifest_hash,
                   package_sample_id=f"{item['ordinal']:03d}", pair_id=item["pair"], listener=listener["person_id"],
                   speaker=roles["speaker"]["person_id"], roles=references, audio_support=audio_support,
                   source_start_s=item["source_start_s"], source_end_s=item["source_end_s"],
                   requested_duration_s=duration, segment_in_context_s=item["segment_in_context_s"],
                   crop_proposals=item["crop_proposals"], crop_quantization="round-half-up x/y/size to native integer pixels",
                   boundary_review_flags=item.get("boundary_review_flags", []), audio_final=item.get("audio_final", {}),
                   admission_pool=item.get("admission_pool"), audio_pool=item.get("audio_pool"),
                   transcript_used_for_annotation=False, context_used_for_annotation=False))
    for key in ("metadata_file", "quality_evidence_file"):
        p = (Path(base) / item[key]).resolve()
        row["source_reference"][key] = dict(path=str(p), sha256=sha(p))
    return prepare_row(row, base)


def prepare_package(items_path, output, media_cfg, preview_ids=()):
    path = Path(items_path).resolve()
    root = Path(output).resolve()
    items = rows(path)
    requested = set(preview_ids)
    if requested - {f"{x['ordinal']:03d}" for x in items}:
        raise ValueError("Unknown preview ordinal")
    report = dict(schema="seamless-annotation-512-v1", source_manifest=str(path), source_sha256=sha(path),
                  total=len(items), ready=0, missing_crop=0, invalid=0, items=[], previews=[],
                  network_called=False, training_admitted=False, audio_role="listener",
                  media_config=media_cfg, model_input_size=[512, 512])
    manifest = []
    for item in items:
        record = dict(sample_id=item["sample_id"], package_sample_id=f"{item['ordinal']:03d}", duration_s=item["duration_s"])
        if (item.get("crop_proposals", {}).get("listener") or {}).get("status") != "proposal_only":
            record.update(status="needs_crop", reason="No measured listener crop")
            report["missing_crop"] += 1
        else:
            try:
                row = adapt(item, path.parent, report["source_sha256"])
                content, video_meta = frames(row["target_video"], row["duration_s"], media_cfg, row["video_transform"])
                _, audio_meta = waveform(row["target_audio"], row["duration_s"], media_cfg)
                record.update(status="av_ready", video_transform=row["video_transform"],
                              sampled_frames=len(video_meta["frames"]), decoded_frames=video_meta["decoded_frames"],
                              video_duration_s=video_meta["decoded_end_s"], audio=audio_meta,
                              duration_s=row["duration_s"], video_minus_audio_s=video_meta["decoded_end_s"]-row["duration_s"],
                              all_frame_sizes_512=all(f["size"] == [512, 512] for f in video_meta["frames"]))
                if record["package_sample_id"] in requested:
                    report["previews"].append(preview(row, content, video_meta, root))
                manifest.append(row)
                report["ready"] += 1
            except (ValueError, KeyError, OSError) as exc:
                record.update(status="invalid", error=str(exc))
                report["invalid"] += 1
        report["items"].append(record)
    report["manifest"] = str(root / "av_manifest.json")
    atomic_json(root / "av_manifest.json", manifest, immutable=True)
    atomic_json(root / "preparation.json", report, immutable=True)
    return dict(report=str(root / "preparation.json"), manifest=report["manifest"], ready=report["ready"],
                missing_crop=report["missing_crop"], failed=report["invalid"], model_called=False)
