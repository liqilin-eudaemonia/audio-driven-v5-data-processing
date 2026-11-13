"""Use recorded media evidence only: no video decode, WAV writing or inference."""
import math
from pathlib import Path

from .common import digest, file_hash, number, percentile, read, union


GATE = dict(version="official-listener-union-gate-v1", padding_seconds=.1, fade_seconds=.01,
            union_sources=["words", "sentences", "vad"], interval_convention="half_open_sample_indices",
            clip_before_padding=True, physical_clip_boundary_fades=False,
            fade_curve="sin_squared", empty_union="equal_length_silence")


def audio_stats(recipe, evidence, source):
    if (recipe.get("schema") != "j-listener-segment-official-union-v1"
            or recipe.get("gate_policy") != GATE or recipe.get("extra_mute_intervals") != []):
        raise ValueError("Unsupported audio policy: do not silently approximate a different recipe")
    if (recipe["sample_id"] != source["sample_id"] or recipe["listener_id"] != source["listener_id"]
            or recipe["source"]["path"] != source["normalized_audio"]["path"]
            or recipe["source"]["sha256"] != source["normalized_audio"]["sha256"]):
        raise ValueError("Audio identity mismatch")
    s = recipe["source"]; sr = s["sample_rate"]
    if type(sr) is not int or sr <= 0 or s.get("channels") != 1 or sr != source["normalized_audio"]["sample_rate"]:
        raise ValueError("Invalid audio geometry")
    first, last = s["first_sample"], s["last_sample"]
    if (type(first) is not int or type(last) is not int or not 0 <= first < last <= s["source_samples"]
            or abs(first / sr - source["source_start_s"]) > 1 / sr
            or abs(last / sr - source["source_end_s"]) > 1 / sr
            or abs(recipe["source_time_origin_s"] - first / sr) > 1e-8):
        raise ValueError("Audio time origin mismatch")
    n = last - first; duration = n / sr
    identity = evidence["audio_identity"]
    if identity.get("recipe_sha256") != digest(recipe):
        raise ValueError("Recipe hash mismatch")
    if (identity.get("source_start_sample") != first or identity.get("source_end_sample") != last
            or type(identity.get("full_silence")) is not bool
            or evidence.get("listener_audio_is_silent") is not identity["full_silence"]):
        raise ValueError("Audio evidence mismatch")
    core = []; skipped = 0; source_counts = {}
    for kind in GATE["union_sources"]:
        intervals = recipe["official_listener"][kind]
        if not isinstance(intervals, list):
            raise ValueError("Missing official intervals is not silence")
        source_counts[kind] = 0
        for entry in intervals:
            a, b = number(entry["start"]), number(entry["end"])
            if b < a:
                raise ValueError("Reversed official interval")
            a, b = max(0., a), min(duration, b)
            if b <= a:
                skipped += 1; continue
            lo = max(0, math.floor(a * sr + 1e-7)); hi = min(n, math.ceil(b * sr - 1e-7))
            if hi > lo:
                core.append([lo, hi]); source_counts[kind] += 1
    core = union(core)
    pad = round(GATE["padding_seconds"] * sr); fade = round(GATE["fade_seconds"] * sr)
    keep = union([[max(0, a - pad), min(n, b + pad)] for a, b in core])
    # Exactly match make_gate's zero-valued fade endpoints, without allocating a waveform.
    positive = []
    for a, b in keep:
        k = min(fade, b - a)
        lo, hi = a + int(k >= 2 and a > 0), b - int(k >= 2 and b < n)
        if hi > lo:
            positive.append([lo, hi])
    positive_n = sum(b - a for a, b in positive)
    if identity.get("positive_samples") != positive_n:
        raise ValueError("Recorded mask support differs from recipe")
    if not positive_n and not identity["full_silence"]:
        raise ValueError("Non-silent waveform without mask support")
    seconds = lambda xs: [[a / sr, b / sr] for a, b in xs]
    return dict(official_core_intervals=seconds(core), official_core_s=sum(b-a for a,b in core)/sr,
                official_bout_count=len(core), official_source_interval_counts=source_counts,
                official_skipped_intervals=skipped, nominal_keep_intervals=seconds(keep),
                nominal_keep_s=sum(b-a for a,b in keep)/sr, mask_positive_s=positive_n/sr,
                waveform_full_silence=identity["full_silence"],
                recipe_sha256=identity["recipe_sha256"],
                note="Mask support is not measured vocal duration; bouts are not word counts.")


def nod_stats(bundle, sid, duration):
    if bundle.get("sample_id") != sid or bundle.get("fps") != 25 or bundle.get("profile") != "recall":
        raise ValueError("Wrong nod source/profile")
    events = bundle["events"]
    if not isinstance(events, list):
        raise ValueError("Missing nod evidence is not absence")
    intervals = []; ids = set()
    for e in events:
        a, b = number(e["start"]), number(e["end"])
        if not 0 <= a < b <= duration or e["id"] in ids:
            raise ValueError("Invalid or duplicate nod evidence")
        ids.add(e["id"]); intervals.append([a, b])
    unique = sorted({tuple(x) for x in intervals})
    support = union(intervals)
    return dict(candidate_interval_count=len(unique), raw_interval_count=len(events),
                candidate_intervals=[list(x) for x in unique], support_intervals=support,
                support_s=sum(b-a for a,b in support), exact_nod_count=None,
                detector_file=bundle.get("detector_file"), detector_sha256=bundle.get("detector_sha256"))


def face_stats(measurement):
    raw = measurement["raw_measurements"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Missing native face measurements")
    short = []
    for r in raw:
        if r.get("detected"):
            w, h = number(r["face_width_native"]), number(r["face_height_native"])
            if min(w, h) <= 0:
                raise ValueError("Invalid native face size")
            short.append(min(w, h))
    focus = measurement["focus"]
    return dict(face_short_side_p10=percentile(short, .1), face_short_side_p50=percentile(short, .5),
                face_sample_detection_fraction=len(short)/len(raw), face_sample_count=len(raw),
                measurement_basis="sampled_native_face_boxes_not_crop_size",
                clarity_low_fraction=focus["low_fraction"],
                clarity_measured_fraction=focus["measured_fraction"],
                laplacian_reference_p80=focus["laplacian_reference_p80"])


def snapshot(record, caption_path, caption_sha, decision=None, *, sample_grid_tolerance=False):
    s = record["source"]; original = record.get("original_source", s); sid = record["sample_id"]
    if s["sample_id"] != sid or original["sample_id"] != sid or s["listener_id"] == s["speaker_id"]:
        raise ValueError("Source/listener identity mismatch")
    for k in ("pair", "split", "group_id", "listener_id", "speaker_id", "source", "source_start_s",
              "source_end_s", "duration_s", "normalized_audio"):
        if s[k] != original[k]:
            raise ValueError("Corrected crop changed non-crop source fields: " + k)
    if (Path(s["source"]["member"]).stem != s["listener_id"] or s["split"] not in ("train", "dev", "test")
            or not s["group_id"] or not isinstance(record["caption"], str)):
        raise ValueError("Invalid identity/split/caption")
    duration = number(s["duration_s"])
    tolerance=1e-7
    if sample_grid_tolerance:
        sr=s['normalized_audio']['sample_rate'];audio=record['audio_recipe']['source']
        if type(sr) is not int or sr<=0:raise ValueError('Invalid sample grid')
        tolerance=1/sr
        if abs((audio['last_sample']-audio['first_sample'])/sr-duration)>tolerance:
            raise ValueError('Declared duration differs from audio sample count')
    if duration <= 0 or abs(number(s["source_end_s"]) - number(s["source_start_s"]) - duration) > tolerance:
        raise ValueError("Source duration mismatch")
    if s["source_start_s"] < 0:
        raise ValueError("Negative source time")
    d = decision or record.get("visual_quality")
    if not d:
        raise ValueError("Quality receipt unavailable; legacy caption is not automatic admission")
    if d["sample_id"] != sid or d["source_sha256"] != digest(original, ascii=True):
        raise ValueError("Quality source mismatch")
    p = Path(d["measurement_file"])
    if file_hash(p) != d["measurement_sha256"]:
        raise ValueError("Quality measurement hash mismatch")
    m = read(p)
    if (m["source_sha256"] != d["source_sha256"] or m["contract_sha256"] != d["contract_sha256"]
            or m["source"] != original or m["status"] != "ok"):
        raise ValueError("Quality measurement provenance mismatch")
    provenance = record.get("crop_provenance")
    if provenance:
        if (provenance["effective_source_sha256"] != digest(s, ascii=True)
                or provenance["original_source_sha256"] != digest(original, ascii=True)
                or provenance["actual_crop"] != s["video_transform"]["crop"]
                or provenance["original_crop"] != original["video_transform"]["crop"]
                or provenance["measurement_sha256"] != d["measurement_sha256"]
                or provenance["quality_contract_sha256"] != d["contract_sha256"]):
            raise ValueError("Crop provenance mismatch")
    flags = []
    if d["decision"] != "retain": flags.append("quality_" + d["decision"])
    if record.get("generation_checks_passed") is not True or record.get("finish_reason") != "stop":
        flags.append("caption_generation_failed")
    if not record["caption"].strip() or record.get("review_flags"):
        flags.append("caption_review_required")
    evidence = record["input_evidence"]
    return dict(sample_id=sid, caption=record["caption"], caption_file=str(Path(caption_path).resolve()),
                caption_file_sha256=caption_sha, source=s, original_source_sha256=digest(original),
                quality=d, crop_provenance=provenance, crop_revision=(provenance or {}).get("revision", "legacy_recorded_crop"),
                face=face_stats(m), audio=audio_stats(record["audio_recipe"], evidence, s),
                nod=nod_stats(evidence["nod_supplement"], sid, duration),
                eligible_for_extraction=not flags, exclusion_reasons=flags,
                original_training_admitted=record.get("training_admitted"), training_admitted=False,
                human_ground_truth=False)
