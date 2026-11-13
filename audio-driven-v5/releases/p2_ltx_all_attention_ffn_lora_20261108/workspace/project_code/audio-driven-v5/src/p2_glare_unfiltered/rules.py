"""Reuse RGB raw detections, keep upstream signal rules, remove quality gates."""
import csv
import math
import importlib.metadata
import pickle
import shutil
from pathlib import Path

from p2_glare_eval.common import ROOT, read, sha, artifact_valid, atomic_json, complete
from p2_glare_eval.detector import EMOTIONS, run_rules
from p2_glare_eval.upstream import verify_sources
from p2_glare_rgb.detector import code_identity as rgb_code, policy as rgb_policy


def code_identity():
    return {**rgb_code(), **{str(p.relative_to(ROOT)): sha(p)
                            for p in Path(__file__).parent.glob('*.py')}}


def policy():
    original = rgb_policy()
    return dict(schema='glare-rgb-no-quality-gate-v1', fps=original['fps'],
                fer_input=original['fer_input'],
                face_selection='first_fer_face_in_detector_order',
                required_frame_coverage=None, min_face_iou=None,
                reject_multiple_faces=False,
                missing_frame_policy='unchanged_reference_reaction_rules',
                fer_missing='empty_csv_values_read_as_zero_by_reference_rules',
                mediapipe_missing='original_detected_mask_and_pose_gap_interpolation',
                reaction_parameters=original['reaction_parameters'],
                nodding_variant=original['nodding_variant'],
                combination=original['combination'],
                score_threshold=original['score_threshold'])


def write_fer(raw, destination, count):
    """FER Video defaults to first_face_only; retain empty rows when no face."""
    if len(raw) != count:
        raise ValueError('Raw FER frame count differs')
    stats = dict(frames=count, fer_no_face_frames=0, fer_multiple_face_frames=0)
    with Path(destination).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_index', *EMOTIONS])
        writer.writeheader()
        for i, row in enumerate(raw):
            if row['frame_index'] != i:
                raise ValueError('Raw FER frame indices are not contiguous')
            faces = row['faces']
            stats['fer_no_face_frames'] += int(not faces)
            stats['fer_multiple_face_frames'] += int(len(faces) > 1)
            values = faces[0]['emotions'] if faces else {k: '' for k in EMOTIONS}
            if faces and not all(isinstance(values[k], (int, float)) and
                                 math.isfinite(values[k]) and 0 <= values[k] <= 1
                                 for k in EMOTIONS):
                raise ValueError('Corrupt FER emotion values')
            writer.writerow(dict(frame_index=i, **{k: values[k] for k in EMOTIONS}))
    return stats


def process(video, source, destination, expected_sha):
    video, source, destination = map(Path, (video, source, destination))
    sources = verify_sources()
    marker = read(source / 'complete.json')
    contract = marker['contract']
    runtime = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'opencv-python')}
    if any(v != contract['runtime']['packages'][k] for k, v in runtime.items()):
        raise ValueError('Run reaction rules in the original detector Python environment')
    if (contract['video_sha256'] != expected_sha or contract['code'] != rgb_code()
            or contract['policy'] != rgb_policy() or contract['upstream'] != sources):
        raise ValueError('RGB source contract changed')
    if not artifact_valid(source, contract) or sha(video) != expected_sha:
        raise ValueError('RGB source or video is corrupt')
    identity = dict(source=str(source), source_marker_sha256=sha(source / 'complete.json'),
                    video_sha256=expected_sha, policy=policy(), code=code_identity(), rule_runtime=runtime)
    if artifact_valid(destination, identity):
        return read(destination / 'quality.json')
    destination.mkdir(parents=True, exist_ok=True)
    count = marker['geometry']['frame_count']
    raw = read(source / 'fer_raw.json')
    quality = write_fer(raw, destination / 'fer.csv', count)
    with (source / 'face.pkl').open('rb') as stream:
        faces = pickle.load(stream)
    if len(faces) != count:
        raise ValueError('MediaPipe frame count differs')
    quality['mediapipe_no_face_frames'] = sum(f.get('detected') is not True for f in faces)
    quality['strict_valid'] = marker['valid']
    quality['strict_joint_invalid_frames'] = count - read(source / 'quality.json')['jointly_valid_frames']
    quality['quality_gates_applied'] = False
    shutil.copyfile(source / 'face.pkl', destination / 'face.pkl')
    run_rules(video, destination, policy(), count)
    if marker['valid']:
        if sha(destination / 'scores.csv') != sha(source / 'scores.csv'):
            raise ValueError('Already strict-valid video changed reaction CSV')
        quality['strict_scores_identical'] = True
    atomic_json(destination / 'quality.json', quality)
    complete(destination, identity, [destination / n for n in
             ('face.pkl', 'fer.csv', 'scores.csv', 'quality.json')],
             geometry=marker['geometry'], quality_gates_applied=False, detector_based=True)
    return quality
