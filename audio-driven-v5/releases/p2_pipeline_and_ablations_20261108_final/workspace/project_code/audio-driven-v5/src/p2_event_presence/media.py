"""Bind existing metric manifests and decode each side without GT timing gates."""
import math
import os
from pathlib import Path
import sys

from .contracts import CLASSES, SCHEMA, digest, read, rows, save, save_rows, sha

ROOT = Path(__file__).resolve().parents[2]
OMNI = ROOT.parent / 'OmniResponse'


def source_identity():
    names = ['scripts/run_listen_native_caption_vocal_pilot.py',
             'scripts/listen_j_corrected_reaction.py', 'scripts/run_glare_nodding_pilot.py',
             'scripts/check_glare_headshake_direction.py', 'scripts/check_glare_expression_gate.py',
             'scripts/p2_event_presence.py',
             'related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py',
             'related/visual_reaction_annotation/nod_headshake/face_detection/face_landmarker_v2_with_blendshapes.task',
             'related/visual_reaction_annotation/reaction_detector/nodding/run_nodding_intensity_mt_v2.py',
             'related/visual_reaction_annotation/reaction_detector/head_shaking/run_head_shaking_intensity_mt.py',
             'related/visual_reaction_annotation/reaction_detector/frowning/run_frowning_intensity_mt.py']
    result = {name: sha(ROOT / name) for name in names}
    for p in (ROOT / 'src/p2_event_presence').glob('*.py'): result[str(p.relative_to(ROOT))] = sha(p)
    for rel in ['evaluation/listener_metrics/media.py', 'evaluation/video/paired_metrics/media.py',
                'evaluation/seamless/reader.py']:
        result[str(OMNI / rel)] = sha(OMNI / rel)
    return result


def bind(manifests, output, config, method='model', draw='0', supports_audio=True, self_check=False):
    output = Path(output).resolve(); policy = read(config); values = []; seen = set()
    if policy.get('schema') != SCHEMA or policy.get('macro_policy') != 'frozen_reference_positive_support':
        raise ValueError('Unsupported event policy')
    if policy.get('pilot_only') is not True: raise ValueError('Formal benchmark requires a separately verified calibrated protocol')
    if (policy.get('visual_fps'), policy.get('cue_fps'), policy.get('audio_sample_rate'),policy.get('temperature')) != (8,25,16000,0.):
        raise ValueError('This release pins annotation8fps/candidates25fps/16k/greedy inference')
    if not 0 <= policy['window_overlap_seconds'] < policy['window_seconds']: raise ValueError('Invalid overlap')
    if output.exists() and (output / 'contract.json').exists(): raise ValueError('Use a NEW event-evaluation directory')
    output.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        for raw in rows(manifest):
            row = dict(raw)
            for name in ['generated', 'generated_audio', 'gt', 'gt_audio', 'gt_asr']:
                if row.get(name): row[name] = str((manifest.parent / row[name]).resolve())
            for key in ['id', 'dataset', 'generated']:
                if not isinstance(row.get(key), str) or not row[key]: raise ValueError('Bound manifest requires ' + key)
            if 'gt' not in row and 'gt_reference' not in row: raise ValueError('Bound GT or gt_reference required')
            duration = row.get('pair_duration_s', row.get('duration_s'))
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('Explicit positive pair_duration_s required')
            row['pair_duration_s'] = float(duration)
            row['method'] = raw.get('method', method); row['draw_id'] = str(raw.get('draw_id', draw))
            row['capabilities'] = raw.get('capabilities', dict(visual=True, audio=supports_audio))
            if set(row['capabilities']) != set(CLASSES) or any(type(v) is not bool for v in row['capabilities'].values()):
                raise ValueError('Explicit boolean visual/audio capabilities required')
            if not row['capabilities']['visual']: raise ValueError('This media adapter requires a generated listener video')
            key = tuple(row[k] for k in ('dataset', 'id', 'method', 'draw_id'))
            if key in seen: raise ValueError('Duplicate frozen manifest pair')
            seen.add(key)
            row_self_check=raw.get('self_check_only',bool(self_check))
            if type(row_self_check) is not bool or (row_self_check and not self_check):raise ValueError('Explicit --self-check required for control rows')
            if not row_self_check:
                if row.get('gt') == row['generated'] or (row.get('gt_audio') and row.get('gt_audio') == row.get('generated_audio')):
                    raise ValueError('GT as generation requires explicit self_check mode')
            row['self_check_only'] = row_self_check
            row['bound_manifest_sha256'] = sha(manifest)
            row['bound_input_sha256'] = {row[k]:sha(row[k]) if Path(row[k]).is_file() else None
                for k in ['generated','generated_audio','gt','gt_audio'] if row.get(k)}
            values.append(row)
    if not values: raise ValueError('Empty manifest')
    policy_identity = digest(policy)
    save_rows(output / 'frozen_pairs.jsonl', values)
    save(output / 'contract.json', dict(schema=SCHEMA, policy=policy, policy_identity=policy_identity,
        frozen_pairs_sha256=sha(output / 'frozen_pairs.jsonl'), source=source_identity(), requested_pairs=len(values),
        self_check_only=all(r['self_check_only'] for r in values), stage='bound', pilot_only=policy['pilot_only'],
        semantics='Final media, per-side independent detection; no GT event matching or generated GT gating.'))
    return values


def verify(work):
    work = Path(work); contract = read(work / 'contract.json')
    if sha(work / 'frozen_pairs.jsonl') != contract['frozen_pairs_sha256']: raise ValueError('Frozen manifest changed')
    if digest(contract['policy']) != contract['policy_identity']: raise ValueError('Policy identity changed')
    if source_identity() != contract['source']: raise ValueError('Source changed; prepare a NEW evaluation directory')
    values=rows(work / 'frozen_pairs.jsonl')
    for row in values:
        for path,expected in row['bound_input_sha256'].items():
            actual=sha(path) if Path(path).is_file() else None
            if actual!=expected:raise ValueError('Bound media changed: '+path)
    return contract,values


def prepare(work):
    """Native paired readers already used by the established evaluation pipeline."""
    contract,values=verify(work)
    sys.path.insert(0,contract['policy']['media_python_path'])
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, str(OMNI))
    from evaluation.listener_metrics.media import read_audio, write_video, grid25
    from evaluation.video.paired_metrics.media import decode
    from evaluation.seamless.reader import load_listener_audio, decode_reference
    work = Path(work); result = []; media_cache = {}
    if (work/'PREPARED.json').exists():
        marker=read(work/'PREPARED.json')
        if sha(work/'prepared_pairs.jsonl')!=marker['prepared_pairs_sha256']:raise ValueError('Prepared manifest changed')
        existing=rows(work/'prepared_pairs.jsonl')
        for row in existing:
            for m in row['prepared_media'].values():
                for field,hashfield in [('visual25','visual25_sha256'),('audio','audio_file_sha256')]:
                    if m.get(field) and sha(m[field])!=m[hashfield]:raise ValueError('Prepared media corrupted')
                for w in m['windows']:
                    if w.get('frames') and sha(w['frames'])!=w['frames_sha256']:raise ValueError('Prepared video window corrupted')
        return existing
    for row in values:
        media = {}; errors = {s: {} for s in ('reference', 'generated')}; duration = row['pair_duration_s']
        video_values = []; waves = {}; audio_audit = {}
        # Independence is essential: broken generation cannot hide a valid GT.
        for side in ('reference', 'generated'):
            try:
                if side == 'reference' and row.get('gt_reference'):
                    frames, fps, audit = decode_reference(row['gt_reference'])
                else:
                    frames, fps, checked = decode(row['gt' if side == 'reference' else 'generated'],
                        row.get('gt_roi' if side == 'reference' else 'generated_roi'))
                    audit = dict(timestamp_checked=checked, source_fps=fps)
                if side == 'reference' and row.get('gt_resize'):
                    if row.get('gt_resize_method') != 'pil_bicubic': raise ValueError('Explicit GT resize policy required')
                    from PIL import Image
                    frames = np.stack([np.asarray(Image.fromarray(f).resize(tuple(row['gt_resize']), Image.Resampling.BICUBIC)) for f in frames])
                if abs(len(frames)/fps-duration) > max(1/fps, .051): raise ValueError('Video clip duration mismatch')
                frames, grid = grid25(frames, fps, duration); video_values.append((side, frames))
                audio_audit[side + '_video'] = dict(source=audit, grid=grid)
            except Exception as exc: errors[side]['visual'] = f'{type(exc).__name__}: {exc}'
            if side == 'generated' and not row['capabilities']['audio']: continue
            try:
                if side == 'reference' and row.get('gt_reference'):
                    wave, sr, a = load_listener_audio(row['gt_reference']); wave = np.asarray(wave).reshape(-1)
                    div = math.gcd(sr, 16000); wave = resample_poly(wave, 16000//div, sr//div).astype(np.float32)
                    audio_audit['reference_recipe'] = a
                else:
                    path = row['gt_audio'] if side == 'reference' else row.get('generated_audio') or row['generated']
                    wave = read_audio(path, duration)
                if abs(len(wave)/16000-duration) > .05: raise ValueError('Audio/pair duration mismatch')
                waves[side] = wave[:round(duration*16000)]
                audio_audit[side + '_audio'] = dict(samples=len(wave), sample_rate=16000, own_full_waveform=True,
                    waveform_sha256=__import__('hashlib').sha256(wave.tobytes()).hexdigest(), gt_event_gate_applied=False)
            except Exception as exc: errors[side]['audio'] = f'{type(exc).__name__}: {exc}'
        for side in ('reference', 'generated'):
            frames = next((v for s, v in video_values if s == side), None); wave = waves.get(side)
            fingerprint = dict(duration_s=duration,
                rgb_sha256=__import__('hashlib').sha256(frames.tobytes()).hexdigest() if frames is not None else None,
                audio_sha256=__import__('hashlib').sha256(wave.tobytes()).hexdigest() if wave is not None else None,
                errors=errors[side])
            opaque = digest(fingerprint); folder = work / 'media' / opaque; folder.mkdir(parents=True, exist_ok=True)
            if opaque in media_cache:
                media[side]=media_cache[opaque];continue
            entry = dict(media_id=opaque, duration_s=duration, folder=str(folder.resolve()), identity=fingerprint,
                         errors=dict(errors[side]), windows=[], full_silence=bool(wave is not None and np.count_nonzero(wave) == 0))
            if frames is not None:
                video = folder / 'visual25.mkv'
                write_video(frames, video)
                entry['visual25'] = str(video.resolve()); entry['visual25_sha256'] = sha(video)
            if wave is not None:
                path = folder / 'listener.wav'
                sf.write(path, wave, 16000, subtype='FLOAT')
                entry.update(audio=str(path.resolve()), audio_file_sha256=sha(path))
            step=contract['policy']['window_seconds']-contract['policy']['window_overlap_seconds']
            starts=[0.]
            while starts[-1]+contract['policy']['window_seconds'] < duration-1e-7: starts.append(starts[-1]+step)
            for start in starts:
                end = min(duration, float(start) + contract['policy']['window_seconds']); win = dict(start_s=float(start), end_s=end)
                if frames is not None:
                    times = np.arange(float(start), end - 1e-7, 1/contract['policy']['visual_fps'])
                    ids = np.floor(times*25+.5-1e-8).astype(int).clip(0,len(frames)-1)
                    path = folder / f'frames8_{len(entry["windows"]):03d}.npy'
                    np.save(path, frames[ids], allow_pickle=False)
                    win.update(frames=str(path.resolve()), frames_sha256=sha(path), frame_times_s=times.tolist())
                entry['windows'].append(win)
            save(folder / 'media.json', entry); media[side] = entry; media_cache[opaque]=entry
        result.append(dict(**row, prepared_media=media, audio_audit=audio_audit))
        print(f'Prepared {row["dataset"]}/{row["id"]}: {errors}', flush=True)
    save_rows(work / 'prepared_pairs.jsonl', result)
    save(work / 'PREPARED.json', dict(pairs=len(result), prepared_pairs_sha256=sha(work/'prepared_pairs.jsonl'),
        policy_identity=contract['policy_identity'], source=contract['source']))
    return result
