"""Window-local observation hints; never convert candidate signals to event labels."""
import math


def smile_trace(poses):
    """Keep missing blendshapes unknown instead of inventing zero smile scores."""
    result = []
    for row in poses:
        shapes = row.get('blendshapes') or {}
        values = [shapes.get(k) for k in ('mouthSmileLeft', 'mouthSmileRight')]
        valid = row.get('detected') and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values)
        result.append(sum(values) / 2 if valid else None)
    return result


def visual_hints(candidates, window):
    """Crop cues and raw smile evidence to the video window actually being judged."""
    start, end = window['start_s'], window['end_s']
    fps = candidates.get('fps', 25)
    lo = max(0, math.ceil(start * fps - 1e-8))
    hi = max(lo, math.ceil(end * fps - 1e-8))
    face = candidates.get('framewise_face_detected')
    hints = dict(frame_coverage=candidates.get('frame_coverage', 0.),
                 coverage_scope='clip_fallback', missing_candidates_are_not_absence=True)
    if face is not None and hi > lo:
        hints.update(frame_coverage=sum(bool(v) for v in face[lo:hi]) / (hi-lo),
                     coverage_scope='current_window')
    hints['cues'] = sorted([
        dict(type=c['type'], start=round(max(start, c['start'])-start, 6),
             end=round(min(end, c['end'])-start, 6))
        for c in candidates.get('cues', [])
        if max(start, c['start']) < min(end, c['end'])
    ], key=lambda c: (c['start'], c['end'], c['type']))
    trace = candidates.get('smile_blendshape', {}).get('framewise_mean_left_right')
    if trace is not None:
        # Half-second summaries bound prompt size while preserving brief peaks.
        # This is a representation interval, not a smile threshold/duration rule.
        bins = []
        for j in range(math.ceil((end-start) / .5 - 1e-8)):
            a, b = start+j*.5, min(end, start+(j+1)*.5)
            ia, ib = max(0, math.ceil(a*fps-1e-8)), max(0, math.ceil(b*fps-1e-8))
            values = [v for v in trace[ia:ib]
                      if isinstance(v, (int, float)) and not isinstance(v, bool)
                      and math.isfinite(v)]
            bins.append([round(a-start, 6), round(b-start, 6),
                         round(sum(values)/len(values), 4) if values else None,
                         round(max(values), 4) if values else None, len(values), ib-ia])
        hints['smile_blendshape'] = dict(
            signal='mean of mouthSmileLeft and mouthSmileRight; not a smile probability',
            columns=['start_s', 'end_s', 'mean', 'peak', 'valid_frames', 'total_frames'],
            time_origin='current_window_start', bins=bins, binary_threshold=None)
    if candidates.get('error'):
        hints['candidate_error'] = candidates['error']
    return hints


def audio_hints(asr, normalize):
    """Match annotation prompts: normalized cues only; raw ASR stays in audit files."""
    texts, observations = [], []
    for value in [asr['transcript'], *asr['nonlexical_syllables']]:
        text, _, observation = normalize(value, None, 'candidate vocalization')
        if text and text not in texts:
            texts.append(text)
        if value and observation not in observations:
            observations.append(observation)
    return dict(normalized_candidate_texts=texts, observations=observations,
                uncertain=asr['uncertain'], official_transcript_used=False)
