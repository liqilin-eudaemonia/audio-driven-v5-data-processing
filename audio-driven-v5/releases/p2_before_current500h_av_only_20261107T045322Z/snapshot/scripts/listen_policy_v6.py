"""Conservative semantic cuts, original-pixel geometry and listener speech screening.

No model inference or action labels. Ambiguous cases are rejected, not repaired
by arbitrary time cuts, crop enlargement, or relabeling untranscribed speech.
"""
import math
import re

POLICY = dict(revision=6, fps=25, max_frames=249, min_seconds=4,
              orientation_policy='reject_non_upright_or_uncertain_raw_video',
              natural_pause_seconds=.65, minimum_crop_side=512,
              max_unknown_burst=.8, max_unknown_seconds=1.2,
              max_unknown_fraction=.15, max_unknown_bursts=2,
              max_listener_fraction=.25, max_listener_bursts=3)

# Pauses after function words are weak evidence of a completed phrase.
DANGLING = set('and or but to the a an of in on at for with because so that this these those '
               'my your his her our their its is was were are be been being have has had '
               'would could should will can we i you they it he she'.split())


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def seconds(intervals):
    return sum(b-a for a, b in merged(intervals))


def reviewed_noise_hits(listener_id, start, end, reviews):
    """Human clip-level exclusions follow source time, even after re-windowing."""
    return [r for r in reviews if r['decision']=='reject' and r['listener_id']==listener_id
            and r['end']>start+1e-7 and r['start']<end-1e-7]


def difference(intervals, covered):
    result = []
    for a, b in merged(intervals):
        for x, y in merged(covered):
            if y <= a or x >= b:
                continue
            if x > a:
                result.append([a, x])
            a = max(a, y)
        if a < b:
            result.append([a, b])
    return result


def listener_screen(person, start, end):
    """VAD without text is uncertainty, never automatic evidence of feedback."""
    vad = merged([[max(start,a), min(end,b)] for a,b in person['vad']])
    segments = [s for s in person['segments'] if s['end'] > start and s['start'] < end]
    reasons = []
    if any(not s['feedback'] for s in segments):
        reasons.append('listener_substantive_speech')
    # Only transcript words explain VAD; a long segment timestamp alone does not.
    covered = [[max(start,w['start']-.08), min(end,w['end']+.08)]
               for s in segments if s['feedback'] for w in s['words']
               if w['end'] > start and w['start'] < end]
    unknown = difference(vad, covered)
    unknown = [v for v in unknown if v[1]-v[0] >= .08]
    duration = end-start
    metrics = dict(listener_vad_seconds=seconds(vad), unknown_seconds=seconds(unknown),
                   unknown_bursts=len(unknown), listener_bursts=len(vad),
                   longest_unknown=max([b-a for a,b in unknown] or [0.]))
    if (metrics['longest_unknown'] > POLICY['max_unknown_burst']
            or metrics['unknown_seconds'] > POLICY['max_unknown_seconds']
            or metrics['unknown_seconds']/duration > POLICY['max_unknown_fraction']
            or len(unknown) > POLICY['max_unknown_bursts']):
        reasons.append('unreliable_listener_transcript_or_repeated_unknown_speech')
    if seconds(vad)/duration > POLICY['max_listener_fraction'] or len(vad) > POLICY['max_listener_bursts']:
        reasons.append('listener_activity_too_dense')
    return dict(accepted=not reasons, reasons=reasons, **metrics)


def natural_clips(start, end, words, speaker_vad=(), listener_activity=()):
    """Complete natural units only; 8k+1 alignment uses verified boundary silence.

    Units crossing an excluded/floor edge are dropped in full. Alignment may
    extend into existing silence by <.32s, but never trims a word or pads media.
    """
    words = sorted([w for w in words if w['end'] > w['start']], key=lambda w:w['start'])
    units, rejected, first = [], [], 0
    for i, w in enumerate(words):
        nxt = words[i+1] if i+1 < len(words) else None
        terminal = bool(re.search(r'[.!?。！？][\s\"\u201d\u2019\)]*$', w.get('text','')))
        token = re.sub(r'[^a-z]', '', w.get('text','').lower())
        pause = (nxt is not None and nxt['start']-w['end'] >= POLICY['natural_pause_seconds']
                 and token not in DANGLING)
        if not (terminal or pause or nxt is None):
            continue
        unit = dict(start=words[first]['start'], end=w['end'], first=first, last=i,
                    reason='sentence_end' if terminal else 'natural_pause' if pause else 'transcript_end')
        if unit['end'] > start and unit['start'] < end:
            if unit['start'] >= start-1e-7 and unit['end'] <= end+1e-7:
                units.append(unit)
            else:
                rejected.append(dict(**unit, rejection='partial_sentence_at_floor_or_exclusion_edge'))
        first = i+1
    out, cursor = [], 0
    max_seconds = POLICY['max_frames']/POLICY['fps']
    while cursor < len(units):
        first_unit = units[cursor]
        choices = []
        for j in range(cursor, len(units)):
            u = units[j]
            # Never bridge a discarded partial sentence.
            if j > cursor and u['first'] != units[j-1]['last']+1:
                break
            a, b = first_unit['start'], u['end']
            if b-a > max_seconds+1e-7:
                break
            if b-a < POLICY['min_seconds']:
                continue
            frames = 8*math.ceil(((b-a)*25-1-1e-7)/8)+1
            duration = frames/25
            extra = max(0., duration-(b-a))
            prev_end = words[first_unit['first']-1]['end'] if first_unit['first'] else start
            next_start = words[u['last']+1]['start'] if u['last']+1 < len(words) else end
            left, right = max(start, prev_end), min(end, next_start)
            # VAD bounds supplement imperfect word timing: extend only through silence.
            for x,y in speaker_vad:
                if y <= a: left = max(left,y)
                elif x < a < y: left = a
                if x >= b: right = min(right,x)
                elif x < b < y: right = b
            left = max(left, out[-1]['end'] if out else start)
            tail = min(extra,max(0.,right-b))
            head = extra-tail
            if head > a-left+1e-7:
                continue
            x,y = a-head,b+tail
            if any(v < x < z or v < y < z for v,z in listener_activity):
                continue
            choices.append((j,dict(start=round(x,9),end=round(y,9),frames=frames,
                duration_s=duration,speech_start=a,speech_end=b,unique_seconds=b-a,
                end_reason=u['reason'],alignment_silence_seconds=extra,
                text=' '.join(w.get('text','') for w in words[first_unit['first']:u['last']+1]))))
        if choices:
            # Prefer a completed sentence even when a later in-sentence pause
            # could produce a longer clip. Length is secondary to a natural end.
            j, clip = max(choices,key=lambda p:(p[1]['end_reason']=='sentence_end',p[1]['duration_s']))
            out.append(clip); cursor=j+1
        else:
            rejected.append(dict(**first_unit,rejection='no_complete_4_to_9_96s_clip_with_silent_frame_alignment'))
            cursor+=1
    return out,rejected


def orientation(keypoints, width, height):
    """Infer raw-pixel quarter turn from face-to-shoulder vectors, vote over time.

    COCO whole-body nose=0, shoulders=5/6. Apply crop in raw coordinates,
    then this rotation with FFmpeg autorotate disabled (tags cannot double-rotate).
    """
    import numpy as np
    k = np.asarray(keypoints)[::5]
    if k.ndim != 3 or k.shape[1] < 7:
        raise ValueError('orientation_invalid_keypoint_shape')
    valid = (k[:,[0,5,6],2] > .5).all(axis=1)
    pts = k[:,[0,5,6],:2]
    valid &= ((pts[:,:,0] >= 0) & (pts[:,:,0] < width) &
              (pts[:,:,1] >= 0) & (pts[:,:,1] < height)).all(axis=1)
    v = k[:,0,:2] - (k[:,5,:2]+k[:,6,:2])/2
    valid &= np.linalg.norm(v,axis=1) > 20
    if valid.mean() < .6 or valid.sum() < 6:
        raise ValueError('orientation_insufficient_landmarks')
    v = v[valid]
    # Positive rotation means clockwise in displayed pixel coordinates.
    angles = np.degrees(np.arctan2(v[:,0],-v[:,1]))
    votes = np.rint(-angles/90).astype(int) % 4
    counts = np.bincount(votes,minlength=4); turn=int(counts.argmax())
    residual = np.abs((angles+turn*90+180)%360-180)
    confidence=float(((votes==turn)&(residual<30)).mean())
    if confidence < .85:
        raise ValueError('orientation_ambiguous')
    return dict(clockwise_degrees=turn*90,confidence=round(confidence,4),
                method='raw_COCO_nose_shoulders',valid_fraction=round(float(valid.mean()),4))


def require_upright(keypoints, width, height, metadata_rotation=0):
    value = orientation(keypoints, width, height)
    if value['clockwise_degrees'] != 0 or abs(metadata_rotation) % 360 > .01:
        raise ValueError('source_orientation_not_upright')
    return value
