"""Recover uncovered listening intervals without changing the frozen V8 run.

Every changed interval must pass fresh QC. A bad parent never supplies a child's
quality verdict. Only localized evidence may be subtracted; global/unknown
quality problems and human holds remain withheld. Geometry never gates recovery.
"""
import hashlib
import math
import re
import subprocess
import numpy as np
import plan_listen_v8_audio_full as old
from listen_policy_v8 import sentence_units, terminal

REVISION = 'local-listening-recovery-1'
GUARD = .15


def detect_shots(source):
    """Native-frame hard-cut proposals; both participant tracks are inspected."""
    process = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1',
        '-filter_threads', '1', '-noautorotate', '-i', old.base.media.uri(source['source']),
        '-vf', 'scale=320:-2,select=gt(scene\\,0.35),metadata=print:file=-',
        '-an', '-vsync', '0', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    times = [float(t) for t in re.findall(r'pts_time:([0-9.]+)', process.stdout)]
    return [t for t in times if .1 < t < source['duration']-.1]


def role_regions(people, limit, shots=()):
    """Scan the WHOLE source; legacy floor nominations do not gate extraction.

    Actual non-backchannel words determine the speaking side. Simultaneous
    content is localized, not allowed to invalidate the surrounding source.
    Pauses of <=3 seconds join only the same speaker; all hard cuts split both
    sides. Brief feedback does not take the floor from the primary speaker.
    """
    spans = [union((max(0., w['start']), min(limit, w['end'])) for w in p['words']
                   if not w.get('feedback') and math.isfinite(w['start']) and math.isfinite(w['end'])
                   and 0 < w['end']-w['start'] <= 1.5) for p in people]
    edges = sorted({0., limit} | {t for group in spans for ab in group for t in ab})
    indices = [0, 0]
    atoms = []
    for a, b in zip(edges, edges[1:]):
        active = []
        for si in (0, 1):
            while indices[si] < len(spans[si]) and spans[si][indices[si]][1] <= a+1e-8:
                indices[si] += 1
            if indices[si] < len(spans[si]) and spans[si][indices[si]][0] < b-1e-8:
                active.append(si)
        state = active[0] if len(active) == 1 else 'overlap' if active else 'silence'
        if atoms and atoms[-1]['state'] == state:
            atoms[-1]['end'] = b
        else:
            atoms.append(dict(start=a, end=b, state=state))
    # Fill small same-speaker pauses only. Overlap is never bridged.
    for i, atom in enumerate(atoms):
        if atom['state'] == 'silence' and 0 < i < len(atoms)-1 and atom['end']-atom['start'] <= 3:
            if atoms[i-1]['state'] in (0, 1) and atoms[i-1]['state'] == atoms[i+1]['state']:
                atom['state'] = atoms[i-1]['state']
    joined = []
    for atom in atoms:
        if joined and joined[-1]['state'] == atom['state']:
            joined[-1]['end'] = atom['end']
        else:
            joined.append(dict(atom))
    regions = []
    for i, atom in enumerate(joined):
        if atom['state'] not in (0, 1):
            continue
        a, b = atom['start'], atom['end']
        if i and joined[i-1]['state'] == 'silence':
            a -= min(.35, (joined[i-1]['end']-joined[i-1]['start'])/2)
        if i+1 < len(joined) and joined[i+1]['state'] == 'silence':
            b += min(.3, (joined[i+1]['end']-joined[i+1]['start'])/2)
        boundaries = [a] + sorted(t for t in set(shots) if a < t < b) + [b]
        regions.extend(dict(start=x, end=y, speaker_index=atom['state'])
                       for x, y in zip(boundaries, boundaries[1:]))
    return regions, [dict(start=x['start'], end=x['end'], reason=
                         'simultaneous_content_speech' if x['state'] == 'overlap' else 'no_primary_speech')
                     for x in joined if x['state'] not in (0, 1)]


def union(intervals):
    out = []
    for a, b in sorted(intervals, key=lambda ab:(ab[0],ab[1])):
        if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
            continue
        if out and a <= out[-1][1] + 1e-7:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def subtract(intervals, blocks):
    result = union(intervals)
    for x, y in union(blocks):
        result = [part for a, b in result for part in
                  ([(a, b)] if y <= a or x >= b else [(a, min(b, x)), (max(a, y), b)])
                  if part[1] > part[0] + 1e-7]
    return result


def seconds(intervals):
    return sum(b-a for a, b in union(intervals))


def semantic_blocks(person):
    """Refine coarse envelopes only with valid word timing AND available VAD.

    All actual words and VAD inside a substantive segment remain blocked. This
    is the interval form of V8's verified_transcript_gap, with a larger guard.
    """
    out = []
    for s in person['segments']:
        if not (s.get('formal_takeover') or
                (not s['feedback'] and (s['word_count'] > 7 or s['end']-s['start'] > 3))):
            continue
        ws = s.get('words', [])
        aligned = s.get('word_aligned') and ws and 'vad' in person
        aligned = aligned and all(math.isfinite(w[k]) for w in ws for k in ('start', 'end'))
        aligned = aligned and all(0 < w['end']-w['start'] <= 1.5 for w in ws)
        aligned = aligned and all(v['start'] >= u['end']-.001 for u, v in zip(ws, ws[1:]))
        spans = [(s['start'], s['end'])]
        if aligned:
            spans = [(w['start'], w['end']) for w in ws]
            spans += [(max(a, s['start']), min(b, s['end'])) for a, b in person['vad']
                      if b > s['start'] and a < s['end']]
        out.extend((a-GUARD, b+GUARD) for a, b in spans)
    return union(out)


def quality_blocks(row):
    """Return source-time exclusion windows, or the whole nonlocalizable parent.

    Evidence times are CLIP RELATIVE. All active reasons must have localization;
    one local event cannot override a second global/unknown reason.
    """
    route = row['audio_final']
    if route['pool'] == 'keep_candidate':
        return [], 'accepted'
    quality = route.get('quality', {})
    reasons = set(quality.get('hard_reasons', [])) | set(quality.get('review_reasons', []))
    if route['pool'] not in ('audio_reject', 'audio_review') or not reasons or any(
            r.startswith('human_') or r == 'prior_source_manual_rejection' for r in route['reasons']):
        return [(row['start'], row['end'])], 'human_or_task_hold'
    windows = list(quality.get('evidence_windows', []))
    # The final router omits windows for some reasons even though QC3/QC4
    # retain them. Recover that evidence rather than treating it as global.
    reference_reasons = reasons & {'strong_reference_match', 'consistent_reference_leakage',
                                    'isolated_reference_match_unresolved'}
    for w in row.get('qc4', {}).get('reference', {}).get('windows', []):
        if w.get('suspicious'):
            windows.extend(dict(start=w['start'],end=w['end'],reason=reason)
                           for reason in reference_reasons)
    if 'local_match_unresolved' in reasons:
        for track in row.get('qc3', {}).values():
            for w in track.get('events', [])+track.get('background', []):
                if 'possible_local_speaker_match' in w.get('decision', {}).get('reasons', []):
                    windows.append(dict(start=w['start'],end=w['end'],reason='local_match_unresolved'))
    located = {w['reason'] for w in windows if math.isfinite(w['start']) and math.isfinite(w['end'])
               and 0 <= w['start'] < w['end'] <= row['duration_s']+1e-5}
    if not reasons <= located:
        return [(row['start'], row['end'])], 'nonlocalizable_quality'
    blocks = [(max(row['start'], row['start']+w['start']-GUARD),
               min(row['end'], row['start']+w['end']+GUARD))
              for w in windows if w['reason'] in reasons]
    return union(blocks), 'localized_quality'


def make_unit(words, original_units):
    a, b = words[0]['start'], words[-1]['end']
    starts = {round(u['start'], 6): u for u in original_units}
    ends = {round(u['end'], 6): u for u in original_units}
    left, right = starts.get(round(a, 6)), ends.get(round(b, 6))
    complete = left is not None and right is not None
    flags = sorted({f for u in original_units if u['end'] > a and u['start'] < b
                    for f in u['semantic_flags']})
    if not complete:
        flags.append('recovered_partial_sentence')
    return dict(start=a, end=b, words=words, text=' '.join(w['text'] for w in words),
                semantic_flags=flags,
                start_evidence=left['start_evidence'] if left else 'recovered_word_boundary',
                end_evidence=right['end_evidence'] if right else 'recovered_word_boundary',
                complete_sentence_boundaries=complete)


def recover_region(people, rms, si, a, b, limit):
    """Try smaller units after failure, including explicit word-boundary fragments.

    Complete sentence ends take priority, then punctuation/pauses, then word
    boundaries. Every attempted cut uses the original PCM acoustic boundary
    checks. Rejection of one end does not prevent trying an earlier end.
    """
    all_words = people[si]['words']
    words = [w for w in all_words if a <= w['start'] and w['end'] <= b
             and 0 < w['end']-w['start'] <= 1.5]
    units = sentence_units(all_words)
    ends = {round(u['end'], 6) for u in units}
    clips, failures = [], []
    i, left = 0, a
    while i < len(words):
        if words[i]['start'] < left-1e-7:
            i += 1
            continue
        choices = []
        for j in range(i, len(words)):
            if j > i and (words[j]['start'] < words[j-1]['end']-.001 or
                          words[j]['start']-words[j-1]['end'] > 3):
                break
            duration = words[j]['end']-words[i]['start']
            if duration > 11.95:
                break
            if duration < 1.6:
                continue
            complete = round(words[j]['end'], 6) in ends
            gap = words[j+1]['start']-words[j]['end'] if j+1 < len(words) else 1
            natural = words[j]['text'].rstrip().endswith((',', ';', ':')) or gap >= .12
            choices.append((0 if complete else 1 if natural else 2,
                            0 if duration <= 10.8 else 1, -duration, j))
        last_error = 'remaining_span_too_short'
        found = False
        for *_, j in sorted(choices):
            unit = make_unit(words[i:j+1], units)
            cut, error = old.acoustic_cut(unit, people, rms, si, limit, left, b)
            if error:
                last_error = error
                continue
            if cut['start'] < left-1e-6 or cut['end'] > b+1e-6:
                last_error = 'cut_outside_safe_region'
                continue
            if any(w['start']+1e-6 < edge < w['end']-1e-6
                   for p in people for w in p['words'] for edge in (cut['start'], cut['end'])):
                last_error = 'boundary_crosses_annotated_word'
                continue
            cut['recovery_complete_sentence'] = unit['complete_sentence_boundaries']
            clips.append(cut)
            left = cut['end']
            i = j+1
            found = True
            break
        if not found:
            failures.append(dict(start=words[i]['start'], end=words[i]['end'],
                                 speaker_index=si, reason=last_error))
            i += 1
    return clips, failures


class SourceContext:
    def __init__(self, item, shot_times=None):
        self.item = item
        loaded = [old.load_person(item[k]) for k in ('a', 'b')]
        self.records = [x[0] for x in loaded]
        self.geometry = [x[1] for x in loaded]
        self.sources = [x[2] for x in loaded]
        self.people = [old.rules.prepare(r, i) for i, r in enumerate(self.records)]
        self.limit = min(s['duration'] for s in self.sources)
        self.floors, self.decisions = old.rules.track_floors(self.people, self.limit)
        for p in self.people:
            for s in p['segments']:
                # Legacy whole-turn role guesses are diagnostic only in this revision.
                s['formal_takeover'] = False
        self.rms = [old.raw_rms(s['source']) for s in self.sources]
        self.shots_by_track = [detect_shots(s) for s in self.sources] if shot_times is None else shot_times
        self.shots = sorted({t for group in self.shots_by_track for t in group if 0 < t < self.limit})
        self.regions, self.source_gaps = role_regions(self.people, self.limit, self.shots)
        import json
        self.reviews = json.loads((old.ROOT/'data_reviews/listener_noise_rejections.json').read_text())['rows']

    def row(self, cut, si):
        li = 1-si
        key = f'{si}:{cut["start"]:.6f}:{cut["end"]:.6f}'
        sid = self.item['pair']+'_local_'+hashlib.sha256(key.encode()).hexdigest()[:12]
        comp = cut['listener_compatibility']
        return dict(cut, sample_id=sid, pair=self.item['pair'], split=self.item['split'],
                    group_id=self.item['group_id'], policy_revision=REVISION,
                    speaker_index=si, listener_index=li, speaker_id=self.records[si]['id'],
                    listener_id=self.records[li]['id'], listener_crop=None, visual_filter_applied=False,
                    face_geometry={role:old.clip_geometry(self.geometry[index], cut['start'], cut['end'])
                                   for role, index in [('speaker', si), ('listener', li)]},
                    listener_events=[dict(start=s['start'], end=s['end'], text=s['text'],
                                         label='feedback_candidate' if s['feedback'] else 'short_speech_unconfirmed')
                                     for s in self.people[li]['segments']
                                     if s['end'] > cut['start'] and s['start'] < cut['end']],
                    boundary_activity=old.boundary_activity(cut['start'], cut['end'], self.people[li]),
                    source_manual_rejections=old.reviewed_noise_hits(self.records[li]['id'], cut['start'], cut['end'], self.reviews),
                    unique_seconds=cut['duration_s'], fps=self.sources[li]['fps'],
                    width=self.sources[li]['width'], height=self.sources[li]['height'],
                    sources={role:self.sources[index]['source'] for role, index in [('speaker', si), ('listener', li)]},
                    orientations=[dict(status='deferred', rotation_degrees=g['rotation_degrees']) for g in self.geometry],
                    source_backed=True, training_admitted=False, audio_qc_status='not_recomputed',
                    end_reason=cut['semantic_end'], frame_adaptation_required=True,
                    listener_screen=dict(accepted=True, deferred_reasons=['listener_short_speech_unconfirmed']
                                         if comp['uncertain_short_speech'] else []),
                    duration_tier='2to3s' if cut['duration_s'] < 3 else 'over10s' if cut['duration_s'] > 10 else 'ge3s')

    def propose(self, old_rows, occupied=(), domain=None, preserve_existing=False):
        keep = [(r['start'], r['end']) for r in old_rows if r['audio_final']['pool'] == 'keep_candidate'] if preserve_existing else []
        blocked_by_role = {0:list(occupied)+keep, 1:list(occupied)+keep}
        evidence = []
        for r in old_rows:
            spans, kind = quality_blocks(r)
            blocked_by_role[r['speaker_index']].extend(spans)
            evidence.extend(dict(start=a, end=b, reason=kind, parent_id=r['sample_id'],
                                 speaker_index=r['speaker_index']) for a, b in spans)
        rows, failures = [], []
        for floor in self.regions:
            si = floor['speaker_index']
            a, b = floor['start'], floor['end']
            if domain is not None:
                if domain['speaker_index'] != si:
                    continue
                a, b = max(a, domain['start']), min(b, domain['end'])
            if b <= a:
                continue
            invalid = [(w['start']-GUARD, w['end']+GUARD) for p in self.people for w in p['words']
                       if math.isfinite(w['start']) and math.isfinite(w['end']) and w['end']-w['start'] > 1.5]
            blockers = semantic_blocks(self.people[1-si]) + blocked_by_role[si] + invalid
            for lo, hi in subtract([(a, b)], blockers):
                if hi-lo < 2:
                    failures.append(dict(start=lo, end=hi, speaker_index=si, reason='safe_region_below_2s'))
                    continue
                cuts, bad = recover_region(self.people, self.rms, si, lo, hi, self.limit)
                failures.extend(bad)
                for cut in cuts:
                    rows.append(self.row(cut, si))
        rows.sort(key=lambda r:(r['start'], r['end']))
        assert all(x['end'] <= y['start']+1e-6 for x, y in zip(rows, rows[1:]))
        assert not any(min(r['end'], b) > max(r['start'], a)+1e-6 for r in rows for a, b in keep)
        return rows, dict(blocked=evidence, unresolved=failures)


def preview_crop(g, a, b):
    """Display proposal only: no size/validity threshold affects audio admission."""
    points = g['arrays'].get('boxes_and_keypoints:keypoints')
    if points is None:
        return dict(status='unmeasured')
    k = points[int(a*g['fps']):math.ceil(b*g['fps']), 23:91]
    valid = ((k[:, :, 2] > .4) & np.isfinite(k).all(axis=2)).sum(axis=1) >= 45
    if not valid.any():
        return dict(status='unmeasured')
    xy = np.where((k[valid, :, 2] > .4)[:, :, None], k[valid, :, :2], np.nan)
    low, high = np.nanpercentile(xy, [3, 97], axis=1)
    fw, fh = np.median(high-low, axis=0)
    left, top = np.percentile(low, 1, axis=0)
    right, chin = np.percentile(high, 99, axis=0)
    head_top = top-.55*fh
    desired = max(fh/.42, right-left+.5*fw, (chin-head_top)/.85)
    side = min(float(min(g['width'], g['height'])), float(desired))
    x = float(np.clip(np.median((low[:, 0]+high[:, 0])/2)-side/2, 0, g['width']-side))
    y = float(np.clip(head_top-.06*side, 0, g['height']-side))
    return dict(status='proposal_only', x=x, y=y, size=side, face_width=float(fw), face_height=float(fh),
                face_area_fraction=float(fw*fh/side**2), valid_fraction=float(valid.mean()),
                fits_estimated_head=bool(x <= left and x+side >= right and y <= head_top and y+side >= chin),
                output_face_height_px=float(fh*512/side), visual_filter_applied=False)
