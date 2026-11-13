"""Five-second pause segmentation for the two-source rerun.

Static derivation of frozen V1 / V2 planners. Only the adjacent-word pause
threshold changes; duration limits, task barriers, acoustic cuts and the two
boundary repairs are retained. No global patching of the frozen modules.
"""
import math
import re
import listen_local_recovery as base
from listen_policy_v8 import terminal, token, TRAILING

REVISION = 'local-listening-boundary-repair-gap5-3'
MAX_WORD_GAP = 5.0
union = base.union
make_unit = base.make_unit
old = base.old


def sentence_units(words):
    """Build on the global word timeline, before any floor/exclusion clipping."""
    ws=sorted([w for w in words if w['end']>w['start']],key=lambda w:(w['start'],w['end']))
    if not ws:return []
    starts={0:'source_transcript_start'}
    ends={}
    for i,(left,right) in enumerate(zip(ws,ws[1:])):
        gap=right['start']-left['end']
        if terminal(left['text']):
            starts[i+1]='after_sentence_terminal';ends[i]='sentence_terminal'
        elif gap>MAX_WORD_GAP:
            starts[i+1]='after_long_gap_unconfirmed';ends[i]='before_long_gap_unconfirmed'
        elif (gap>=.35 and left['text'].rstrip().endswith((',', ';', ':'))
              and re.match(r'^["“\']?[A-Z]',right['text'])
              and token(left['text']) not in TRAILING):
            # ASR sometimes writes "..., well, The concept ...". This is an
            # explicit clause hypothesis, never silently called a full sentence.
            starts[i+1]='inferred_clause_start';ends[i]='inferred_clause_end'
    positions=sorted(starts)
    units=[]
    for k,i in enumerate(positions):
        j=positions[k+1]-1 if k+1<len(positions) else len(ws)-1
        subset=ws[i:j+1];end_kind=ends.get(j,'sentence_terminal' if terminal(ws[j]['text']) else 'transcript_end_unconfirmed')
        flags=[]
        if starts[i] in ('after_long_gap_unconfirmed','inferred_clause_start'):
            flags.append(starts[i])
        if starts[i]=='source_transcript_start' and not re.match(r'^["“\']?[A-Z]',ws[i]['text']):
            flags.append('source_sentence_start_unconfirmed')
        if end_kind!='sentence_terminal':flags.append(end_kind)
        if any(w['end']-w['start']>1.5 for w in subset):flags.append('word_timestamp_over_1_5s')
        if any(b['start']<a['end']-.001 for a,b in zip(subset,subset[1:])):flags.append('word_timestamp_conflict')
        units.append(dict(start=subset[0]['start'],end=max(w['end'] for w in subset),
            words=subset,text=' '.join(w['text'] for w in subset),first_word_index=i,last_word_index=j,
            start_evidence=starts[i],end_evidence=end_kind,semantic_flags=sorted(set(flags))))
    return units


def _role_regions(people, limit, shots=()):
    """Scan the WHOLE source; legacy floor nominations do not gate extraction.

    Actual non-backchannel words determine the speaking side. Simultaneous
    content is localized, not allowed to invalidate the surrounding source.
    Pauses of <=5 seconds join only the same speaker; all hard cuts split both
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
        if atom['state'] == 'silence' and 0 < i < len(atoms)-1 and atom['end']-atom['start'] <= MAX_WORD_GAP:
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


def _greedy_recover_region(people, rms, si, a, b, limit):
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
                          words[j]['start']-words[j-1]['end'] > MAX_WORD_GAP):
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


def role_regions(people, limit, shots=()):
    regions, gaps = _role_regions(people, limit, shots)
    for i, region in enumerate(regions):
        si = region['speaker_index']; end = region['end']
        bound = min([limit] + [r['start'] for r in regions[i+1:]] +
                    [t for t in shots if t >= end-1e-7])
        words = [w for w in people[si]['words'] if not w.get('feedback') and
                 region['start'] <= w['start'] < end and w['end'] <= end]
        if not words:
            continue
        speech_end = max(w['end'] for w in words)
        for feedback in people[1-si].get('segments', []):
            if (feedback.get('feedback') and not feedback.get('long_feedback_timing_uncertain')
                and feedback['start'] < speech_end+.2 and
                speech_end < feedback['end'] <= speech_end+1.5 and
                end < feedback['end'] <= bound):
                region['end'] = max(region['end'], min(bound, feedback['end']+.3))
        if region['end'] > end+1e-7:
            region['boundary_repair'] = dict(reason='complete_nearby_feedback_tail', previous_end=end)
    assert all(a['end'] <= b['start']+1e-6 for a, b in zip(regions, regions[1:]))
    assert all(not r['start'] < t < r['end'] for r in regions for t in shots)
    return regions, gaps


def recover_region(people, rms, si, a, b, limit):
    clips, failures = _greedy_recover_region(people, rms, si, a, b, limit)
    words = [w for w in people[si]['words'] if a <= w['start'] and w['end'] <= b
             and 0 < w['end']-w['start'] <= 1.5]
    units = sentence_units(people[si]['words'])

    def covered(w, rows):
        return any(c['start'] <= w['start']+1e-6 and c['end'] >= w['end']-1e-6 for c in rows)

    def cut(ws, left, right):
        if not ws or ws[-1]['end']-ws[0]['start'] < 1.6:
            return None
        unit = base.make_unit(ws, units)
        c, error = base.old.acoustic_cut(unit, people, rms, si, limit, left, right)
        if error or c['start'] < left-1e-6 or c['end'] > right+1e-6:
            return None
        if any(w['start']+1e-6 < edge < w['end']-1e-6
               for p in people for w in p['words'] for edge in (c['start'], c['end'])):
            return None
        c['recovery_complete_sentence'] = unit['complete_sentence_boundaries']
        return c

    # Work backwards so replacing one clip with two does not invalidate earlier
    # indices. The search is bounded to one retained clip and its trailing gap.
    for idx in range(len(clips)-1, -1, -1):
        previous = clips[idx]
        right = clips[idx+1]['start'] if idx+1 < len(clips) else b
        ws = [w for w in words if w['start'] >= previous['speech_start']-1e-7 and w['end'] <= right+1e-7]
        if not ws or not any(not covered(w, [previous]) for w in ws):
            continue
        if ws[-1]['end']-ws[0]['start'] > 23.9 or any(
            v['start'] < u['end']-.001 or v['start']-u['end'] > MAX_WORD_GAP for u, v in zip(ws, ws[1:])):
            continue
        left = clips[idx-1]['end'] if idx else a
        replacement = None
        whole = cut(ws, left, right)
        if whole:
            replacement = [whole]
        else:
            ends = {round(u['end'], 6) for u in units}
            def rank(k):
                natural = ws[k-1]['text'].rstrip().endswith((',', ';', ':')) or ws[k]['start']-ws[k-1]['end'] >= .12
                return (0 if round(ws[k-1]['end'], 6) in ends else 1 if natural else 2,
                        abs(ws[k-1]['end']-previous['speech_end']))
            for k in sorted(range(1, len(ws)), key=rank):
                first = cut(ws[:k], left, right)
                second = cut(ws[k:], first['end'], right) if first else None
                if second:
                    replacement = [first, second]
                    break
        if replacement is None:
            continue
        assert all(covered(w, replacement) for w in ws)
        for c in replacement:
            c['boundary_repair'] = dict(reason='rebalance_greedy_short_tail',
                previous_start=previous['start'], previous_end=previous['end'])
        clips[idx:idx+1] = replacement
    failures = [f for f in failures if not covered(f, clips)]
    assert all(x['end'] <= y['start']+1e-6 for x, y in zip(clips, clips[1:]))
    return clips, failures


class SourceContext(base.SourceContext):
    def __init__(self, item, shot_times=None):
        super().__init__(item, shot_times)
        self.regions, self.source_gaps = role_regions(self.people, self.limit, self.shots)

    def row(self, cut, si):
        return dict(super().row(cut, si), policy_revision=REVISION)

    def propose(self, old_rows, occupied=(), domain=None, preserve_existing=False):
        # Same barriers as V1; only role regions and recover_region differ.
        keep = [(r['start'], r['end']) for r in old_rows if r['audio_final']['pool'] == 'keep_candidate'] if preserve_existing else []
        blocked_by_role = {0:list(occupied)+keep, 1:list(occupied)+keep}
        evidence = []
        for row in old_rows:
            spans, kind = base.quality_blocks(row)
            blocked_by_role[row['speaker_index']].extend(spans)
            evidence.extend(dict(start=a, end=b, reason=kind, parent_id=row['sample_id'],
                                 speaker_index=row['speaker_index']) for a, b in spans)
        rows, failures = [], []
        invalid = [(w['start']-base.GUARD, w['end']+base.GUARD) for p in self.people for w in p['words']
                   if math.isfinite(w['start']) and math.isfinite(w['end']) and w['end']-w['start'] > 1.5]
        for region in self.regions:
            si = region['speaker_index']; a, b = region['start'], region['end']
            if domain is not None:
                if domain['speaker_index'] != si:
                    continue
                a, b = max(a, domain['start']), min(b, domain['end'])
            if b <= a:
                continue
            blockers = base.semantic_blocks(self.people[1-si]) + blocked_by_role[si] + invalid
            for lo, hi in base.subtract([(a, b)], blockers):
                if hi-lo < 2:
                    failures.append(dict(start=lo, end=hi, speaker_index=si, reason='safe_region_below_2s'))
                    continue
                cuts, bad = recover_region(self.people, self.rms, si, lo, hi, self.limit)
                failures.extend(bad)
                rows.extend(self.row(cut, si) for cut in cuts)
        rows.sort(key=lambda r:(r['start'], r['end']))
        assert all(x['end'] <= y['start']+1e-6 for x, y in zip(rows, rows[1:]))
        assert not any(min(r['end'], b) > max(r['start'], a)+1e-6 for r in rows for a, b in keep)
        return rows, dict(blocked=evidence, unresolved=failures)
