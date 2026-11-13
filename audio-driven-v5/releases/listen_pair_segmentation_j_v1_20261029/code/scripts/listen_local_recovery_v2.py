"""Boundary repair for the two-source rerun; frozen V1 files stay reproducible.

Retain the original acoustic cutter and all semantic/quality barriers. Only
rebalance a greedy tail when every word in the replacement can be retained,
and extend a role's tail for nearby annotated feedback within unclaimed time.
"""
import math
import listen_local_recovery as base

REVISION = 'local-listening-boundary-repair-2'


def role_regions(people, limit, shots=()):
    regions, gaps = base.role_regions(people, limit, shots)
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
    clips, failures = base.recover_region(people, rms, si, a, b, limit)
    words = [w for w in people[si]['words'] if a <= w['start'] and w['end'] <= b
             and 0 < w['end']-w['start'] <= 1.5]
    units = base.sentence_units(people[si]['words'])

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
            v['start'] < u['end']-.001 or v['start']-u['end'] > 3 for u, v in zip(ws, ws[1:])):
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
