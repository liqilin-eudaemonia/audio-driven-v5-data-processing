"""Trial: attach otherwise uncovered short feedback to the same person's next turn.

The existing local-feedback/sentence-priority planner runs first. Only initial
candidates may gain a prefix; endings and other candidates never move. Quality
recovery children cannot expand back into their parent's excluded audio.
"""
import math
import copy
import listen_local_recovery_localbc as prior

REVISION = 'local-feedback-head-attachment-5'
MAX_HEAD_GAP = .3
MAX_WORD_GAP = prior.MAX_WORD_GAP
base = prior.base
prepare_people = prior.prepare_people
role_regions = prior.role_regions


def overlap(a, b, x, y):
    return min(b, y) > max(a, x) + 1e-6


def valid_words(words):
    return bool(words) and all(math.isfinite(w[k]) for w in words for k in ('start', 'end')) and all(
        0 < w['end']-w['start'] <= 1.5 for w in words) and all(
        v['start'] >= u['end']-.001 for u, v in zip(words, words[1:]))


def attach_heads(ctx, rows, protected=()):
    """Return new rows plus an audit; existing rows/word classifications stay intact."""
    revised = list(rows)
    audit = []
    for index, row in enumerate(revised):
        si = row['speaker_index']
        first = row['speech_start']
        candidates = [s for s in ctx.people[si]['segments']
            if s.get('feedback') and s.get('word_aligned')
            and not s.get('long_feedback_timing_uncertain')
            and 0 < s['end']-s['start'] <= 3
            and -1e-6 <= first-s['end'] <= MAX_HEAD_GAP+1e-6
            and s['start'] < row['start']-1e-6 and valid_words(s.get('words', []))]
        for event in sorted(candidates, key=lambda s:s['end'], reverse=True):
            entry = dict(start=event['start'], end=event['end'], text=event['text'],
                         speaker_index=si, candidate_id=row['sample_id'],
                         gap_seconds=round(first-event['end'],6), attached=False)
            # Any partial coverage is also excluded: do not duplicate a reaction
            # or move one that already belongs to the other participant's pair.
            if any(overlap(event['start'], event['end'], r['start'], r['end']) for r in revised):
                continue
            audit.append(entry)
            words = [w for w in ctx.people[si]['words']
                     if event['start']-1e-6 <= w['start'] and w['end'] <= row['speech_end']+1e-6]
            if not valid_words(words):
                entry['reason'] = 'unreliable_word_alignment';continue
            left = max([0.] + [r['end'] for r in revised[:index]] +
                       [t for t in ctx.shots if t <= first])
            barriers = list(protected) + prior.semantic_blocks(ctx.people[1-si])
            # Keep all substantive words of the other person out of the newly
            # attached prefix, even short utterances permitted elsewhere.
            barriers += [(w['start']-base.GUARD,w['end']+base.GUARD)
                         for w in ctx.people[1-si]['words'] if not w.get('feedback')
                         and math.isfinite(w['start']) and math.isfinite(w['end'])
                         and overlap(w['start']-base.GUARD,w['end']+base.GUARD,event['start'],first)]
            barriers += [(w['start']-base.GUARD,w['end']+base.GUARD)
                         for p in ctx.people for w in p['words']
                         if math.isfinite(w['start']) and math.isfinite(w['end']) and w['end']-w['start']>1.5]
            safe = next(((a,b) for a,b in base.subtract([(left,row['end'])],barriers)
                         if a <= event['start']+1e-6 and b >= row['end']-1e-6),None)
            if safe is None:
                entry['reason'] = 'protected_boundary_or_other_speaker';continue
            unit = base.make_unit(words,prior.sentence_units(ctx.people[si]['words']))
            cut,error = base.old.acoustic_cut(unit,ctx.people,ctx.rms,si,ctx.limit,*safe)
            if error:
                entry['reason'] = error;continue
            if (cut['start'] < safe[0]-1e-6 or abs(cut['end']-row['end'])>1e-6
                or cut['start'] > event['start']+1e-6 or cut['start'] >= row['start']-1e-6
                or not 2-1e-6 <= cut['duration_s'] <= 11.95+1e-6):
                entry['reason'] = 'would_change_end_or_violate_bounds';continue
            if any(w['start']+1e-6 < edge < w['end']-1e-6
                   for p in ctx.people for w in p['words'] for edge in (cut['start'],cut['end'])):
                entry['reason'] = 'boundary_crosses_annotated_word';continue
            if any(overlap(cut['start'],cut['end'],r['start'],r['end'])
                   for j,r in enumerate(revised) if j!=index):
                entry['reason'] = 'would_overlap_existing_candidate';continue
            cut['recovery_complete_sentence'] = unit['complete_sentence_boundaries']
            cut['head_attachment'] = dict(feedback_start=event['start'],feedback_end=event['end'],
                feedback_text=event['text'],gap_seconds=entry['gap_seconds'],
                previous_start=row['start'],previous_end=row['end'],previous_sample_id=row['sample_id'])
            cut['head_attachment_original'] = copy.deepcopy(row)
            if row.get('boundary_repair'):cut['boundary_repair']=row['boundary_repair']
            revised[index] = ctx.row(cut,si)
            entry.update(attached=True,reason='uncovered_feedback_attached_to_same_speaker',
                         new_start=cut['start'],new_end=cut['end'],new_sample_id=revised[index]['sample_id'])
            break
    assert all(a['end']<=b['start']+1e-6 for a,b in zip(revised,revised[1:]))
    return revised,audit


class SourceContext(prior.SourceContext):
    def row(self,cut,si):
        return dict(super().row(cut,si),policy_revision=REVISION)

    def propose(self,old_rows,occupied=(),domain=None,preserve_existing=False):
        rows,diagnostic = super().propose(old_rows,occupied,domain,preserve_existing)
        if domain is not None:
            return rows,diagnostic
        protected=list(occupied)
        for r in old_rows:
            spans,_=base.quality_blocks(r);protected.extend(spans)
            if preserve_existing and r['audio_final']['pool']=='keep_candidate':
                protected.append((r['start'],r['end']))
        rows,audit=attach_heads(self,rows,protected)
        diagnostic['head_attachment']=audit
        return rows,diagnostic


def quality_fallback(plan, assessed):
    """A failed optional prefix never replaces the original full candidate."""
    if assessed['audio_final']['pool']!='keep_candidate' and plan.get('head_attachment_original'):
        original=copy.deepcopy(plan['head_attachment_original'])
        original['head_attachment_fallback']=dict(rejected_expansion_id=plan['sample_id'],
            reason='prefix_extension_failed_unchanged_quality_rules')
        return [original]
    return None
