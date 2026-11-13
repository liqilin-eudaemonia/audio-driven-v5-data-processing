"""Two-source trial: protect sentence endings and review unsafe recovery fragments.

Keep H's five-second hard pause, local feedback, acoustic/QC rules and head
attachment. A bounded soft-pause lookahead protects a following full sentence.
No production module is modified and no quality hold is automatically cleared.
"""
import math
import listen_local_recovery_localbc as prior
import listen_local_recovery_head_attach as head

REVISION = 'sentence-integrity-two-source-trial-6'
MAX_WORD_GAP = prior.MAX_WORD_GAP
base = prior.base
prepare_people = prior.prepare_people
role_regions = prior.role_regions
quality_fallback = head.quality_fallback
sentence_units = prior.sentence_units


def safe_cut(ws, units, people, rms, si, limit, left, right):
    if not ws or ws[-1]['end']-ws[0]['start'] < 1.6:
        return None
    unit = base.make_unit(ws, units)
    cut, error = base.old.acoustic_cut(unit, people, rms, si, limit, left, right)
    if error or cut['start'] < left-1e-6 or cut['end'] > right+1e-6:
        return None
    if any(w['start']+1e-6 < edge < w['end']-1e-6
           for person in people for w in person['words'] for edge in (cut['start'], cut['end'])):
        return None
    cut['recovery_complete_sentence'] = unit['complete_sentence_boundaries']
    return cut


def greedy_with_lookahead(people, rms, si, a, b, limit, enabled=True):
    """Only reconsider a <=3-word, <1.6s leading prefix before a 1–5s pause.

    Trigger only if prefix+next sentence exceeds the hard maximum while the
    following sentence independently passes the original acoustic cutter.
    Unpairable prefixes stay explicit in diagnostics; they are not claimed kept.
    """
    words = [w for w in people[si]['words'] if a <= w['start'] and w['end'] <= b
             and 0 < w['end']-w['start'] <= 1.5]
    units = sentence_units(people[si]['words'])
    terminals = {round(u['end'], 6) for u in units if u['end_evidence']=='sentence_terminal'}
    if enabled:
        for k in range(1, min(4, len(words))):
            prefix = words[:k]
            if prefix[-1]['end']-prefix[0]['start'] >= 1.6:
                continue
            gap = words[k]['start']-prefix[-1]['end']
            if not 1.0 <= gap <= MAX_WORD_GAP:
                continue
            stop = next((j for j in range(k, len(words)) if round(words[j]['end'], 6) in terminals), None)
            if stop is None or words[stop]['end']-words[0]['start'] <= 11.95:
                continue
            suffix = words[k:stop+1]
            if any(v['start'] < u['end']-.001 or v['start']-u['end'] > MAX_WORD_GAP
                   for u,v in zip(suffix,suffix[1:])):
                continue
            boundary = (prefix[-1]['end']+words[k]['start'])/2
            if not safe_cut(suffix, units, people, rms, si, limit, boundary, b):
                continue
            first, failures = prior._greedy_recover_region(people, rms, si, a, boundary, limit)
            rest, errors = prior._greedy_recover_region(people, rms, si, boundary, b, limit)
            if not rest or rest[0]['semantic_end']!='sentence_terminal':
                continue
            rest[0]['soft_pause_rebalance'] = dict(gap_seconds=gap, prefix_text=' '.join(w['text'] for w in prefix),
                prefix_start=prefix[0]['start'], prefix_end=prefix[-1]['end'],
                reason='short_prefix_would_split_following_complete_sentence')
            if rest[0]['semantic_start']=='recovered_word_boundary':
                rest[0]['semantic_start']='after_soft_pause_unconfirmed'
            for failure in failures:
                failure['text']=' '.join(w['text'] for w in prefix)
                failure['note']='Short prefix not admitted as a pair; source words retained in diagnostics.'
            return first+rest, failures+errors
    return prior._greedy_recover_region(people, rms, si, a, b, limit)


def recover_region(people, rms, si, a, b, limit, soft_pause=True):
    clips, failures = greedy_with_lookahead(people, rms, si, a, b, limit, soft_pause)
    words = [w for w in people[si]['words'] if a <= w['start'] and w['end'] <= b
             and 0 < w['end']-w['start'] <= 1.5]
    units = sentence_units(people[si]['words'])
    def covered(w, rows):
        return any(c['start']<=w['start']+1e-6 and c['end']>=w['end']-1e-6 for c in rows)
    def cut(ws, left, right):
        return safe_cut(ws, units, people, rms, si, limit, left, right)
    for idx in range(len(clips)-1, -1, -1):
        previous=clips[idx];right=clips[idx+1]['start'] if idx+1<len(clips) else b
        ws=[w for w in words if w['start']>=previous['speech_start']-1e-7 and w['end']<=right+1e-7]
        if not ws or not any(not covered(w,[previous]) for w in ws):continue
        if ws[-1]['end']-ws[0]['start']>23.9 or any(v['start']<u['end']-.001 or v['start']-u['end']>MAX_WORD_GAP for u,v in zip(ws,ws[1:])):continue
        left=clips[idx-1]['end'] if idx else a
        replacement=None;whole=cut(ws,left,right)
        if whole:
            replacement=[whole]
        else:
            kinds={round(u['end'],6):u['end_evidence'] for u in units}
            def rank(k):
                natural=ws[k-1]['text'].rstrip().endswith((',', ';', ':')) or ws[k]['start']-ws[k-1]['end']>=.12
                return prior.end_priority(kinds.get(round(ws[k-1]['end'],6)),natural),abs(ws[k-1]['end']-previous['speech_end'])
            for k in sorted(range(1,len(ws)),key=rank):
                first=cut(ws[:k],left,right);second=cut(ws[k:],first['end'],right) if first else None
                if second:replacement=[first,second];break
        if replacement is None:continue
        # The original F fix must not trade a selected sentence end for a
        # trailing fragment of the next sentence, even if more words fit.
        if previous['semantic_end']=='sentence_terminal' and replacement[-1]['semantic_end']!='sentence_terminal':
            previous['sentence_end_protection']=dict(rejected_end=replacement[-1]['end'],
                rejected_text=replacement[-1]['text'],reason='preserve_terminal_instead_of_appending_next_sentence_fragment')
            continue
        assert all(covered(w,replacement) for w in ws)
        for c in replacement:
            c['boundary_repair']=dict(reason='rebalance_greedy_short_tail',previous_start=previous['start'],previous_end=previous['end'])
        if previous.get('soft_pause_rebalance'):
            replacement[0]['soft_pause_rebalance']=previous['soft_pause_rebalance']
        clips[idx:idx+1]=replacement
    failures=[f for f in failures if not covered(f,clips)]
    assert all(x['end']<=y['start']+1e-6 for x,y in zip(clips,clips[1:]))
    return clips,failures


def recovery_is_complete(row):
    # Conservative evidence for this trial; this is not an English grammar
    # classifier. Ambiguous recoveries preserve their parent for review.
    return row['semantic_end']=='sentence_terminal' and row['semantic_start'] in {
        'after_sentence_terminal','source_transcript_start','inferred_clause_start','after_soft_pause_unconfirmed'}


class SourceContext(head.SourceContext):
    def row(self, cut, si):
        return dict(super().row(cut,si),policy_revision=REVISION)

    def propose(self, old_rows, occupied=(), domain=None, preserve_existing=False):
        keep=[(r['start'],r['end']) for r in old_rows if r['audio_final']['pool']=='keep_candidate'] if preserve_existing else []
        blocked={0:list(occupied)+keep,1:list(occupied)+keep};evidence=[]
        for row in old_rows:
            spans,kind=base.quality_blocks(row);blocked[row['speaker_index']].extend(spans)
            evidence.extend(dict(start=a,end=b,reason=kind,parent_id=row['sample_id'],speaker_index=row['speaker_index']) for a,b in spans)
        rows=[];failures=[]
        invalid=[(w['start']-base.GUARD,w['end']+base.GUARD) for p in self.people for w in p['words']
                 if math.isfinite(w['start']) and math.isfinite(w['end']) and w['end']-w['start']>1.5]
        for region in self.regions:
            si=region['speaker_index'];a,b=region['start'],region['end']
            if domain is not None:
                if domain['speaker_index']!=si:continue
                a,b=max(a,domain['start']),min(b,domain['end'])
            if b<=a:continue
            blockers=prior.semantic_blocks(self.people[1-si])+blocked[si]+invalid
            for lo,hi in base.subtract([(a,b)],blockers):
                if hi-lo<2:
                    failures.append(dict(start=lo,end=hi,speaker_index=si,reason='safe_region_below_2s'));continue
                cuts,bad=recover_region(self.people,self.rms,si,lo,hi,self.limit,soft_pause=domain is None)
                rows.extend(self.row(c,si) for c in cuts);failures.extend(bad)
        rows.sort(key=lambda r:(r['start'],r['end']))
        assert all(x['end']<=y['start']+1e-6 for x,y in zip(rows,rows[1:]))
        diagnostic=dict(blocked=evidence,unresolved=failures)
        if domain is not None:
            if rows and any(not recovery_is_complete(r) for r in rows):
                rejected=[{k:r[k] for k in ('start','end','text','semantic_start','semantic_end')} for r in rows]
                domain['semantic_recovery_hold']=dict(reason='recovery_does_not_preserve_complete_units',proposed_children=rejected)
                diagnostic['unresolved'].append(dict(start=domain['start'],end=domain['end'],reason='original_candidate_preserved_for_review_instead_of_incomplete_recovery',proposed_children=rejected))
                return [],diagnostic
            return rows,diagnostic
        protected=list(occupied)
        for row in old_rows:
            spans,_=base.quality_blocks(row);protected.extend(spans)
            if preserve_existing and row['audio_final']['pool']=='keep_candidate':protected.append((row['start'],row['end']))
        rows,audit=head.attach_heads(self,rows,protected);diagnostic['head_attachment']=audit
        return rows,diagnostic
