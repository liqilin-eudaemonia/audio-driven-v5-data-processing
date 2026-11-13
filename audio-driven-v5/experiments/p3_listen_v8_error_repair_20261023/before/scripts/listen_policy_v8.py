"""Complete semantic units first, then dual-track acoustic cuts for LISTEN only.

No arbitrary word-gap windows, no response-stage extension, no waveform edits.
Uncertain or overlong units remain explicitly recorded for review.
"""
import math
import re
import numpy as np

POLICY = dict(revision=8, min_seconds=2., ordinary_max_seconds=10., max_seconds=11.95,
    extension_reason='preserve_complete_speaker_unit_and_its_acoustic_boundaries',
    head_target=.2, head_search=.35, tail_target=.3, tail_search=1.5,
    quiet_min_seconds=.06, next_voice_guard=.08, minimum_crop_side=448,
    max_adjacent_word_gap=3., onset_policy='retain_and_label_not_automatically_exclude',
    quality_policy='old_QC_not_inherited_after_boundary_changes', frame_alignment_filter=False)
TRAILING = set('and or but to the a an of in on at for with because so that my your his her our their is was were are be have has had would could should will can'.split())


def token(text):
    return re.sub('[^a-z]', '', text.lower())


def terminal(text):
    t=text.rstrip().rstrip('"\'”’)]')
    return (bool(re.search(r'[.!?。！？]$', t)) and not t.endswith('...')
            and t.lower() not in {'mr.','mrs.','ms.','dr.','prof.','etc.'})


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
        elif gap>3:
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


def listener_compatibility(person,a,b):
    feedback=[];uncertain=[];blockers=[]
    for s in person['segments']:
        if s['end']<=a+1e-6 or s['start']>=b-1e-6:continue
        item={k:s[k] for k in ('start','end','text','id')}
        if s.get('formal_takeover'):blockers.append(item)
        elif s['feedback']:feedback.append(item)
        elif s['word_count']<=7 and s['end']-s['start']<=3:
            uncertain.append(item)
        else:blockers.append(item)
    return dict(feedback=feedback,uncertain_short_speech=uncertain,substantive=blockers)


def group_units(units,listener):
    """Cover complete units with nonoverlapping groups; exceptional >10s units
    stand alone. No content word is dropped to meet a duration budget.
    """
    units=sorted(units,key=lambda u:(u['start'],u['end']))
    n=len(units);best=[None]*(n+1);prev=[None]*(n+1);best[0]=(0.,0.)
    for i in range(n):
        u=units[i]
        skip=(best[i][0]+sum(w['end']-w['start'] for w in u['words']),best[i][1])
        if best[i+1] is None or skip<best[i+1]:best[i+1]=skip;prev[i+1]=(i,False)
        for j in range(i,n):
            if j>i and (units[j]['first_word_index']!=units[j-1]['last_word_index']+1
                         or units[j]['start']-units[j-1]['end']>3):break
            a,b=u['start'],units[j]['end'];d=b-a
            if d>11.95+1e-7:break
            if j>i and d>9.5:break  # reserve natural margins; long allowance is not a packing target
            if d<1.6:continue  # acoustic margins may make the final interval >=2 seconds
            if listener_compatibility(listener,a,b)['substantive']:break
            score=(best[i][0],best[i][1]+1+max(0,5-d)*.1)
            if best[j+1] is None or score<best[j+1]:best[j+1]=score;prev[j+1]=(i,True)
    groups=[];unresolved=[];j=n
    while j:
        i,keep=prev[j];parts=units[i:j]
        if keep:
            words=[w for u in parts for w in u['words']]
            groups.append(dict(start=parts[0]['start'],end=parts[-1]['end'],words=words,
                text=' '.join(w['text'] for w in words),start_evidence=parts[0]['start_evidence'],
                end_evidence=parts[-1]['end_evidence'],
                semantic_flags=sorted({f for u in parts for f in u['semantic_flags']}),
                sentence_units=len(parts)))
        else:
            u=parts[0];reason='complete_unit_exceeds_11_95s' if u['end']-u['start']>11.95 else 'complete_unit_too_short_or_blocked'
            unresolved.append(dict(start=u['start'],end=u['end'],text=u['text'],reason=reason))
        j=i
    return list(reversed(groups)),list(reversed(unresolved))


def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))


def quiet_point(times,levels,thresholds,lower,upper,nominal):
    if lower>upper:return None
    choices=[]
    for a,b in runs(np.all(levels<=thresholds,axis=1)):
        if b-a<6:continue
        lo=max(lower,float(times[a]+.025));hi=min(upper,float(times[b-1]-.025))
        if lo<=hi:choices.append(float(np.clip(nominal,lo,hi)))
    return min(choices,key=lambda v:abs(v-nominal)) if choices else None


def acoustic_cut(unit,people,rms,speaker,limit,left_bound=0.,right_bound=None):
    """Choose shared A/V bounds from raw 10ms RMS. Formal listener response
    never becomes part of the target. Uncertain quiet cuts retain review flags.
    """
    li=1-speaker;a=unit['start'];b=unit['end'];right_bound=limit if right_bound is None else right_bound
    flags=list(unit['semantic_flags']);compat=listener_compatibility(people[li],a,b)
    if compat['substantive']:return None,'listener_substantive_speech'
    if compat['uncertain_short_speech']:flags.append('listener_short_speech_unconfirmed')
    content_start=a;content_end=b
    # Preserve a recognized micro-reaction straddling either speaker boundary.
    for s in people[li]['segments']:
        if not s['feedback']:continue
        if s.get('long_feedback_timing_uncertain') and s['end']>a and s['start']<b:
            flags.append('listener_feedback_timing_unconfirmed')
        if s['start']<a<s['end'] and s['start']>=a-.35:content_start=min(content_start,s['start'])
        if s['start']<b+.2 and s['end']>b and s['end']<=b+1.5:
            content_end=max(content_end,s['end'])
    # A nearby VAD tail may be laughter without text. Preserve its duration but
    # retain an uncertainty flag; VAD is not a positive reaction label.
    for x,y in people[li]['vad']:
        if x<b<y and y<=b+1.5 and y>content_end:
            content_end=y;flags.append('untranscribed_listener_tail')
    if any(w['start']>=b-1e-7 and w['start']<content_end-1e-7
           and w['end']>b+1e-7 for w in people[speaker]['words']):
        return None,'listener_tail_overlaps_next_speaker_sentence'
    for s in people[li]['segments']:
        if s['feedback']:continue
        if s['start']<content_end and s['end']>b+1e-6:
            if s.get('formal_takeover') or s['word_count']>7 or s['end']-s['start']>3:
                return None,'tail_would_include_formal_listener_response'
            flags.append('listener_short_speech_unconfirmed')
    previous=[w['end'] for p in people for w in p['words'] if w['end']<=content_start+1e-7]
    following=[w['start'] for p in people for w in p['words'] if w['start']>=content_end-1e-7 and w['end']>content_end+1e-7]
    prev=max(previous,default=0.);nxt=min(following,default=limit)
    first=max(0,int((a-3)/.01));last=min(min(len(r) for r in rms),math.ceil((content_end+2)/.01))
    times=(np.arange(first,last)+.5)*.01
    if not len(times):return None,'missing_acoustic_context'
    values=np.stack([r[first:last] for r in rms],axis=1)
    levels=20*np.log10(np.maximum(values,1e-10));thresholds=[]
    for role,p in enumerate(people):
        active=np.zeros(len(times),bool)
        for w in p['words']:
            if w['end']>times[0] and w['start']<times[-1]:active|=(times>=w['start'])&(times<w['end'])
        speech=values[active,role]
        floor=float(np.percentile(levels[:,role],15))
        if len(speech)<2:
            # A quiet listener has no local speech reference. Background P90
            # must not be treated as speech (that makes the quiet threshold
            # lower than the background itself). Use real whole-track words.
            global_active=np.zeros(len(rms[role]),bool)
            for w in p['words']:
                lo=max(0,math.ceil(w['start']/.01-.5));hi=min(len(global_active),math.ceil(w['end']/.01-.5))
                if hi>lo:global_active[lo:hi]=True
            speech=rms[role][global_active]
        if len(speech):
            ref=20*np.log10(max(float(np.sqrt(np.mean(speech**2))),1e-10))
            thresholds.append(max(-80,min(ref-12,max(ref-24,floor+6,-70))))
        else:
            # Relative quietness is a boundary criterion, not a clean-audio
            # assertion. Stationary noise remains subject to independent QC.
            thresholds.append(max(-80,min(-20,floor+6)))
    thresholds=np.array(thresholds)
    lower=max(0.,left_bound,a-.35,min(content_start,prev+.02))
    upper=content_start-.02;nominal=max(lower,content_start-.2)
    # If physical onset precedes the annotated first word, keep a safety margin.
    for x,y in runs((times>=max(lower,a-.2))&(times<=a+.1)&(levels[:,speaker]>thresholds[speaker]+3)):
        if y-x>=4:upper=min(upper,float(times[x]-.035));break
    start=quiet_point(times,levels,thresholds,lower,upper,nominal)
    if start is None:start=min(content_start,max(lower,content_start-.2));flags.append('start_no_quiet_cut')
    if content_end-start>11.95:
        # Tighten margins only; never remove a retained word or micro-reaction.
        candidate=quiet_point(times,levels,thresholds,max(lower,content_end-11.95),upper,content_end-11.95)
        start=candidate if candidate is not None else max(start,content_end-11.95)
        if candidate is None:flags.append('start_margin_tightened_without_quiet_cut')
    if start>content_start+1e-7:return None,'complete_content_exceeds_11_95s'
    guard=nxt
    for role,p in enumerate(people):
        if any(abs(w['start']-nxt)<1e-5 for w in p['words']):
            for x,y in runs(levels[:,role]>thresholds[role]+3):
                if y-x>=4 and max(content_end+.02,nxt-.2)<=times[x]<=nxt+.1:
                    guard=min(guard,float(times[x]-.005));break
    hard_upper=min(limit,right_bound,b+1.5,start+11.95,nxt)
    upper=min(hard_upper,guard-.08)
    lower=content_end+.12;nominal=min(content_end+.3,upper)
    # Detect remaining speaker tail energy after its final aligned word.
    for x,y in runs((times>=b)&(times<=min(content_end+.3,upper))&(levels[:,speaker]>thresholds[speaker]+3)):
        if y-x>=4:lower=max(lower,float(times[y-1]+.065))
    end=quiet_point(times,levels,thresholds,lower,upper,nominal)
    if end is None:
        if hard_upper<content_end-1e-7:return None,'no_room_for_complete_tail_before_next_voice'
        if upper<content_end:
            upper=content_end;flags.append('next_voice_guard_too_close')
        end=max(content_end,min(content_end+.3,upper));flags.append('end_no_quiet_cut')
        if end-content_end<.12:flags.append('insufficient_tail_room')
    if end-start<2-1e-7:return None,'complete_unit_below_2s'
    assert end-start<=11.95+1e-6 and start<=a+1e-6 and end>=b-1e-6
    for edge,name in [(start,'start'),(end,'end')]:
        if any(w['start']+1e-6<edge<w['end']-1e-6 for p in people for w in p['words']):flags.append(name+'_crosses_annotated_voice')
        if any(x+1e-6<edge<y-1e-6 for x,y in people[li]['vad']):flags.append(name+'_crosses_listener_vad')
    return dict(start=round(start,6),end=round(end,6),duration_s=round(end-start,6),
        speech_start=a,speech_end=b,text=unit['text'],semantic_start=unit['start_evidence'],
        semantic_end=unit['end_evidence'],boundary_review_flags=sorted(set(flags)),
        start_padding_s=round(a-start,6),tail_s=round(end-b,6),
        quiet_threshold_dbfs=thresholds.tolist(),next_voice_start=nxt,
        extended_for_completeness=end-start>10+1e-7,
        boundary_status='review' if flags else 'ready_for_audio_qc',listener_compatibility=compat),None
