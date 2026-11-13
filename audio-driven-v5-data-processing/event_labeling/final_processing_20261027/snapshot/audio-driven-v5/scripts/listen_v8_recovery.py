"""Isolated yield experiment; no changes to frozen production rules or admission.

Only ignore a transcript envelope in a gap corroborated by aligned words AND
VAD. Boundary warnings are graded, not erased. Identity uncertainty never
becomes automatic permission to use a clip.
"""
import math
import listen_policy_v8 as original

REVISION='v8-recovery-evaluation-1'
BASE_COMPATIBILITY=original.listener_compatibility
BASE_ACOUSTIC_CUT=original.acoustic_cut
SOFT_BOUNDARY=frozenset(('start_no_quiet_cut','end_no_quiet_cut',
    'insufficient_tail_room','next_voice_guard_too_close',
    'start_margin_tightened_without_quiet_cut'))
IDENTITY_REASONS=frozenset(('reaction_identity_uncertain','listener_speech_identity_unconfirmed'))

def intersects(a,b,x,y):return min(b,y)>max(a,x)+1e-6

def verified_transcript_gap(segment,person,a,b):
    words=segment.get('words',[])
    if not segment.get('word_aligned') or not words or 'vad' not in person:return False
    if any(not math.isfinite(w[k]) for w in words for k in ('start','end')):return False
    if any(not 0<w['end']-w['start']<=1.5 for w in words):return False
    if any(v['start']<u['end']-.001 for u,v in zip(words,words[1:])):return False
    # A short timing buffer prevents treating alignment jitter as silence.
    lo,hi=a-.08,b+.08
    return not any(intersects(lo,hi,w['start'],w['end']) for w in words) and not any(
        intersects(lo,hi,x,y) for x,y in person['vad'])

def listener_compatibility(person,a,b):
    removed=[];segments=[]
    for segment in person['segments']:
        if intersects(a,b,segment['start'],segment['end']) and verified_transcript_gap(segment,person,a,b):
            removed.append(segment['id'])
        else:segments.append(segment)
    value=BASE_COMPATIBILITY(dict(person,segments=segments),a,b)
    value['ignored_verified_gap_segments']=removed
    return value

def install_planner_patch():
    import plan_listen_v8 as planner
    original.listener_compatibility=listener_compatibility
    planner.listener_compatibility=listener_compatibility
    planner.acoustic_cut=acoustic_cut

def acoustic_cut(unit,people,rms,speaker,limit,left_bound=0.,right_bound=None):
    listener=people[1-speaker];a,b=unit['start'],unit['end'];left=a;right=b
    # Match the frozen exporter's complete-reaction extension before testing a
    # gap. Keep words and VAD untouched so real future voices still bound cuts.
    for segment in listener['segments']:
        if not segment['feedback']:continue
        if segment['start']<a<segment['end'] and segment['start']>=a-.35:left=min(left,segment['start'])
        if segment['start']<b+.2 and segment['end']>b and segment['end']<=b+1.5:right=max(right,segment['end'])
    for x,y in listener['vad']:
        if x<b<y and y<=b+1.5:right=max(right,y)
    removed=[s['id'] for s in listener['segments'] if not s['feedback'] and
        intersects(left,right,s['start'],s['end']) and verified_transcript_gap(s,listener,left,right)]
    revised=list(people);revised[1-speaker]=dict(listener,segments=[s for s in listener['segments'] if s['id'] not in removed])
    cut,error=BASE_ACOUSTIC_CUT(unit,revised,rms,speaker,limit,left_bound,right_bound)
    if cut is not None:cut['asr_gap_tail_exemptions']=removed
    return cut,error

def boundary_grade(row):
    flags=set(row.get('boundary_review_flags',[]))
    if not (2-1e-6<=row['duration_s']<=11.95+1e-6):return 'hard_review'
    if row['start']>row['speech_start']+1e-6 or row['end']<row['speech_end']-1e-6:return 'hard_review'
    if flags-SOFT_BOUNDARY:return 'hard_or_semantic_review'
    return 'soft_review' if flags else 'clear'

def analysis(row):
    pool=row['release']['pool'];grade=boundary_grade(row)
    reasons={r for role in ('listener','speaker') for r in row['qc3'][role]['reasons']}
    reasons.update(row['qc4']['audio_reasons'])
    has_identity=bool(reasons&IDENTITY_REASONS)
    other=reasons-IDENTITY_REASONS
    human_hold=any(r.startswith('human_') or r=='prior_source_manual_rejection' for r in row['release']['reasons'])
    content=bool(row['qc4']['task_reasons'])
    boundary_recovery=(pool=='boundary_review' and grade=='soft_review'
        and row['audio_qc_status']=='keep' and not row['qc4']['audio_reasons']
        and not content and not human_hold and not row.get('source_manual_rejections'))
    identity_only=(has_identity and not other and pool=='repair_reserve' and not content
        and not human_hold and not row.get('source_manual_rejections') and grade in ('clear','soft_review'))
    if has_identity:
        stratum=('identity_only_'+grade if identity_only else
            'identity_with_reference_match' if any('match' in r or 'leakage' in r for r in other) else
            'identity_with_other_holds')
    else:stratum=None
    return dict(revision=REVISION,boundary_grade=grade,boundary_recovery_candidate=boundary_recovery,
        identity_review_stratum=stratum,identity_only_review_potential=identity_only,
        audio_reasons=sorted(reasons),training_admitted=False,
        requires_listening_review=bool(boundary_recovery or has_identity))

def union_seconds(intervals):
    total=0.;right=None
    for a,b in sorted(intervals):
        if b<=a:continue
        if right is None or a>=right:total+=b-a;right=b
        elif b>right:total+=b-right;right=b
    return total

def incremental_seconds(existing,added):
    return max(0.,union_seconds(existing+added)-union_seconds(existing))
