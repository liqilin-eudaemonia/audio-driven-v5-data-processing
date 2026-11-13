"""QC2 review triage. No automatic hard rejection based on AST or acoustics."""
from copy import deepcopy
from listen_audio_quality import acoustic_flags

POLICY = dict(revision='audio-qc-2', human_score=.20, competing_noise_score=.15,
              suspect_noise_score=.30, suspect_noise_to_human_ratio=2.,
              quiet_event_rms_dbfs=-42., quiet_event_peak_dbfs=-28.,
              automatic_hard_rejection=False,
              note='Fixed before independent sampling. Suspected noise and ambiguity both need human review; no training admission seal.')


def event_decision(event):
    old=event['decision'];f=event['features']
    prediction={k:old[k] for k in ('human_score','speech_score','noise_score','top_events') if k in old}
    flags=acoustic_flags(f)
    human,noise=old['human_score'],old['noise_score']
    warnings=list(flags)
    if not event.get('word_support',False) and old['speech_score']>=.25:
        warnings.append('speech_without_transcript_is_not_noise_evidence')
    # A model/acoustic agreement prioritizes listening; never declares source noise as fact.
    if (noise>=POLICY['suspect_noise_score'] and noise>=2*max(human,.01) and flags):
        status='suspect_noise';why=['model_noise_and_acoustics_agree_review_required']
    elif human>=POLICY['human_score'] and noise<POLICY['competing_noise_score']:
        status='keep';why=['human_vocal_evidence_no_noise_dominance']
    elif (f['rms_dbfs']<POLICY['quiet_event_rms_dbfs'] and
          f['peak_dbfs']<POLICY['quiet_event_peak_dbfs'] and not flags and noise<.15):
        status='keep';why=['weak_activity_not_confirmed_reaction']
    else:
        status='review';why=['insufficient_or_conflicting_event_evidence']
    return dict(status=status,accepted=status=='keep',reasons=why,warnings=warnings,
                word_support=event.get('word_support',False),**prediction)


def retriage(check):
    result=deepcopy(check);result['legacy_qc1_reasons']=result.pop('reasons',[])
    for event in result['events']:
        event['legacy_qc1_decision']=event['decision']
        event['decision']=event_decision(event)
    statuses=[e['decision']['status'] for e in result['events']]
    status='suspect_noise' if 'suspect_noise' in statuses else 'review' if 'review' in statuses else 'keep'
    if 'missing_audible_speaker' in result['legacy_qc1_reasons']:status='review'
    result.update(status=status,accepted=status=='keep',policy=POLICY,
                  reasons=sorted({r for e in result['events'] if e['decision']['status']!='keep'
                                  for r in e['decision']['reasons']}))
    return result


def row_decision(checks,listener_screen):
    states=[c['status'] for c in checks.values()]
    if 'suspect_noise' in states:return 'suspect_noise'
    if 'review' in states or listener_screen.get('deferred_reasons'):return 'review'
    return 'keep'
