"""Severity-aware small-batch trial; thresholds are provisional, not calibrated."""
import listen_v8_audio_first as previous

REVISION='v8-audio-severity-review-2'
IDENTITY_WARNINGS=previous.IDENTITY_WARNINGS
POLICY=dict(revision=REVISION,repeat_min_correlation=.30,repeat_min_event_seconds=.12,
    repeat_min_event_rms_dbfs=-35.,repeat_delay_span_ms=10.,repeat_min_separation_seconds=.25,
    loud_match_correlation=.45,loud_match_event_seconds=.25,loud_match_rms_dbfs=-20.,
    weak_match_max_event_seconds=.25,weak_match_max_rms_dbfs=-28.,
    severe_listener_floor_dbfs=-40.,mild_stationary_floor_dbfs=-42.,
    isolated_speaker_music_max_score=.35,speaker_music_min_speech_score=.60,
    mild_stationary_max_score=.35,mild_stationary_max_rms_dbfs=-35.,
    thresholds_validated=False,training_admitted=False)
STATIONARY={'White noise','Pink noise','Static','Hum','Mains hum','Air conditioning','Mechanical fan'}
TRANSIENT={'Clicking','Clickety-clack','Crackle','Crack','Burst, pop','Bang','Knock','Tap','Scratch','Scrape','Distortion'}

def evidence(row):
    hard=[];review=[];soft=[];events=[];windows=[]
    q=row['qc3']['listener'];ref=row['qc4']['reference']
    for w in q['events']:
        p=w.get('paired',{})
        if not p.get('available'):continue
        events.append(dict(start=w['start'],end=w['end'],correlation=p['abs_wave_correlation'],
            delay_ms=p['listener_delay_ms'],rms_dbfs=w['features']['rms_dbfs'],
            reference_rms_dbfs=p['speaker_rms_dbfs'],match_rms_dbfs=p['listener_rms_dbfs']))
    candidates=[w for w in events if w['correlation']>=POLICY['repeat_min_correlation']
        and w['end']-w['start']>=POLICY['repeat_min_event_seconds']
        and w['rms_dbfs']>=POLICY['repeat_min_event_rms_dbfs']
        and w['reference_rms_dbfs']>-45 and w['match_rms_dbfs']>-45]
    repeated=[]
    for a in candidates:
        for b in candidates:
            if b['start']<=a['start'] or b['start']-a['start']<POLICY['repeat_min_separation_seconds']:continue
            if b['start']<a['end']:continue
            if abs(a['delay_ms']-b['delay_ms'])<=POLICY['repeat_delay_span_ms']:
                repeated.append([a,b])
    if repeated:
        hard.append('repeated_audible_delayed_match')
        windows.extend(dict(w,role='listener',reason='repeated_audible_delayed_match') for pair in repeated for w in pair)
    loud=[w for w in events if w['correlation']>=POLICY['loud_match_correlation']
        and w['end']-w['start']>=POLICY['loud_match_event_seconds']
        and w['rms_dbfs']>=POLICY['loud_match_rms_dbfs'] and w['reference_rms_dbfs']>-45]
    if loud:
        hard.append('loud_local_speaker_match')
        windows.extend(dict(w,role='listener',reason='loud_local_speaker_match') for w in loud)
    if 'strong_reference_match' in ref['reasons'] or 'consistent_reference_leakage' in ref['reasons']:
        hard.extend(ref['reasons'])
    elif 'isolated_reference_match_requires_review' in ref['reasons']:
        local=[w for w in events if w['correlation']>=.45]
        if local and all(w['end']-w['start']<=POLICY['weak_match_max_event_seconds'] and w['rms_dbfs']<=POLICY['weak_match_max_rms_dbfs'] for w in local):
            soft.append('brief_weak_reference_match')
        else:review.append('isolated_reference_match_unresolved')
    if q['floor20_dbfs']>POLICY['severe_listener_floor_dbfs']:
        hard.append('high_listener_background_floor')
    for role,track in row['qc3'].items():
        music_background=[w for w in track['background'] if 'background_environmental_evidence' in w['decision']['reasons'] and w['panns']['noise_label']=='Music']
        for kind,items in (('event',track['events']),('background',track['background'])):
            for w in items:
                reasons=set(w['decision']['reasons'])-IDENTITY_WARNINGS
                p=w['panns'];f=w['features']
                for reason in reasons:
                    if reason in ('strong_local_speaker_match','two_model_environmental_contamination'):
                        hard.append(reason);windows.append(dict(start=w['start'],end=w['end'],role=role,reason=reason))
                    elif reason=='possible_local_speaker_match':
                        weak=(role=='listener' and w['end']-w['start']<=POLICY['weak_match_max_event_seconds'] and f['rms_dbfs']<=POLICY['weak_match_max_rms_dbfs'])
                        (soft if weak else review).append('brief_weak_local_match' if weak else 'local_match_unresolved')
                    elif reason in ('background_environmental_evidence','environmental_sound_evidence','audible_broadband_background'):
                        if reason=='audible_broadband_background' or p['noise_label'] in STATIONARY:
                            mild=(p['noise_score']<POLICY['mild_stationary_max_score'] and
                                (f['rms_dbfs']<=POLICY['mild_stationary_max_rms_dbfs'] or
                                 (p['human_label']=='Speech' and p['human_score']>=.60)))
                            if role=='listener' and track['floor20_dbfs']<=POLICY['mild_stationary_floor_dbfs'] and mild:
                                soft.append('mild_stationary_background')
                            else:review.append('stationary_background_unresolved')
                        elif p['noise_label'] in TRANSIENT:
                            hard.append('intrusive_transient_noise');windows.append(dict(start=w['start'],end=w['end'],role=role,reason='intrusive_transient_noise'))
                        elif role=='speaker' and kind=='background' and p['noise_label']=='Music' and len(music_background)==1 and p['noise_score']<POLICY['isolated_speaker_music_max_score'] and p['human_score']>=POLICY['speaker_music_min_speech_score'] and p['human_label']=='Speech':
                            soft.append('isolated_speaker_music_prediction')
                        else:
                            review.append('environmental_prediction_unresolved');windows.append(dict(start=w['start'],end=w['end'],role=role,reason='environmental_prediction_unresolved'))
                    else:review.append(reason)
        for reason in set(track['reasons'])-IDENTITY_WARNINGS:
            if reason=='elevated_between_event_floor_requires_review':
                if role=='listener' and track['floor20_dbfs']<=POLICY['mild_stationary_floor_dbfs']:soft.append('mild_stationary_background')
                elif role!='listener' or track['floor20_dbfs']<=POLICY['severe_listener_floor_dbfs']:review.append('stationary_background_unresolved')
            elif not any(reason in w['decision']['reasons'] for w in track['events']+track['background']):review.append(reason)
    known_reference={'strong_reference_match','consistent_reference_leakage','isolated_reference_match_requires_review'}
    review.extend(set(row['qc4']['audio_reasons'])-known_reference-IDENTITY_WARNINGS)
    if row['audio_qc_status']=='exclude' and not hard:review.append('original_audio_exclusion_unresolved')
    unique={ (w['role'],w['start'],w['end'],w['reason']):w for w in windows }
    return dict(hard_reasons=sorted(set(hard)),review_reasons=sorted(set(review)),soft_reasons=sorted(set(soft)),
        repeat_pairs=repeated,evidence_windows=list(unique.values()),listener_floor20_dbfs=q['floor20_dbfs'])

def route(row,feedback=()):
    old=previous.route(row,feedback);quality=evidence(row)
    # Preserve human interval decisions and exact approvals before applying the trial.
    human=old['human_exact_acceptance'] or any(r.startswith('human_') or r=='prior_source_manual_rejection' for r in old['reasons'])
    clean=dict(row,qc3={role:dict(q,reasons=[]) for role,q in row['qc3'].items()},
        qc4=dict(row['qc4'],audio_reasons=[]),audio_qc_status='keep')
    guard=previous.route(clean,feedback)
    if human:pool,reasons=old['pool'],old['reasons']
    elif quality['hard_reasons']:pool,reasons='audio_reject',quality['hard_reasons']
    elif quality['review_reasons']:pool,reasons='audio_review',quality['review_reasons']
    elif guard['pool']!='keep_candidate':pool,reasons=guard['pool'],guard['reasons']
    else:pool,reasons='keep_candidate',['passed_severity_trial']
    return dict(old,revision=REVISION,pool=pool,reasons=reasons,quality=quality,
        advisory_reasons=sorted(set(old['advisory_reasons']+quality['soft_reasons'])),
        note='Three-way provisional routing. Review is withheld, not automatically admitted. Exact human decisions override only the reviewed interval. No waveform or crop changes.')
