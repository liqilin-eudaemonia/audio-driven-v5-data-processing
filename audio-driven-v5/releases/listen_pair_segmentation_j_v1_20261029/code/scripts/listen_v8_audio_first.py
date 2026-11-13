"""Independent audio-first routing trial; frozen QC4 evidence remains intact."""
import math
from listen_v8_release import exact,overlap

REVISION='v8-audio-first-review-1'
IDENTITY_WARNINGS=frozenset(('reaction_identity_uncertain','listener_speech_identity_unconfirmed',
    'audible_track_without_identified_events'))

def route(row,feedback=()):
    audio=set(row['qc4']['audio_reasons'])
    for role in ('speaker','listener'):audio.update(row['qc3'][role]['reasons'])
    contamination=audio-IDENTITY_WARNINGS
    comp=row.get('listener_compatibility',{})
    warnings=sorted(set(row.get('boundary_review_flags',[])) | (audio & IDENTITY_WARNINGS)
        | ({'listener_content_not_confirmed_backchannel'} if comp.get('uncertain_short_speech') else set()))
    negative=[f for f in feedback if f['disposition']!='accepted_exact' and overlap(row,f)]
    positives=[f for f in feedback if f['disposition']=='accepted_exact' and exact(row,f)]
    old_human=[r for r in row['release']['reasons'] if r.startswith('human_') or r=='prior_source_manual_rejection']
    if negative:
        pool='audio_reject' if any(f['disposition']=='repair_reserve' for f in negative) else 'task_hold'
        reasons=sorted({'human_'+f['disposition'] for f in negative})
    elif row.get('source_manual_rejections') or old_human:
        pool='human_hold';reasons=old_human or ['prior_source_manual_rejection']
    elif positives:
        pool='keep_candidate';reasons=['user_accepted_exact_sample']
    elif contamination:
        pool='audio_reject';reasons=sorted(contamination)
    elif row['audio_qc_status']=='exclude':
        pool='audio_reject';reasons=['unresolved_original_audio_exclusion']
    elif not all(math.isfinite(row[k]) for k in ('start','end','duration_s')) or not 2-1e-6<=row['duration_s']<=11.95+1e-6:
        pool='source_hold';reasons=['invalid_clip_interval']
    elif row['start']>row['speech_start']+1e-6 or row['end']<row['speech_end']-1e-6:
        pool='task_hold';reasons=['speaker_words_outside_clip']
    elif comp.get('substantive'):
        pool='task_hold';reasons=['confirmed_listener_substantive_speech']
    elif set(row['qc4']['task_reasons'])-{'listener_content_not_confirmed_backchannel'}:
        pool='task_hold';reasons=list(row['qc4']['task_reasons'])
    else:
        pool='keep_candidate';reasons=['no_detected_audio_contamination']
    return dict(revision=REVISION,pool=pool,reasons=reasons,advisory_reasons=warnings,
        audio_contamination_reasons=sorted(contamination),human_exact_acceptance=bool(positives) and not negative and not old_human and not row.get('source_manual_rejections'),
        matching_human_samples=[f['sample_id'] for f in negative+positives],training_admitted=False,
        note='Suspected audio contamination is withheld, not confirmed ground truth. Boundary/identity/short-response uncertainty is advisory. Exact human acceptance does not generalize to other clips.')
