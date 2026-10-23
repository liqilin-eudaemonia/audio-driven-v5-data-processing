"""Release routing for reviewed QC4. No waveform changes or automatic rescue."""
REVISION='qc4-reviewed-routing-1'
POOLS=('candidate','repair_reserve','role_review','content_review','boundary_review','excluded')

def overlap(row,f):
    return row['listener_id']==f['listener_id'] and min(row['end'],f['end'])>max(row['start'],f['start'])+1e-6

def exact(row,f):
    return row['sample_id']==f['sample_id'] and row['speaker_id']==f['speaker_id'] and row['listener_id']==f['listener_id'] and abs(row['start']-f['start'])<1e-6 and abs(row['end']-f['end'])<1e-6

def route(row,feedback):
    hits=[f for f in feedback if overlap(row,f)]
    negative=[f for f in hits if f['disposition']!='accepted_exact']
    positives=[f for f in feedback if f['disposition']=='accepted_exact' and exact(row,f)]
    q=row['qc4'];reasons=[]
    # Negative intervals take precedence; approval never transfers to a larger cut.
    if negative:
        priorities=('role_review','content_review','repair_reserve')
        pool=next(p for p in priorities if any(f['disposition']==p for f in negative))
        reasons=['human_'+pool]
    elif row.get('source_manual_rejections'):
        pool='excluded';reasons=['prior_source_manual_rejection']
    elif positives:
        pool='candidate';reasons=['user_accepted_exact_sample']
    elif q['task_reasons']:
        pool='content_review';reasons=list(q['task_reasons'])
    elif q['audio_reasons'] or row['audio_qc_status'] in ('review','exclude'):
        pool='repair_reserve';reasons=list(q['audio_reasons'])+['original_audio_'+row['audio_qc_status']]
    elif q['status']=='candidate':
        pool='candidate';reasons=['passed_reviewed_automatic_rules']
    elif q['status']=='exclude':
        pool='excluded';reasons=['original_exclusion']
    else:
        pool='boundary_review';reasons=['remaining_boundary_or_human_hold']
    return dict(revision=REVISION,pool=pool,reasons=reasons,
        human_exact_acceptance=bool(positives) and not negative,
        matching_human_samples=[f['sample_id'] for f in negative+positives],
        secondary_audio_concerns=list(q['audio_reasons']),
        repair_authorized=False,training_admitted=False,
        note='Repair reserve includes suspected/unconfirmed audio issues; not all are proven crosstalk. No reserve pool is automatically admitted or repaired.')
