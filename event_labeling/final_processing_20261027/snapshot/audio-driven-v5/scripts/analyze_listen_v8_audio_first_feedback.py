"""Describe human/model disagreements without changing release rules."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_listen.common import read,write,sha
from listen_v8_recovery import union_seconds

SOURCE=ROOT/'data_previews/seamless_listen_v8_audio_first/samples.json'
OUT=ROOT/'experiments/p3_listen_v8_audio_first_20261023/feedback_analysis'

def main():
    data=read(SOURCE);rows=data['samples'];OUT.mkdir(parents=True,exist_ok=True)
    expected={6:'naturalistic_V03_S1654_I00000402_v8_s0_002',18:'naturalistic_V03_S2131_I00000130_v8_s1_007',
        19:'naturalistic_V03_S1451_I00000307_v8_s1_008',22:'naturalistic_V03_S1058_I00000028_v8_s0_017',
        23:'naturalistic_V02_S5281_I00000280_v8_s1_000'}
    feedback=[]
    for n,sid in expected.items():
        r=rows[n-1];assert r['sample_id']==sid
        f={k:r[k] for k in ('sample_id','pair','speaker_id','listener_id','start','end')}
        f.update(page_number=n,disposition='repair_reserve' if n==6 else 'accepted_exact',
            note='User reports speaker bleed' if n==6 else 'User says acceptable despite automatic rejection')
        feedback.append(f)
    record=dict(review='v8-audio-first-comparison',sample_manifest=str(SOURCE),sample_manifest_sha256=sha(SOURCE),
        selection_sha256=data['selection_sha256'],rows=feedback,
        unmentioned_samples='No new exact approval inferred. Other displayed exclusions are comparison references.',
        training_authorized=False,full_rollout_of_new_policy_authorized=False)
    dest=ROOT/'data_reviews/listener_v8_audio_first_feedback_20261023.json'
    if dest.exists():assert read(dest)==record
    else:write(dest,record)
    analysis=[]
    for n in (6,17,18,19,20,21,22,23,24):
        r=rows[n-1];q=r['qc3']['listener'];ref=r['qc4']['reference']
        event_matches=[];background=[]
        for w in q['events']:
            paired=w.get('paired',{})
            if paired.get('available'):
                event_matches.append(dict(event_start=w['start'],event_end=w['end'],event_duration=w['end']-w['start'],
                    event_rms_dbfs=w['features']['rms_dbfs'],event_peak_dbfs=w['features']['peak_dbfs'],
                    match_window_seconds=paired['window_seconds'],correlation=paired['abs_wave_correlation'],
                    delay_ms=paired['listener_delay_ms'],triggered_local_match=any('speaker_match' in x for x in w['decision']['reasons'])))
        for role in ('speaker','listener'):
            for w in r['qc3'][role]['background']:
                if w['decision']['reasons']:
                    background.append(dict(role=role,start=w['start'],end=w['end'],rms_dbfs=w['features']['rms_dbfs'],
                        peak_dbfs=w['features']['peak_dbfs'],noise_label=w['panns']['noise_label'],noise_score=w['panns']['noise_score'],
                        human_label=w['panns']['human_label'],human_score=w['panns']['human_score'],reasons=w['decision']['reasons']))
        suspicious=[w for w in ref['windows'] if w['suspicious']]
        triggered=[w for w in event_matches if w['triggered_local_match']]
        analysis.append(dict(page_number=n,sample_id=r['sample_id'],duration_s=r['duration_s'],
            user_label='explicit_bad_bleed' if n==6 else 'explicit_acceptable' if n in (18,19,22,23) else 'other_excluded_comparison',
            automatic=r['audio_first'],listener_floor20_dbfs=q['floor20_dbfs'],listener_rms_dbfs=q['rms_dbfs'],
            speaker_reasons=r['qc3']['speaker']['reasons'],listener_reasons=q['reasons'],
            event_matches=event_matches,background_triggers=background,
            reference_max=ref['max_correlation'],reference_suspicious_windows=len(suspicious),
            reference_suspicious_window_union_seconds=union_seconds([(w['start'],w['end']) for w in suspicious]),
            local_match_event_union_seconds=union_seconds([(w['event_start'],w['event_end']) for w in triggered]),
            local_max_correlation=max((w['correlation'] for w in event_matches),default=0)))
    floor_difference=rows[23]['qc3']['listener']['floor20_dbfs']-rows[22]['qc3']['listener']['floor20_dbfs']
    write(OUT/'comparison.json',dict(source_sha256=sha(SOURCE),feedback_sha256=sha(dest),rows=analysis,
        floor24_minus_floor23_db=floor_difference,
        limitations=['Human audibility judgments are supplied by the user; this analysis does not claim independent listening.',
            'Classifier scores are not calibrated noise severity.',
            'Overlapping model windows are not independent events or exact contamination durations.',
            'Levels are measured on separately normalized preview tracks, not original acoustic leakage ratios.',
            'These examples do not establish a validated threshold separating acceptable and unacceptable audio.'],
        production_rules_changed=False))
    print(json.dumps(dict(samples=len(analysis),explicit_feedback=len(feedback),floor24_minus_floor23_db=floor_difference,
        output=str(OUT/'comparison.json')),ensure_ascii=False))

if __name__=='__main__':main()
