from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p2_full50h_restore.annotations import recover
from test_semantic_prompt import fixture,decision


def plan():
 return dict(decisions=[decision('wide','support','Smiles throughout',['local']),decision('local','keep','Smiles broadly'),decision('hair','keep','Raises both hands and adjusts her hair.')],conflicts=[])


def test_safe_groups_merge_and_audio_is_unchanged():
 r,_=fixture();before=copy.deepcopy(r);events,a=recover(r,plan())
 assert r==before and not a['fallback_source_ids']
 assert len([e for e in events if e['modality']=='visual'])==2
 assert [e for e in events if e['modality']=='audio']==[e for e in r['events'] if e['modality']=='audio']


def test_failed_group_retains_original_but_independent_group_is_compiled():
 r,_=fixture();p=plan();p['decisions'][0]['targets']=['wide']
 events,a=recover(r,p)
 assert 'wide' in a['fallback_source_ids']
 assert next(e for e in events if e['event_id']=='wide')==r['events'][0]
 assert 'hair' in a['safe_source_ids']
 assert next(e for e in events if e['event_id']=='semantic-hair')['description']==r['events'][2]['description']
 assert set(a['safe_source_ids'])|set(a['fallback_source_ids'])=={'wide','local','hair'}


def test_no_proposal_preserves_entire_original_record():
 r,_=fixture();events,a=recover(r,None)
 assert events==r['events'] and a['mode']=='original_only'


def test_disjoint_group_does_not_delete_independent_sources():
 r,_=fixture();r['events'][0]['start']=3;r['events'][0]['end']=5
 events,a=recover(r,plan())
 assert set(a['fallback_source_ids'])=={'wide','local'}
 assert {e['event_id'] for e in events if e['modality']=='visual'}=={'wide','local','semantic-hair'}


def test_missing_source_decision_is_retained():
 r,_=fixture();p=plan();p['decisions'].pop();events,a=recover(r,p)
 assert 'hair' in a['fallback_source_ids']
 assert next(e for e in events if e['event_id']=='hair')==r['events'][2]


def test_invalid_conflicts_preserve_original():
 r,_=fixture();p=plan();p['conflicts']=[dict(source_event_ids=['bad'],selected_event_ids=['local'],reason='bad')]
 events,a=recover(r,p);assert events==r['events']
