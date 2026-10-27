from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.open_verbatim_subset import integrate_open


def observation(i,start,end,window):
    e=dict(event_id=i,modality='visual',behavior='Briefly closes eyes.',start=start,end=end,
           categories=[],mapping_status='out_of_scope',review_flags=[])
    d=dict(open_event_id=i,source_candidates=[dict(id=i,sources=[dict(source='qwen3_vl',window_id=window)])])
    return e,d


def test_all_visual_descriptions_use_existing_cross_window_dedup_without_union():
    values=[observation('v1',1,1.3,'full'),observation('v2',1.02,1.31,'w0'),observation('v3',3,3.3,'full')]
    ev=dict(sample_id='s',original_events=[v[0] for v in values],decisions=[v[1] for v in values])
    base=dict(sample_id='s',duration_s=5,source={});audio=dict(event_id='a1',modality='audio',description='Yeah',start=2,end=2.3,intervals=[dict(start=2,end=2.1),dict(start=2.2,end=2.3)],categories=['listener_vocalization'])
    record=dict(sample_id='s',audio_events=[audio],provenance={});before=deepcopy((base,ev,record))
    result=integrate_open(base,ev,record);visual=[e for e in result['events'] if e['modality']=='visual']
    assert len(visual)==2 and len(visual[0]['source_observation_ids'])==2
    assert (visual[0]['start'],visual[0]['end']) in [(1,1.3),(1.02,1.31)]
    assert all(e['description']=='Briefly closes eyes.' and e['categories']==[] for e in visual)
    assert [e for e in result['events'] if e['modality']=='audio']==[audio]
    assert (base,ev,record)==before


def test_same_pass_repetitions_remain_distinct():
    values=[observation('v1',1,1.3,'full'),observation('v2',1.02,1.31,'full')]
    ev=dict(sample_id='s',original_events=[v[0] for v in values],decisions=[v[1] for v in values])
    result=integrate_open(dict(sample_id='s',duration_s=5,source={}),ev,dict(sample_id='s',audio_events=[],provenance={}))
    assert len(result['events'])==2
