from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.semantic_prompt import validate_proposal,materialize,render_prompt,compile_plan

def fixture():
    r=dict(sample_id='sample',duration_s=6,events=[
        dict(event_id='wide',modality='visual',start=0,end=5,description='Smiles throughout.'),
        dict(event_id='local',modality='visual',start=0,end=2,description='Smiles broadly, indicating engagement.'),
        dict(event_id='hair',modality='visual',start=1,end=2,description='Raises both hands and adjusts her hair.'),
        dict(event_id='okay1',modality='audio',start=1,end=1.2,description='okay',text='okay'),
        dict(event_id='okay2',modality='audio',start=3,end=3.5,description='okay',text='okay',intervals=[dict(start=3,end=3.2),dict(start=3.3,end=3.5)])])
    a=dict(visual_events=[dict(id='smile',description='Smiles broadly.',timing_source_id='local',evidence=[dict(source_event_id='local',quote='Smiles broadly'),dict(source_event_id='wide',quote='Smiles throughout')]),
        dict(id='hair',description='Raises both hands and adjusts her hair.',timing_source_id='hair',evidence=[dict(source_event_id='hair',quote='Raises both hands and adjusts her hair.')])],
        source_decisions=[dict(source_event_id='wide',disposition='redundant',target_event_ids=['smile'],reason='Broad summary supports local smile; duration differs.'),
            dict(source_event_id='local',disposition='represented',target_event_ids=['smile'],reason='Observable smile.'),
            dict(source_event_id='hair',disposition='represented',target_event_ids=['hair'],reason='Different simultaneous action.')],
        conflicts=[dict(source_event_ids=['wide','local'],selected_event_ids=['smile'],reason='Wide duration only a summary; exact support remains uncertain.')])
    return r,a

def test_source_and_audio_are_immutable_and_no_interval_union():
    r,a=fixture();before=deepcopy(r);out=materialize(r,a,dict(accepted=True,issues=[]),{})
    assert r==before and out['source_record']==r
    assert [e for e in out['events'] if e['modality']=='audio']==[e for e in r['events'] if e['modality']=='audio']
    smile=next(e for e in out['events'] if e['event_id']=='semantic-smile')
    assert (smile['start'],smile['end'])==(0,2)
    text=render_prompt(out);assert text.count('Audio response: "okay"')==2
    assert '[3–3.2; 3.3–3.5 seconds]' in text
    assert len([e for e in out['events'] if e['modality']=='audio'])==2
    assert 'indicating engagement' not in text

@pytest.mark.parametrize('mutation',[
    lambda a:a['visual_events'][0].update(start=0),
    lambda a:a['visual_events'][0].update(timing_source_id='invented'),
    lambda a:a['visual_events'][0]['evidence'][0].update(quote='Nods in agreement'),
    lambda a:a['source_decisions'].pop(),
    lambda a:a['source_decisions'][0].update(target_event_ids=[]),
    lambda a:a['visual_events'][0]['evidence'].pop(0),
])
def test_invalid_sources_and_invented_boundaries_rejected(mutation):
    r,a=fixture();mutation(a)
    with pytest.raises(ValueError):validate_proposal(r,a)

def test_broad_record_cannot_bridge_two_occurrences():
    r,a=fixture();r['events'].append(dict(event_id='later',modality='visual',start=3,end=4,description='Smiles again.'))
    a['visual_events'][0]['timing_source_id']='wide';a['visual_events'][0]['evidence'].append(dict(source_event_id='later',quote='Smiles again'))
    a['source_decisions'].append(dict(source_event_id='later',disposition='redundant',target_event_ids=['smile'],reason='Same smile'))
    with pytest.raises(ValueError,match='disjoint'):validate_proposal(r,a)

def test_semantic_rejection_never_publishes():
    r,a=fixture()
    review=dict(accepted=False,issues=[dict(kind='unsupported_action',source_event_ids=['local'],output_event_ids=['smile'],reason='Not supported')])
    with pytest.raises(ValueError,match='Semantic review'):materialize(r,a,review,{})

def test_empty_visual_preserves_repeated_audio():
    r,a=fixture();r['events']=[e for e in r['events'] if e['modality']=='audio']
    out=materialize(r,dict(visual_events=[],source_decisions=[],conflicts=[]),dict(accepted=True,issues=[]),{})
    assert out['events']==r['events']

def test_plan_support_never_becomes_a_second_instruction():
    r,_=fixture()
    plan=dict(decisions=[dict(source_event_id='wide',action='support',description='',quote='Smiles throughout',targets=['local'],reason='Broad summary'),
        dict(source_event_id='local',action='keep',description='Smiles broadly.',quote='Smiles broadly',targets=[],reason='Local expression'),
        dict(source_event_id='hair',action='keep',description='Adjusts her hair.',quote='adjusts her hair',targets=[],reason='Independent action')],conflicts=[])
    answer=compile_plan(r,plan)
    assert len(answer['visual_events'])==2
    assert {e['id'] for e in answer['visual_events']}=={'local','hair'}
    assert answer['visual_events'][0]['evidence'][1]['source_event_id']=='wide'
    plan['decisions'][0]['description']='Smiles throughout.'
    with pytest.raises(ValueError,match='cannot emit'):compile_plan(r,plan)

def test_plan_preserves_verbatim_subject_and_rejects_dropped_nod():
    r=dict(sample_id='s1400',duration_s=8,events=[
        dict(event_id='fade',modality='visual',start=5.6,end=5.9,description='Smile fades into a more neutral, contented expression, suggesting continued positive reception.'),
        dict(event_id='nod',modality='visual',start=5.6,end=6.9,description='The listener maintains a slight smile or neutral expression with a subtle nod, suggesting attentive listening.')])
    bad_plan=dict(decisions=[
        dict(source_event_id='fade',action='keep',description='Smile fades.',quote='Smile fades',targets=[],reason='Kept short'),
        dict(source_event_id='nod',action='support',description='',quote='neutral expression',targets=['fade'],reason='Overlaps')],conflicts=[])
    with pytest.raises(ValueError,match='physical action'):compile_plan(r,bad_plan,strict=True)
    repaired=compile_plan(r,bad_plan)
    assert len(repaired['visual_events'])==1
    assert repaired['visual_events'][0]['id']=='nod'
    assert repaired['visual_events'][0]['description']=='The listener maintains a slight smile or neutral expression with a subtle nod.'


