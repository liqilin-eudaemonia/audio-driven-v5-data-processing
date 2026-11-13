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
    a['visual_events'][0]['description']='Smiles throughout.'
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

def decision(sid, action, quote, targets=None):
    return dict(source_event_id=sid,action=action,quote=quote,targets=targets or [],reason='Evidence selection')


def test_plan_uses_quote_and_retains_subject():
    r,_=fixture()
    plan=dict(decisions=[decision('wide','support','Smiles throughout',['local']),
        decision('local','keep','Smiles broadly'),decision('hair','keep','Raises both hands and adjusts her hair.')],conflicts=[])
    original=deepcopy(plan);answer=compile_plan(r,plan)
    assert plan==original
    assert len(answer['visual_events'])==2
    assert answer['visual_events'][1]['description']=='Raises both hands and adjusts her hair.'
    plan['decisions'][1]['description']='Rewritten smile.'
    with pytest.raises(ValueError):compile_plan(r,plan)


def nod_fixture():
    r=dict(sample_id='s1400',duration_s=8,events=[
        dict(event_id='fade',modality='visual',start=5.6,end=5.9,description='Smile fades into a more neutral expression, suggesting continued positive reception.'),
        dict(event_id='nod',modality='visual',start=5.6,end=6.9,description='The listener maintains a slight smile or neutral expression with a subtle nod, suggesting attentive listening.')])
    bad=dict(decisions=[decision('fade','keep','Smile fades into a more neutral expression'),
        decision('nod','support','The listener maintains a slight smile or neutral expression with a subtle nod',['fade'])],conflicts=[])
    return r,bad


def test_dropped_nod_is_rejected_without_silent_repair():
    r,bad=nod_fixture();before=deepcopy(bad)
    with pytest.raises(ValueError,match='physical action'):compile_plan(r,bad)
    assert bad==before
    with pytest.raises(ValueError,match='disabled'):compile_plan(r,bad,strict=False)
    bad['decisions'][0].update(action='support',targets=['nod'])
    bad['decisions'][1].update(action='keep',targets=[])
    out=compile_plan(r,bad)
    assert len(out['visual_events'])==1
    assert out['visual_events'][0]['description']=='The listener maintains a slight smile or neutral expression with a subtle nod.'


@pytest.mark.parametrize('quote',[
    'maintains a slight smile or neutral expression with a subtle nod',
    'The listener maintains a slight smile',
    'The Listener maintains a slight smile or neutral expression with a subtle nod',
    'The listener maintains a slight smile or neutral expression with a subtle nod, suggesting attentive listening.',
])
def test_bad_prefix_rewrite_missing_action_or_inference_rejected(quote):
    r,bad=nod_fixture();bad['decisions'][0].update(action='support',targets=['nod'])
    bad['decisions'][1].update(action='keep',targets=[],quote=quote)
    with pytest.raises(ValueError):compile_plan(r,bad)


def test_inactive_cannot_hide_action():
    r,bad=nod_fixture();bad['decisions'][1]=decision('nod','non_observable',r['events'][1]['description'])
    with pytest.raises(ValueError,match='physical action'):compile_plan(r,bad)


def test_visible_showing_teeth_and_later_action_are_not_cut():
    from listen_benchmark.semantic_prompt import visible_prefix
    full='The listener smiles, showing teeth, suggesting enjoyment.'
    assert visible_prefix(full,'The listener smiles, showing teeth')=='The listener smiles, showing teeth.'
    with pytest.raises(ValueError):visible_prefix(full,'The listener smiles')
    with pytest.raises(ValueError,match='physical action'):
        visible_prefix('The listener smiles, indicating agreement while nodding.','The listener smiles')
    # Inferred laughter is not an observed action to add to the output.
    assert visible_prefix('He tilts his head, suggesting he is laughing.','He tilts his head')=='He tilts his head.'


def test_support_cannot_chain_or_rewrite_and_materialize_cannot_bypass():
    r,bad=nod_fixture();bad['decisions'][0].update(action='support',targets=['nod'])
    with pytest.raises(ValueError,match='kept'):compile_plan(r,bad)
    bad['decisions'][1].update(action='keep',targets=[])
    proposal=compile_plan(r,bad);proposal['visual_events'][0]['description']='He nods.'
    with pytest.raises(ValueError,match='exactly equal'):materialize(r,proposal,dict(accepted=True,issues=[]),{})


def test_actual_s1400_tail_preserves_subtle_nod():
    import json
    path=Path(__file__).resolve().parents[1]/'experiments/listen_semantic_prompt_pilot_20261027/inputs.jsonl'
    r=next(j['record'] for j in map(json.loads,path.open()) if 'S1400' in j['job_id'])
    r['events']=[e for e in r['events'] if e['event_id'] in ('event-0013','event-0014')]
    quote='Maintaining a slight smile or neutral expression with a subtle nod'
    answer=compile_plan(r,dict(decisions=[decision('event-0013','support',quote,['event-0014']),decision('event-0014','keep',quote)],conflicts=[]))
    assert answer['visual_events'][0]['description']==quote+'.'
    result=materialize(r,answer,dict(accepted=True,issues=[]),{})
    assert result['events'][0]['end']==7.881067


def test_quote_options_only_allow_complete_original_prefixes():
    from listen_benchmark.semantic_prompt import quote_options, plan_schema
    text='The listener smiles, showing teeth, suggesting enjoyment.'
    options=quote_options(text)
    assert options and all('showing teeth' in q and 'suggesting' not in q for q in options)
    assert all(text.startswith(q) for q in options)
    r,_=nod_fixture();schema=plan_schema(r)
    assert schema['properties']['decisions']['minItems']==2
    for branch in schema['properties']['decisions']['prefixItems']:
        assert 'description' not in branch['properties']
        assert branch['properties']['quote']['enum']


def test_decoder_excludes_target_that_cannot_cover_nod():
    from listen_benchmark.semantic_prompt import plan_schema
    r,_=nod_fixture()
    branches=plan_schema(r)['properties']['decisions']['prefixItems']
    nod_support=[b for b in branches if b['properties']['source_event_id']['enum']==['nod']
                 and 'support' in b['properties']['action']['enum']]
    assert not nod_support  # The only other source has no nod.
    fade_support=next(b for b in branches if b['properties']['source_event_id']['enum']==['fade']
                      and 'support' in b['properties']['action']['enum'])
    assert fade_support['properties']['targets']['items']['enum']==['nod']
    assert fade_support['properties']['targets']['maxItems']==1


def test_semantic_review_sees_each_active_instruction_once():
    from listen_benchmark.semantic_prompt import review_payload
    r,bad=nod_fixture();bad['decisions'][0].update(action='support',targets=['nod'])
    bad['decisions'][1].update(action='keep',targets=[])
    answer=compile_plan(r,bad);before=deepcopy(answer)
    payload=review_payload(r,answer)
    assert answer==before and len(payload['proposal']['visual_events'])==1
    event=payload['proposal']['visual_events'][0]
    assert 'evidence' not in event and set(event['source_event_ids'])=={'fade','nod'}
    assert len(answer['visual_events'][0]['evidence'])==2  # complete evidence persists
