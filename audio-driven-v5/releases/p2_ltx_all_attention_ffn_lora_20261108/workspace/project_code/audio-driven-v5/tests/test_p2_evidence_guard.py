import copy
from p2_ltx_events.evidence_guard import guard_row
from p2_clip_events.labels import tensors
NAMES=('visual:head nodding','visual:eyebrow raising','visual:frowning','visual:surprise','audio:speech','audio:vocalization','audio:audible laughter')

def sample():
    return dict(positive=[],unknown=[],allow_weak=True,caption_sha256='a'*64),dict(audio_identity={'full_silence':False},nod_supplement={'events':[{'start':.1,'end':.5}]},reaction_supplement={'cues':[{'start':.2,'end':.6,'observation':'brows raised'}]},vocal_supplement=[{'start':.3,'end':.4}])

def test_conflicting_weak_negatives_masked_without_new_positives():
    row,e=sample();old=copy.deepcopy(row);new=guard_row(row,e,1.,NAMES)
    assert row==old and new['positive']==[]
    assert set(new['unknown'])=={NAMES[0],NAMES[1],*NAMES[4:]}
    target=tensors(new,NAMES,.1)
    assert all(not target[k]['known'].item() for k in new['unknown'])
    assert target['visual:frowning']['known'].item() and target['visual:surprise']['known'].item()

def test_positives_unknown_and_incomplete_rows_preserved():
    row,e=sample();row['positive']=[NAMES[0],NAMES[4]];row['unknown']=[NAMES[1]]
    assert guard_row(row,e,1.,NAMES)==row
    row,e=sample();row['allow_weak']=False
    assert guard_row(row,e,1.,NAMES)==row

def test_out_of_clip_cues_and_physical_brow_lowering_do_not_create_labels():
    row,e=sample();e['nod_supplement']['events']=[{'start':2.,'end':3.}];e['vocal_supplement']=[]
    e['reaction_supplement']['cues'][0]['observation']='brow lowering'
    assert guard_row(row,e,1.,NAMES)==row

def test_confirmed_silence_masks_audio_positive_not_negative():
    row,e=sample();row['positive']=[NAMES[4]];e['audio_identity']['full_silence']=True
    new=guard_row(row,e,1.,NAMES)
    assert NAMES[4] not in new['positive'] and NAMES[4] in new['unknown']
    assert NAMES[5] not in new['unknown'] and NAMES[6] not in new['unknown']

def test_open_audio_positive_not_collapsed_into_all_negative_closed_vocabulary():
    row,e=sample();e['vocal_supplement']=[];row['positive']=['audio:verbal affirmation']
    new=guard_row(row,e,1.,NAMES)
    assert new['positive']==row['positive'] and set(NAMES[4:])<=set(new['unknown'])

def test_baseline_mentions_are_masked_not_assumed_absent():
    row,e=sample();e['nod_supplement']['events']=[];e['reaction_supplement']['cues']=[];e['vocal_supplement']=[]
    stats={'evidence_ledger':[{'raw':{'modality':'audio'},'canonical':'speech','disposition':'baseline'},
                              {'raw':{'modality':'visual'},'canonical':'head nodding','disposition':'baseline'}]}
    new=guard_row(row,e,1.,NAMES,statistics=stats)
    assert set(new['unknown'])=={'audio:speech','visual:head nodding'} and not new['positive']
    row['positive']=['audio:speech']
    new=guard_row(row,e,1.,NAMES,statistics=stats)
    assert new['positive']==['audio:speech'] and 'audio:speech' not in new['unknown']


def test_preflight_rejects_changed_config_even_with_same_data_and_code():
    import pytest
    from p2_listen.contracts import digest
    from p2_ltx_events import VERSION
    from p2_ltx_events.training import check_preflight
    c={'text_condition':'guided','loss':'av_event_feature','stage':1,'event_supervision':{'min_positive':5000}}
    pre=dict(configuration_sha256=digest(c),schema=VERSION,source={},data_identity={},experiment_mode=['guided','av_event_feature',1])
    check_preflight(c,pre,{},{});changed=copy.deepcopy(c);changed['event_supervision']['min_positive']=10000
    with pytest.raises(ValueError,match='preflight'):check_preflight(changed,pre,{},{})

def test_guard_rules_and_vocabulary_are_frozen():
    import pytest
    from pathlib import Path
    from p2_listen.contracts import sha
    from p2_ltx_events import evidence_guard
    from p2_ltx_events.numeric_labels import validate_guard
    m={'evidence_guard':{'policy':evidence_guard.POLICY,'classes':list(NAMES),'code_sha256':sha(Path(evidence_guard.__file__))}}
    validate_guard(m,NAMES)
    with pytest.raises(ValueError,match='rules/classes'):validate_guard(m,NAMES[::-1])
    m['evidence_guard']['code_sha256']='0'*64
    with pytest.raises(ValueError,match='rules/classes'):validate_guard(m,NAMES)
