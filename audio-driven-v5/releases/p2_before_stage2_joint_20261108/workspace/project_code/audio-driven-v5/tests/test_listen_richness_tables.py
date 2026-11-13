"""Quantity units, source preservation, and no sample loss on partial extraction failures."""
from pathlib import Path
import copy
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_listen_richness_pilot import aggregate, report, repetition


def snap(caption):
    return dict(sample_id="test",caption=caption,caption_file="original.json",crop_revision="verified",
        source=dict(pair="pair",listener_id="person",split="train",duration_s=8.,source_start_s=0.,source_end_s=8.),
        face=dict(face_short_side_p10=300.,face_short_side_p50=320.),
        audio=dict(waveform_full_silence=True,official_core_s=0.,official_bout_count=0),
        nod=dict(candidate_interval_count=0,support_s=0.))


def b(name,refs=None,ident="b1",form="change",polarity="positive",modality="visual",repeat=None):
    return dict(id=ident,behavior=name,source_ids=refs or ["s001"],form=form,polarity=polarity,modality=modality,
        repeat=repeat or dict(kind="not_stated",count=None,source_ids=[]))


def run(caption,bs):
    return aggregate(snap(caption),dict(behaviors=bs,relations=[]))


@pytest.mark.parametrize("text,want",[("At two seconds she nods.",None),("She nods once.",1),
    ("She nods downward twice during the clip.",2),("She nods repeatedly.",None),
    ("She nods once, then nods again.",None)])
def test_clock_numbers_are_not_counts_and_unknown_is_not_zero(text,want):
    assert repetition("head nodding",[text],[])["count"]==want


def test_distinct_mentions_are_derived_not_literal():
    n=repetition("head nodding",["She performs a single nod.","This nod repeats around seven seconds."],[])
    h=repetition("humming",["She produces a hum, before emitting another similar sound around seven seconds."],[])
    assert (n["kind"],n["count"])==("derived_from_distinct_mentions",2)
    assert (h["kind"],h["count"])==("derived_from_distinct_mentions",2)


def test_repeated_state_and_synonyms_do_not_duplicate_counts_or_detector():
    text="She maintains a smile. She keeps smiling. She nods twice."
    s=snap(text);s["nod"]["candidate_interval_count"]=4
    out=aggregate(s,dict(behaviors=[b("smiling",["s001"],form="state"),b("smile",["s002"],"b2",form="state"),
        b("head nodding",["s003"],"b3")],relations=[]))
    assert out["counted_type_count"]==2
    assert out["known_occurrences_by_type"]=={"visual:head nodding":2}
    assert out["state_group_count"]==1 and out["nod_counted_once"]==1


def test_failed_count_references_are_preserved_and_repaired_without_mutation():
    text="She briefly closes her eyes and tilts her head downward before reopening her eyes. This sequence of closing her eyes and lowering her head occurs once during the clip."
    value=b("head tilting",repeat=dict(kind="explicit",count=1,source_ids=["s002"]))
    original=copy.deepcopy(value);out=run(text,[value])
    assert value==original and out["evidence_ledger"][0]["raw"]==original
    assert out["evidence_ledger"][0]["reference_repaired"]
    assert out["known_occurrences_by_type"]=={"visual:head tilting":1}
    assert out["retained"]


def test_unknown_open_behavior_does_not_become_fixed_category_or_zero():
    r=run("She shrugs her shoulders.",[b("shoulder shrugging")])
    c=report([r])["catalog"]["visual:shoulder shrugging"]
    assert r["unknown_occurrence_types"]==["visual:shoulder shrugging"]
    assert c["supported_occurrences_sum"] is None and c["unknown_occurrence_samples"]==1


def test_negative_baseline_and_pending_are_retained_not_positive_counts():
    text="She maintains a steady gaze. She does not nod. She wears glasses."
    r=run(text,[b("gaze shifting",form="state"),b("head nodding",["s002"],"b2",polarity="negative"),
               b("adjusting glasses",["s003"],"b3")])
    assert r["retained"] and r["counted_type_count"]==0 and r["pending_behavior_count"]==1
    assert not r["zero_counted_types_is_verified_absence"]
    assert report([r])["catalog"]["visual:adjusting glasses"]["pending_sample_count"]==1


def test_audio_claim_on_muted_waveform_keeps_other_behaviors_and_source():
    r=run('She nods once. She says "Yeah".',[b("head nodding"),b("speech",["s002"],"b2",modality="audio")])
    assert r["retained"] and r["known_occurrences_by_type"]=={"visual:head nodding":1}
    assert r["pending_behavior_count"]==1 and len(r["evidence_ledger"])==2


def test_state_groups_not_summed_into_occurrence_total():
    r=run("She maintains a smile.",[b("smiling",form="state")])
    c=report([r])["catalog"]["visual:smiling"]
    assert c["state_samples"]==1 and c["supported_occurrences_sum"] is None
    assert c["unknown_occurrence_samples"]==0


def test_continuous_audio_is_not_counted_as_one_visual_state_group():
    s=snap('He continues speaking and says "Exactly".')
    s["audio"].update(waveform_full_silence=False,official_core_s=1.,official_bout_count=1)
    r=aggregate(s,dict(behaviors=[b("speech",form="state",modality="audio")],relations=[]))
    assert r["state_group_count"]==0
    assert r["groups"][0]["quantity"]["count"] is None
    assert r["groups"][0]["quantity"]["unit"]=="occurrence"


def test_repetition_of_other_action_not_borrowed():
    r=repetition("gaze shifting",["She shifts her gaze while blinking occasionally."],[])
    assert r["kind"]=="unknown" and r["count"] is None


def test_spoken_word_count_not_confused_with_mouth_repetitions():
    r=repetition("speech",['Her mouth opens several times, producing audible laughter and a single spoken word, "Yeah".'],[])
    assert (r["kind"],r["count"])==("literal",1)
