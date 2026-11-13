import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from listen_benchmark.train_subset import CATEGORIES, Selection, person_keys, score, select_rows, audit_catalog


def row(i=0, face=300, shake=False):
    r = dict(sample_id=f"s{i:03}", pair=f"pair{i}", split="train", source_start_s=0, source_end_s=10,
             duration_s=10, face_short_side_p10=face, face_valid_fraction=1, label_issue_flag=0,
             non_smile_event_count=5)
    r.update({c+"_count": 1 for c in CATEGORIES})
    r["head_shake_count"] = int(shake)
    return r


def test_caps_do_not_reward_smile_fragmentation_or_change_original_counts():
    a=row(); b=copy.deepcopy(a); b["smile_count"]=10000
    assert score(a) == score(b)
    assert b["smile_count"] == 10000
    b["head_nod_count"]=3; v=score(b)
    b["head_nod_count"]=10000
    assert score(b) == v


def test_duration_taxonomy_core_and_rare_supplement_are_reproducible():
    pool=[row(i) for i in range(30)] + [row(99,170,True)]
    original=copy.deepcopy(pool)
    selected,policy=select_rows(pool,hours=.05,class_minimum=2)
    seconds=sum(r["duration_s"] for r in selected)
    assert 180 <= seconds < 190
    assert len({r['sample_id'] for r in selected}) == len(selected)
    assert any(r["selection_reason"]=="supplement:head_shake" and r["sample_id"]=="s099" for r in selected)
    assert all(r["face_short_side_p10"]>=224 for r in selected if r["selection_reason"].startswith("core"))
    assert all(any(r[c+"_count"]>0 for r in selected) for c in CATEGORIES)
    assert pool == original
    assert select_rows(list(reversed(pool)),hours=.05,class_minimum=2)[0] == selected


def test_overlapping_pair_spans_excluded_even_with_different_listener():
    s=Selection(2); a=row(); assert s.add(a,"core")
    b=row(1); b["pair"]=a["pair"]; b.update(source_start_s=5,source_end_s=15,listener_id="other")
    assert not s.add(b,"core")
    b.update(source_start_s=10,source_end_s=20)
    assert s.add(b,"core")
    c=row(2); c.update(pair=a["pair"],source_start_s=20,source_end_s=30)
    assert s.reason(c)=="pair_cap"


def test_never_admit_eval_rows_or_relax_insufficient_pool():
    a=row(); a["split"]="test"
    with pytest.raises(ValueError,match="Ineligible"):
        select_rows([a],hours=.05,class_minimum=2)
    with pytest.raises(ValueError,match="Insufficient"):
        select_rows([row()],hours=50)


def test_explicit_large_dataset_policy_preserves_default_and_overlap_checks():
    pool=[row(i,face=180,shake=i==0) for i in range(30)]
    with pytest.raises(ValueError,match="Insufficient"):
        select_rows(pool,hours=.05,class_minimum=2)
    chosen,policy=select_rows(pool,hours=.05,class_minimum=2,core_face_floor=160,max_per_pair=24)
    assert sum(r['duration_s'] for r in chosen)>=180
    assert policy['floors']['core_face_p10']==160 and policy['max_per_pair']==24
    with pytest.raises(ValueError,match="authorized minimum"):
        select_rows(pool,hours=.05,class_minimum=2,core_face_floor=128)


def test_split_audit_normalizes_ab_and_preserves_version_namespace():
    assert person_keys("V00_S0001_I1_P0123A")==("V00_S0001","V00_P0123")
    a=dict(pair="a",group_id="g1",split="train",a={"id":"V00_S0001_I1_P0123A"},b={"id":"V00_S0001_I1_P0456"})
    b=dict(pair="b",group_id="g2",split="dev",a={"id":"V00_S0002_I2_P0123B"},b={"id":"V00_S0002_I2_P0789"})
    with pytest.raises(ValueError,match="Cross-split identity"):
        audit_catalog({"rows":[a,b]})
    b['a']['id']="V01_S0002_I2_P0123B"; b['b']['id']="V01_S0002_I2_P0789"
    pairs,counts=audit_catalog({"rows":[a,b]})
    assert counts==dict(train=1,dev=1)
