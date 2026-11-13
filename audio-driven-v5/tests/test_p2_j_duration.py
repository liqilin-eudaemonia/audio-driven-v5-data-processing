from collections import Counter
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from p2_j_duration_execution import make_pairs,Schedule


@pytest.mark.parametrize('size,steps,start',[(285638,8927,73),(1031,33,3),(17,4,1)])
def test_prefix_membership_ordinals_and_tail(size,steps,start):
    rows=[dict(requested_duration_s=2+(i*37%1000)/100) for i in range(size)]
    plain=make_pairs(rows,steps,32,23,start,'random')
    aligned=make_pairs(rows,steps,32,23,start,'duration')
    assert plain[:start*32]==aligned[:start*32]
    assert Counter(plain)==Counter(aligned)
    for offset in range(start*32,len(plain),512):
        assert Counter(plain[offset:offset+512])==Counter(aligned[offset:offset+512])
    assert aligned==make_pairs(rows,steps,32,23,start,'duration')


def test_32_rank_sharding_and_resume_preserve_position():
    rows=[dict(requested_duration_s=2+i/100) for i in range(1031)]
    obj=object.__new__(Schedule);obj.size=1031;obj.batch=32;obj.seed=23
    obj.pairs=make_pairs(rows,33,32,23,3,'duration')
    for step in [1,3,4,19,33]:
        pieces=[obj.indices(1031,step,32,23,rank,32) for rank in range(32)]
        assert [v[0] for v in pieces]==obj.pairs[(step-1)*32:step*32]
    with pytest.raises(ValueError):obj.indices(1031,34,32,23)


def test_duration_reduces_batch_max_without_short_to_long_curriculum():
    rows=[dict(requested_duration_s=2+(i*37%1000)/100) for i in range(2048)]
    a=make_pairs(rows,64,32,23,0,'random');b=make_pairs(rows,64,32,23,0,'duration')
    longest=lambda p:[max(rows[i]['requested_duration_s'] for i,_ in p[k:k+32]) for k in range(0,len(p),32)]
    assert sum(longest(b))<sum(longest(a))
    assert longest(b)[:16]!=sorted(longest(b)[:16])
