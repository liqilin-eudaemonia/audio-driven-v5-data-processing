import importlib.util,random
from collections import Counter
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/p2_duration_execution.py'
spec=importlib.util.spec_from_file_location('duration_execution',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def rows(n):return [dict(requested_duration_s=2+(i*37%101)/10) for i in range(n)]

@pytest.mark.parametrize('n,steps',[(29975,1249),(17,4),(384,16),(385,17)])
def test_exact_membership_rng_and_budget(n,steps):
    data=rows(n);plain=m.schedule(data,steps,24,20261023,'random');aligned=m.schedule(data,steps,24,20261023)
    assert Counter(plain)==Counter(aligned)
    assert len(aligned)==steps*24
    assert len({ordinal for _,ordinal in aligned})==steps*24
    assert aligned==m.schedule(data,steps,24,20261023)
    # Each window uses exactly the original examples and original per-example RNG ordinal.
    for start in range(0,len(plain),384):assert Counter(plain[start:start+384])==Counter(aligned[start:start+384])

def test_baseline_matches_native_epoch_shuffle():
    actual=m.schedule(rows(29),4,24,77,'random')
    for index,ordinal in actual:
        epoch,offset=divmod(ordinal,29);order=list(range(29));random.Random(77+epoch).shuffle(order)
        assert index==order[offset]

def test_resume_and_world_sharding():
    obj=object.__new__(m.Schedule);obj.size=29975;obj.batch=24;obj.seed=23;obj.pairs=m.schedule(rows(obj.size),1249,24,23)
    for world in [1,6,8,24]:
        for step in [1,16,17,1249]:
            per_rank=[obj.indices(obj.size,step,24,23,rank,world) for rank in range(world)]
            recovered=[per_rank[rank][micro] for micro in range(24//world) for rank in range(world)]
            assert recovered==obj.pairs[(step-1)*24:step*24]
    with pytest.raises(ValueError):obj.indices(obj.size,1250,24,23)
    with pytest.raises(ValueError):obj.indices(obj.size,1,24,23,24,24)

def test_alignment_reduces_longest_without_global_curriculum():
    data=rows(768);plain=m.schedule(data,32,24,99,'random');aligned=m.schedule(data,32,24,99)
    longest=lambda seq:[max(data[i]['requested_duration_s'] for i,_ in seq[start:start+24]) for start in range(0,len(seq),24)]
    assert sum(longest(aligned))<sum(longest(plain))
    assert longest(aligned)[:16]!=sorted(longest(aligned)[:16])
