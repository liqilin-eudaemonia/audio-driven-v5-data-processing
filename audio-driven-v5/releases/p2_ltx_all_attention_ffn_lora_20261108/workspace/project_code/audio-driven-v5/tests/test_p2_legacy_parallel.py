import sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from p2_legacy_parallel import commands
from p2_listen.training import indices

@pytest.mark.parametrize('world',[6,12,24])
def test_original_global_sample_order_and_ordinals(world):
    expected=sorted(item for rank in range(24) for item in indices(30415,106,24,20261023,rank,24))
    actual=sorted(item for rank in range(world) for item in indices(30415,106,24,20261023,rank,world))
    assert expected==actual

@pytest.mark.parametrize('slots,world', [([0],6),([1,3],12),([0,1,2,3],24)])
def test_launch_gpu_ownership(slots,world):
    cluster=dict(nodes=[dict(host=str(i),gpus=list(range(8))) for i in range(3)],network={},python='python')
    before=copy.deepcopy(cluster);cc,cmd=commands(cluster,slots,'config',None,30001)
    assert cluster==before and len(cmd)==3
    for n,c in zip(cc['nodes'],cmd):
        assert n['gpus']==sorted(x for s in slots for x in [s*2,s*2+1])
        assert '--nproc_per_node='+str(world//3) in c
    assert not any('--resume' in c for c in cmd)

def test_unsupported_three_slots_rejected():
    with pytest.raises(ValueError):commands({'nodes':[]},[0,1,2],'config',None,30001)
