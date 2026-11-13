import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pytest
from p2_listen.contracts import read,atomic_json,sha
from p2_whisper_continue30 import make_remaining,normalized,Schedule


def config(arm,tmp_path):
    pp=ROOT/f'configs/p2_whisper_{arm}_20261106.json';p=read(pp);prefix=4000 if arm=='event' else 0
    pairs,proof=make_remaining(p,prefix);plan=tmp_path/'schedule.json';atomic_json(plan,dict(pairs=pairs,proof=proof))
    c=copy.deepcopy(p);c['output']=str(tmp_path/'out');c['training'].update(global_batch=30,steps=proof['steps']);c['duration_sampling'].update(window=480,start_step=prefix,manifest=str(plan),sha256=sha(plan))
    c['continuation30']=dict(parent_config=str(pp),parent_config_sha256=sha(pp),prefix_step=prefix,rng_parent_ranks=list(range(24,31))+[7]+list(range(8,24))+list(range(6)))
    return c,p,pairs,proof


@pytest.mark.parametrize('arm,steps,remain,pad',[('event',8134,124000,20),('feature',8267,248000,10)])
def test_exact_remaining_members_noise_and_all_ranks(tmp_path,arm,steps,remain,pad):
    from p2_j_duration_execution import Schedule as Original
    c,p,pairs,proof=config(arm,tmp_path);assert normalized(c)==p
    assert proof['steps']==steps and proof['remaining']==remain and proof['padding']==pad
    original=Original(read(p['ablation_reference']['config']));size=len(original.rows) if hasattr(original,'rows') else 285638
    real=original.pairs[:248000][c['continuation30']['prefix_step']*31:]
    assert sorted(x for x in pairs if x[1]<size)==sorted(real)
    s=Schedule(c)
    for step in [s.prefix+1,s.steps]:
        selected=[s.indices(s.size,step,30,s.seed,r,30)[0] for r in range(30)]
        assert selected==pairs[(step-s.prefix-1)*30:(step-s.prefix)*30]
    bad=copy.deepcopy(c);bad['loss_weights']['event']=.1
    with pytest.raises(ValueError):normalized(bad)
    with pytest.raises(ValueError):s.indices(s.size,s.prefix,30,s.seed,0,30)
    assert len(set(c['continuation30']['rng_parent_ranks']))==30


def test_install_keeps_old_configs_and_validates_new(tmp_path):
    c,p,_,_=config('event',tmp_path)
    from p2_whisper_continue30 import install
    install()
    from p2_whisper_ablation import config as C
    C.validate(c);C.validate(p);C.validate_baseline(c,read(c['baseline_config']))
    assert C.Schedule(c).steps==8134
    assert C.Schedule(p).steps==8000
