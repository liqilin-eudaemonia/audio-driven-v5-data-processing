import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import pytest

RELEASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RELEASE/'src'))
from p2_listen import preflight as p


@pytest.fixture
def fake(tmp_path,monkeypatch):
    dep=tmp_path/'dependency';dep.write_text('original')
    meta=dict(binding={'schema':'test','config':1},rows=[{'sample_id':'x'}],targets={})
    result=dict(signatures=['original'],canonical=[dict(provenance={})],statistics={'head_nod':{'positive':1}})
    calls=[]
    def audit(c,m):
        calls.append(1)
        return dict(binding=m['binding'],dependencies=p.snapshot([str(dep)]),result=copy.deepcopy(result))
    monkeypatch.setattr(p,'full_audit',audit)
    return dep,meta,result,calls,dict(output=str(tmp_path)),tmp_path/'receipts'


def test_receipt_reuse(fake):
    dep,m,result,calls,c,cache=fake
    a,ra=p.get_audit(c,m,cache);b,rb=p.get_audit(c,m,cache)
    assert a==b==result and len(calls)==1
    assert ra['mode']=='full' and rb['mode']=='reused' and rb['full_audits']==0


@pytest.mark.parametrize('kind',['payload','same_size_restored_mtime','symlink'])
def test_dependency_changes_invalidate(fake,kind):
    dep,m,result,calls,c,cache=fake
    if kind=='symlink':
        actual=dep.with_name('actual');dep.rename(actual);dep.symlink_to(actual)
    p.get_audit(c,m,cache);before=dep.stat()
    if kind=='symlink':
        other=dep.with_name('other');other.write_text('original');dep.unlink();dep.symlink_to(other)
    else:
        dep.write_text('modified')
        if kind=='same_size_restored_mtime':os.utime(dep,ns=(before.st_atime_ns,before.st_mtime_ns))
    _,r=p.get_audit(c,m,cache)
    assert r['mode']=='rebuilt' and len(calls)==2


def test_parent_symlink_changes_invalidate(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    (a/'value').write_text('same');(b/'value').write_text('same')
    link=tmp_path/'cache';link.symlink_to(a,target_is_directory=True)
    prior=p.snapshot([str(link/'value')]);link.unlink();link.symlink_to(b,target_is_directory=True)
    assert not p.unchanged(prior)


@pytest.mark.parametrize('kind',['body','marker','missing'])
def test_corrupt_receipts_rejected(fake,kind):
    dep,m,result,calls,c,cache=fake;p.get_audit(c,m,cache);key=p.digest(m['binding'])
    receipt=cache/(key+'.json');marker=cache/(key+'.complete.json')
    if kind=='body':receipt.write_text('{}')
    elif kind=='marker':p.atomic_json(marker,dict(schema='bad',binding=key,sha256=p.sha(receipt)))
    else:receipt.unlink()
    with pytest.raises((ValueError,OSError)):p.get_audit(c,m,cache)
    assert len(calls)==1


def test_failure_never_publishes_complete(fake,monkeypatch):
    dep,m,result,calls,c,cache=fake
    def fail(*args):raise ValueError('corrupt real payload')
    monkeypatch.setattr(p,'full_audit',fail)
    with pytest.raises(ValueError):p.get_audit(c,m,cache)
    assert not list(cache.glob('*.complete.json'))


def test_concurrent_jobs_do_one_full_audit(fake):
    from concurrent.futures import ThreadPoolExecutor
    dep,m,result,calls,c,cache=fake
    with ThreadPoolExecutor(max_workers=4) as pool:
        rs=list(pool.map(lambda _:p.get_audit(c,m,cache),range(4)))
    assert len(calls)==1 and sum(r['full_audits'] for _,r in rs)==1


def test_binding_change_requires_new_receipt(fake):
    dep,m,result,calls,c,cache=fake;p.get_audit(c,m,cache)
    changed=copy.deepcopy(m);changed['binding']['config']=2
    assert p.get_audit(c,changed,cache)[1]['mode']=='full' and len(calls)==2


def test_bound_manifest_hash_and_parse_same_bytes(tmp_path,monkeypatch):
    f=tmp_path/'m.json';f.write_text('{"a":1}')
    data,h,state=p.bound_json(f);assert data=={'a':1} and h==p.sha(f)
    original=Path.read_bytes
    def changing(self):
        value=original(self)
        if self==f:self.write_text('{"a":2}')
        return value
    monkeypatch.setattr(Path,'read_bytes',changing)
    with pytest.raises(ValueError,match='changed'):p.bound_json(f)


def test_same_session_rejected(fake,monkeypatch):
    dep,m,result,calls,c,cache=fake
    monkeypatch.setattr(p,'metadata',lambda cfg:m)
    p.distributed_audit(c,0,1,'session_test1',2)
    with pytest.raises(FileExistsError):p.distributed_audit(c,0,1,'session_test1',2)


def test_rank_failure_propagates_to_all(fake,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    dep,m,result,calls,c,cache=fake
    def fail(c,m,cache_dir=None):raise ValueError('rank0 audit error')
    monkeypatch.setattr(p,'metadata',lambda cfg:m);monkeypatch.setattr(p,'get_audit',fail)
    def worker(rank):
        with pytest.raises(RuntimeError,match='rank0 audit error'):p.distributed_audit(c,rank,2,'failure_session',4)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,range(2)))


def test_distributed_only_rank0_audits(fake,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    dep,m,result,calls,c,cache=fake
    monkeypatch.setattr(p,'metadata',lambda cfg:m)
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda r:p.distributed_audit(c,r,3,'three_ranks_test',4),range(3)))
    assert len(calls)==1 and all(v[0]['statistics']==result['statistics'] for v in values)


def test_missing_rank_times_out(fake,monkeypatch):
    dep,m,result,calls,c,cache=fake;monkeypatch.setattr(p,'metadata',lambda cfg:m)
    with pytest.raises(RuntimeError,match='timed out'):p.distributed_audit(c,0,2,'missing_rank_test',.2)


def test_training_cache_loader_remains_hash_checked(tmp_path):
    from p2_listen.cache import save_part,load_part
    import torch
    folder=tmp_path/'cache';ident={'x':1};save_part(folder,ident,{'x':torch.ones(2)})
    raw=bytearray((folder/'payload.pt').read_bytes());raw[100]^=1;(folder/'payload.pt').write_bytes(raw)
    with pytest.raises(ValueError,match='Corrupt'):load_part(folder,ident)


def test_original_cache_modules_keep_original_paths():
    from p2_listen import features,contracts
    assert Path(features.__file__).resolve()==p.ROOT/'src/p2_listen/features.py'
    assert contracts.ROOT==p.ROOT


def test_ordered_io_preserves_statistics_order():
    def variable_latency(i):
        time.sleep((7-i%8)*.001)
        return i
    assert list(p.ordered_io(variable_latency,list(range(150))))==list(range(150))


def test_full_audit_rejects_mutation_before_receipt(tmp_path,monkeypatch):
    from p2_listen import training_data,features,annotations
    dep=tmp_path/'event.json';dep.write_text('{"positive":1}')
    c=dict(output=str(tmp_path),events=dict(pos_weight_cap=10,K_train=['head_nod']))
    meta=dict(binding={'key':1},manifest_states={},rows=[{'sample_id':'one'}],targets={'one':{}})
    monkeypatch.setattr(p,'dependencies',lambda c,m:[str(dep)])
    monkeypatch.setattr(p,'observation',lambda o:None);monkeypatch.setattr(p,'target',lambda t,o:None)
    def load_sample(*args):
        dep.write_text('{"positive":2}')
        return (None,None,None,None,object())
    monkeypatch.setattr(training_data,'load_training_sample',load_sample)
    monkeypatch.setattr(features,'load_text',lambda *args:None)
    monkeypatch.setattr(p,'read',lambda path:{'provenance':{}})
    monkeypatch.setattr(annotations,'class_statistics',lambda *args:{})
    with pytest.raises(ValueError,match='changed during full audit'):
        p.get_audit(c,meta,tmp_path/'receipts')
    assert not list((tmp_path/'receipts').glob('*.complete.json'))


def test_large_recent_dependency_requires_quiescence(tmp_path,monkeypatch):
    dep=tmp_path/'payload.pt';dep.write_bytes(b'x'*65537)
    state=p.snapshot([str(dep)])
    latest=max(state[str(dep)]['mtime_ns'],state[str(dep)]['ctime_ns'])/1e9
    sleeps=[]
    monkeypatch.setattr(p.time,'time',lambda:latest+.5)
    monkeypatch.setattr(p.time,'sleep',sleeps.append)
    with pytest.raises(ValueError,match='still being written'):p.stable_snapshot([str(dep)])
    assert sleeps and 1<=sleeps[0]<=2
