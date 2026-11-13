import sys,copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import pytest
from p2_listen.contracts import read
import p2_whisper_topology31 as new
import p2_whisper_event_feature_train31 as old


def cluster():
    c=read(ROOT/'experiments/p2_whisper_event_20261106/cluster.json')
    c['nodes'][0].update(name='node11',node='node11',host='172.29.189.119')
    return c


def test_rank_rng_slots_and_reserved_card_preserved():
    original=read(ROOT/'experiments/p2_whisper_event_20261106/cluster.json')
    assert [n['ranks'] for n in new.layout(cluster())]==[n['ranks'] for n in old.layout(original)]
    assert sum(len(n['gpus']) for n in new.layout(cluster()))==31
    assert new.layout(cluster())[-1]['gpus']==list(range(7))
    bad=cluster();bad['nodes'][-1]['gpus'].append(7)
    with pytest.raises(ValueError):new.layout(bad)


def test_new_commands_use_wrapper_and_leave_old_launcher_intact(monkeypatch):
    c=read(ROOT/'configs/p2_whisper_event_20261106.json');original=old.layout
    monkeypatch.setattr(new.cli,'read',lambda _:cluster())
    cmds=new.commands(c,ROOT/'experiments/p2_whisper_event_20261106/step_004000.pt')
    assert old.layout is original
    assert len(cmds)==4
    assert all(str(ROOT/'scripts/p2_whisper_topology31.py') in cmd for cmd in cmds)
    assert all('--resume' in cmd for cmd in cmds)
    assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6' in cmds[-1]


@pytest.mark.parametrize('observed_step',[4000,4001])
def test_migration_gates_checkpoint_old_exit_idle_and_exact_resume(monkeypatch,tmp_path,observed_step):
    import json
    import migrate_p2_whisper_at4000 as migration
    from p2_listen.contracts import atomic_json,sha
    root=tmp_path;w=root/'migration';q=root/'queue';o=root/'experiments/p2_whisper_event_20261106'
    feature=root/'experiments/p2_whisper_feature_20261106'
    for p in [w,q,o,feature]:p.mkdir(parents=True)
    atomic_json(q/'controller.json',dict(pid=1,start_ticks='test'))
    atomic_json(o/'status.json',dict(step=observed_step))
    ck=o/'step_004000.pt';ck.write_bytes(b'committed model adam and rng')
    atomic_json(ck.with_suffix('.complete.json'),dict(step=4000,sha256=sha(ck)))
    original=read(ROOT/'experiments/p2_whisper_event_20261106/cluster.json')
    new_cluster=cluster()
    for arm,out in [('event',o),('feature',feature)]:
        atomic_json(out/'cluster.json',original);atomic_json(w/f'{arm}_new_cluster.json',new_cluster)
    for name,value in [('ROOT',root),('W',w),('Q',q),('O',o)]:monkeypatch.setattr(migration,name,value)
    monkeypatch.setattr(migration,'prepare',lambda:None)
    monkeypatch.setattr(migration,'live',lambda _:False)
    monkeypatch.setattr(migration.time,'sleep',lambda _:None)
    probes=[]
    def idle(c):
        probes.append(c);return dict(all_idle=len(probes)>1)
    monkeypatch.setattr(migration,'idle_snapshot',idle)
    launched=[]
    class Child:
        pid=456
        def poll(self):return None
    def start(cmd,**kwargs):
        assert len(probes)==2
        assert not (o/'STOP').exists()
        assert read(o/'cluster.json')==read(feature/'cluster.json')==new_cluster
        launched.append(cmd);logs=root/'worker_logs';logs.mkdir()
        atomic_json(q/'state.json',dict(resume=str(ck),logs=str(logs.parent)))
        # Production puts receipts under logs/ranks.
        ranks=root/'ranks';ranks.mkdir()
        for name,count in [('node11',8),('node12',8),('node4',8),('node8',7)]:
            atomic_json(ranks/f'{name}_workers.json',dict(node=name,pids=list(range(count))))
        atomic_json(o/'status.json',dict(step=4001))
        return Child()
    monkeypatch.setattr(migration.subprocess,'Popen',start)
    if observed_step==4001:
        with pytest.raises(ValueError,match='exact4000'):migration.run()
        assert not launched and read(o/'cluster.json')==original
    else:
        migration.run()
        assert len(launched)==1 and read(w/'RESUME_VERIFIED.json')['success']
