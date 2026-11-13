import importlib.util
from pathlib import Path
import json

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('whisper_train_queue',ROOT/'scripts/p2_whisper_train_after_cache.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
from p2_listen.contracts import atomic_json,sha


def test_cache_must_be_full_hashed_and_speaker(tmp_path):
    assert not queue.cache_ready(tmp_path,2)
    atomic_json(tmp_path/'protocol.json',dict(samples=2))
    value=dict(success=True,samples=2,audio_source='speaker',protocol_sha256=sha(tmp_path/'protocol.json'))
    atomic_json(tmp_path/'COMPLETE.json',value);assert queue.cache_ready(tmp_path,2)
    assert not queue.cache_ready(tmp_path,3)
    value['audio_source']='listener';atomic_json(tmp_path/'COMPLETE.json',value)
    assert not queue.cache_ready(tmp_path,2)


def test_initial_gate_checks_pure_av_gradients_and_all_rank_longest_probe(tmp_path):
    rows=[dict(step=i,av=1.,av_video=.6,av_audio=.4,weighted_total=1.,event=0.,feature_video=0.,
        feature_audio=0.,grad_norm=dict(connector=1.,text_lora=1.,whisper=1.),seconds=1.) for i in (1,2)]
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for rank in range(32):
        atomic_json(tmp_path/'memory_probe'/f'rank{rank:03d}.json',dict(rank=rank,duration_s=32.,peak_reserved_gib=100.))
    assert queue.verify_initial(tmp_path)['pure_av']
    rows[1]['event']=.1;(tmp_path/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='Only AV'):queue.verify_initial(tmp_path)


def test_output_stop_allows_checkpoint_write_without_cancelling_cluster(tmp_path,monkeypatch):
    import sys
    import time
    from types import SimpleNamespace
    from p2_whisper_av import checkpoint
    work=tmp_path/'queue';work.mkdir();out=tmp_path/'output';out.mkdir()
    config=tmp_path/'config.json';cluster=tmp_path/'cluster.json'
    atomic_json(config,{});atomic_json(cluster,dict(python='unused'))
    atomic_json(work/'queue.json',dict(output=str(out),cache=str(tmp_path/'cache'),samples=2,
        config=str(config),config_sha256=sha(config),cluster=str(cluster),cluster_sha256=sha(cluster)))
    atomic_json(out/'run.json',{});atomic_json(out/'step_000002.complete.json',{})
    atomic_json(out/'GPU_VALIDATION.json',dict(success=True))
    monkeypatch.setattr(queue,'cache_ready',lambda *args:True)
    monkeypatch.setattr(queue,'execute',lambda *args:None)
    monkeypatch.setattr(queue,'resource_checks',lambda *args:[dict(free=True)])
    monkeypatch.setattr(checkpoint,'checked',lambda *args:dict(identity=dict(world=32,experiment={})))
    monkeypatch.setitem(sys.modules,'p2_j_whisper_av',SimpleNamespace(commands=lambda *args,**kw:[]))
    def launch(*args,cancel_event,**kwargs):
        (out/'STOP').write_text('Finish writing checkpoint')
        # Cross the real monitor polling boundary while the trainer saves.
        time.sleep(16)
        assert not cancel_event.is_set()
        atomic_json(out/'status.json',dict(step=209,complete=False))
    monkeypatch.setitem(sys.modules,'listen_cluster',SimpleNamespace(launch=launch))
    queue.run(work)
    assert json.loads((work/'state.json').read_text())['phase']=='stopped'
