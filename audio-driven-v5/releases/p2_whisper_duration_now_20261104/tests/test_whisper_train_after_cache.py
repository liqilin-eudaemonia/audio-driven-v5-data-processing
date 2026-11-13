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
