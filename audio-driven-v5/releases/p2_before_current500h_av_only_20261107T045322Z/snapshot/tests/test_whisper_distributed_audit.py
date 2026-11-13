import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('whisper_audit',ROOT/'scripts/p2_whisper_cache_audit_cluster.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def prepare(tmp_path):
    config=tmp_path/'config.json';cache=tmp_path/'cache'
    m.atomic_json(config,dict(whisper_cache=str(cache)))
    m.atomic_json(cache/'protocol.json',dict(sample_ids=['a','b','c'],samples=3))
    for rank,ids in [(0,['a','c']),(1,['b'])]:
        m.atomic_json(cache/'distributed_audit'/f'rank{rank:02d}.json',dict(success=True,rank=rank,world=2,
            protocol_sha256=m.sha(cache/'protocol.json'),code_sha256=m.sha(m.__file__),
            records=[[sid,2,'receipt-'+sid] for sid in ids]))
    return config,cache


def test_merge_reconstructs_canonical_serial_receipt(tmp_path):
    config,cache=prepare(tmp_path);result=m.merge(config,2)
    h=hashlib.sha256()
    for sid in ['a','b','c']:h.update((sid+':receipt-'+sid+'\n').encode())
    assert result['cache_receipts_sha256']==h.hexdigest()
    assert result['samples']==3 and result['tokens']==6 and result['audio_source']=='speaker'


def test_missing_or_wrong_shard_never_publishes_complete(tmp_path):
    config,cache=prepare(tmp_path);p=cache/'distributed_audit/rank01.json'
    r=m.read(p);r['records']=[];m.atomic_json(p,r)
    with pytest.raises(ValueError,match='mismatched'):m.merge(config,2)
    assert not (cache/'COMPLETE.json').exists()
