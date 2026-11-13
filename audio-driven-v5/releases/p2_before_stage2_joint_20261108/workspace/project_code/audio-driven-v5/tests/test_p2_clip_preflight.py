import json
import sqlite3
from types import SimpleNamespace

import pytest
from p2_clip_events import backend, VERSION
from p2_clip_events.labels import LabelReader
from p2_listen.contracts import ROOT, read, atomic_json, digest, sha


def fixture(tmp_path,monkeypatch):
    from p2_j_cache import data,runtime
    c=read(ROOT/'configs/p2_j_clip_events_20261101/guided_av_event_s1.json')
    c.update(output=str(tmp_path/'run'),clip_labels=str(tmp_path/'labels'))
    labels=tmp_path/'labels';labels.mkdir()
    entries={'s0':'source0','s1':'source1'}
    with sqlite3.connect(labels/'labels.sqlite') as conn:
        conn.execute('CREATE TABLE labels (ordinal INTEGER PRIMARY KEY,sid TEXT,value TEXT)')
        for i,(sid,source) in enumerate(entries.items()):
            conn.execute('INSERT INTO labels VALUES (?,?,?)',(i,sid,json.dumps({'source_digest':source})))
    names=[n for group in c['event_supervision']['classes'].values() for n in group]
    atomic_json(labels/'manifest.json',dict(schema=VERSION,database_sha256=sha(labels/'labels.sqlite'),
        full_train=False,statistics={n:dict(enabled=True,pos_weight=1.) for n in names}))
    work=tmp_path/'cache'
    def identity(m,item,part,assets):return dict(sid=item['sample_id'],part=part)
    for sid in entries:
        for part in ('vae','text_guided','input_plan'):
            folder=work/'cache'/part/sid;value=identity(None,{'sample_id':sid},part,None)
            atomic_json(folder/'complete.json',dict(identity=digest(value),contract=value))
            (folder/'payload.pt').write_bytes(b'payload')
    dataset=SimpleNamespace(spec=c,labels=LabelReader(labels),rows=[dict(sample_id=s) for s in entries],
        reader=SimpleNamespace(manifest=dict(entries=entries),work=work,assets={p:{} for p in ('vae','text_guided','input_plan')},part=lambda *a:None))
    monkeypatch.setattr(backend,'Dataset',lambda c:dataset)
    monkeypatch.setattr(backend,'source_identity',lambda:{})
    monkeypatch.setattr(backend,'data_identity',lambda c:{})
    monkeypatch.setattr(data,'sample',lambda w,m,sid:{'sample_id':sid})
    monkeypatch.setattr(runtime,'cache_identity',identity)
    return c,work


def test_serial_and_parallel_check_the_same_rows(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    a=backend.preflight(c,processes=1);b=backend.preflight(c,processes=2)
    for key in ('samples','statistics','data_identity','source','experiment_mode'):assert a[key]==b[key]
    assert a['samples']==2


@pytest.mark.parametrize('processes',[1,2])
def test_corrupt_marker_fails_including_child_process(tmp_path,monkeypatch,processes):
    c,work=fixture(tmp_path,monkeypatch)
    path=work/'cache/input_plan/s1/complete.json';value=read(path);value['identity']='corrupt';atomic_json(path,value)
    with pytest.raises(ValueError,match='Invalid cache'):backend.preflight(c,processes=processes)
    assert not (tmp_path/'run/preflight.json').exists()
