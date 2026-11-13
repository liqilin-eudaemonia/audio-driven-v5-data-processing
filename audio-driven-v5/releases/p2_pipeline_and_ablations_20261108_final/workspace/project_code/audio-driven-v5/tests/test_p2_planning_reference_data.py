import hashlib
import json
import pickle
import sqlite3

import numpy as np
from PIL import Image
import pytest

from p2_planning_sft import reference_data as mod


@pytest.fixture
def sealed(tmp_path):
    root=tmp_path/'data';root.mkdir()
    c=sqlite3.connect(root/'index.sqlite')
    c.execute('create table samples(sid text primary key,split text,file text,offset integer,bytes integer,duration real,group_id text,pair text)')
    summary={'splits':{},'manifests':{}}
    for split,n in [('train',4),('dev',3)]:
        path=root/(split+'.jsonl')
        with path.open('wb') as f:
            for i in range(n):
                sid=f'{split}-{i}'
                row=dict(sample_id=sid,split=split,duration_s=.5,group_id=split,pair=split,
                         inputs=dict(speaker_audio={},listener_first_frame={}),target=dict(text='The listener smiles.'))
                raw=(json.dumps(row)+'\n').encode();offset=f.tell();f.write(raw)
                c.execute('insert into samples values(?,?,?,?,?,?,?,?)',(sid,split,path.name,offset,len(raw),.5,split,split))
        summary['splits'][split]={'counts':{'keep':n},'clip_hours':{'keep':n*.5/3600}}
        summary['manifests'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    c.commit();c.close()
    (root/'crop_config.json').write_text('{}')
    (root/'summary.json').write_text(json.dumps(summary))
    seal=tmp_path/'seal.json'
    mod.prepare(root,seal,expected_train=4,expected_dev=3,diagnostic_count=2)
    return root,seal


def test_indexed_selection_and_pickle_reopen(sealed):
    _,seal=sealed
    train=mod.ReferencePlanningDataset(seal,'train')
    assert len(train)==4 and train.record(2)['sample_id']=='train-2'
    other=pickle.loads(pickle.dumps(train))
    assert other._stream is None and other.record(1)['sample_id']=='train-1'
    train.close();other.close()
    a=mod.ReferencePlanningDataset(seal,'dev',selection='diagnostic')
    b=mod.ReferencePlanningDataset(seal,'dev',selection='diagnostic')
    assert a.ids==b.ids and len(a)==2
    assert not set(a.ids)&set(train.ids)
    with pytest.raises(ValueError):mod.ReferencePlanningDataset(seal,'test')


def test_modified_manifest_rejected(sealed):
    root,seal=sealed
    with (root/'train.jsonl').open('ab') as f:f.write(b'\n')
    with pytest.raises(ValueError,match='Prepared input changed'):mod.ReferencePlanningDataset(seal,'train')


def test_bad_offsets_fail_full_preflight(sealed,tmp_path):
    root,_=sealed
    with sqlite3.connect(root/'index.sqlite') as c:c.execute('update samples set offset=1 where sid="train-0"')
    with pytest.raises(ValueError,match='byte range'):mod.prepare(root,tmp_path/'new.json',expected_train=4,expected_dev=3)


def test_source_groups_cannot_overlap(sealed,tmp_path):
    root,_=sealed
    # Match manifest and index, but deliberately break source isolation.
    with sqlite3.connect(root/'index.sqlite') as c:
        for split in ('train','dev'):
            p=root/(split+'.jsonl');rows=[json.loads(x) for x in p.read_text().splitlines()]
            with p.open('wb') as f:
                for row in rows:
                    row['group_id']='shared';b=(json.dumps(row)+'\n').encode();pos=f.tell();f.write(b)
                    c.execute('update samples set group_id=?,offset=?,bytes=? where sid=?',('shared',pos,len(b),row['sample_id']))
    summary=json.loads((root/'summary.json').read_text())
    for split in ('train','dev'):
        summary['manifests'][split+'.jsonl']['sha256']=hashlib.sha256((root/(split+'.jsonl')).read_bytes()).hexdigest()
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='group overlap'):mod.prepare(root,tmp_path/'new.json',expected_train=4,expected_dev=3)


def test_true_resampling_and_target_preservation(sealed,monkeypatch):
    _,seal=sealed
    wave=np.sin(2*np.pi*440*np.arange(24000)/48000).astype('float32')
    def fake(row,crop):
        return dict(inputs=dict(speaker_audio=wave,speaker_sample_rate=48000,listener_first_frame=Image.new('RGB',(512,512))),
                    target_text=row['target']['text'],audit={'future_listener_media_in_inputs':False})
    monkeypatch.setattr(mod,'load_example',fake)
    d=mod.ReferencePlanningDataset(seal,'train');item=d[0]
    assert item['audio'].shape==(8000,) and item['image'].size==(512,512)
    assert item['target']=='The listener smiles.' and not item['audit']['future_listener_media_in_inputs']
    freq=np.argmax(abs(np.fft.rfft(item['audio'])))*16000/8000
    assert freq==440
    d.close()
