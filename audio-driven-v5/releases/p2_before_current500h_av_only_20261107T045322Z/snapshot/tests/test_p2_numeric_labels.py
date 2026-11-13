"""Numeric cache parity, loss gradients, stale sources and immutable publication."""
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from p2_listen.contracts import atomic_json, digest, sha, read
from p2_clip_events import VERSION
from p2_clip_events.labels import LabelReader, tensors
from p2_clip_events.model import event_sums
from p2_ltx_events.numeric_labels import prepare_numeric, NumericLabels

NAMES = ('visual:head nodding', 'visual:smiling', 'audio:speech')


def fixture(tmp_path, weight=.1):
    source=tmp_path/'source';folder=source/'clip_labels';folder.mkdir(parents=True)
    rows=[dict(sample_id='positive',positive=[NAMES[0]],unknown=[],allow_weak=True),
          dict(sample_id='uncertain',positive=[],unknown=[NAMES[0]],allow_weak=True),
          dict(sample_id='incomplete',positive=[NAMES[2]],unknown=[],allow_weak=False),
          dict(sample_id='conflict',positive=[],unknown=list(NAMES),allow_weak=True)]
    for row in rows:row['source_digest']=digest(row['sample_id'])
    ids=[r['sample_id'] for r in rows]
    atomic_json(source/'manifest.json',dict(training_ids=ids,entries={r['sample_id']:r['source_digest'] for r in rows},source_split_counts={'train':3,'dev':1}))
    with sqlite3.connect(folder/'labels.sqlite') as db:
        db.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,sid TEXT UNIQUE,value TEXT)')
        db.executemany('INSERT INTO labels VALUES(?,?,?)',[(i,r['sample_id'],json.dumps(r)) for i,r in enumerate(rows)])
    atomic_json(folder/'manifest.json',dict(schema=VERSION,database_sha256=sha(folder/'labels.sqlite'),samples=len(ids),
        cache_manifest_sha256=sha(source/'manifest.json'),absence_weight=weight,label_policy='test-policy'))
    config=dict(current_cache_index='fixture',current_source=str(source),clip_labels=str(folder),
                event_supervision=dict(classes={'video':list(NAMES[:2]),'audio':list(NAMES[2:])}))
    return config,rows,ids


@pytest.mark.parametrize('weight',[0.,.1,1.])
def test_exact_targets_loss_and_gradient_parity(tmp_path,weight):
    c,rows,ids=fixture(tmp_path,weight);out=tmp_path/'numeric';prepare_numeric(c,out)
    labels=LabelReader(c['clip_labels']);reader=NumericLabels(out,labels,NAMES,ids)
    try:
        for i,row in enumerate(rows):
            expected=tensors(row,NAMES,weight)
            # Runtime must not consult or convert the sparse JSON labels.
            with patch.object(labels,'row',side_effect=AssertionError('sparse read')):
                actual=reader.events(i,row['sample_id'],row['source_digest'])
            for name in NAMES:
                for key in ('y','known','weight'):
                    assert torch.equal(expected[name][key],actual['clip/'+name][key])
            z=torch.tensor([.7],requires_grad=True)
            logits={name:z for name in NAMES};pos={name:1.7 for name in NAMES}
            a,ac=event_sums(logits,expected,pos)
            b,bc=event_sums(logits,{k.removeprefix('clip/'):v for k,v in actual.items()},pos)
            assert all(torch.equal(ac[k],bc[k]) for k in ac)
            la=sum(a.values());lb=sum(b.values())
            assert torch.equal(la,lb)
            assert torch.equal(torch.autograd.grad(la,z,retain_graph=True)[0],torch.autograd.grad(lb,z)[0])
            actual['clip/'+NAMES[0]]['y'].fill_(99)
            assert torch.equal(reader.events(i,row['sample_id'],row['source_digest'])['clip/'+NAMES[0]]['y'],expected[NAMES[0]]['y'])
    finally:labels.conn.close()


def test_reject_stale_order_classes_sources_and_payload(tmp_path):
    c,rows,ids=fixture(tmp_path);out=tmp_path/'numeric';prepare_numeric(c,out)
    labels=LabelReader(c['clip_labels'])
    try:
        for names,order in [(NAMES[::-1],ids),(NAMES,ids[::-1])]:
            with pytest.raises(ValueError,match='identity'):
                NumericLabels(out,labels,names,order)
        reader=NumericLabels(out,labels,NAMES,ids)
        with pytest.raises(ValueError,match='sample/index'):reader.events(0,ids[1],rows[0]['source_digest'])
        with pytest.raises(ValueError,match='Stale'):reader.events(0,ids[0],'f'*64)
        with pytest.raises(FileExistsError):prepare_numeric(c,out)
        with (out/'y.npy').open('r+b') as stream:
            stream.seek(-1,2);stream.write(b'X')
        with pytest.raises(ValueError,match='checksum'):NumericLabels(out,labels,NAMES,ids)
    finally:labels.conn.close()


def test_failed_prepare_never_publishes(tmp_path):
    c,rows,ids=fixture(tmp_path);source=Path(c['current_source'])
    m=read(source/'manifest.json');m['entries'][ids[0]]='f'*64;atomic_json(source/'manifest.json',m)
    lm=read(Path(c['clip_labels'])/'manifest.json');lm['cache_manifest_sha256']=sha(source/'manifest.json');atomic_json(Path(c['clip_labels'])/'manifest.json',lm)
    out=tmp_path/'numeric'
    with pytest.raises(ValueError,match='Stale'):prepare_numeric(c,out)
    assert not out.exists() and not list(tmp_path.glob('numeric.tmp-*'))
