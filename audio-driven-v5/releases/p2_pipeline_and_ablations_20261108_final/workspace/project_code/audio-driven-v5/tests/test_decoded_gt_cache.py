import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p2_testsets_eval.decoded_gt import DecodedGTCache


def setup(tmp_path, fail=False):
    archive = tmp_path/'source.tar';archive.write_bytes(b'original')
    code = tmp_path/'reader.py';code.write_text('original sealed reader')
    frames = np.random.default_rng(42).integers(0,256,(2,512,512,3),dtype=np.uint8)
    audit = dict(frames=2,rgb_sha256=hashlib.sha256(memoryview(frames)).hexdigest(),source_pts_s=[1.,1.04])
    calls = []
    def decode(reference):
        calls.append(reference)
        if fail:raise RuntimeError('decode failed')
        return frames.copy(),25.,dict(audit)
    def resolve(reference):
        if reference.get('row_sha256')!='sealed':raise ValueError('Frozen row changed')
        return {},dict(source=dict(source=dict(archive=str(archive)),crop=reference.get('crop',1)),crop_provenance='final')
    reader = SimpleNamespace(__file__=str(code),decode_reference=decode,resolve=resolve)
    return DecodedGTCache(tmp_path/'cache',reader),dict(sample_id='id',row_sha256='sealed'),frames,audit,calls,archive


def test_exact_pixels_audit_and_hit_isolation(tmp_path):
    cache,ref,frames,audit,calls,_=setup(tmp_path)
    a,_,first=cache.get(ref);b,_,second=cache.get(ref)
    np.testing.assert_array_equal(a,frames);np.testing.assert_array_equal(a,b)
    assert first==second==audit and len(calls)==1
    first['source_pts_s'][0]=900
    assert cache.get(ref)[2]==audit
    assert not b.flags.writeable and cache.report()['hits']==2


def test_changed_crop_or_source_never_reuses_old_cache(tmp_path):
    cache,ref,_,_,calls,archive=setup(tmp_path)
    cache.get(ref);cache.get(dict(ref,crop=2));archive.write_bytes(b'changed source archive')
    cache.get(ref)
    assert len(calls)==3
    with pytest.raises(ValueError,match='Frozen row'):cache.get(dict(ref,row_sha256='wrong'))


def test_concurrent_requests_decode_once_and_no_partial_publication(tmp_path):
    cache,ref,frames,_,calls,_=setup(tmp_path)
    with ThreadPoolExecutor(max_workers=3) as pool:
        out=list(pool.map(cache.get,[ref]*3))
    assert len(calls)==1
    for value in out:np.testing.assert_array_equal(value[0],frames)
    other=tmp_path/'failure';other.mkdir()
    broken,ref,*_=setup(other,fail=True)
    with pytest.raises(RuntimeError,match='decode failed'):broken.get(ref)
    assert not list(broken.root.glob('*/*/metadata.json'))
    assert not list(broken.root.glob('*/*/frames.npy'))


def test_corrupted_pixels_are_rejected_and_reader_restored(tmp_path):
    cache,ref,_,_,_,_=setup(tmp_path);cache.get(ref)
    path=next(cache.root.glob('*/*/frames.npy'))
    changed=np.load(path,mmap_mode='r+');changed[0,0,0,0]^=1;changed.flush();del changed
    with pytest.raises(ValueError,match='pixel checksum'):cache.get(ref)
    original=cache.reader.decode_reference
    with pytest.raises(RuntimeError):
        with cache.installed():
            assert cache.reader.decode_reference==cache.get
            raise RuntimeError('test cleanup')
    assert cache.reader.decode_reference==original
