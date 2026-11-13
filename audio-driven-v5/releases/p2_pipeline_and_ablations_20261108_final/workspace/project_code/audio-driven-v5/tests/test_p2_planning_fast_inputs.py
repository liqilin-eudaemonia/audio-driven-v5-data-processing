import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from p2_listen.contracts import digest,sha
from p2_planning_sft.fast_media import FastInputs,decode_first_frame
from listen_response_sft.data import first_frame


def test_real_decoder_preserves_keyframe_boundary_and_nearest_pts(tmp_path):
    import av
    path=tmp_path/'source.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p';stream.codec_context.gop_size=25
        rng=np.random.default_rng(22)
        for i in range(80):
            rgb=rng.integers(0,256,size=(120,160,3),dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    source=dict(source=dict(archive=str(path),offset=0,byte_size=path.stat().st_size),
                video_transform=dict(source_size=[160,120],output_size=[512,512],resample='bicubic',crop=dict(x=20,y=10,width=90,height=90)))
    for start in (0,.02,.04,.99,1,1.02,1.04,1.999,2.001,3):
        row=dict(source,source_start_s=start)
        old,ot=first_frame(row);new,nt=decode_first_frame(row,4)
        assert ot==nt and np.array_equal(np.asarray(old),np.asarray(new)),start
    from p2_planning_sft.isolated_decode import IsolatedDecoder
    decoder=IsolatedDecoder(log_dir=tmp_path/'logs')
    try:
        for start in (0,.02,.99,1.02,2.001,3):
            row=dict(source,source_start_s=start)
            old,ot=first_frame(row);new,nt=decoder(row)
            assert ot==nt and np.array_equal(np.asarray(old),np.asarray(new)),start
    finally:decoder.close()


def test_hung_decoder_is_killed_and_original_mode_retries(tmp_path):
    import sys,time,os
    from p2_planning_sft.isolated_decode import IsolatedDecoder
    helper=tmp_path/'helper.py'
    helper.write_text('''import sys,json,time,os,hashlib
import numpy as np
for line in sys.stdin:
 r=json.loads(line)
 if r['mode']=='auto':
  open(r['source']['pid_file'],'w').write(str(os.getpid()))
  time.sleep(60)
 a=np.full((512,512,3),7,dtype=np.uint8)
 np.save(r['output'],a)
 print(json.dumps(dict(pts=1.02,rgb_sha256=hashlib.sha256(a.tobytes()).hexdigest())),flush=True)
''')
    decoder=IsolatedDecoder(timeout=2,command=[sys.executable,str(helper)],log_dir=tmp_path/'logs')
    try:
        t=time.monotonic();image,pts=decoder(dict(pid_file=str(tmp_path/'pid')))
        assert time.monotonic()-t<8 and pts==1.02
        assert np.all(np.asarray(image)==7)
        assert not Path('/proc', (tmp_path/'pid').read_text()).exists()
    finally:decoder.close()


@pytest.fixture
def legacy(tmp_path):
    for name in ('samples','media_audit','references'):(tmp_path/name).mkdir()
    sid='sample-1';source=dict(video_transform=dict(crop=dict(x=1,y=2,width=40,height=40)))
    item=dict(source=source,split='train')
    (tmp_path/'samples'/f'{sid}.json').write_text(json.dumps(item))
    image=Image.new('RGB',(512,512),(1,2,3));image.save(tmp_path/'references'/f'{sid}.png')
    audit=dict(sample_id=sid,split='train',final_crop=source['video_transform']['crop'],reference_from_first_frame=True,
               source_video_used=True,crop_revision='v',first_rgb_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),source_frame_pts_s=[2.0])
    (tmp_path/'media_audit'/f'{sid}.json').write_text(json.dumps(audit))
    index=tmp_path/'index.sqlite'
    with sqlite3.connect(index) as c:
        c.execute('create table frames(sid text primary key,digest text)');c.execute('insert into frames values(?,?)',(sid,digest(item)))
    seal=tmp_path/'seal.json';st=index.stat();seal.write_text(json.dumps(dict(root=str(tmp_path),index=str(index),index_bytes=st.st_size,index_mtime_ns=st.st_mtime_ns)))
    config=dict(version='verified-png-auto4-v1',decode_threads=4,legacy_seal=str(seal),legacy_seal_sha256=sha(seal))
    return FastInputs(config),dict(sample_id=sid,split='train'),dict(source=source,crop_provenance=dict(revision='v')),tmp_path


def test_legacy_cache_matches_source_and_checks_pixels(legacy):
    reader,row,resolved,root=legacy
    image,pts=reader.legacy_frame(row,resolved);assert pts==2 and image.size==(512,512)
    wrong=copy.deepcopy(resolved);wrong['source']['video_transform']['crop']['x']=2
    assert reader.legacy_frame(row,wrong) is None
    Image.new('RGB',(512,512),(9,9,9)).save(root/'references'/'sample-1.png')
    with pytest.raises(ValueError,match='RGB changed'):reader.legacy_frame(row,resolved)


def test_unindexed_legacy_id_does_not_reuse_image(legacy):
    reader,row,resolved,_=legacy
    assert reader.legacy_frame(dict(row,sample_id='missing'),resolved) is None


def test_corrupt_legacy_source_refuses_reuse(legacy):
    reader,row,resolved,root=legacy
    (root/'samples'/'sample-1.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):reader.legacy_frame(row,resolved)
