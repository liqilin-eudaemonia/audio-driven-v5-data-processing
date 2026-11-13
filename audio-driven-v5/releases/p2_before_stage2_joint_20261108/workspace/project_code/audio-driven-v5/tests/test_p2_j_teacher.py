import math
import numpy as np
import pytest
import torch
from p2_j_teacher.media import selected_frames,frame_indices
from p2_j_teacher.core import validate_geometry
from p2_listen.layout import make_layout


def test_selected_teacher_frames_match_native_25fps_crop(tmp_path):
    import av
    path=tmp_path/'source.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=25);stream.width=160;stream.height=120;stream.pix_fmt='yuv420p';stream.codec_context.gop_size=25
        rng=np.random.default_rng(123)
        for _ in range(85):
            image=rng.integers(0,256,(120,160,3),dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    for start,duration in ((0,.13),(.017,.77),(.99,2.1),(2.92,.44)):
        item=dict(sample_id='test',source=dict(source=dict(archive=str(path),offset=0,byte_size=path.stat().st_size),
          duration_s=duration,source_start_s=start,video_transform=dict(source_size=[160,120],output_size=[512,512],
          resample='bicubic',crop=dict(x=23,y=12,width=90,height=90))))
        old,ot=selected_frames(item,None,(224,224),True)
        new,nt=selected_frames(item,None,(224,224),False)
        assert np.array_equal(old,new) and ot==nt
        assert len(new)==2*math.ceil(duration*4-1e-7)


def test_tail_and_ties_use_training_grid():
    assert frame_indices(.13)==[0,3]
    assert frame_indices(.5)==[0,3,6,9]
    assert max(frame_indices(1.01))==25


def test_teacher_layout_masks_keep_silence_and_reject_tail():
    layout=make_layout(.3,(2,2));value={}
    for mod in ('video','audio'):
        value[mod]=torch.zeros(*layout[mod+'_valid'].shape,3)
        value[mod+'_valid']=layout[mod+'_target_valid'].clone()
    validate_geometry(value,layout,3)
    assert value['audio_valid'].sum()==2
    value['audio_valid'][0,-1]=True
    with pytest.raises(ValueError,match='padded tail'):validate_geometry(value,layout,3)


def test_dynamic_claim_no_duplicate_and_recovers_released_lock(tmp_path):
    from p2_j_teacher.runner import claim
    jobs=[['a'],['b']]
    a=claim(tmp_path,jobs);b=claim(tmp_path,jobs)
    assert a[0]=='000000' and b[0]=='000001'
    assert claim(tmp_path,jobs) is None
    a[2].close()
    c=claim(tmp_path,jobs);assert c[0]=='000000'
    (tmp_path/'done').mkdir();(tmp_path/'done'/'000000.json').write_text('{}')
    c[2].close();b[2].close()
    d=claim(tmp_path,jobs);assert d[0]=='000001';d[2].close()
