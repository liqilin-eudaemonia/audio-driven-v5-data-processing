import sys
import unittest
import tempfile
from pathlib import Path
import av
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_benchmark.frame_crop import crop_original_frame,crop_original_frame_roi


class RoiCropTests(unittest.TestCase):
    def test_real_decoder_chroma_location_is_preserved(self):
        # Decoded H264 carries chroma siting that a newly allocated VideoFrame
        # does not inherit. Synthetic from_ndarray frames alone miss this bug.
        for fmt in ['yuv420p','yuv420p10le']:
            with self.subTest(fmt=fmt),tempfile.TemporaryDirectory() as d:
                path=Path(d)/'source.mp4'
                with av.open(str(path),'w') as out:
                    stream=out.add_stream('libx264',rate=25);stream.width=642;stream.height=482;stream.pix_fmt=fmt
                    stream.options={'crf':'18','threads':'1'}
                    for i in range(3):
                        rgb=np.random.default_rng(i).integers(0,256,(482,642,3),dtype=np.uint8)
                        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=i
                        for packet in stream.encode(frame):out.mux(packet)
                    for packet in stream.encode():out.mux(packet)
                with av.open(str(path)) as inp:
                    for frame in inp.decode(video=0):
                        tr=dict(source_size=[642,482],crop=dict(x=33,y=32,width=306,height=306),output_size=[512,512],resample='bicubic')
                        self.assertEqual(crop_original_frame(frame,tr).tobytes(),crop_original_frame_roi(frame,tr).tobytes())

    def test_pixels_identical_at_edges_odd_offsets_and_color_modes(self):
        for fmt in ['yuv420p','yuv420p10le','rgb24']:
            rgb=np.random.default_rng(78).integers(0,256,(482,642,3),dtype=np.uint8)
            f=av.VideoFrame.from_ndarray(rgb,format='rgb24').reformat(format=fmt)
            for space in [1,2,5,6,9]:
                for color_range in [0,1,2]:
                    f.colorspace=space;f.color_range=color_range
                    for x,y,size in [(0,0,333),(17,63,333),(309,149,333),(0,0,480)]:
                        tr=dict(source_size=[642,482],crop=dict(x=x,y=y,width=size,height=size),output_size=[512,512],resample='bicubic')
                        with self.subTest(fmt=fmt,space=space,range=color_range,x=x,y=y):
                            self.assertEqual(crop_original_frame(f,tr).tobytes(),crop_original_frame_roi(f,tr).tobytes())


if __name__=='__main__':unittest.main()
