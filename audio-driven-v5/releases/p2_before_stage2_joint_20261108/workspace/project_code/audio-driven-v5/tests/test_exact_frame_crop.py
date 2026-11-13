import io
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import av
import numpy as np
from listen_benchmark.frame_crop import crop_original_frame
from listen_labeling.media import crop_resize


class CropTests(unittest.TestCase):
    def test_rgb_yuv_formats_edges_and_odd_crop_coordinates_match(self):
        rng=np.random.default_rng(42)
        for width,height in [(640,480),(2160,3840)]:
            rgb=rng.integers(0,256,(height,width,3),dtype=np.uint8)
            base=av.VideoFrame.from_ndarray(rgb,format='rgb24')
            for fmt in ['rgb24','yuv420p','yuv444p','yuv420p10le']:
                frame=base.reformat(format=fmt)
                for x,y,size in [(0,0,320),(17,23,333),(width-384,height-384,384)]:
                    t={'source_size':[width,height],'crop':{'x':x,'y':y,'width':size,'height':size},
                       'output_size':[512,512],'resample':'bicubic'}
                    with self.subTest(dimensions=(width,height),format=fmt,crop=(x,y,size)):
                        old=crop_resize(frame.to_image(),t);new=crop_original_frame(frame,t)
                        self.assertEqual(old.tobytes(),new.tobytes())
                        encoded=[]
                        for image in [old,new]:
                            out=io.BytesIO();image.save(out,format='JPEG',quality=85);encoded.append(out.getvalue())
                        self.assertEqual(*encoded)

    def test_invalid_crop_and_wrong_source_dimensions_still_fail(self):
        frame=av.VideoFrame.from_ndarray(np.zeros((480,640,3),dtype=np.uint8),format='rgb24')
        t={'source_size':[640,480],'crop':{'x':-1,'y':0,'width':384,'height':384},
           'output_size':[512,512],'resample':'bicubic'}
        with self.assertRaises(ValueError):crop_original_frame(frame,t)
        t['crop']['x']=0;t['source_size']=[1280,720]
        with self.assertRaisesRegex(ValueError,'source size mismatch'):crop_original_frame(frame,t)


if __name__=='__main__':unittest.main()
