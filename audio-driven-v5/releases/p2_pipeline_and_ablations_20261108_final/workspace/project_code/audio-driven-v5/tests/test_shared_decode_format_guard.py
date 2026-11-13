"""Production media guard: unsupported RGB conversion must fall back exactly."""
import sys,tempfile,unittest
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import prepare


class FormatGuardTests(unittest.TestCase):
    def test_shared_native_is_only_written_for_verified_8bit_format(self):
        rgb=np.random.default_rng(8).integers(0,256,(480,640,3),dtype=np.uint8)
        transform=dict(source_size=[640,480],crop=dict(x=17,y=23,width=333,height=333),
                       output_size=[512,512],resample='bicubic')
        row=dict(normalized_audio=dict(sample_rate=48000),source_start_s=0.,source_end_s=.12,
                 video_transform=transform,source={})
        for formats,declared_duration in [(('yuv420p',)*4,.12),(('yuv420p10le',)*4,.12),
                        (('yuv420p','yuv420p10le','yuv420p10le','yuv420p10le'),.12),
                        (('yuv420p',)*4,.120001)]:
            row['duration_s']=declared_duration
            frames=[]
            for i,fmt in enumerate(formats):
                f=av.VideoFrame.from_ndarray(rgb,format='rgb24').reformat(format=fmt)
                f.pts=i;f.time_base=Fraction(1,25);frames.append(f)
            stream=SimpleNamespace(width=640,height=480,average_rate=25,
                                   time_base=Fraction(1,25),codec_context=SimpleNamespace())
            container=SimpleNamespace(streams=SimpleNamespace(video=[stream]),
                                      seek=lambda *a,**k:None,decode=lambda *_:iter(frames))
            @contextmanager
            def opened(_):yield container,'test_original'
            with self.subTest(formats=formats),tempfile.TemporaryDirectory() as d,patch('listen_benchmark.full_raw.open_source',opened):
                original=prepare(row,'visual')
                optimized=prepare(row,'visual',fast_crop=True,native_dump=d)
                self.assertEqual(original['meta'],optimized['meta'])
                self.assertEqual([im.tobytes() for im in original['images']],
                                 [im.tobytes() for im in optimized['images']])
                self.assertEqual((Path(d)/'manifest.json').exists(),len(set(formats))==1 and formats[0]=='yuv420p' and declared_duration==.12)


if __name__=='__main__':unittest.main()
