"""Frame selection and grouping tests use real encoded videos, not mock outputs."""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from glare_shared_decode import decode_group, group_rows
from p2_j_cache.data import decode_frames


class SharedDecode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.video = self.directory / 'source.mp4'
        with av.open(str(self.video), 'w') as out:
            stream = out.add_stream('libx264rgb', rate=30)
            stream.width, stream.height = 640, 640
            stream.pix_fmt = 'rgb24'
            stream.options = {'crf':'0', 'preset':'fast', 'threads':'1'}
            for i in range(180):
                x = np.zeros((640, 640, 3), np.uint8)
                x[:, :, 0] = i
                x[:, :, 1] = np.arange(640, dtype=np.uint16)[None, :] % 256
                f = av.VideoFrame.from_ndarray(x, format='rgb24')
                for packet in stream.encode(f):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, sid, start, duration=1.1, x=20):
        source = dict(source=dict(archive=str(self.video), member='source.mp4', offset=0,
                      byte_size=self.video.stat().st_size), source_start_s=start,
                      source_end_s=start+duration, duration_s=duration,
                      normalized_audio={'sample_rate':48000},
                      video_transform=dict(source_size=[640,640],output_size=[512,512],
                        resample='bicubic',crop=dict(x=x,y=30,width=570,height=570)))
        return dict(sample_id=sid, duration_s=duration, source=source)

    def test_adjacent_pairs_share_container_and_preserve_exact_nearest_grid(self):
        rows = [self.row('a', .717), self.row('b', 1.851, x=40)]
        result = decode_group(rows, self.directory/'shared')
        self.assertEqual(result['original_opens'], 1)
        for row, item in zip(rows, result['items']):
            expected, pts = decode_frames(row['source'])
            actual = np.load(Path(item['folder'])/'lia_rgb.npy')
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(item['pts'], pts)

    def test_native_detector_lane_excludes_prestart_and_postend(self):
        row = self.row('a', .717)
        result = decode_group([row], self.directory/'shared')
        import json
        meta = json.loads((Path(result['items'][0]['folder'])/'native/manifest.json').read_text())
        self.assertGreaterEqual(min(meta['times']),0.)
        self.assertLess(max(meta['times']),row['duration_s'])
        self.assertTrue(all(b>a for a,b in zip(meta['times'],meta['times'][1:])))

    def test_distant_pairs_are_not_merged(self):
        rows=[self.row('a',0.),self.row('b',3.)]
        self.assertEqual(len(group_rows(rows)),2)

    def test_group_size_and_total_duration_are_bounded(self):
        rows=[self.row(str(i),i*.5,duration=8.) for i in range(5)]
        groups=group_rows(rows)
        self.assertTrue(all(len(g)<=2 and sum(r['duration_s'] for r in g)<=16 for g in groups))


if __name__=='__main__':
    unittest.main()
