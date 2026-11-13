"""Exercise real FFmpeg EOF conversion and bounded shared-queue scheduling."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from cache_listen_corpus import bounded_results,configure_media_export
from p3_listen import media


class ExportTests(unittest.TestCase):
    def test_eof_preserved_but_short_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                'testsrc2=size=32x32:rate=30:duration=2','-an','-c:v','libx264',
                '-threads','1',str(root/'source.mp4')],check=True)
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                'sine=frequency=440:sample_rate=48000:duration=2',str(root/'source.wav')],check=True)
            archive=root/'source.tar'
            with tarfile.open(archive,'w') as tar:
                for ext in ('mp4','wav'):tar.add(root/f'source.{ext}',arcname=f'source.{ext}')
            with tarfile.open(archive) as tar:
                member=tar.getmember('source.mp4')
                source=dict(archive=str(archive),member=member.name,offset=member.offset_data,byte_size=member.size)
            row=dict(sample_id='eof',split='train',start=1.238,duration_s=.76,frames=19,
                width=32,height=32,fps=25,listener_crop=dict(size=32,x=0,y=0),
                sources=dict(speaker=source,listener=source))
            with self.assertRaisesRegex(ValueError,'geometry mismatch'):
                media.export(root,row)
            with patch.object(media,'command',media.command):
                configure_media_export(media)
                media.export(root,row)
                _,marker=media.verify_media(root,row)
                self.assertEqual(marker['frames'],19)
                short=dict(row,sample_id='too_short',start=1.32)
                with self.assertRaisesRegex(ValueError,'geometry mismatch'):
                    media.export(root,short)

    def test_bounded_queue_visits_every_row_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            result=list(bounded_results(pool,lambda x:x,range(101),6))
        self.assertEqual(sorted(result),list(range(101)))


if __name__=='__main__':unittest.main()
