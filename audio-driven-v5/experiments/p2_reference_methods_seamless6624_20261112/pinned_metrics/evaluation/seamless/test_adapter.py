import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .reader import SCHEMA,frame_count,load_record
from .bind import match
from ..video.paired_metrics.run import read_manifest
from ..video.paired_metrics.media import load_pair


class SeamlessTests(unittest.TestCase):
    def test_frame_grid_no_eight_frame_padding(self):
        self.assertEqual(frame_count(2.),50)
        self.assertEqual(frame_count(2.015),51)
        with self.assertRaises(ValueError):frame_count(float('nan'))

    def test_reference_integrity_and_test_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'crop.json';cfg.write_text('{}')
            source=root/'source.jsonl'
            raw=(json.dumps(dict(sample_id='sid',split='test'))+'\n').encode();source.write_bytes(raw)
            ref=dict(schema=SCHEMA,sample_id='sid',manifest=str(source),offset=0,bytes=len(raw),
                row_sha256=hashlib.sha256(raw).hexdigest(),crop_config=str(cfg),
                crop_config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),reader_contract={'code':'digest'})
            with patch('evaluation.seamless.reader.reader_contract',return_value={'code':'digest'}):
                row,_=load_record(ref);self.assertEqual(row['split'],'test')
                with self.assertRaisesRegex(ValueError,'identity'):load_record(dict(ref,sample_id='wrong'))
                source.write_bytes(raw.replace(b'test',b'dev '))
                with self.assertRaisesRegex(ValueError,'changed'):load_record(ref)

    def test_missing_unknown_wrong_timeline_rejected(self):
        refs=[dict(id='a')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'gen.mp4').write_bytes(b'fixture')
            good=dict(id='a',dataset='seamless',generated='gen.mp4',target_role='listener',timeline='same_pair_window')
            self.assertEqual(len(match(refs,[good],root)),1)
            for items in ([],[dict(good,generated=None)],[dict(good,id='b')],[good,good],
                          [dict(good,timeline='after_speaker')],[dict(good,dataset='responsenet')]):
                with self.assertRaises(ValueError):match(refs,items,root)

    def test_lazy_manifest_and_generic_pair_dispatch(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'gen.mp4'
            out=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(16,16))
            for _ in range(4):out.write(np.zeros((16,16,3),np.uint8))
            out.release()
            source=dict(id='sid',dataset='seamless',generated='gen.mp4',
                        gt_reference=dict(manifest='source.jsonl',crop_config='crop.json'))
            path=root/'input.json';path.write_text(json.dumps([source]))
            row=read_manifest(path)[0]
            self.assertEqual(row['gt_reference']['manifest'],str(root/'source.jsonl'))
            fake=np.zeros((4,16,16,3),np.uint8)
            with patch('evaluation.seamless.reader.decode_reference',return_value=(fake,25.,{'rgb_sha256':'checked'})):
                gt,gen,meta=load_pair(row)
                self.assertEqual(gt.shape,gen.shape)
                self.assertEqual(meta['gt_sha256_kind'],'decoded_full_gt_rgb_sequence')
                with self.assertRaises(ValueError):load_pair(dict(row,gt=str(video)))


if __name__=='__main__':unittest.main()
