"""No GPU required: fixed32 shard coverage, live claims and pinned weight identity."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p2_current_dual_eval_20261108 as D
from p2_testsets_eval.protocol import read,save,partition
from p2_testsets_eval.shared_items import claim_items
from p2_testsets_eval.runtime import complete


class Dual32Tests(unittest.TestCase):
    def test_fixed400_repartition_without_sample_changes(self):
        rows=read(D.FEATURE/'shared_items.json');parts=partition(rows,32)
        self.assertEqual(len(parts),32)
        flattened=[r['key'] for chunk in parts for r in chunk]
        self.assertEqual(len(flattened),len(set(flattened)))
        self.assertEqual(set(flattened),{r['key'] for r in rows})
        for chunk in parts:self.assertEqual({r['dataset'] for r in chunk},{'seamless','responsenet'})

    def test_active_legacy_claim_is_not_repeated_by_helpers(self):
        with tempfile.TemporaryDirectory() as temp:
            work=Path(temp);rows=[dict(key='seamless/a',geometry=dict(native_frames=20)),
                                  dict(key='seamless/b',geometry=dict(native_frames=10))]
            src=work/'items.json';save(src,rows)
            p=dict(shared_item_queue=True,shared_items_manifest=str(src),
                   shared_items_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),total_samples=2)
            arm=dict(name='arm')
            legacy=claim_items(work,[],p,arm,'render');first=next(legacy)
            helper=claim_items(work,[],p,arm,'render');second=next(helper)
            self.assertNotEqual(first['key'],second['key'])
            for row in [first,second]:
                out=work/'latents/arm'/row['key'];out.mkdir(parents=True,exist_ok=True)
                f=out/'payload.pt';f.write_bytes(b'preserved');complete(out,[f])
            legacy.close();helper.close()
            self.assertEqual(list(claim_items(work,[],p,arm,'render')),[])

    def test_feature_export_is_pinned_and_source_identity_is_preserved(self):
        root=ROOT/'experiments/p2_current_feature_whisper6000_eval32_20261108'
        value=read(root/'exports/feature/RECEIPT.json')
        self.assertTrue(value['all_model_tensors_bitwise_equal'])
        self.assertEqual(D.stamp(value['model_checkpoint']),value['model_stamp'])
        self.assertEqual(value['original_sha256'],read(D.INFO['feature']['train']/'step_006000.complete.json')['sha256'])

    def test_pinned_export_change_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'weights.pt';path.write_bytes(b'old')
            save(root/'exports/feature/RECEIPT.json',dict(model_checkpoint=str(path),model_stamp=D.stamp(path)))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'Inference export changed'):
                D.pinned(root,'feature')


if __name__=='__main__':unittest.main()
