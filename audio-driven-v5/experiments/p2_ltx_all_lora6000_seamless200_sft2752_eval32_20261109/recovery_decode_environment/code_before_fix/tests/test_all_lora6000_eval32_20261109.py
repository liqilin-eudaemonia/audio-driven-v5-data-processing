import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import p2_all_lora6000_eval_20261109 as E
from p2_testsets_eval.protocol import read,save
spec=importlib.util.spec_from_file_location('entry',E.CLI)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class EvalTests(unittest.TestCase):
    def test_same200_and_only_expanded_lora_checkpoint(self):
        p=read(E.WORK/'protocol.json');parent=read(E.PARENT/'protocol.json')
        self.assertEqual(p['seed'],137);self.assertEqual(list(p['datasets']),['seamless'])
        self.assertEqual(p['generation'],parent['generation'])
        self.assertEqual(p['planner_adapter_identity'],parent['planner_adapter_identity'])
        self.assertEqual(p['arms'][0]['original_sha256'],read(E.TRAIN/'step_006000.complete.json')['sha256'])
        self.assertEqual(p['checkpoint_identity']['ltx_lora_report']['projections'],1632)
        self.assertEqual(p['checkpoint_identity']['experiment']['loss_weights']['event'],.05)
        source={x['key'] for x in read(E.PARENT/'shared_items.json') if x['dataset']=='seamless'}
        target=read(E.WORK/'shared_items.json');self.assertEqual({x['key'] for x in target},source)
        parts=[x for rank in range(32) for x in read(E.WORK/'shards'/f'{rank:02d}.json')]
        self.assertEqual(len(parts),200);self.assertEqual({x['key'] for x in parts},source)

    def test_all32_physical_gpus_are_eligible(self):
        c=read(E.WORK/'cluster.json')
        self.assertEqual([n['name'] for n in c['nodes']],['node12','node8','node4','node11'])
        self.assertTrue(all(n['gpus']==list(range(8)) for n in c['nodes']))

    def test_changed_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);p=work/'model.pt';p.write_bytes(b'original')
            save(work/'export/RECEIPT.json',dict(checkpoint=str(p),stamp=E.stamp(p)))
            p.write_bytes(b'changed')
            with self.assertRaises(AssertionError):E.pinned(work)

    def test_missing_last_sample_cannot_pass_full_bind(self):
        b=entry.Binder.__new__(entry.Binder);b.committed=set(range(199));b.p=dict(total_samples=200)
        with self.assertRaises(AssertionError):b.canonical_audit()


if __name__=='__main__':unittest.main()
