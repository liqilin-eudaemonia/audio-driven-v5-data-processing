import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT.parent/'OmniResponse')]
import p2_all_lora5000_eval_20261109 as E
from p2_testsets_eval.protocol import read,save
spec=importlib.util.spec_from_file_location('entry',E.CLI)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class EvalTests(unittest.TestCase):
    def test_native_decoders_are_isolated_from_metric_vendor_packages(self):
        for stage in ['render','decode']:
            self.assertEqual(entry.worker_environment(stage,0)['PYTHONPATH'],'')
            self.assertEqual(entry.worker_environment(stage,0)['CUDA_VISIBLE_DEVICES'],'4')
        self.assertEqual(entry.worker_environment('metric',3)['PYTHONPATH'],
                         '/private/liqi_workspace/.cache/paired_video_metrics/python')
        self.assertEqual(entry.worker_environment('metric',3)['CUDA_VISIBLE_DEVICES'],'7')

    def test_same200_and_only_expanded_lora_checkpoint(self):
        p=read(E.WORK/'protocol.json');parent=read(E.PARENT/'protocol.json')
        self.assertEqual(p['seed'],137);self.assertEqual(list(p['datasets']),['seamless'])
        self.assertEqual(p['generation'],parent['generation'])
        self.assertEqual(p['planner_adapter_identity'],parent['planner_adapter_identity'])
        self.assertEqual(p['arms'][0]['original_sha256'],read(E.TRAIN/'step_005000.complete.json')['sha256'])
        self.assertEqual(p['checkpoint_identity']['ltx_lora_report']['projections'],1632)
        self.assertEqual(p['checkpoint_identity']['experiment']['loss_weights']['event'],.05)
        source={x['key'] for x in read(E.PARENT/'shared_items.json') if x['dataset']=='seamless'}
        from p2_ltx_all_lora.config import source_identity
        self.assertEqual(source_identity(),p['checkpoint_identity']['source'])
        target=read(E.WORK/'shared_items.json');self.assertEqual({x['key'] for x in target},source)
        parts=[x for rank in range(4) for x in read(E.WORK/'shards'/f'{rank:02d}.json')]
        self.assertEqual(len(parts),200);self.assertEqual({x['key'] for x in parts},source)

    def test_all4_physical_gpus_are_eligible(self):
        c=read(E.WORK/'cluster.json')
        self.assertEqual([n['name'] for n in c['nodes']],['node11'])
        self.assertEqual([n['gpus'] for n in c['nodes']],[[4,5,6,7]])
        for physical in range(4,32):
            with self.assertRaises(AssertionError):entry.worker_environment('render',physical)

    def test_changed_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);p=work/'model.pt';p.write_bytes(b'original')
            save(work/'export/RECEIPT.json',dict(checkpoint=str(p),stamp=E.stamp(p)))
            p.write_bytes(b'changed')
            with self.assertRaises(AssertionError):E.pinned(work)

    def test_missing_last_sample_cannot_pass_full_bind(self):
        b=entry.Binder.__new__(entry.Binder);b.committed=set(range(199));b.p=dict(total_samples=200)
        with self.assertRaises(AssertionError):b.canonical_audit()


class SamplerTests(unittest.TestCase):
    def test_native_render_math_unchanged(self):
        import ast
        native=(ROOT/'scripts/p2_ltx6000_latest_sft_dual200_20261107.py').read_text()
        tree=ast.parse(native)
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='render')
        class Step(ast.NodeTransformer):
            def visit_Constant(self,n):
                return ast.copy_location(ast.Constant(value=5000),n) if n.value==6000 else n
        expected=ast.dump(Step().visit(fn),include_attributes=False)
        tree=ast.parse(Path(E.__file__).read_text())
        actual=ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='render'),include_attributes=False)
        self.assertEqual(expected,actual)

if __name__=='__main__':unittest.main()
