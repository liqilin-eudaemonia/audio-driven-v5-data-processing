from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from listen_benchmark.execution_profile import execution_for_gpu,engine_graph_kwargs
import listen_j_annotation_supervisor as supervisor


class ExecutionProfileTests(unittest.TestCase):
    def base(self):return {'batch_size':512,'tokens':32768,'frames':16384,'memory':.85,'eager':True}
    def trial(self):return {'gpu_execution_overrides':{'7':{'memory':.8,'eager':False,
        'decode_graph_capture_sizes':[1,2,4,8,16,32,64,128]}}}

    def test_default_path_unchanged(self):
        base=self.base();out=execution_for_gpu(base,{},0)
        self.assertEqual(out,base);self.assertIsNot(out,base)
        self.assertEqual(engine_graph_kwargs(out),{'enforce_eager':True})

    def test_only_selected_gpu_changes_and_no_aliases(self):
        base=self.base();config=self.trial()
        self.assertEqual(execution_for_gpu(base,config,0),base)
        selected=execution_for_gpu(base,config,7)
        self.assertEqual(selected['memory'],.8)
        for name in ['batch_size','tokens','frames']:self.assertEqual(selected[name],base[name])
        selected['decode_graph_capture_sizes'].append(256)
        self.assertEqual(config['gpu_execution_overrides']['7']['decode_graph_capture_sizes'][-1],128)
        self.assertEqual(base['memory'],.85)

    def test_decode_only_graph_and_compile_disabled(self):
        out=engine_graph_kwargs(execution_for_gpu(self.base(),self.trial(),7))
        self.assertFalse(out['enforce_eager'])
        self.assertEqual(out['compilation_config']['mode'],0)
        self.assertEqual(out['compilation_config']['cudagraph_mode'],'FULL_DECODE_ONLY')
        self.assertEqual(out['compilation_config']['max_cudagraph_capture_size'],128)

    def test_renderer_override_is_per_gpu_and_supervisor_checks_it(self):
        config={'renderer_workers':2,'gpu_execution_overrides':{'7':{'renderer_workers':4}}}
        normal=execution_for_gpu(self.base(),config,0);trial=execution_for_gpu(self.base(),config,7)
        self.assertEqual(normal['renderer_workers'],2);self.assertEqual(trial['renderer_workers'],4)
        for key in self.base():self.assertEqual(normal[key],trial[key])
        child={'pid':123};heartbeat={'pid':123,'revision':10,'completed_clips':50,'execution':trial}
        with patch.object(supervisor,'alive',return_value=True):
            self.assertTrue(supervisor.warm_ready(heartbeat,child,trial,7))
            self.assertFalse(supervisor.warm_ready(heartbeat,child,normal,7))
        for workers in [True,1,3,8,'4']:
            with self.assertRaises(ValueError):
                execution_for_gpu(self.base(),{'gpu_execution_overrides':{'7':{'renderer_workers':workers}}},7)

    def test_process_decoder_only_changes_requested_gpu_and_requires_restart(self):
        cfg={'gpu_execution_overrides':{'7':{'decode_backend':'process'}}}
        target=execution_for_gpu(self.base(),cfg,7)
        self.assertNotIn('decode_backend',execution_for_gpu(self.base(),cfg,0))
        with patch.object(supervisor,'alive',return_value=True):
            h={'pid':1,'revision':11,'completed_clips':50,'execution':target}
            self.assertTrue(supervisor.warm_ready(h,{'pid':1},target,7))
            self.assertFalse(supervisor.warm_ready(h,{'pid':1},self.base(),7))
        with self.assertRaises(ValueError):
            execution_for_gpu(self.base(),{'gpu_execution_overrides':{'7':{'decode_backend':'fork'}}},7)

    def test_invalid_or_semantic_overrides_fail(self):
        invalid=[{'tokens':1},{'fps':4},{'memory':float('nan')},{'memory':True},
            {'memory':1.0},{'eager':False},{'eager':False,'decode_graph_capture_sizes':[2,1]},
            {'eager':False,'decode_graph_capture_sizes':[1,1]},
            {'eager':False,'decode_graph_capture_sizes':[True,2]},
            {'eager':False,'decode_graph_capture_sizes':[1024]},
            {'eager':True,'decode_graph_capture_sizes':[1]}]
        for override in invalid:
            with self.subTest(override=override),self.assertRaises(ValueError):
                execution_for_gpu(self.base(),{'gpu_execution_overrides':{'7':override}},7)
        with self.assertRaises(ValueError):execution_for_gpu(self.base(),{'gpu_execution_overrides':{'8':{}}},7)

    def test_supervisor_uses_same_target_and_detects_graph_mismatch(self):
        target=execution_for_gpu(self.base(),self.trial(),7)
        child={'pid':123,'start_ticks':'test'}
        heartbeat={'pid':123,'revision':9,'completed_clips':50,'execution':target}
        with patch.object(supervisor,'alive',return_value=True):
            self.assertTrue(supervisor.warm_ready(heartbeat,child,target,8))
            self.assertFalse(supervisor.warm_ready(heartbeat,child,self.base(),8))
            bad=dict(heartbeat,execution=dict(target,eager=True))
            self.assertFalse(supervisor.warm_ready(bad,child,target,8))


if __name__=='__main__':unittest.main()
