import ast,copy,inspect,unittest
from unittest.mock import patch
import gpu7_resume as resume

class GPU7ResumeChecks(unittest.TestCase):
    def test_resident_model_does_not_require_second_model_allocation(self):
        policy=dict(gpu_memory_utilization=.36,gpu_reserve_gib=16)
        total=143771
        self.assertFalse(resume.memory_available(50000,total,policy,False))
        self.assertTrue(resume.memory_available(50000,total,policy,True))
        self.assertFalse(resume.memory_available(15000,total,policy,True))
        self.assertTrue(resume.memory_available(100000,total,policy,False))

    def test_adapted_guard_reads_live_model_cache(self):
        cache={}
        adapted=resume.adapted_infer(lambda:bool(cache))
        self.assertFalse(adapted.__globals__['model_loaded']())
        cache['frozen_model']=object()
        self.assertTrue(adapted.__globals__['model_loaded']())
        self.assertIn('memory_available(free, total, policy, model_loaded())',adapted.__adapted_source__)

    def test_overlay_changes_only_memory_and_physical_placement(self):
        policy=dict(physical_gpu=7,node11_gpu_uuid='canonical',gpu_memory_utilization=.8,cpu_offload_gb=0,
                    seed=137,max_output_tokens=768,max_model_len=16384,model='frozen',temperature=0.)
        original=copy.deepcopy(policy)
        with patch.object(resume,'validate_gpu'):out=resume.runtime_policy(policy)
        self.assertEqual(policy,original)
        self.assertEqual({k for k in policy if policy[k]!=out[k]},
                         {'node11_gpu_uuid','gpu_memory_utilization','cpu_offload_gb'})
        self.assertEqual(out['node11_gpu_uuid'],resume.UUID)

    def test_native_generation_and_model_call_preserved(self):
        import p2_event_presence.inference as native
        adapted=ast.parse(resume.adapted_infer().__adapted_source__);original=ast.parse(inspect.getsource(native.infer))
        def section(tree,name):return next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name==name)
        self.assertEqual(ast.dump(section(adapted,'generate')),ast.dump(section(original,'generate')))
        def model(tree):return next(x for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='LLM')
        self.assertEqual(ast.dump(model(adapted)),ast.dump(model(original)))

    def test_detector_model_loaded_once_and_mismatched_args_rejected(self):
        calls=[]
        def factory(**kwargs):calls.append(kwargs);return object()
        load,cache=resume.persistent_factory(factory)
        first=load(model='same',dtype='bfloat16');self.assertIs(first,load(dtype='bfloat16',model='same'))
        self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):load(model='different',dtype='bfloat16')

if __name__=='__main__':unittest.main()
