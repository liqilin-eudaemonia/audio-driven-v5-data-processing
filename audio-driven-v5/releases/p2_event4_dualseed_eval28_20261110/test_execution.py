import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[2]/'src')]
import driver
import parallel_events as events

class ExecutionChecks(unittest.TestCase):
    def test_only_event_four_and_two_seeds(self):
        self.assertEqual(driver.NAMES,('baseline','event001','event020','event100'))
        self.assertEqual(driver.SEEDS,(137,238))
        self.assertEqual(len(driver.NAMES)*len(driver.SEEDS)*64,512)
        self.assertEqual(len({driver.folder(n,s)for n in driver.NAMES for s in driver.SEEDS}),8)

    def test_exact_28_allocation_preserves_node11_reserved_cards(self):
        cluster=dict(nodes=[dict(host=str(i),gpus=list(range(8 if i<3 else 4)))for i in range(4)])
        assigned=[events.assignment(cluster,r)for r in range(28)]
        self.assertEqual([g for n,g in assigned if n['host']=='3'],[0,1,2,3])
        self.assertEqual(len({(n['host'],g)for n,g in assigned}),28)
        for r in (-1,28,31):
            with self.assertRaises(ValueError):events.assignment(cluster,r)

    def test_media_dedup_assignment_has_no_cross_worker_duplicates(self):
        media=[dict(media_id=str(i),identity=str(i))for i in range(128)]
        parts=events.shards(media+media[:64]);flat=[m['media_id']for part in parts for m in part]
        self.assertEqual(len(flat),128);self.assertEqual(len(set(flat)),128)
        self.assertTrue(all(parts));self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
        with self.assertRaises(ValueError):events.shards(media+[dict(media_id='0',identity='changed')])

    def test_native_infer_only_adapts_device_binding(self):
        self.assertTrue(callable(events.stage_adapter()))
        function=events.adapt_infer()
        import ast,inspect,p2_event_presence.inference as native
        generated=next(n for n in ast.walk(ast.parse(function.__adapted_source__))if isinstance(n,ast.FunctionDef)and n.name=='generate')
        original=next(n for n in ast.walk(ast.parse(inspect.getsource(native.infer)))if isinstance(n,ast.FunctionDef)and n.name=='generate')
        self.assertEqual(ast.dump(generated),ast.dump(original))
        with patch.object(events.inspect,'getsource',return_value='def infer(work):pass'):
            with self.assertRaises(ValueError):events.adapt_infer()

    def test_per_seed_mean_and_grouped_uncertainty_reused(self):
        sys.path.insert(0,str(driver.LEGACY))
        from dual_seed_compare import compare
        from test_promotion_policy import fixtures
        import promotion_policy as policy
        membership,good=fixtures();_,bad=fixtures(False)
        with patch.dict(policy.DEFAULT,bootstrap_draws=100):
            result=compare({n:{137:copy.deepcopy(good),238:copy.deepcopy(bad)}for n in driver.NAMES},membership,copy.deepcopy(policy.DEFAULT))
        self.assertEqual(len(result['mean_metrics']),4)
        self.assertTrue(all(x['reaction']==.5 for x in result['mean_metrics'].values()))
        self.assertEqual(result['independent_source_groups'],12)
        self.assertFalse(result['automatic_promotion'])

if __name__=='__main__':unittest.main(verbosity=2)
