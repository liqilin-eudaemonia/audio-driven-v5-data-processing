import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_listen_v8_cluster as c
class ClusterTests(unittest.TestCase):
    def test_budget_rejects_128_per_node(self):
        with self.assertRaises(ValueError):c.dimensions(2,8)
        with self.assertRaises(ValueError):c.dimensions(4,4)
        self.assertEqual(c.dimensions(4,2)['preparation_slots'],64)
        with self.assertRaises(ValueError):c.dimensions(8,1)
        with self.assertRaises(ValueError):c.dimensions(8,2)
    def test_unique_global_ranks(self):
        ranks=[c.worker_rank(n['index'],i) for n in c.configuration()['nodes'] for i in range(64)]
        self.assertEqual(len(ranks),len(set(ranks)))
    def test_claim_skip_completed_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'SCHED',Path(tmp)),patch.object(c,'OUT',Path(tmp)):
            p=Path(tmp);(p/'claims').mkdir();(p/'results').mkdir()
            c.write(p/'queue_state.json',dict(cursor=0,total=3));c.write(p/'control.json',dict(stop=False));c.write(p/'results/a.json',dict(complete=True))
            self.assertEqual(c.claim(['a','b','c'],32),'b')
            c.write(p/'control.json',dict(stop=True));self.assertIsNone(c.claim(['a','b','c'],33))
            self.assertEqual(c.read(p/'queue_state.json')['cursor'],2)
            c.write(p/'control.json',dict(stop=False));self.assertEqual(c.claim(['a','b','c'],33),'c');self.assertTrue(c.exhausted())
    def test_memory_pressure_checks_available_not_free(self):
        node=dict(cpu_busy_percent=20,memory={'MemAvailable':200,'MemTotal':1000},gpus=[dict(total_mib=143771,memory_mib=40000)])
        self.assertEqual(c.pressure(node),[])
        node['memory']['MemAvailable']=99;self.assertIn('ram',c.pressure(node))
        node['cpu_busy_percent']=90;self.assertIn('cpu',c.pressure(node))
        node['gpus'][0]['memory_mib']=140000;self.assertIn('gpu_memory',c.pressure(node))
    def test_all_benchmark_stages_fit_budget(self):
        for models,prep in c.configuration()['stages']:self.assertLessEqual(c.dimensions(models,prep)['preparation_slots'],64)
    def test_fixed_setting_skips_sweep(self):
        cfg=c.configuration()
        self.assertEqual(cfg['fixed_setting'],[4,2])
        self.assertEqual(c.initial_stage_index(cfg),len(cfg['stages']))
    def test_metadata_read_retries_transient_failures(self):
        with patch.object(c,'raw_read',side_effect=[FileNotFoundError(),json.JSONDecodeError('partial','',0),{'cursor':12}]),patch.object(c.time,'sleep') as sleep:
            self.assertEqual(c.read(Path('state.json')),{'cursor':12})
            self.assertEqual(sleep.call_count,2)
    def test_metadata_read_surfaces_persistent_failure(self):
        with patch.object(c,'raw_read',side_effect=FileNotFoundError()) as read,patch.object(c.time,'sleep'):
            with self.assertRaises(FileNotFoundError):c.read(Path('state.json'))
            self.assertEqual(read.call_count,9)
    def test_prep_increase_does_not_drain_or_wait(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'SCHED',Path(tmp)),patch.object(c.time,'sleep') as sleep:
            p=Path(tmp);c.write(p/'control.json',dict(stop=False,nodes={n['name']:c.dimensions(1,1) for n in c.configuration()['nodes']}))
            c.set_stage(1,4,'benchmark_1x4')
            control=c.read(p/'control.json');self.assertFalse(control['drain']);self.assertTrue(all(d['preparation_slots']==32 for d in control['nodes'].values()));sleep.assert_not_called()
    def test_more_models_wait_for_reduced_prefetch_without_idling(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'SCHED',Path(tmp)):
            p=Path(tmp);(p/'progress').mkdir();(p/'nodes').mkdir()
            c.write(p/'control.json',dict(stop=False,nodes={n['name']:c.dimensions(1,8) for n in c.configuration()['nodes']}))
            c.write(p/'progress/test.json',dict(node='node17',local_rank=0,pending_preparation=8))
            def finish_one(_):
                control=c.read(p/'control.json');self.assertFalse(control['drain']);self.assertEqual(control['nodes']['node17']['model_workers'],8);self.assertEqual(control['nodes']['node17']['prep_per_worker'],4)
                c.write(p/'progress/test.json',dict(node='node17',local_rank=0,pending_preparation=4))
            with patch.object(c.time,'sleep',side_effect=finish_one):c.set_stage(2,4,'benchmark_2x4')
            self.assertEqual(c.read(p/'control.json')['nodes']['node17']['preparation_slots'],64)
if __name__=='__main__':unittest.main()
