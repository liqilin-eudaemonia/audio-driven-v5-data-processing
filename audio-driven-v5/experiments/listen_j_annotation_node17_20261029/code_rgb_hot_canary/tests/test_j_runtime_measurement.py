import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH=Path(__file__).resolve().parents[1]/'experiments/listen_j_annotation_node17_20261029/measure_runtime_window.py'
spec=importlib.util.spec_from_file_location('runtime_measurement',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class RuntimeMeasurementTests(unittest.TestCase):
    def snapshots(self):
        workers={f'gpu-{i}':{'pid':100+i,'start_ticks':str(i),'live':True,
            'heartbeat_matches':True,'completed':100,'failed':0,'revision':7,
            'decode_workers':8,'decode_slots':80,'execution':{'tokens':32768}}
            for i in range(8)}
        workers['nod-0']={'pid':200,'start_ticks':'200','live':True,
            'heartbeat_matches':True,'completed':100,'execution':{}}
        first={'at':1.,'workers':workers,'gpus':[{'utilization':80}],
            'expected_nod_workers':1,'guards_live':True,
            'tuning':{'decode_workers':8,'decode_slots':80},'nod_decode_slots':80,
            'rolling':False,'warming':[],'cpu_usage_usec':0,'oom_kill':0}
        last=copy.deepcopy(first);last['at']=101.;last['cpu_usage_usec']=8_000_000_000
        for w in last['workers'].values():w['completed']+=50
        return [first,last]

    def test_fixed_live_processes_use_worker_counts_not_source_receipt_bursts(self):
        out=module.summarize(self.snapshots(),7)
        self.assertTrue(out['valid_for_throughput_comparison'])
        self.assertEqual(out['clips_per_s'],{'vlm':4.,'nod':.5})
        self.assertEqual(out['used_cpu_cores'],80.)

    def test_restart_invalidates_even_if_count_exceeds_old_worker_count(self):
        data=self.snapshots();data[1]['workers']['gpu-0'].update(pid=999,start_ticks='new',completed=1000)
        out=module.summarize(data,7)
        self.assertFalse(out['valid_for_throughput_comparison']);self.assertIsNone(out['clips_per_s'])

    def test_transient_death_inside_window_cannot_be_hidden_by_healthy_end(self):
        first,last=self.snapshots();middle=copy.deepcopy(first);middle['at']=50.
        middle['workers']['gpu-0']['live']=False
        self.assertFalse(module.summarize([first,middle,last],7)['valid_for_throughput_comparison'])

    def test_mixed_revision_requires_explicit_exact_canary_override(self):
        rows=self.snapshots()
        for row in rows:row['workers']['gpu-7']['revision']=8
        self.assertIsNone(module.summarize(rows,7)['clips_per_s'])
        out=module.summarize(rows,7,{'gpu-7':8})
        self.assertTrue(out['valid_for_throughput_comparison'])
        self.assertEqual(out['per_gpu_clips_per_s']['gpu-7'],.5)
        rows[-1]['workers']['gpu-6']['revision']=8
        self.assertIsNone(module.summarize(rows,7,{'gpu-7':8})['clips_per_s'])

    def test_rolling_path_change_invalidates_even_if_revision_is_unchanged(self):
        rows=self.snapshots();rows[-1]['workers']['gpu-7']['rolling_inference']=True
        self.assertIsNone(module.summarize(rows,7)['clips_per_s'])

    def test_rgb_runtime_selection_change_invalidates_the_window(self):
        rows=self.snapshots();rows[-1]['rgb_reuse_gpus']=[6]
        out=module.summarize(rows,7)
        self.assertIn('RGB converter selection changed',out['invalid_reasons'])
        rows[0]['rgb_reuse_gpus']=[6]
        out=module.summarize(rows,7)
        self.assertTrue(out['valid_for_throughput_comparison'])
        self.assertEqual(out['rgb_reuse_gpus'],[6])

    def test_partial_rollout_unadopted_tuning_oom_or_execution_change_rejected(self):
        changes=[lambda d:d.update(warming=['gpu-1']),
                 lambda d:d['workers']['gpu-0'].update(revision=5),
                 lambda d:d['workers']['gpu-0'].update(decode_workers=4),
                 lambda d:d.update(oom_kill=1),
                 lambda d:d['workers']['gpu-0'].update(execution={'tokens':65536}),
                 lambda d:d.update(nod_decode_slots=64)]
        for change in changes:
            with self.subTest(change=change):
                rows=self.snapshots();change(rows[1]);self.assertIsNone(module.summarize(rows,7)['clips_per_s'])

    def test_workload_uses_initial_claims_journal_range_and_exact_completion_window(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder);lines=[json.dumps({'job_id':j}).encode()+b'\n' for j in ['a','b','b','c']]
            (work/'ready.jsonl').write_bytes(b''.join(lines))
            source={'video_transform':{'source_size':[1920,1080]}}
            for job,times in [('a',[0,1]),('b',[50,101,105]),('c',[30])]:
                rows=[]
                for index,at in enumerate(times):
                    sid=f'{job}-{index}';rows.append({'sample_id':sid,'source':source})
                    module.write(module.result_path(work,'vlm',sid),{'sample_id':sid,'source':source,
                        'completed_at':at,'duration_s':2,'execution':{'rolling_inference':job=='b'},'records':[{'status':'valid',
                        'finish_reason':'stop','prompt_tokens':100,'completion_tokens':20}]})
                module.write(work/'jobs'/f'{job}.json',{'rows':rows})
            first={'at':1.,'vlm_queue_range':{'offset':len(lines[0]),'inflight':['a']}}
            last={'at':101.,'vlm_queue_range':{'offset':sum(map(len,lines[:3])),'inflight':[]}}
            out=module.workload_summary(work,first,last)
            self.assertEqual(out['source_jobs'],2);self.assertEqual(out['candidate_clips'],5)
            self.assertEqual(out['clips'],2);self.assertEqual(out['video_seconds'],4)
            self.assertEqual(out['requests'],2);self.assertEqual(out['completion_tokens'],40)
            self.assertEqual(out['by_execution_variant']['rolling_groups']['clips'],1)
            self.assertEqual(out['by_execution_variant']['bounded_batches']['clips'],1)

    def test_nod_workload_preserves_missing_evidence_and_bounds_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            lines=[json.dumps({'job_id':j}).encode()+b'\n' for j in ['a','b','b','not_read']]
            (work/'ready.jsonl').write_bytes(b''.join(lines))
            source={'video_transform':{'source_size':[3840,2160]}}
            for job,at,status in [('a',1,'ok'),('b',50,'unavailable')]:
                module.write(work/'jobs'/f'{job}.json',{'rows':[{'sample_id':job,'source':source}]})
                module.write(module.result_path(work,'nod',job),{
                    'sample_id':job,'source':source,'completed_at':at,'duration_s':5,
                    'status':status,'performance':{'render_s':4,'face_s':2,'decode_lock_wait_s':0}})
            first={'at':1.,'nod_queue_range':{'offset':len(lines[0]),'inflight':['a']}}
            last={'at':101.,'nod_queue_range':{'offset':sum(map(len,lines[:3])),'inflight':[]}}
            out=module.nod_workload_summary(work,first,last)
            self.assertEqual(out['clips'],2)
            self.assertEqual(out['statuses'],{'ok':1,'unavailable':1})
            self.assertEqual(out['video_seconds'],10)
            self.assertEqual(out['source_sizes'],{'3840x2160':2})
            self.assertEqual(out['performance']['render_s']['median'],4)
            # End is exclusive; the omitted journal entry has no job file.
            last['at']=50
            self.assertEqual(module.nod_workload_summary(work,first,last)['clips'],1)
            last['at']=101
            path=module.result_path(work,'nod','b');bad=module.read(path)
            bad['source']={'different':'identity'};module.write(path,bad)
            with self.assertRaisesRegex(ValueError,'provenance'):
                module.nod_workload_summary(work,first,last)

    def test_old_window_does_not_invent_nod_input_workload(self):
        out=module.nod_workload_summary(Path('/not-read'),{'at':1},{'at':2})
        self.assertFalse(out['available'])


if __name__=='__main__':unittest.main()
