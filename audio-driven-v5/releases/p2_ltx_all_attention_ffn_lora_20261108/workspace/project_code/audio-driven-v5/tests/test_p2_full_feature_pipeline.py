import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_fusion import atomic,file_sha,read
from listen_benchmark.feature_pipeline import publish_shard,source_row,select_jobs,make_label,receipt_sample_clock


class FeaturePipelineTests(unittest.TestCase):
    def test_nominal_clock_requires_exact_source_and_sample_endpoints(self):
        from copy import deepcopy
        s=dict(sample_id='s',duration_s=4.7937,source_start_s=293.124,source_end_s=297.9177,normalized_audio=dict(sample_rate=48000))
        a,z=round(s['source_start_s']*48000),round(s['source_end_s']*48000)
        c=dict(sample_id='s',duration_s=round((z-a)/48000,9),source=s)
        media=dict(meta=dict(audio_sources=dict(listener=dict(sample_rate=48000,start_sample=a,end_sample=z))))
        r=dict(source=deepcopy(s));before=deepcopy(r)
        adapted,audit=receipt_sample_clock(r,c,media)
        self.assertEqual(r,before);self.assertEqual(adapted['source']['duration_s'],c['duration_s']);self.assertFalse(audit['raw_receipt_modified'])
        bad=deepcopy(media);bad['meta']['audio_sources']['listener']['end_sample']+=1
        with self.assertRaisesRegex(ValueError,'clock identity'):receipt_sample_clock(r,c,bad)
        bad=deepcopy(r);bad['source']['sample_id']='other'
        with self.assertRaisesRegex(ValueError,'clock identity'):receipt_sample_clock(bad,c,media)
    def test_periodic_report_waits_for_all_raw_branches(self):
        spec=importlib.util.spec_from_file_location('feature_monitor_test',ROOT/'scripts/manage_p2_full_features.py');manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)/'features';raw=Path(t)/'raw';cfg=dict(raw_run=str(raw),monitor_start_after_raw=True)
            atomic(run/'config.json',cfg);branches={b:dict(total_shards=1024,shards_done=1024) for b in ['visual','audio','asr']};branches['visual']['shards_done']=1023;atomic(raw/'status.json',dict(branches=branches))
            self.assertFalse(manager.raw_finished(cfg));self.assertEqual(manager.monitor_once(run)['status'],'waiting_for_raw_completion');self.assertFalse((run/'monitor/latest.json').exists())
            branches['visual']['shards_done']=1024;atomic(raw/'status.json',dict(branches=branches));self.assertTrue(manager.raw_finished(cfg))
    def test_publish_requires_finished_fusion_and_valid_hash(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);run=p/'features';fusion=p/'fusion';raw=p/'raw'
            atomic(run/'config.json',dict(fusion_run=str(fusion),raw_run=str(raw),job_samples=1))
            self.assertIsNone(publish_shard(run,'0000'))
            for name,value in [('labels',dict(sample_id='s')),('evidence',dict(sample_id='s'))]:
                path=fusion/f'{name}/0000.jsonl';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
            files={f'{name}/0000.jsonl':dict(sha256=file_sha(fusion/f'{name}/0000.jsonl')) for name in ['labels','evidence']}
            atomic(fusion/'state/fusion/0000.json',dict(done=True,files=files,counts=dict(samples=1)))
            for b in ['visual','audio']:
                path=raw/f'raw/{b}/results/0000.jsonl';path.parent.mkdir(parents=True);path.write_text('')
            result=publish_shard(run,'0000');self.assertEqual(result['jobs'],['0000-0000']);self.assertEqual(publish_shard(run,'0000'),result)
            (run/'state/published/0000.json').unlink();(fusion/'labels/0000.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):publish_shard(run,'0000')

    def test_source_line_hash_rejects_wrong_clip(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);p=run/'source.jsonl';full=dict(sample_id='s',split='train',sources=dict(listener={'member':'a'}),normalized_audio=dict(listener=dict(path='a.wav')),listener_id='L',speaker_id='S');line=(json.dumps(full)+'\n').encode();p.write_bytes(line)
            atomic(run/'source_index/COMPLETE.json',dict(files={str(p):dict(file_id=0,identity=dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))}))
            with sqlite3.connect(run/'source_index/offsets.sqlite') as db:db.execute('CREATE TABLE offsets(file_id,line,offset)');db.execute('INSERT INTO offsets VALUES(0,1,0)')
            source=dict(source_manifest=str(p),source_line_1based=1,source_line_sha256=hashlib.sha256(line).hexdigest(),sample_id='s',split='train',source={'member':'a'},normalized_audio=dict(path='a.wav'))
            self.assertEqual(source_row(run,source),full)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):source_row(run,dict(source,source_line_sha256='bad'))

    def test_feature_queue_never_uses_unprepared_jobs(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);atomic(run/'jobs/not-ready.json',{});atomic(run/'state/media/ready.json',dict(done=True));atomic(run/'state/media/done.json',dict(done=True));atomic(run/'state/features/done.json',dict(done=True))
            self.assertEqual(select_jobs(run,'features'),['ready'])

    def test_missing_full_receipts_produce_no_known_background(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);cfg=read(ROOT/'configs/p2_listen_weak_background_v1.json');atomic(run/'p2_config.json',cfg)
            candidate=dict(schema='listen-existing-result-fusion-verbatim-v2',sample_id='s',duration_s=2.,events=[],training_admitted=False,raw_completion_status='partial_raw')
            target=dict(media_hashes=dict(target_video='a'*64,target_audio='b'*64))
            record=dict(candidate=candidate,evidence=dict(original_events=[],semantic_issues=[]),receipts=dict(visual=None,audio=None))
            with patch('p2_listen.weak_labels.full_receipt',return_value=(None,'missing')):
                label=make_label(run,record,dict(meta={},target=target))
            self.assertFalse(label['training_admitted']);self.assertTrue(all(not spans for spans in label['coverage'].values()));self.assertTrue(all(spans==[[0.,2.]] for spans in label['unknown'].values()))

    def test_resident_parity_checks_values_and_dtype(self):
        import torch
        from listen_benchmark.resident_features import compare_trees
        x={'z':torch.ones(1,2,dtype=torch.bfloat16),'metadata':{'frames':57}};compare_trees(x,x)
        with self.assertRaisesRegex(AssertionError,'tensor differs'):compare_trees(x,dict(x,z=torch.zeros(1,2,dtype=torch.bfloat16)))
        with self.assertRaisesRegex(AssertionError,'tensor differs'):compare_trees(x,dict(x,z=torch.ones(1,2,dtype=torch.float32)))

    def test_node_handoff_waits_for_fusion_and_free_memory(self):
        spec=importlib.util.spec_from_file_location('feature_manager_test',ROOT/'scripts/manage_p2_full_features.py');manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)
        for upstream,busy,expected in [(False,False,0),(True,True,0),(True,False,1)]:
            with self.subTest(upstream=upstream,busy=busy),tempfile.TemporaryDirectory() as t:
                run=Path(t)/'features';fusion=Path(t)/'fusion';calls=[]
                cfg=dict(fusion_run=str(fusion),nodes={'node12':dict(jobs=[dict(gpu=0,rank=0)])},worker_stall_seconds=1200,media_workers_initial=0,media_workers_max=0,media_ready_backlog=512,feature_workers_initial=2,feature_workers_max=4,gpu_headroom_mib=16000,poll_seconds=10)
                atomic(run/'config.json',cfg);atomic(run/'queues.json',dict(media=[],features=['job']));atomic(run/'CANARY.json',dict(passed=True,peak_reserved_mib=30000))
                if upstream:atomic(fusion/'COMPLETE.json',dict(done=True))
                def fake_launch(run,cfg,role,worker,**kw):
                    calls.append(role);return dict(pid=99999999,role=role,worker=worker,gpu=kw.get('gpu'),launched_at=1)
                tele=dict(gpus={0:dict(memory_total_mib=140000,memory_used_mib=1000 if busy else 0)},cpu_ticks=[],utc=1)
                with patch.object(manager,'telemetry',return_value=tele),patch.object(manager,'launch',side_effect=fake_launch),patch.object(manager.time,'sleep',side_effect=lambda _: (run/'STOP').touch()):manager.node_loop(run,'node12')
                self.assertEqual(len(calls),expected)

    def test_partial_node_never_launches_on_unassigned_cards(self):
        spec=importlib.util.spec_from_file_location('feature_partial_node_test',ROOT/'scripts/manage_p2_full_features.py');manager=importlib.util.module_from_spec(spec);spec.loader.exec_module(manager)
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)/'features';fusion=Path(t)/'fusion';calls=[]
            cfg=dict(fusion_run=str(fusion),nodes={'node8':dict(jobs=[dict(gpu=4,rank=100)])},worker_stall_seconds=1200,media_workers_initial=0,media_workers_max=0,media_ready_backlog=512,feature_workers_initial=2,feature_workers_max=4,gpu_headroom_mib=16000,poll_seconds=10)
            atomic(run/'config.json',cfg);atomic(run/'queues.json',dict(media=[],features=['job']));atomic(run/'CANARY.json',dict(passed=True,peak_reserved_mib=30000));atomic(fusion/'COMPLETE.json',dict(done=True))
            def launch(run,cfg,role,worker,**kw):
                calls.append(kw['gpu']);return dict(pid=99999999,role=role,worker=worker,gpu=kw['gpu'],launched_at=1)
            tele=dict(gpus={g:dict(memory_total_mib=140000,memory_used_mib=0) for g in range(8)},cpu_ticks=[],utc=1)
            with patch.object(manager,'telemetry',return_value=tele),patch.object(manager,'launch',side_effect=launch),patch.object(manager.time,'sleep',side_effect=lambda _: (run/'STOP').touch()):manager.node_loop(run,'node8')
            self.assertEqual(calls,[4]);self.assertEqual(set(read(run/'nodes/node8/status.json')['telemetry']['gpus']),{'4'})


if __name__=='__main__':unittest.main()
