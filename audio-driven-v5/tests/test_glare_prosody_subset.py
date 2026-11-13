import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from glare_prosody_subset import (ChunkCache,IntegrityError,audit_raw,generation_rows,pair_vector,
                                read,required_chunks,save,track_identity,sha,lock)
from prepare_glare_prosody_subset import source_pair
from run_glare_prosody_subset_cache import filtered_groups


class ProsodySubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cache=ChunkCache(self.tmp.name,{'test':'contract'})
        self.track=dict(track_id='speaker-a',audio={'sha256':'wavehash','sample_rate':48000})
        self.pair=dict(sid='example',row_sha='row',split='train',duration=1.,start=2.,end=3.)

    def commit(self,index,raw='NULL'):
        gen=generation_rows([[10,11,99]],[99],64)[0];gen.update(eos_ids=[99],max_new_tokens=64)
        return self.cache.commit(self.track,index,raw,gen,audit_raw(raw)['normalized_text'],'pcmhash',[])

    def test_half_open_grid_including_short_pairs(self):
        self.assertEqual(required_chunks(0.,2.4),[0])
        self.assertEqual(required_chunks(2.4,4.8),[1])
        self.assertEqual(required_chunks(.2,.8),[0])
        self.assertEqual(required_chunks(2.3999,2.4001),[0,1])
        self.assertEqual(required_chunks(100.8,103.2),[42])

    def test_explicit_null_and_below_threshold_not_errors(self):
        self.assertEqual(audit_raw('NULL')['status'],'explicit_null')
        self.assertEqual(audit_raw('0.00, 1.00, 0.30')['status'],'below_threshold')
        self.assertTrue(audit_raw('0.00, 2.40, 0.80')['accepted'])

    def test_official_collapsed_null_never_masks_invalid_raw(self):
        for raw in ('Sorry, provide audio.','```\nNULL\n```',r'0.00, 0.50, 0.80\n0.60, 1.00, 0.70',
                    'NULL\n0.00, 0.50, 0.80','0.00, 2.50, 0.80','0.00, 0.10, 1.20','',
                    '0.00, 1.00, 0.80\n0.50, 1.20, 0.70'):
            with self.subTest(raw=raw):self.assertFalse(audit_raw(raw)['accepted'])
        self.assertFalse(audit_raw('NULL',truncated=True)['accepted'])

    def test_per_row_eos_not_batch_padding(self):
        rows=generation_rows([[1,99]+[0]*62,[1]*64],[99,98],64)
        self.assertTrue(rows[0]['eos_seen']);self.assertFalse(rows[0]['truncated'])
        self.assertTrue(rows[1]['truncated']);self.assertEqual(rows[0]['tokens'],2)
        self.assertFalse(generation_rows([[1]*63+[98]],[99,98],64)[0]['truncated'])

    def test_cross_chunk_local_mapping_preserves_fractional_boundary(self):
        p=dict(self.pair,start=2.379,end=2.579,duration=.2)
        records={0:dict(analysis=audit_raw('2.38, 2.40, 0.80')),
                 1:dict(analysis=audit_raw('0.00, 0.05, 0.60'))}
        scores,valid,events=pair_vector(p,records)
        self.assertEqual(events[0]['start'],.001)
        self.assertEqual(events[1]['start'],.021)
        self.assertEqual(scores.shape,(5,1));self.assertTrue(valid.all())
        # Adjacent intervals share a local raster bin; native later assignment wins.
        self.assertEqual(scores[0,0],np.float32(.6));self.assertEqual(scores[1,0],np.float32(.6))
        self.assertFalse(scores[2:].any())

    def test_short_pair_can_be_covered_without_its_own_full_chunk(self):
        p=dict(self.pair,start=.1,end=.6,duration=.5)
        self.commit(0)
        decision=self.cache.admit_pair(p,self.track,'pool')
        self.assertEqual(decision['decision'],'keep')
        with np.load(decision['intensity']) as v:
            self.assertEqual(v['intensity'].shape,(13,1));self.assertTrue(v['intensity_valid'].all())

    def test_missing_is_pending_failed_is_excluded(self):
        self.commit(0)
        self.assertEqual(self.cache.admit_pair(self.pair,self.track,'pool')['decision'],'pending')
        self.cache.error(self.track,1,'uncovered_source_tail','tail')
        result=self.cache.admit_pair(self.pair,self.track,'pool')
        self.assertEqual(result['decision'],'exclude')
        self.assertEqual(result['failures'][0]['status'],'uncovered_source_tail')
        self.assertFalse(self.cache.pair_paths(self.pair)[0].exists())

    def test_immutable_cache_resume_and_corruption_detection(self):
        self.commit(0);self.commit(1,'0.00, 0.60, 0.80')
        first=self.cache.admit_pair(self.pair,self.track,'pool')
        second=self.cache.admit_pair(self.pair,self.track,'pool')
        self.assertTrue(second['reused']);self.assertEqual(first['intensity_receipt_sha256'],second['intensity_receipt_sha256'])
        with self.assertRaises(IntegrityError):self.commit(0)
        receipt=Path(first['intensity_receipt']);r=read(receipt)
        with Path(first['intensity']).open('wb') as f:
            np.savez(f,intensity=np.zeros((25,1),np.float32),intensity_valid=np.ones((25,1),bool))
        r['payload_sha256']=sha(first['intensity']);save(receipt,r)
        with self.assertRaises(IntegrityError):self.cache.admit_pair(self.pair,self.track,'pool')

    def test_retry_only_execution_and_wrong_normalizer_rejected(self):
        self.cache.error(self.track,0,'input_error_exhausted','temporary')
        self.assertTrue(self.cache.retry_execution(self.track,0));self.assertIsNone(self.cache.load(self.track,0))
        self.commit(0);self.assertFalse(self.cache.retry_execution(self.track,0))
        with self.assertRaises(IntegrityError):
            self.cache.commit(self.track,1,'NULL',{},'0.00, 1.00, 0.80','hash',[])

    def test_raw_and_eos_tampering_detected_on_resume(self):
        self.commit(0);path=self.cache.paths(self.track,0)[1];r=read(path)
        r['generation']['truncated']=True;save(path,r)
        with self.assertRaises(IntegrityError):self.cache.load(self.track,0)

    def test_invalid_split_and_duration_cannot_enter_training(self):
        for p in (dict(self.pair,split='test'),dict(self.pair,duration=2.)):
            with self.assertRaises(IntegrityError):self.cache.admit_pair(p,self.track,'pool')

    def test_shared_contract_lock_excludes_other_plan(self):
        other=ChunkCache(self.tmp.name,{'test':'contract'})
        with lock(self.cache.root/'mutation.lock',False):
            with self.assertRaises(BlockingIOError):
                with lock(other.root/'mutation.lock',False):pass

    def test_survivor_group_routing_preserves_identity(self):
        groups=[dict(gid=4,worker=1,ordinals='[0, 1]',pairs=2,seconds=5.),dict(gid=5,ordinals='[2]',pairs=1,seconds=4.)]
        result=filtered_groups(groups,{1},{0:2.,1:3.,2:4.})
        self.assertEqual(result,[dict(gid=4,worker=1,ordinals='[1]',pairs=1,seconds=3.)])
        self.assertEqual(groups[0]['ordinals'],'[0, 1]')

    def test_real_source_role_identity_and_swap_guard(self):
        parent=ROOT/'experiments/glare_current500h_nonprosody_cache_20261110'
        if not (parent/'jobs.sqlite').exists():self.skipTest('Real pool unavailable')
        import sqlite3
        db=sqlite3.connect(parent/'jobs.sqlite');db.row_factory=sqlite3.Row;job=dict(db.execute('select * from jobs limit 1').fetchone());db.close()
        p=read(parent/'protocol.json')
        with open(p['selection'],'rb') as f:f.seek(job['offset']);row=json.loads(f.readline())
        result=read(row['source']['source_result']);pair,track=source_pair(row,job,result)
        self.assertEqual(pair['split'],'train');self.assertNotEqual(track['audio']['path'],row['source']['normalized_audio']['path'])
        broken=copy.deepcopy(result);clip=broken['clips'][row['source']['source_clip_index']]
        clip['normalized_audio']['speaker']=clip['normalized_audio']['listener']
        with self.assertRaises(IntegrityError):source_pair(row,job,broken)


    def test_entrypoint_extract_assemble_and_zero_generation_resume(self):
        import sqlite3
        from types import SimpleNamespace
        from unittest.mock import patch
        import prepare_glare_prosody_subset as cli
        plan=Path(self.tmp.name)/'plan';plan.mkdir()
        source=plan/'selection.jsonl'
        row=dict(sample_id='example',split='train',source={'speaker_id':'correct'},duration_s=1.)
        raw=(json.dumps(row)+'\n').encode();source.write_bytes(raw)
        import hashlib
        pair=dict(self.pair,start=.1,end=1.1,ordinal=0,offset=0,
                  row_sha=hashlib.sha256(raw).hexdigest(),track_id=self.track['track_id'])
        db=sqlite3.connect(plan/'index.sqlite')
        db.executescript('CREATE TABLE tracks(track_id TEXT,data TEXT);CREATE TABLE pairs(sid TEXT,ordinal INTEGER,data TEXT);CREATE TABLE chunks(track_id TEXT,chunk_index INTEGER);')
        db.execute('INSERT INTO tracks VALUES(?,?)',(self.track['track_id'],json.dumps(self.track)))
        db.execute('INSERT INTO pairs VALUES(?,?,?)',(pair['sid'],0,json.dumps(pair)))
        db.execute('INSERT INTO chunks VALUES(?,?)',(self.track['track_id'],0));db.commit();db.close()
        p=dict(cache_root=self.tmp.name,extractor={'test':'contract'},selection=str(source),
               selection_sha256=sha(source),pairs=1)
        save(plan/'protocol.json',p)
        args=SimpleNamespace(plan=plan,all=False,limit=1,batch_size=16,retry_execution=False)
        calls=[]
        class FakeRuntime:
            def __init__(self,*_):pass
            def waveform(self,track):return np.zeros(38400,np.float32)
            def load(self):pass
            def infer(self,waves):
                calls.append(len(waves));gen=generation_rows([[1,99]],[99],64)[0]
                gen.update(eos_ids=[99],max_new_tokens=64)
                return [('NULL',gen,'NULL') for _ in waves]
        with patch.object(cli,'protocol',return_value=p),patch('glare_prosody_source_runtime.OfficialRuntime',FakeRuntime),patch.object(cli.signal,'signal'):
            cli.extract(args);self.assertEqual(calls,[1]);cli.assemble(args)
            latest=read(plan/'LATEST.json');out=Path(latest['export'])
            summary=read(out/'SUMMARY.json');self.assertEqual(summary['counts'],{'keep':1})
            kept=json.loads((out/'retained_train.jsonl').read_text())
            self.assertEqual(kept['split'],'train');self.assertEqual(kept['source'],row['source'])
            self.assertEqual((out/'retained_dev.jsonl').read_text(),'')
            receipt=kept['glare_prosody']['intensity_receipt_sha256']
            cli.extract(args);self.assertEqual(calls,[1]);cli.assemble(args)
            other=Path(read(plan/'LATEST.json')['export'])
            self.assertEqual(json.loads((other/'retained_all.jsonl').read_text())['glare_prosody']['intensity_receipt_sha256'],receipt)
            self.assertEqual(read(plan/'status.json')['new'],0)


    def test_downstream_refuses_active_original_controller(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import os
        import run_glare_prosody_subset_cache as bridge
        old=Path(self.tmp.name)/'old';old.mkdir();(old/'workers').mkdir()
        parent=Path(self.tmp.name)/'parent';parent.mkdir()
        work=Path(self.tmp.name)/'survivors';work.mkdir()
        export=Path(self.tmp.name)/'export';export.mkdir();save(export/'SEAL.json',dict(files={}))
        for path in (old/'protocol.json',old/'groups.sqlite',parent/'jobs.sqlite'):path.write_text('frozen')
        scope=dict(code_sha256=sha(bridge.__file__),export=str(export),export_seal_sha256=sha(export/'SEAL.json'),
                   parent_protocol_sha256=sha(old/'protocol.json'),groups_sha256=sha(old/'groups.sqlite'),
                   parent_jobs_sha256=sha(parent/'jobs.sqlite'))
        save(work/'scope.json',scope)
        with patch.object(bridge,'OLD',old),patch.object(bridge,'PARENT',parent),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='7'):
            for occupied in (old/'run_all.lock',old/'controller.lock',old/'workers/gpu7.lock'):
                with self.subTest(lock=str(occupied)),lock(occupied,False):
                    with self.assertRaises(BlockingIOError):bridge.run(SimpleNamespace(work=work))

    def test_feature_workspace_cannot_overwrite_existing_protocol(self):
        from unittest.mock import patch
        import run_glare_prosody_subset_cache as bridge
        root=Path(self.tmp.name)
        old=root/'old';parent=root/'parent';plan=root/'plan';export=plan/'exports';cache=root/'cache'
        existing=root/'other-experiment';existing.mkdir();protocol_file=existing/'protocol.json';protocol_file.write_text('preserve')
        with patch.object(bridge,'OLD',old),patch.object(bridge,'PARENT',parent):
            for target in (old,parent,plan,export,cache,old/'nested',root,existing):
                with self.subTest(work=str(target)):
                    with self.assertRaises(IntegrityError):bridge.validate_work_target(target,plan,export,cache)
            bridge.validate_work_target(root/'new-feature-work',plan,export,cache)
        self.assertEqual(protocol_file.read_text(),'preserve')


    def test_sample_quantized_duration_preserves_real_subsample_rounding(self):
        pair=dict(self.pair,start=224.971,end=227.864216,duration=2.8932083333333334)
        # The metadata end is rounded to microseconds; the real audio is48k sample-quantized.
        self.assertGreater(abs(pair['end']-pair['start']-pair['duration']),1e-6)
        self.assertEqual(self.cache.admit_pair(pair,self.track,'pool')['decision'],'pending')
        with self.assertRaises(IntegrityError):self.cache.admit_pair(dict(pair,duration=pair['duration']+.001),self.track,'pool')


if __name__=='__main__':unittest.main()
