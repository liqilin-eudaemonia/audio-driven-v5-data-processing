"""Cache completion means all frozen inputs, without restarting GPU tests."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import glare_cache_cluster as c
import glare_cache_cluster_worker as worker
import glare_cluster_frozen_audio as audio
from assemble_glare_training_caches import collect,assemble
from glare_prosody_subset import canonical


class AllCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.work=Path(self.tmp.name)
        self.old=self.work/'shared';self.old.mkdir();self.parent=self.work/'parent';self.parent.mkdir()
        self.cfg=dict(work=str(self.work),shared_lia=str(self.old),source_plan=str(self.work/'source'))
        self.p=dict(schema='native-shared',code={},selection_sha256='sel')
        self.fp=dict(schema='native-frozen',view='first frame excluded; no history',dtype='float32',pre_projection=True)
        self.m=dict(run_id='test',selection_sha256='sel',cache_contract={},shared_protocol=self.p,
                    frozen_audio_protocol=self.fp,parent=str(self.parent),pairs=1,required_caches=['all'])
        self.job=dict(sid='sample',row_sha='source-sha',duration=4.,split='train')
    def tearDown(self):self.tmp.cleanup()
    def frozen(self,job=None,root=None):
        job=job or self.job;root=root or self.old/'frozen_audio'
        payload,receipt=audio.cache_paths(root,job['sid']);payload.parent.mkdir(parents=True,exist_ok=True);payload.write_bytes(b'features')
        frames=__import__('math').ceil(job['duration']*25-1e-7)-1
        c.save(receipt,dict(identity=dict(sample_id=job['sid'],row_sha256=job['row_sha'],protocol_sha256=canonical(self.fp)),
                shape=[frames,9216],waveform_samples=frames*640,pre_projection=True,payload_sha256=c.sha(payload)))
        return payload,receipt
    def full_entry(self):
        identity=dict(schema=self.p['schema'],sample_id=self.job['sid'],row_sha256=self.job['row_sha'],
                      selection_sha256='sel',code_sha256=canonical({}))
        folder=self.old/'cache'/hashlib.sha256(b'sample').hexdigest()[:2];folder.mkdir(parents=True)
        for kind in ['lia','audio','reaction','statistics']:
            payload=folder/('sample.'+kind+'.payload');payload.write_bytes(b'tensor')
            c.save(folder/('sample.'+kind+'.json'),dict(identity=identity,payload=str(payload),payload_sha256=c.sha(payload)))
        self.frozen()
        payload=self.work/'intensity.npz';payload.write_bytes(b'official-null-zero')
        receipt=self.work/'intensity.json'
        c.save(receipt,dict(identity=dict(pair=dict(sid='sample',row_sha='source-sha',split='train',frames=100),source_contract_id=canonical({})),
                            training_condition_usable=True,policy='official',payload_sha256=c.sha(payload)))
        row=dict(sample_id='sample',split='train',glare_prosody=dict(intensity=str(payload),intensity_receipt=str(receipt),
                         intensity_receipt_sha256=c.sha(receipt)))
        return row
    def test_existing_frozen_cache_reuses_without_importing_gpu_runtime(self):
        self.frozen()
        with patch.dict(sys.modules,{'glare_audio_execution_v2':None,'torch':None}):
            result=audio.ensure(self.cfg,self.m,dict(gpus=[0,1,2,3]),2,self.work/'slot',[self.job],[])
        self.assertEqual(result['reused'],1);self.assertEqual(result['completed'],0);self.assertTrue(result['validated'])
    def test_wrong_source_or_audio_window_is_not_silently_reused(self):
        _,receipt=self.frozen();r=c.read(receipt);r['waveform_samples']-=1;c.save(receipt,r)
        with self.assertRaises(AssertionError):audio.cached(self.old/'frozen_audio',self.job,self.fp)
        self.frozen();bad=dict(self.job,row_sha='another-source')
        with self.assertRaises(AssertionError):audio.cached(self.old/'frozen_audio',bad,self.fp)
    def test_explicit_payload_hashing_detects_damage(self):
        payload,_=self.frozen();self.assertTrue(audio.cached(self.old/'frozen_audio',self.job,self.fp,True))
        payload.write_bytes(b'changed')
        with self.assertRaises(AssertionError):audio.cached(self.old/'frozen_audio',self.job,self.fp,True)
    def test_missing_audio_calls_native_engine_on_assigned_gpu_and_writes_canonical_cache(self):
        created=[];destinations=[];outer=self
        flags=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True))
        class Engine:
            def __init__(self,gpu,assigned_gpus):created.append((gpu,assigned_gpus))
            def write(self,job,audio_,n,hidden,emotion,root):destinations.append(Path(root));outer.frozen(job,Path(root))
            def run(self,jobs,root,asynchronous,reuse):
                assert flags.cuda.matmul.allow_tf32 is False and flags.cudnn.allow_tf32 is False
                for job in jobs:self.write(job,None,None,None,None,root)
                return dict(stage='finished',failed=0,completed=len(jobs))
        modules={'glare_audio_execution_v2':types.SimpleNamespace(Engine=Engine),
                 'glare_frozen_audio_cache':types.SimpleNamespace(WORK=None),'torch':types.SimpleNamespace(backends=flags)}
        with patch.dict(sys.modules,modules):
            result=audio.ensure(self.cfg,self.m,dict(gpus=[0,1,2,3]),2,self.work/'slot',[self.job],[])
        self.assertEqual(created,[(2,(0,1,2,3))]);self.assertEqual(destinations,[self.old/'frozen_audio'])
        self.assertEqual(result['completed'],1);self.assertTrue(flags.cuda.matmul.allow_tf32 and flags.cudnn.allow_tf32)
        audio.cache_paths(self.old/'frozen_audio',self.job['sid'])[0].unlink()
        with patch.dict(sys.modules,modules):
            repaired=audio.ensure(self.cfg,self.m,dict(gpus=[0,1,2,3]),2,self.work/'slot',[self.job],[])
        self.assertEqual(repaired['completed'],1)
        self.assertEqual(len(list((self.old/'frozen_audio/retry_history').glob('*.json'))),1)
    def test_stop_marker_prevents_gpu_initialization_even_when_cache_missing(self):
        (self.work/'STOP').touch()
        with patch.dict(sys.modules,{'glare_audio_execution_v2':None,'torch':None}):
            result=audio.ensure(self.cfg,self.m,dict(gpus=[0,1,2,3]),0,self.work/'slot',[self.job],[])
        self.assertTrue(result['stopped']);self.assertFalse(result['validated'])
    def test_registry_requires_all_features_and_contains_masks_and_preprojection(self):
        row=self.full_entry();entry=collect(self.cfg,self.m,self.job,row,True)
        self.assertEqual(set(entry),{'lia','audio','reaction','statistics','frozen_audio','prosody','view'})
        self.assertEqual(entry['frozen_audio']['wav2vec_shape'],[99,9216]);self.assertTrue(entry['frozen_audio']['pre_projection'])
        self.assertEqual(entry['view']['target_frames'],99)
        audio.cache_paths(self.old/'frozen_audio','sample')[0].unlink()
        with self.assertRaises(AssertionError):collect(self.cfg,self.m,self.job,row)
    def test_lia_task_does_not_commit_before_frozen_audio_finishes(self):
        db=sqlite3.connect(self.work/'routing.sqlite');db.execute('CREATE TABLE tasks(stage TEXT,task_id INTEGER,units INTEGER,weight REAL,items INTEGER)')
        db.execute("INSERT INTO tasks VALUES('lia',0,1,4.,1)");db.commit();db.close()
        native=types.SimpleNamespace(stage='finished')
        with patch.object(worker,'configure_lia',return_value=lambda task:dict(stage='finished',failed=0,completed=1,reused=0)),\
             patch.object(audio,'task_jobs',return_value=[self.job]),\
             patch.object(audio,'ensure',return_value=dict(stopped=True,validated=False)):
            worker.run_lia(self.cfg,self.m,dict(name='node11'),0,self.work/'slot',c.TaskQueue(self.work,'test','lia'))
        self.assertEqual(c.TaskQueue(self.work,'test','lia').summary()['done_tasks'],0)
    def test_final_registry_is_published_only_when_all_inputs_are_present(self):
        row=self.full_entry();source=Path(self.cfg['source_plan']);source.mkdir()
        (source/'retained_all.jsonl').write_text(json.dumps(row)+'\n')
        c.save(source/'SUMMARY.json',dict(counts=dict(keep=1)));c.save(source/'SEAL.json',{})
        c.save(source/'LATEST.json',dict(export=str(source),policy='official',seal_sha256=c.sha(source/'SEAL.json')))
        c.save(self.work/'manifest.json',self.m)
        db=sqlite3.connect(self.parent/'jobs.sqlite');db.execute('CREATE TABLE jobs(sid TEXT,row_sha TEXT,duration REAL,split TEXT)')
        db.execute('INSERT INTO jobs VALUES(?,?,?,?)',tuple(self.job.values()));db.commit();db.close()
        result=assemble(self.cfg,self.m)
        self.assertTrue(result['summary']['cache_registry_complete'])
        self.assertFalse(result['summary']['production_training_reader_ready'])
        (self.work/'TRAINING_CACHES_LATEST.json').unlink()
        audio.cache_paths(self.old/'frozen_audio','sample')[0].unlink()
        with self.assertRaises(AssertionError):assemble(self.cfg,self.m)
        self.assertFalse((self.work/'TRAINING_CACHES_LATEST.json').exists())
    def test_adding_audio_models_does_not_rehash_unchanged_qwen_weights(self):
        old=self.work/'old-weight';new=self.work/'audio-weight';old.write_bytes(b'old');new.write_bytes(b'audio')
        proof=self.work/'nodes/node11/assets.json'
        c.save(proof,dict(run_id='old-run',assets_sha256={str(old):c.sha(old)},stats={str(old):c.stat_key(old)}))
        m=dict(run_id='new-run',assets_sha256={str(old):c.sha(old),str(new):c.sha(new)})
        real_sha=c.sha;hashed=[]
        def observed(path):hashed.append(str(path));return real_sha(path)
        with patch.object(c,'sha',side_effect=observed):c.node_assets(self.cfg,m,dict(name='node11'))
        self.assertEqual(hashed,[str(new)]);c.verify_assets(m,proof)


if __name__=='__main__':unittest.main()
