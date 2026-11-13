import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_glare_source_prosody_full as engine
from glare_prosody_subset import ChunkCache,generation_rows,read,CHUNK_SAMPLES
from glare_prosody_source_runtime import waveform_sha


class PackedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.plan=Path(self.tmp.name)/'plan';self.plan.mkdir()
        db=sqlite3.connect(self.plan/'index.sqlite')
        db.executescript('CREATE TABLE tracks(track_id TEXT,data TEXT);CREATE TABLE chunks(track_id TEXT,chunk_index INTEGER);')
        self.tracks=[]
        for i in range(5):
            track=dict(track_id='track'+str(i),number=i,audio=dict(sha256='wav'+str(i)))
            self.tracks.append(track);db.execute('INSERT INTO tracks VALUES(?,?)',(track['track_id'],json.dumps(track)))
            db.executemany('INSERT INTO chunks VALUES(?,?)',[(track['track_id'],j) for j in range(4)])
        db.commit();db.close()
        self.cache=ChunkCache(Path(self.tmp.name)/'cache',{'contract':'unchanged'})
        self.protocol=dict(chunks=20,tracks=5)
        self.batch_sizes=[];self.raw_wave={}
        for t in self.tracks:
            self.raw_wave[t['track_id']]=np.full(4*CHUNK_SAMPLES,t['number']+.25,np.float32)
        outer=self
        class Runtime:
            def load(self):pass
            def infer(self,waves):
                outer.batch_sizes.append(len(waves))
                g=generation_rows([[4,99]],[99],64)[0];g.update(eos_ids=[99],max_new_tokens=64)
                return [('NULL',g,'NULL') for _ in waves]
        self.runtime=Runtime()

    def prepare(self,runtime,cache,track,indices,retry_execution=False):
        needed=[];reused={}
        for i in indices:
            old=cache.load(track,i)
            if old:reused['explicit_null']=reused.get('explicit_null',0)+1
            else:needed.append(i)
        return dict(track=track,indices=needed,reused=reused,attempts=[],
                    wave=self.raw_wave[track['track_id']] if needed else None,error=None,prepare_s=.001)

    def run_engine(self,**kw):
        with patch.object(engine,'prepare_source',self.prepare),patch.object(engine.signal,'signal'):
            return engine.run_engine(self.plan,self.protocol,self.runtime,self.cache,**kw)

    def test_packs_cross_source_batches_preserves_exact_pcm_and_resumes(self):
        state=self.run_engine(loaders=2,batch_size=16,queue_size=32)
        self.assertEqual(self.batch_sizes,[16,4]);self.assertEqual(state['new'],20)
        self.assertEqual(state['stage'],'complete');self.assertTrue(state['producer_drained'])
        for track in self.tracks:
            for i in range(4):
                record=self.cache.load(track,i)
                self.assertEqual(record['waveform_sha256'],waveform_sha(self.raw_wave[track['track_id']][i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES]))
        state=self.run_engine(loaders=2,batch_size=16,queue_size=32)
        self.assertEqual(state['new'],0);self.assertEqual(state['reused'],20)
        self.assertEqual(self.batch_sizes,[16,4])

    def test_bounded_run_commits_exact_limit_then_full_resume(self):
        state=self.run_engine(limit=7,loaders=2,batch_size=16,queue_size=16)
        self.assertEqual(state['new'],7);self.assertEqual(self.batch_sizes,[7])
        self.assertEqual(state['stage'],'bounded_run_complete');self.assertTrue(state['producer_drained'])
        state=self.run_engine(loaders=2,batch_size=16,queue_size=16)
        self.assertEqual(state['new'],13);self.assertEqual(state['reused'],7)
        self.assertEqual(state['stage'],'complete')

    def test_source_failure_is_not_null_and_producer_failure_aborts(self):
        original=self.prepare
        def bad(runtime,cache,track,indices,retry_execution=False):
            r=original(runtime,cache,track,indices,retry_execution)
            if track['number']==0:r.update(error='IO exhausted',wave=None)
            return r
        with patch.object(engine,'prepare_source',bad),patch.object(engine.signal,'signal'):
            state=engine.run_engine(self.plan,self.protocol,self.runtime,self.cache,batch_size=16)
        self.assertEqual(state['new_counts']['input_error_exhausted'],4)
        self.assertFalse(self.cache.load(self.tracks[0],0)['analysis']['accepted'])
        with patch.object(engine,'prepare_source',side_effect=ValueError('broken identity')),patch.object(engine.signal,'signal'):
            with self.assertRaises(RuntimeError):engine.run_engine(self.plan,self.protocol,self.runtime,self.cache)
        self.assertEqual(read(self.plan/'status.json')['stage'],'failed')


    def test_oom_releases_failed_traceback_before_subdivision(self):
        import gc
        import weakref
        import torch
        held=weakref.WeakSet()
        class Payload:pass
        original=self.runtime.infer
        def infer(waves):
            if len(waves)>4:
                payload=Payload();held.add(payload)
                raise torch.cuda.OutOfMemoryError('simulate failed CUDA allocation')
            gc.collect()
            self.assertFalse(held,'Failed batch traceback still retains CUDA payload before child retry')
            return original(waves)
        self.runtime.infer=infer
        with patch.object(torch.cuda,'empty_cache'):
            state=self.run_engine(loaders=2,batch_size=16,queue_size=32)
        self.assertEqual(state['new'],20);self.assertEqual(state['stage'],'complete')
        self.assertNotIn('inference_error_exhausted',state['counts'])
        self.assertEqual(state['effective_batch_size'],4)
        self.assertEqual(self.batch_sizes,[4]*5)


    def test_early_stop_cannot_admit_legal_null_as_rejected(self):
        from glare_prosody_subset import IntegrityError
        original=self.runtime.infer
        def malicious(waves):
            return [(raw,dict(gen,early_reject=True),native) for raw,gen,native in original(waves)]
        self.runtime.infer=malicious
        with self.assertRaises(IntegrityError):self.run_engine(loaders=2,batch_size=16,queue_size=32)
        self.assertEqual(read(self.plan/'status.json')['stage'],'failed')
        self.assertFalse(list((self.cache.root/'chunks').glob('*/*.json')))


if __name__=='__main__':unittest.main()
