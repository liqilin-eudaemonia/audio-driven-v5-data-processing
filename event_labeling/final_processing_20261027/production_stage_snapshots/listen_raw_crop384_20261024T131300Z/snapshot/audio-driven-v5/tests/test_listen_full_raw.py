import copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import compact, views, SourceBytes
spec=importlib.util.spec_from_file_location('raw_worker',ROOT/'scripts/run_listen_full_raw.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
spec_subset=importlib.util.spec_from_file_location('raw_subset',ROOT/'scripts/subset_listen_full_raw.py')
subset=importlib.util.module_from_spec(spec_subset);spec_subset.loader.exec_module(subset)

def fixture():
    src=dict(archive='/a.tar',member='listener.mp4',offset=0,byte_size=100)
    return dict(sample_id='s1',pair='p1',split='train',admission_pool='onset_reserve',audio_final={'pool':'keep_candidate'},
        listener_id='listener',speaker_id='speaker',listener_index=0,speaker_index=1,sources={'listener':src},
        normalized_audio={'listener':dict(contract={'source':src},path='/listener.wav',sha256='hash',sample_rate=48000)},
        start=2.,end=8.,duration_s=6.,boundary_activity={'onset_activity':True},boundary_review_flags=['start_no_quiet_cut'],
        face_geometry={'listener':dict(width=1080,height=1920)},
        crop_proposals={'listener':dict(status='proposal_only',x=100.4,y=200.6,size=500.2)})

class RawTests(unittest.TestCase):
    def test_member_reader_cannot_leak_adjacent_tar_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'archive';p.write_bytes(b'prefixMEMBERsuffix')
            with SourceBytes(dict(archive=str(p),offset=6,byte_size=6)) as f:
                self.assertEqual(f.read(100),b'MEMBER');self.assertEqual(f.read(1),b'')
                f.seek(-3,2);self.assertEqual(f.read(),b'BER')
                f.seek(100);self.assertEqual(f.read(),b'')
                with self.assertRaises(ValueError):f.seek(-1)
    def test_retry_input_failure_keeps_original_raw_record(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw';raw=(json.dumps(dict(record_kind='result',request_id='x@input',status='preparation_failed',error={'type':'InvalidDataError'}))+'\n').encode();p.write_bytes(raw)
            self.assertIn('x@input',worker.recover(p)[0])
            self.assertEqual(worker.recover(p,True)[0],{})
            self.assertEqual(p.read_bytes(),raw)
    def test_subset_uses_unrounded_native_crop_and_requires_valid_transform(self):
        row=compact(fixture(),'/manifest',1,'hash')
        for size,expected in [(383.9,False),(384,True),(500,True),(float('nan'),False)]:
            row['crop_proposal']['size']=size
            self.assertEqual(subset.eligible(row,384),expected)
        row['crop_proposal']['size']=500;row['video_transform']=None
        self.assertFalse(subset.eligible(row,384))
    def test_subset_raw_reuse_keeps_bytes_and_excludes_other_samples(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';target=Path(d)/'target'
            (source/'raw/visual/results').mkdir(parents=True)
            (target/'queue/shards').mkdir(parents=True)
            parent=dict(sample_id='keep',source_line_sha256='hash',duration_s=2)
            (target/'queue/shards/0000.jsonl').write_text(json.dumps(parent)+'\n')
            (target/'queue/index.json').write_text(json.dumps(dict(shards={'0000':1},scope={})))
            kept=b'{"record_kind": "result", "sample_id":"keep", "request_id":"keep@full", "source":{"source_line_sha256":"hash"}, "status":"schema_pass", "raw_response":"original words"}\n'
            excluded=json.dumps(dict(record_kind='result',sample_id='outside')).encode()+b'\n'
            (source/'raw/visual/results/0000.jsonl').write_bytes(excluded+kept)
            with self.assertRaises(ValueError):subset.import_raw(source,target,384)
            (source/'RAW_FROZEN').touch();subset.import_raw(source,target,384)
            self.assertEqual((target/'raw/visual/results/0000.jsonl').read_bytes(),kept)
            state=json.loads((target/'raw/visual/state/0000.json').read_text())
            self.assertTrue(state['done']);self.assertEqual(state['requests'],1)
    def test_onset_pool_and_original_boundary_are_preserved(self):
        r=fixture();before=copy.deepcopy(r);out=compact(r,'/manifest',5,'hash')
        self.assertEqual(r,before);self.assertEqual(out['boundary_activity'],r['boundary_activity'])
        self.assertEqual((out['source_start_s'],out['source_end_s']),(2.,8.))
        self.assertEqual(out['video_transform']['crop'],dict(x=100,y=201,width=500,height=500))
        self.assertEqual(out['video_transform']['output_size'],[512,512])
    def test_invalid_crop_does_not_drop_valid_audio_or_use_full_frame(self):
        r=fixture();r['crop_proposals']['listener']['size']=2000
        out=compact(r,'/manifest',1,'hash');self.assertIsNone(out['video_transform'])
        self.assertTrue(out['crop_error']);self.assertEqual(out['normalized_audio']['path'],'/listener.wav')
    def test_mixed_or_other_person_audio_is_rejected(self):
        r=fixture();r['normalized_audio']['listener']['contract']['source']=dict(member='speaker.mp4')
        with self.assertRaises(ValueError):compact(r,'/manifest',1,'hash')
    def test_dev_test_or_audio_reject_is_not_silently_admitted(self):
        r=fixture();r['split']='test'
        with self.assertRaises(ValueError):compact(r,'/manifest',1,'hash')
        r=fixture();r['audio_final']['pool']='review'
        with self.assertRaises(ValueError):compact(r,'/manifest',1,'hash')
    def test_whole_clip_and_overlapping_views_keep_explicit_offsets(self):
        self.assertEqual(views(3),[('full',0.,3)])
        self.assertEqual(views(4.2),[('full',0.,4.2),('w000',0.,3.),('w001',1.,4.),('w002',2.,4.2)])
    def test_resume_does_not_count_transcript_as_completed_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl'
            rows=[dict(record_kind='asr_transcript',request_id='x'),dict(record_kind='result',request_id='y',status='schema_pass')]
            p.write_bytes((''.join(json.dumps(r)+'\n' for r in rows)).encode()+b'{"interrupted":')
            done,counts=worker.recover(p)
            self.assertEqual(done,{'y':'schema_pass'});self.assertEqual(counts['schema_pass'],1)
            self.assertEqual(len(p.read_text().splitlines()),2)

if __name__=='__main__':unittest.main()
