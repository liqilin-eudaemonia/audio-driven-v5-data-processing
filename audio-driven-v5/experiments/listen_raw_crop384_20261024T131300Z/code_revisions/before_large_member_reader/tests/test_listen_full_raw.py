import copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.full_raw import compact, views
spec=importlib.util.spec_from_file_location('raw_worker',ROOT/'scripts/run_listen_full_raw.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

def fixture():
    src=dict(archive='/a.tar',member='listener.mp4',offset=0,byte_size=100)
    return dict(sample_id='s1',pair='p1',split='train',admission_pool='onset_reserve',audio_final={'pool':'keep_candidate'},
        listener_id='listener',speaker_id='speaker',listener_index=0,speaker_index=1,sources={'listener':src},
        normalized_audio={'listener':dict(contract={'source':src},path='/listener.wav',sha256='hash',sample_rate=48000)},
        start=2.,end=8.,duration_s=6.,boundary_activity={'onset_activity':True},boundary_review_flags=['start_no_quiet_cut'],
        face_geometry={'listener':dict(width=1080,height=1920)},
        crop_proposals={'listener':dict(status='proposal_only',x=100.4,y=200.6,size=500.2)})

class RawTests(unittest.TestCase):
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
