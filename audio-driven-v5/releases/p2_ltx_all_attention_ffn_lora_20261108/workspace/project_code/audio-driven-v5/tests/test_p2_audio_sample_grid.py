import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audio_grid',ROOT/'scripts/repair_p2_audio_sample_grid.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class AudioGridTests(unittest.TestCase):
    def fixture(self,run):
        dest=run/'media/s';dest.mkdir(parents=True);source=dest/'speaker.wav'
        wave=np.linspace(-.5,.5,96001,dtype=np.float32);sf.write(source,wave,48000,subtype='PCM_16')
        obs=dict(sample_id='s',speaker_audio=str(source),requested_duration_s=round(96001/48000,9),media_hashes=dict(speaker_audio=mod.file_sha(source)))
        row=dict(sample_id='s',observation=obs,target=dict(sample_id='s',untouched=True))
        mod.atomic(dest/'COMPLETE.json',dict(observation=obs,target=row['target'],meta=dict(sha256={'speaker.wav':mod.file_sha(source)})))
        path=run/'manifests/shards/job.json';mod.atomic(path,[row]);mod.atomic(run/'state/media/job.json',dict(done=True,samples=1,manifest_sha256=mod.file_sha(path)))
        return row,source,path

    def test_exact_float_derivative_preserves_raw_and_target_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);row,source,path=self.fixture(run);original=source.read_bytes()
            result=mod.repair_job(run,'job');self.assertEqual(result['repaired'],1)
            after=mod.read(path)[0];self.assertEqual(after['target'],row['target']);self.assertEqual(source.read_bytes(),original)
            raw,_=sf.read(source,dtype='float32');expected=resample_poly(raw,1,3).astype(np.float32)[:32000]
            actual,rate=sf.read(after['observation']['speaker_audio'],dtype='float32');self.assertEqual(rate,16000);np.testing.assert_array_equal(actual,expected)
            self.assertEqual(mod.read(run/'state/media/job.json')['manifest_sha256'],mod.file_sha(path));before=path.read_bytes();mod.repair_job(run,'job');self.assertEqual(path.read_bytes(),before)

    def test_busy_job_and_existing_cache_are_protected(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);row,_,_=self.fixture(run)
            with mod.lock(run/'locks/worker-features-job.lock'):
                self.assertIsNone(mod.repair_job(run,'job'))
            mod.atomic(run/'cache/input/s/complete.json',{})
            with self.assertRaisesRegex(ValueError,'completed feature caches'):mod.repair_job(run,'job')

    def test_interrupted_commit_replays_journal(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);self.fixture(run);original=mod.atomic
            def interrupted(path,value):
                if str(path).endswith('state/media/job.json'):raise RuntimeError('simulated crash')
                return original(path,value)
            with patch.object(mod,'atomic',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'simulated crash'):mod.repair_job(run,'job')
            self.assertFalse(mod.read(run/'audio_grid/jobs/job.json').get('committed',False))
            mod.repair_job(run,'job');self.assertTrue(mod.read(run/'audio_grid/jobs/job.json')['committed'])
            self.assertEqual(mod.read(run/'state/media/job.json')['manifest_sha256'],mod.file_sha(run/'manifests/shards/job.json'))

if __name__=='__main__':unittest.main()
