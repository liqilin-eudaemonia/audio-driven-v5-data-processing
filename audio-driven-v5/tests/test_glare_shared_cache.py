"""Read-only admission tests for real shared-cache data and grouping."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_glare_shared_cache as cache


class SharedCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = cache.load_protocol()
        old = ROOT/'experiments/baseline_training_realdata_20261109'
        cls.old = old
        d = cache.read(old/'DATA_PROTOCOL.json')
        cls.ids = d['diagnostic_train'] + d['diagnostic_validation']
        db = sqlite3.connect(cache.PARENT/'jobs.sqlite')
        db.row_factory = sqlite3.Row
        cls.jobs = {sid:dict(db.execute('SELECT * FROM jobs WHERE sid=?',(sid,)).fetchone()) for sid in cls.ids}
        db.close()

    def test_group_partition_has_exact_members_no_duplicates(self):
        db = sqlite3.connect(cache.WORK/'groups.sqlite')
        seen = set()
        for ordinals,n,seconds in db.execute('SELECT ordinals,pairs,seconds FROM groups'):
            ids=json.loads(ordinals)
            self.assertEqual(len(ids),n)
            self.assertLessEqual(n,2)
            self.assertLessEqual(seconds,16.0000001)
            self.assertFalse(seen.intersection(ids))
            seen.update(ids)
        self.assertEqual(seen,set(range(284218)))
        db.close()

    def test_canary_audio_matches_original_input_exactly(self):
        for sid in self.ids:
            old=torch.load(self.old/'features/glare'/(sid+'.pt'),weights_only=True,map_location='cpu')
            new=torch.load(cache.path_for(sid,'audio.pt'),weights_only=True,map_location='cpu')
            self.assertTrue(torch.equal(old['audio'],new['audio']))

    def test_existing_nod_scores_and_unknown_masks_are_preserved(self):
        old = cache.base.load_protocol(cache.PARENT)
        for sid in self.ids:
            row=cache.base.load_row(self.jobs[sid],old)
            evidence=cache.base.evidence(row)
            prod=cache.read(evidence['detector_file'])
            with np.load(cache.path_for(sid,'reaction.npz')) as value:
                np.testing.assert_array_equal(value['reaction'][:,4],np.asarray(prod['scores'],np.float32))
                np.testing.assert_array_equal(value['reaction_valid'][:,4],np.asarray(prod['pose']['detected'],bool))
                self.assertEqual(list(value['classes']),self.p['reaction_classes'])
                self.assertEqual(value['reaction_valid'].dtype,bool)
                self.assertTrue(np.isfinite(value['reaction']).all())
                if prod.get('fer_status')!='ok':
                    self.assertFalse(value['reaction_valid'][:,2:4].any())

    def test_complete_marker_does_not_claim_prosody_or_training(self):
        for sid in self.ids:
            self.assertTrue(cache.complete(self.jobs[sid],self.p))
            mark=cache.read(cache.path_for(sid,'complete.json'))
            self.assertEqual(mark['prosody'],'deferred')
            self.assertFalse(mark['full_glare_training_admitted'])

    def test_corrupted_payload_is_rejected_on_resume(self):
        old_work=cache.WORK
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cache.WORK=Path(tmp)
                job=self.jobs[self.ids[0]]
                payload=Path(tmp)/'payload.bin'
                payload.write_bytes(b'valid')
                cache.finish_stage(job,'audio',payload,self.p)
                self.assertTrue(cache.stage_ok(job,'audio',self.p))
                payload.write_bytes(b'changed')
                with self.assertRaisesRegex(AssertionError,'corrupt'):
                    cache.stage_ok(job,'audio',self.p)
        finally:
            cache.WORK=old_work


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
