"""Exercise failure-sensitive temporal and statistical choices on real ViCo IDs."""
import csv
import json
from pathlib import Path
import sys
import unittest
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_avatarforcing import WORK, timestamp_map, read, sha
from vico_unified_evaluate import cluster_bootstrap, OLD, DY
import vico_unified_evaluate as evaluator


class UnifiedProtocolTests(unittest.TestCase):
    def test_distinct_listener_pairs_are_never_collapsed_by_audio_key(self):
        spec=read(OLD/'protocol.json')['datasets']['vico']
        with open(spec['annotation']) as f:rows=list(csv.DictReader(f))
        actual=read(WORK/'avatarforcing/protocol.json')['samples']
        self.assertEqual({r['uuid'] for r in rows},{s['id'] for s in actual})
        self.assertEqual(len(actual),43)
        self.assertEqual(len({s['id'].split('.')[0] for s in actual}),40)
        self.assertTrue(all(r['video_id']==r['uuid'].rsplit('_',2)[0] for r in rows))

    def test_short_and_long_timestamps_preserve_complete_target_grid(self):
        for s in read(WORK/'avatarforcing/protocol.json')['samples']:
            native=int(np.ceil(round(s['duration']*16000)*25/16000))
            mapping=timestamp_map(s['frames'],native)
            self.assertEqual(len(mapping),s['frames'])
            self.assertEqual(mapping[0],0)
            self.assertTrue((np.diff(mapping)>=0).all())
            self.assertLessEqual(abs(mapping/25-np.arange(s['frames'])/30).max(),.04)
        with self.assertRaises(ValueError):timestamp_map(0,2)

    def test_source_cluster_uncertainty_does_not_treat_duplicate_clips_as_independent(self):
        ids=['source_A_000001_000002.'+str(i) for i in range(20)]+['source_B_000001_000002.'+str(i) for i in range(20)]
        values=[-1.]*20+[1.]*20
        result=cluster_bootstrap(ids,values)
        self.assertEqual(result['source_groups'],2)
        self.assertEqual(result['clips'],40)
        self.assertEqual(result['ci95'],[-1.,1.])
        self.assertEqual(result,cluster_bootstrap(ids,values))

    def test_unsupported_clips_are_excluded_with_coverage(self):
        result=cluster_bootstrap(['a_1_2.0','b_1_2.0','c_1_2.0'],[None,2.,2.])
        self.assertEqual((result['mean'],result['ci95'],result['clips']),(2.,[2.,2.],2))
        self.assertIsNone(cluster_bootstrap(['a_1_2.0'],[None])['mean'])

    def test_sealed_old_results_are_preserved(self):
        p=read(WORK/'avatarforcing/protocol.json')
        for path,h in p['old_results'].items():self.assertEqual(sha(path),h,path)
        self.assertEqual(sha(DY/'protocol.json'),p['source_protocol_sha256'])

    def test_concurrent_prefetch_and_controller_launch_only_one_metric_job(self):
        calls=[];barrier=threading.Barrier(2)
        def child(argv,env,stdout,stderr,check):
            calls.append(argv)
            time.sleep(.1)
            stdout.write('psnr: 20.0\nssim: 0.75\n')
        def score():
            barrier.wait();return evaluator.score_image('base_metrics')
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(evaluator,'AF',Path(tmp)),mock.patch.object(evaluator.subprocess,'run',child):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a,b=[f.result() for f in [pool.submit(score),pool.submit(score)]]
            self.assertEqual(len(calls),1)
            self.assertEqual(a,b)
            self.assertEqual(a['values'],{'psnr':20.,'ssim':.75})


if __name__=='__main__':unittest.main()
