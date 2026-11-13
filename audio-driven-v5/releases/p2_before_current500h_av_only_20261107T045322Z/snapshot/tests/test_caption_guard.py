from concurrent.futures import Future
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from listen_benchmark.caption_guard import (
    PREPARE_LIMIT_S,
    archive_status,
    caption_file,
    clip_identity,
    completion_rows,
    due_prepares,
    ordered_prefetch_shutdown,
    reaction_cache_problems,
    reconcile_completion,
    record_stage_timeout,
    release_after_children_exit,
    republish_caption_jobs,
    write_completion_status,
)
from listen_benchmark.j_annotation import JobQueue, result_path, write
from listen_benchmark.j_omni_crop import REVISION


def source():
    return {
        'pair': 'pair',
        'crop_proposal': {'revision': REVISION},
        'video_transform': {'crop': [1, 2, 3, 4]},
    }


class TestCaptionGuard(unittest.TestCase):
    def test_old_cache_stays_when_the_identity_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            sid = 'clip-a'
            old = result_path(work, 'caption', sid)
            write(old, {'caption': 'previous', 'cache_identity': {'schema': 'stale'}})
            current = caption_file(work, source(), 'measurement-a', sid)
            other = caption_file(work, source(), 'measurement-b', sid)
            self.assertNotEqual(current, old)
            self.assertNotEqual(current, other)
            write(current, {'cache_identity': clip_identity(source(), 'measurement-a'), 'caption': 'new'})
            self.assertEqual(read_caption(old)['caption'], 'previous')
            self.assertFalse(other.exists())

    def test_failed_samples_are_not_a_successful_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows = [
                {'sample_id': 'a', 'source': source()},
                {'sample_id': 'a', 'source': source()},
                {'sample_id': 'b', 'source': source()},
            ]
            status = reconcile_completion(work, rows)
            self.assertEqual(status['status'], 'completed_with_failures')
            self.assertEqual(status['duplicate_ids'], ['a'])
            self.assertIn('b', status['missing_ids'])
            self.assertNotEqual(status['status'], 'succeeded')

            passed = {'sample_id': 'c', 'source': source()}
            path = caption_file(work, passed['source'], None, 'c')
            write(path, {
                'cache_identity': clip_identity(passed['source'], None),
                'generation_checks_passed': True,
                'review_flags': [],
            })
            clean = reconcile_completion(work, [passed])
            self.assertEqual(clean['status'], 'succeeded')
            self.assertEqual(clean['counts']['pass'], 1)

    def test_timeout_saves_a_diagnostic_before_isolating_the_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self.assertEqual(record_stage_timeout(work, 'clip-a', 'prepare', {'error': 'slow'}, limit=2), 'restart')
            self.assertFalse(result_path(work, 'failed', 'clip-a').exists())
            self.assertEqual(record_stage_timeout(work, 'clip-a', 'prepare', {'error': 'slow'}, limit=2), 'isolate')
            self.assertTrue(result_path(work, 'failed', 'clip-a').exists())
            self.assertEqual(len(list((work / 'diagnostics').glob('clip-a-prepare-*.json'))), 2)

    def test_prefetch_releases_the_claim_only_after_settle_and_can_resume(self):
        log = []

        def stop_reader():
            log.append('stop')

        def settle():
            log.append('settle')

        def release():
            self.assertEqual(log, ['stop', 'settle'])
            log.append('release')

        ordered_prefetch_shutdown(stop_reader, settle, release)
        self.assertEqual(log, ['stop', 'settle', 'release'])

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write(work / 'jobs' / 'pair.json', {'job_id': 'pair', 'rows': []})
            (work / 'ready.jsonl').write_text('{"job_id": "pair"}\n')
            queue = JobQueue(work, 'caption')
            job, handle = queue.claim('worker-a')
            self.assertEqual(job, 'pair')
            self.assertIsNone(queue.claim('worker-b'))
            handle.close()
            resumed, resumed_handle = queue.claim('worker-b')
            self.assertEqual(resumed, 'pair')
            resumed_handle.close()

    def test_a_prepare_future_aged_1000_seconds_is_due_immediately(self):
        fut = Future()
        pending = {fut: {'submitted_at': time.monotonic() - 1000, 'seq': 1}}
        started = time.monotonic()
        found = due_prepares(pending, time.monotonic(), PREPARE_LIMIT_S)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(found, [fut])
        self.assertFalse(fut.done())

    def test_locks_release_only_after_the_child_exits(self):
        proc = multiprocessing.Process(target=time.sleep, args=(30,))
        proc.start()
        released = []

        def release():
            self.assertFalse(proc.is_alive())
            released.append('release')

        still = release_after_children_exit([proc], release, grace_s=3)
        self.assertEqual(still, [])
        self.assertEqual(released, ['release'])
        self.assertFalse(proc.is_alive())

    def test_fixed_manifest_keeps_samples_that_never_entered_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            kept = {'sample_id': 'kept', 'source': source()}
            write(work / 'jobs' / 'pair.json', {'job_id': 'pair', 'rows': [kept]})
            path = caption_file(work, kept['source'], None, 'kept')
            write(path, {
                'cache_identity': clip_identity(kept['source'], None),
                'generation_checks_passed': True,
                'review_flags': [],
            })
            (work / 'fixed_manifest.jsonl').write_text(
                '{"sample_id": "kept", "source": %s}\n{"sample_id": "never-queued"}\n'
                % __import__('json').dumps(kept['source'])
            )
            rows = completion_rows(work)
            self.assertEqual([row['sample_id'] for row in rows], ['kept', 'never-queued'])
            status = write_completion_status(work)
            self.assertEqual(status['status'], 'completed_with_failures')
            self.assertEqual(status['missing_ids'], ['never-queued'])
            self.assertEqual(status['counts']['pass'], 1)

    def test_stale_caption_receipt_is_republished_without_touching_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            row = {'sample_id': 'clip-a', 'source': source()}
            write(work / 'jobs' / 'pair.json', {'job_id': 'pair', 'rows': [row]})
            write(work / 'reaction_cache' / 'keep.json', {'events': []})
            (work / 'ready.jsonl').write_text('{"job_id": "pair"}\n')
            queue = JobQueue(work, 'caption')
            job, handle = queue.claim('worker-a')
            queue.finish(job, receipt={'job_id': job, 'clips': 1})
            handle.close()
            self.assertIsNone(queue.claim('worker-b'))
            self.assertEqual(republish_caption_jobs(work), ['pair'])
            self.assertTrue((work / 'reaction_cache' / 'keep.json').exists())
            self.assertTrue((work / 'caption' / 'completed' / 'pair.json.stale').exists())
            self.assertFalse((work / 'caption' / 'completed' / 'pair.json').exists())
            again, again_handle = queue.claim('worker-b')
            self.assertEqual(again, 'pair')
            again_handle.close()

    def test_reaction_cache_rejects_a_wrong_model_hash(self):
        problems = reaction_cache_problems({
            'parameters': {
                'headshake_direction_gate': {'min_yaw_range_deg': 0},
                'expression_gate_v3': {},
            },
            'model_sha256': '0' * 64,
        })
        self.assertIn('headshake_gate', problems)
        self.assertIn('expression_gate', problems)
        self.assertIn('model_sha256', problems)
        live = next(
            (ROOT / 'experiments/listen_j_reaction_vocal_omni_20261103/reaction_cache/reaction').rglob('*.json'),
            None,
        )
        if live is not None:
            self.assertEqual(reaction_cache_problems(__import__('json').loads(live.read_text())), [])

    def test_archive_status_is_separate_from_workspace_drift(self):
        report = archive_status(
            ROOT / 'releases/listen_j_nod_omni_caption_v1_20261103',
            ROOT,
        )
        self.assertTrue(report['archive_unchanged'])
        self.assertEqual(report['archive_mismatches'], [])
        self.assertIn('workspace_consistent', report)
        self.assertIn('workspace_mismatches', report)


def read_caption(path):
    import json
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    unittest.main()
