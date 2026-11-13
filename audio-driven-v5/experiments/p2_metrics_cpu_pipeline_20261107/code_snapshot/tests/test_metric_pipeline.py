import sys
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
import importlib.util

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT.parent/'OmniResponse')]
from p2_testsets_eval.metric_pipeline import MetricPipeline
from evaluation.video.paired_metrics.core import pixel_scores


class PipelineTests(unittest.TestCase):
    def runner(self, count=4, fail_load=None, fail_pixel=False):
        rng = np.random.default_rng(137)
        rows = [dict(id=str(i)) for i in range(count)]
        frames = {r['id']: (rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8),
                            rng.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)) for r in rows}
        def load(row):
            if row['id'] == fail_load:
                raise ValueError('decode failed')
            return (*frames[row['id']], dict(id=row['id']))
        def pixel(a, b):
            if fail_pixel:
                raise ValueError('pixel failed')
            return pixel_scores(a, b)
        return SimpleNamespace(read_manifest=lambda _: rows, load_pair=load, pixel_scores=pixel), frames

    def test_native_frame_values_and_order_exact(self):
        runner, frames = self.runner()
        original = runner.load_pair
        pipeline = MetricPipeline(runner, depth=2, pixel_workers=2)
        with pipeline.installed():
            for row in runner.read_manifest(None):
                gt, gen, meta = runner.load_pair(row)
                self.assertEqual(meta['id'], row['id'])
                self.assertIs(gt, frames[row['id']][0])
                for a, b in zip(gt, gen):
                    self.assertEqual(runner.pixel_scores(a, b), pixel_scores(a, b))
        self.assertIs(runner.load_pair, original)
        self.assertTrue(pipeline.report()['closed'])

    def test_load_failure_does_not_strand_remaining_rows(self):
        runner, _ = self.runner(fail_load='1')
        with MetricPipeline(runner).installed():
            for row in runner.read_manifest(None):
                if row['id'] == '1':
                    with self.assertRaisesRegex(ValueError, 'decode failed'):
                        runner.load_pair(row)
                else:
                    self.assertEqual(runner.load_pair(row)[2]['id'], row['id'])

    def test_pixel_failure_stays_in_pixel_stage(self):
        runner, _ = self.runner(count=1, fail_pixel=True)
        with MetricPipeline(runner).installed():
            row = runner.read_manifest(None)[0]
            gt, gen, meta = runner.load_pair(row)
            self.assertEqual(meta['id'], row['id'])
            with self.assertRaisesRegex(ValueError, 'pixel failed'):
                runner.pixel_scores(gt[0], gen[0])

    def test_next_pair_prepares_while_current_is_available(self):
        runner, _ = self.runner(count=3)
        native = runner.load_pair
        second_started = threading.Event()
        def load(row):
            if row['id'] == '1':
                second_started.set()
            return native(row)
        runner.load_pair = load
        with MetricPipeline(runner, depth=1).installed():
            rows = runner.read_manifest(None)
            gt, gen, _ = runner.load_pair(rows[0])
            self.assertTrue(second_started.wait(5), 'next CPU read did not overlap')
            self.assertEqual(runner.pixel_scores(gt[0], gen[0]), pixel_scores(gt[0], gen[0]))

    def test_disabled_pixel_stage_does_not_run_pixels(self):
        runner, _ = self.runner(count=1, fail_pixel=True)
        pipeline = MetricPipeline(runner, compute_pixels=False)
        with pipeline.installed():
            row = runner.read_manifest(None)[0]
            runner.load_pair(row)
        self.assertEqual(pipeline.events[0]['pixel_frames'], 0)
        self.assertIsNone(pipeline.events[0]['pixel_error'])

    def test_unprepared_frame_is_rejected(self):
        runner, _ = self.runner(count=1)
        with MetricPipeline(runner).installed():
            row = runner.read_manifest(None)[0]
            gt, gen, _ = runner.load_pair(row)
            with self.assertRaisesRegex(RuntimeError, 'outside the prepared pair'):
                runner.pixel_scores(gt[0].copy(), gen[0])

    def test_queue_wrapper_only_changes_video_commands(self):
        spec = importlib.util.spec_from_file_location('pipelined_cli', ROOT/'scripts/p2_pipelined_testsets_eval.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ['--manifest', 'same.jsonl', '--batch-size', '64']
        direct = ['python', '-m', 'evaluation.video.paired_metrics.run', *args]
        cached = ['python', str(ROOT/'scripts/p2_cached_video_metrics.py'), '--decoded-gt-cache', 'cache', *args]
        audio = ['python', '-m', 'evaluation.listener_metrics.run', *args]
        self.assertEqual(module.pipelined_command(direct)[4:], args)
        self.assertEqual(module.pipelined_command(cached)[4:], cached[2:])
        self.assertIs(module.pipelined_command(audio), audio)


if __name__ == '__main__':
    unittest.main()
