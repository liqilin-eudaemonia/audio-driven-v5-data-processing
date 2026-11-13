"""CPU regression tests for batching, restart safety and bounded prefetch."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import listen_visual_native8_nod as runner
from listen_benchmark.full_raw import views
from listen_benchmark.visual_schedule import assign_rows, estimated_work, prefetched, window_batches


def row(sid, duration):
    return {'sample_id': sid, 'source': {'source_start_s': .123,
            'source_end_s': .123 + duration, 'normalized_audio': {'sample_rate': 48000}}}


class FakeEngine:
    def __init__(self, fail_call=None, truncated=False):
        self.calls = []
        self.fail_call = fail_call
        self.truncated = truncated

    def generate(self, entries, params, use_tqdm):
        self.calls.append((entries, params))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated interruption')
        outputs = []
        for entry, duration in zip(entries, params):
            assert entry['end'] - entry['start'] == duration
            text = json.dumps({'events': [{'start': 0, 'end': min(.2, duration),
                'modality': 'visual', 'behavior': entry['view']}]})
            outputs.append(SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
                text=text, token_ids=[3], finish_reason='length' if self.truncated else 'stop')]))
        return outputs


def fake_prepare(processor, whole, start, end, view):
    return {'view': view, 'start': start, 'end': end}, {'prompt': f'{view}:{end-start}'}


class SchedulingTests(unittest.TestCase):
    def test_balanced_complete_deterministic_and_less_skewed(self):
        rows = [row(str(i), 20 if i % 3 == 0 else 3) for i in range(18)]
        balanced = assign_rows(rows, 3)
        self.assertEqual(balanced, assign_rows(rows, 3))
        self.assertEqual(sorted(r['sample_id'] for part in balanced for r in part),
                         sorted(r['sample_id'] for r in rows))
        load = lambda parts: max(sum(estimated_work(r) for r in part) for part in parts)
        self.assertLess(load(balanced), load(assign_rows(rows, 3, 'round-robin')))

    def test_frame_budget_preserves_long_request(self):
        windows = views(32)
        batches = list(window_batches(windows, 4, 256))
        self.assertEqual(batches[0], [('full', 0., 32)])
        self.assertCountEqual([w for batch in batches for w in batch], windows)
        self.assertTrue(all(len(batch) <= 4 for batch in batches))

    def test_short_clip_is_not_duplicated(self):
        self.assertEqual(list(window_batches(views(2.1), 4)), [[('full', 0., 2.1)]])

    def test_prefetch_overlaps_and_uses_one_reader(self):
        next_started = threading.Event()
        def load(item):
            if item == 1:
                next_started.set()
            return item * 10
        stream = prefetched(range(3), load, 1)
        try:
            self.assertEqual(next(stream), (0, 0))
            self.assertTrue(next_started.wait(2), 'Next decode did not start during consumer work')
            self.assertEqual(list(stream), [(1, 10), (2, 20)])
        finally:
            stream.close()

    def test_prefetch_failure_propagates(self):
        def load(item):
            if item == 1:
                raise ValueError('bad source')
            return item
        with self.assertRaisesRegex(ValueError, 'bad source'):
            list(prefetched(range(3), load, 2))
        self.assertFalse(any(t.name.startswith('visual-decode') for t in threading.enumerate()))

    def test_prefetch_disabled(self):
        self.assertEqual(list(prefetched(range(3), lambda x: x, 0)), [(0, 0), (1, 1), (2, 2)])


class BatchResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = SimpleNamespace(output=Path(self.tmp.name), batch_size=4, batch_frame_budget=256)
        self.whole = {8: {'duration_s': 7.}}
        self.patch = patch('listen_benchmark.visual_native8.prepare_input', side_effect=fake_prepare)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def run_sample(self, engine):
        return runner.annotate_sample(self.args, None, engine, row('sample', 7), self.whole, lambda d: d)

    def test_batched_and_serial_have_same_raw_events_and_sources(self):
        batched = self.run_sample(FakeEngine())
        self.assertEqual(batched['performance']['batch_sizes'], [4, 2])
        (self.args.output / 'partial/sample.json').unlink()
        self.args.batch_size = 1
        serial = self.run_sample(FakeEngine())
        for a, b in zip(batched['records'], serial['records']):
            for key in ('view', 'start', 'end', 'raw', 'parsed', 'status', 'prompt'):
                self.assertEqual(a[key], b[key])
        self.assertEqual([r['view'] for r in batched['records']], [w[0] for w in views(7)])

    def test_interrupted_batch_resumes_only_missing_windows(self):
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.run_sample(FakeEngine(fail_call=2))
        self.assertFalse((self.args.output / 'results/sample.json').exists())
        partial = json.loads((self.args.output / 'partial/sample.json').read_text())
        self.assertEqual(len(partial['records']), 4)
        engine = FakeEngine()
        result = self.run_sample(engine)
        self.assertEqual(result['performance']['resumed_windows'], 4)
        self.assertEqual(result['performance']['new_windows'], 2)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len({r['view'] for r in result['records']}), 6)
        again = FakeEngine()
        self.run_sample(again)
        self.assertFalse(again.calls)

    def test_truncated_responses_are_retained_as_invalid(self):
        result = self.run_sample(FakeEngine(truncated=True))
        self.assertTrue(all(r['status'] == 'invalid' and r['raw'] for r in result['records']))

    def test_bad_checkpoint_is_rejected_before_inference(self):
        self.run_sample(FakeEngine())
        path = self.args.output / 'partial/sample.json'
        saved = json.loads(path.read_text())
        saved['records'][0]['start'] = .5
        path.write_text(json.dumps(saved))
        engine = FakeEngine()
        with self.assertRaisesRegex(ValueError, 'window mismatch'):
            self.run_sample(engine)
        self.assertFalse(engine.calls)

    def test_incomplete_batch_does_not_checkpoint_or_publish(self):
        engine = FakeEngine()
        engine.generate = lambda *args, **kwargs: []
        with self.assertRaisesRegex(RuntimeError, 'Incomplete inference batch'):
            self.run_sample(engine)
        self.assertFalse((self.args.output / 'partial/sample.json').exists())


if __name__ == '__main__':
    unittest.main()
