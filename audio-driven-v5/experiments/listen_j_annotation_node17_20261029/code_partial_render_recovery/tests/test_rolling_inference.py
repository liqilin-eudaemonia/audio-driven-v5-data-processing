import itertools
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.rolling_inference import RollingBatchEngine, VllmStepDriver, rolling_groups


class Driver:
    def __init__(self):
        self.counter = itertools.count();self.active = {};self.added = [];self.aborted = []

    def add(self, prepared, params, engine_inputs):
        ids = []
        for entry, _ in prepared:
            key = str(next(self.counter));ids.append(key)
            self.added.append((entry, tuple(x[0] for x in self.active.values())))
            self.active[key] = [entry, 30 if entry == 'slow' else 1]
        return ids

    def step(self):
        time.sleep(.001)  # Simulate blocking engine output and let producers run.
        result = []
        for key, state in list(self.active.items()):
            state[1] -= 1
            if state[1] == 0:
                del self.active[key]
                result.append(SimpleNamespace(request_id=key, finished=True, value=state[0]))
        return result

    def abort(self, ids):
        self.aborted.extend(ids)
        for key in ids:self.active.pop(key, None)


def submit(engine, names, counts=None):
    return engine.submit_inference([(x, {}) for x in names], [None] * len(names),
                                   [None] * len(names), counts or [2] * len(names))


class RollingTests(unittest.TestCase):
    def wait_queued(self, engine, count):
        deadline = time.monotonic() + 2
        while len(engine.waiting) < count and time.monotonic() < deadline:time.sleep(.001)
        self.assertEqual(len(engine.waiting), count)

    def drain(self, engine, futures):
        deadline = time.monotonic() + 2
        while not all(f.done() for f in futures):
            self.assertLess(time.monotonic(), deadline)
            engine.step();time.sleep(.001)

    def test_refill_preserves_batch_order_and_both_budgets(self):
        driver = Driver();engine = RollingBatchEngine(driver, 3, 6)
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(submit, engine, ['slow', 'fast']);self.wait_queued(engine, 1)
            b = pool.submit(submit, engine, ['new']);self.wait_queued(engine, 2)
            try:
                self.drain(engine, [a, b])
                self.assertEqual([r.value for r in a.result()], ['slow', 'fast'])
                self.assertEqual([r.value for r in b.result()], ['new'])
                self.assertIn(('new', ('slow',)), driver.added)
                self.assertEqual(engine.refills_while_busy, 1)
                self.assertLessEqual(engine.peak_requests, 3)
                self.assertLessEqual(engine.peak_frames, 6)
            finally:engine.close()

    def test_frame_budget_defers_next_batch(self):
        driver = Driver();engine = RollingBatchEngine(driver, 3, 4)
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(submit, engine, ['slow'], [4]);self.wait_queued(engine, 1)
            b = pool.submit(submit, engine, ['new'], [2]);self.wait_queued(engine, 2)
            try:self.drain(engine, [a, b])
            finally:engine.close()
        self.assertIn(('new', ()), driver.added)
        self.assertEqual(engine.refills_while_busy, 0)

    def test_close_unblocks_active_and_queued_waiters(self):
        driver = Driver();engine = RollingBatchEngine(driver, 1, 2)
        with ThreadPoolExecutor(2) as pool:
            a = pool.submit(submit, engine, ['slow']);self.wait_queued(engine, 1)
            b = pool.submit(submit, engine, ['new']);self.wait_queued(engine, 2)
            engine.step();engine.close()
            for f in (a, b):
                with self.assertRaisesRegex(RuntimeError, 'closed'):f.result(timeout=2)
        self.assertEqual(driver.aborted, ['0'])
        with self.assertRaisesRegex(RuntimeError, 'closed'):submit(engine, ['later'])

    def test_invalid_capacity_and_off_thread_engine_calls(self):
        engine = RollingBatchEngine(Driver(), 2, 4)
        with self.assertRaisesRegex(ValueError, 'capacity'):submit(engine, ['x'], [5])
        with ThreadPoolExecutor(1) as pool:
            with self.assertRaisesRegex(RuntimeError, 'owner'):pool.submit(engine.step).result()
        engine.close()

    def test_group_finalization_in_source_order_despite_out_of_order_results(self):
        driver = Driver();engine = RollingBatchEngine(driver, 4, 8);stop = threading.Event()
        groups = iter([{'i': i} for i in range(3)])
        def annotate(group, loaded, engine):return submit(engine, ['slow' if group['i'] == 0 else str(group['i'])])
        results = list(rolling_groups(groups, lambda g: (g, [g['i']]), annotate, engine, stop))
        self.assertEqual([g['i'] for g, _, _ in results], [0, 1, 2])
        self.assertIn(('1', ('slow',)), driver.added)
        self.assertTrue(engine.closed)

    def test_stop_drains_started_groups_without_admitting_prefetched_group(self):
        driver = Driver();engine = RollingBatchEngine(driver, 4, 8);stop = threading.Event()
        groups = iter([{'i': i} for i in range(10)])
        def annotate(group, loaded, engine):return submit(engine, ['slow' if group['i'] == 0 else str(group['i'])])
        completed = []
        for group, _, _ in rolling_groups(groups, lambda g: (g, [g['i']]), annotate, engine, stop):
            completed.append(group['i']);stop.set()
        self.assertEqual(completed, [0, 1])

    def test_stop_arriving_during_first_load_does_not_admit_partial_group(self):
        engine=RollingBatchEngine(Driver(),4,8);stop=threading.Event();calls=[]
        def load(group):
            stop.set()
            return group,['partially-rendered']
        def annotate(*args):calls.append(args)
        self.assertEqual(list(rolling_groups(iter([1,2]),load,annotate,engine,stop)),[])
        self.assertEqual(calls,[])
        self.assertTrue(engine.closed)

    def test_annotation_failure_closes_pipeline(self):
        engine = RollingBatchEngine(Driver(), 4, 8)
        def fail(*args):raise ValueError('annotation failure')
        with self.assertRaisesRegex(ValueError, 'annotation failure'):
            list(rolling_groups(iter([1, 2]), lambda g: (g, []), fail, engine, threading.Event()))
        self.assertTrue(engine.closed)

    def test_vllm_external_ids_not_randomized_internal_ids(self):
        calls = [];aborts = []
        def add(key, prompt, sampling, **kwargs):
            calls.append((key, prompt, sampling.output_kind));return key + '-randomized'
        driver = object.__new__(VllmStepDriver);driver.final_only = 'FINAL_ONLY'
        driver.llm = SimpleNamespace(request_counter=itertools.count(),
            _resolve_mm_lora=lambda *args: None,
            llm_engine=SimpleNamespace(add_request=add, abort_request=lambda ids, internal: aborts.append((ids, internal))))
        ids = driver.add([('unused', {})] * 2, [SimpleNamespace(), SimpleNamespace()], [{'type': 'a'}, {'type': 'b'}])
        self.assertEqual(ids, ['0', '1'])
        self.assertEqual([x[2] for x in calls], ['FINAL_ONLY', 'FINAL_ONLY'])
        driver.abort(ids);self.assertEqual(aborts, [(['0', '1'], False)])

    def test_vllm_partial_enqueue_failure_aborts_every_attempted_external_id(self):
        aborts=[]
        def add(key,*args,**kwargs):
            if key=='1':raise ValueError('enqueue failure')
            return key+'-internal'
        driver=object.__new__(VllmStepDriver);driver.final_only='FINAL_ONLY'
        driver.llm=SimpleNamespace(request_counter=itertools.count(),_resolve_mm_lora=lambda *args:None,
            llm_engine=SimpleNamespace(add_request=add,abort_request=lambda ids,internal:aborts.append((ids,internal))))
        with self.assertRaisesRegex(ValueError,'enqueue failure'):
            driver.add([('unused',{})]*2,[SimpleNamespace(),SimpleNamespace()],[{'type':'a'},{'type':'b'}])
        self.assertEqual(aborts,[(['0','1'],False)])

    def test_partial_background_render_during_stop_preserves_order_and_metadata(self):
        from listen_benchmark.engine_input_renderer import complete_engine_inputs
        metadata={'frames_indices':[0,1,2]}
        prepared=[({'prompt':str(i),'multi_modal_data':{'video':[('pixels',metadata)]}}, {}) for i in range(3)]
        cached={'type':'cached-middle'};seen=[]
        def render(entries):
            seen.extend(entry['prompt'] for entry in entries)
            for entry in entries:entry['multi_modal_data']['video'][0][1]['frames_indices'].append(2)
            return [{'type':entry['prompt']} for entry in entries]
        original=[None,cached,None]
        out=complete_engine_inputs(SimpleNamespace(render_cmpl=render),prepared,original)
        self.assertEqual(seen,['0','2'])
        self.assertEqual([x['type'] for x in out],['0','cached-middle','2'])
        self.assertIs(out[1],cached)
        self.assertEqual(metadata,{'frames_indices':[0,1,2]})
        self.assertEqual(original,[None,cached,None])

    def test_incomplete_fallback_render_rejected_before_enqueue(self):
        driver=object.__new__(VllmStepDriver);driver.final_only='FINAL_ONLY';calls=[]
        driver.llm=SimpleNamespace(renderer=SimpleNamespace(render_cmpl=lambda entries:[]),
            llm_engine=SimpleNamespace(add_request=lambda *a,**k:calls.append(a)))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            driver.add([({'prompt':'missing'}, {})],[SimpleNamespace()],[None])
        self.assertEqual(calls,[])


if __name__ == '__main__':unittest.main()
