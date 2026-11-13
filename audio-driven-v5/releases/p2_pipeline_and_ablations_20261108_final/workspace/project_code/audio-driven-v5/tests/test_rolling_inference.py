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

    def test_four_groups_refill_past_a_long_first_group_without_reordering_or_exceeding_budgets(self):
        deadline=time.monotonic()+2
        class HeldFirstDriver(Driver):
            def step(inner):
                if time.monotonic()>=deadline:raise AssertionError('Fourth group was not admitted')
                if not any(entry=='3' for entry,_ in inner.added):
                    for state in inner.active.values():
                        if state[0]=='slow':state[1]=2
                return super().step()
        driver=HeldFirstDriver();engine=RollingBatchEngine(driver,2,4);progress=[]
        def annotate(group,loaded,proxy):return submit(proxy,['slow' if group==0 else str(group)])
        results=list(rolling_groups(iter(range(6)),lambda g:(g,[g]),annotate,engine,
                                    threading.Event(),max_active_groups=4,progress=progress.append))
        self.assertEqual([g for g,_,_ in results],list(range(6)))
        self.assertTrue(any(entry=='3' and 'slow' in active for entry,active in driver.added))
        self.assertLessEqual(engine.peak_requests,2)
        self.assertLessEqual(engine.peak_frames,4)
        self.assertEqual(max(x['rolling_active_groups'] for x in progress),4)
        self.assertTrue(all(x['rolling_group_capacity']==4 for x in progress))
        self.assertTrue(engine.closed)

    def test_four_group_stop_drains_started_groups_and_does_not_admit_prefetch(self):
        driver=Driver();engine=RollingBatchEngine(driver,4,8);stop=threading.Event();loaded=[]
        def load(g):loaded.append(g);return g,[g]
        def annotate(g,values,proxy):return submit(proxy,['slow' if g==0 else str(g)])
        def progress(row):
            if row['rolling_active_groups']==4:stop.set()
        out=list(rolling_groups(iter(range(10)),load,annotate,engine,stop,
                                max_active_groups=4,progress=progress))
        self.assertEqual([g for g,_,_ in out],[0,1,2,3])
        self.assertLessEqual(len(loaded),5)
        self.assertFalse(any(entry=='4' for entry,_ in driver.added))
        self.assertTrue(engine.closed)

    def test_group_capacity_rejects_unbounded_or_ambiguous_values(self):
        for capacity in [0,1,3,8,True,'4']:
            engine=RollingBatchEngine(Driver(),4,8)
            try:
                with self.subTest(capacity=capacity),self.assertRaisesRegex(ValueError,'capacity'):
                    list(rolling_groups(iter([]),lambda g:(g,[]),lambda *args:None,
                                        engine,threading.Event(),max_active_groups=capacity))
            finally:engine.close()

    def test_stop_drains_started_groups_without_admitting_prefetched_group(self):
        driver = Driver();engine = RollingBatchEngine(driver, 4, 8);stop = threading.Event()
        groups = iter([{'i': i} for i in range(10)])
        def annotate(group, loaded, engine):return submit(engine, ['slow' if group['i'] == 0 else str(group['i'])])
        completed = []
        for group, _, _ in rolling_groups(groups, lambda g: (g, [g['i']]), annotate, engine, stop):
            completed.append(group['i']);stop.set()
        self.assertEqual(completed, [0, 1])

    def test_stop_arriving_during_first_load_does_not_admit_partial_group(self):
        owner=threading.get_ident();release=threading.Event()
        class ControlledStop(threading.Event):
            checks=0
            def is_set(self):
                if threading.get_ident()==owner:
                    self.checks+=1
                    if self.checks==2:
                        release.set()
                        return False  # Stop arrives after this check, during result().
                return super().is_set()
        engine=RollingBatchEngine(Driver(),4,8);stop=ControlledStop();calls=[]
        def load(group):
            self.assertTrue(release.wait(2))
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
