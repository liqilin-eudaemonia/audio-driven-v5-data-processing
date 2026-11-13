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
from listen_benchmark.visual_batch import annotate_group
from listen_benchmark.visual_media_cache import cached_media


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

    def test_cross_clip_batch_has_no_source_mix_and_resumes(self):
        whole={8:{'duration_s':7.,'images':[None]*56}}
        def prepare(processor,media,start,end,view):
            return fake_prepare(processor,media,start,end,view)
        with patch('listen_benchmark.visual_batch.prepare_input',side_effect=prepare):
            engine=FakeEngine(fail_call=2)
            inputs=[(row('a',7),whole),(row('b',7),whole)]
            def run(engine):
                return annotate_group(inputs,None,engine,lambda d:d,self.args.output,8,1024,runner.save,'hash')
            with self.assertRaisesRegex(RuntimeError,'interruption'):run(engine)
            first={sid:json.loads((self.args.output/f'partial/{sid}.json').read_text()) for sid in ('a','b')}
            self.assertEqual(sum(len(x['records']) for x in first.values()),8)
            engine=FakeEngine();results=run(engine)
            self.assertEqual(sum(r['performance']['new_windows'] for r in results),4)
            for result in results:
                self.assertEqual([r['view'] for r in result['records']],[w[0] for w in views(7)])
                self.assertTrue(all(r['parsed']['events'][0]['behavior']==r['view'] for r in result['records']))
            self.assertEqual(len(engine.calls),1)

    def test_exact_media_cache_reuses_and_invalidates_source(self):
        archive=self.args.output/'source.tar';archive.write_bytes(b'fixture')
        source=dict(row('a',7)['source'],source={'archive':str(archive)})
        calls=[]
        def decode():
            calls.append(1)
            return {8:{'images':[b'exact pixels'],'metadata':'unaltered'}},{'decode_s':1,'decode_lock_wait_s':0}
        a,ma=cached_media(source,self.args.output/'cache',decode)
        b,mb=cached_media(source,self.args.output/'cache',decode)
        self.assertEqual(a,b);self.assertEqual(len(calls),1);self.assertTrue(mb['media_cache_hit'])
        archive.write_bytes(b'changed source fixture')
        cached_media(source,self.args.output/'cache',decode)
        self.assertEqual(len(calls),2)

    def test_precompiled_windows_preserve_input_and_bypass_foreground_builder(self):
        whole={8:{'duration_s':7.,'images':[None]*56}}
        whole['_prepared_inputs']={view:{'interval':(start,end),
            'input':fake_prepare(None,whole,start,end,view)} for view,start,end in views(7)}
        engine=FakeEngine()
        with patch('listen_benchmark.visual_batch.prepare_input',side_effect=AssertionError('foreground prepare')):
            results=annotate_group([(row('a',7),whole)],None,engine,lambda d:d,
                self.args.output,512,16384,runner.save,'hash')
        self.assertEqual(len(results[0]['records']),len(views(7)))
        self.assertTrue(all(r['parsed']['events'][0]['behavior']==r['view'] for r in results[0]['records']))

    def test_precompiled_mismatched_window_is_rejected(self):
        whole={8:{'duration_s':7.,'images':[None]*56},
            '_prepared_inputs':{'full':{'interval':(0.,6.),'input':None}}}
        engine=FakeEngine()
        with self.assertRaisesRegex(ValueError,'Precompiled input interval mismatch'):
            annotate_group([(row('a',7),whole)],None,engine,lambda d:d,
                self.args.output,512,16384,runner.save,'hash')
        self.assertFalse(engine.calls)

    def test_rolling_cross_group_preserves_windows_raw_text_and_resume(self):
        from listen_benchmark.rolling_inference import RollingBatchEngine,rolling_groups
        from test_rolling_inference import Driver
        class Responses(Driver):
            def add(self, prepared, params, engine_inputs):
                ids=super().add(prepared,params,engine_inputs)
                for key in ids:
                    if self.active[key][0]['view']=='full':self.active[key][1]=20
                return ids
            def step(self):
                outputs=super().step()
                for out in outputs:
                    entry=out.value
                    text=json.dumps({'events':[{'start':0,'end':.2,'modality':'visual',
                        'behavior':entry['sid']+':'+entry['view']}]})
                    out.prompt_token_ids=[1,2]
                    out.outputs=[SimpleNamespace(text=text,token_ids=[3],
                        finish_reason='length' if entry['sid']=='b' and entry['view']=='full' else 'stop')]
                return outputs
        groups=[]
        for sid,duration in [('a',7.),('b',5.)]:
            whole={8:{'duration_s':duration,'images':[None]*round(duration*8)}}
            whole['_prepared_inputs']={v:{'interval':(s,e),'input':(
                {'sid':sid,'view':v,'start':s,'end':e},{'prompt':sid+':'+v})}
                for v,s,e in views(duration)}
            groups.append([(row(sid,duration),whole)])
        def run(driver):
            engine=RollingBatchEngine(driver,8,512)
            def annotate(group,loaded,proxy):
                return annotate_group(loaded,None,proxy,lambda d:d,self.args.output,
                    8,512,runner.save,'config-hash')
            return list(rolling_groups(iter(groups),lambda g:(g,g),annotate,engine,threading.Event()))
        first=run(Responses())
        for _,loaded,results in first:
            sid=loaded[0][0]['sample_id'];duration=loaded[0][1][8]['duration_s']
            records=results[0]['records']
            self.assertEqual([(r['view'],r['start'],r['end']) for r in records],views(duration))
            for r in records:
                self.assertEqual(r['prompt'],sid+':'+r['view'])
                self.assertEqual(json.loads(r['raw'])['events'][0]['behavior'],sid+':'+r['view'])
                self.assertEqual(r['status'],'invalid' if sid=='b' and r['view']=='full' else 'valid')
        resume=Responses();again=run(resume)
        self.assertFalse(resume.added)
        self.assertEqual([r[2][0]['records'] for r in first],[r[2][0]['records'] for r in again])

    def test_rendered_inputs_route_without_reprocessing_and_keep_windows(self):
        whole={8:{'duration_s':7.,'images':[None]*56}}
        whole['_prepared_inputs']={view:{'interval':(start,end),
            'input':fake_prepare(None,whole,start,end,view),
            'engine_input':{'view':view,'start':start,'end':end}}
            for view,start,end in views(7)}
        backend=FakeEngine()
        with patch('listen_benchmark.engine_input_renderer.generate_engine_inputs',
                   side_effect=lambda llm,entries,params:backend.generate(entries,params,False)) as generate:
            results=annotate_group([(row('a',7),whole)],None,object(),lambda d:d,
                self.args.output,512,16384,runner.save,'hash')
        self.assertEqual(generate.call_count,1)
        self.assertTrue(all(r['parsed']['events'][0]['behavior']==r['view'] for r in results[0]['records']))

    def test_partial_rendering_is_rejected_before_inference(self):
        whole={8:{'duration_s':7.,'images':[None]*56}}
        whole['_prepared_inputs']={view:{'interval':(start,end),
            'input':fake_prepare(None,whole,start,end,view)} for view,start,end in views(7)}
        whole['_prepared_inputs']['full']['engine_input']={'type':'multimodal'}
        engine=FakeEngine()
        with self.assertRaisesRegex(ValueError,'mix rendered'):
            annotate_group([(row('a',7),whole)],None,engine,lambda d:d,
                self.args.output,512,16384,runner.save,'hash')
        self.assertFalse(engine.calls)


if __name__ == '__main__':
    unittest.main()
