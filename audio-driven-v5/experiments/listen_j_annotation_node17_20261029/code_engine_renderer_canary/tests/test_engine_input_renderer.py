import asyncio
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from listen_benchmark.engine_input_renderer import EngineInputRenderer, generate_engine_inputs, estimate_render_frames, copy_video_metadata


class FakeRenderer:
    model_config = SimpleNamespace(renderer_num_workers=2,
        multimodal_config=SimpleNamespace(mm_processor_cache_gb=0))

    def __init__(self):
        self.threads = []
        self.finished = []

    async def render_cmpl_async(self, entries):
        self.threads.append(threading.get_ident())
        await asyncio.sleep(entries[0]['delay'])
        self.finished.append(entries[0]['name'])
        return [{'type': 'multimodal', 'name': entry['name']} for entry in entries]


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.version = patch('listen_benchmark.engine_input_renderer.version', return_value='0.19.0')
        self.version.start()
        self.addCleanup(self.version.stop)

    def test_out_of_order_cpu_completion_preserves_each_source_and_closes(self):
        renderer = FakeRenderer()
        pool = EngineInputRenderer(renderer)
        a = {'full': {'input': ({'name': 'a', 'delay': .04}, {'evidence': 'a'})}}
        b = {'full': {'input': ({'name': 'b', 'delay': 0}, {'evidence': 'b'})}}
        try:
            pending_a, pending_b = pool.submit(a), pool.submit(b)
            pool.attach(pending_a)
            pool.attach(pending_b)
            self.assertEqual(renderer.finished, ['b', 'a'])
            self.assertEqual(a['full']['engine_input']['name'], 'a')
            self.assertEqual(b['full']['engine_input']['name'], 'b')
            self.assertEqual(a['full']['input'][1], {'evidence': 'a'})
            self.assertTrue(all(t != threading.get_ident() for t in renderer.threads))
        finally:
            pool.close()
        self.assertFalse(pool.thread.is_alive())
        self.assertTrue(pool.loop.is_closed())
        pool.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            pool.submit(a)

    def test_incomplete_renderer_output_is_not_attached(self):
        renderer = FakeRenderer()
        async def incomplete(entries):
            return []
        renderer.render_cmpl_async = incomplete
        pool = EngineInputRenderer(renderer)
        prepared = {'full': {'input': ({'name': 'a'}, {})}}
        try:
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                pool.attach(pool.submit(prepared))
            self.assertNotIn('engine_input', prepared['full'])
        finally:
            pool.close()

    def test_thread_unsafe_cache_and_unknown_version_are_rejected(self):
        renderer = FakeRenderer()
        renderer.model_config = SimpleNamespace(renderer_num_workers=2,
            multimodal_config=SimpleNamespace(mm_processor_cache_gb=4))
        with self.assertRaisesRegex(ValueError, 'cache_gb=0'):
            EngineInputRenderer(renderer)
        with patch('listen_benchmark.engine_input_renderer.version', return_value='0.20.0'):
            with self.assertRaisesRegex(RuntimeError, '0.19.0'):
                EngineInputRenderer(FakeRenderer())

    def test_engine_submission_uses_original_order_and_parameters(self):
        calls = []
        llm = SimpleNamespace(_render_and_run_requests=lambda **kw: calls.append(kw) or 'outputs')
        inputs = [{'type': 'multimodal', 'id': 2}, {'type': 'multimodal', 'id': 1}]
        params = [object(), object()]
        with patch.dict(sys.modules, {'vllm.outputs': SimpleNamespace(RequestOutput=object)}):
            self.assertEqual(generate_engine_inputs(llm, inputs, params), 'outputs')
        self.assertEqual(list(calls[0]['prompts']), inputs)
        self.assertIs(calls[0]['params'], params)
        self.assertFalse(calls[0]['use_tqdm'])

    def test_memory_estimate_counts_every_overlapping_window_and_padding(self):
        def row(duration):
            return {'sample_id':'sample','source':{'source_start_s':0.,'source_end_s':duration,
                                                  'normalized_audio':{'sample_rate':48000}}}
        self.assertEqual(estimate_render_frames([row(7)]),176)
        self.assertEqual(estimate_render_frames([row(.125)]),2)
        self.assertGreater(estimate_render_frames([row(32)]*20),12288)

    def test_metadata_padding_cannot_mutate_source_evidence(self):
        frames=object()
        entry={'prompt':'unchanged','multi_modal_data':{'video':[(frames,{'frames_indices':[0,3,6]})]},
               'mm_processor_kwargs':{'size':{'shortest_edge':65536}}}
        copied=copy_video_metadata(entry)
        copied['multi_modal_data']['video'][0][1]['frames_indices'].append(6)
        copied['mm_processor_kwargs']['size']['shortest_edge']=1
        self.assertIs(copied['multi_modal_data']['video'][0][0],frames)
        self.assertEqual(entry['multi_modal_data']['video'][0][1]['frames_indices'],[0,3,6])
        self.assertEqual(entry['mm_processor_kwargs']['size']['shortest_edge'],65536)


if __name__ == '__main__':
    unittest.main()
