"""Bounded CPU renderer bridge for the pinned vLLM 0.19.0 offline API.

Only the background loop renders inputs. Engine submission stays on its owning
thread. Callers bound pending clip futures and retain all source evidence.
"""
import asyncio
import copy
from importlib.metadata import version
import math
import threading
import time


def estimate_render_frames(rows):
    from .full_raw import views
    from .visual_schedule import source_duration
    return sum(2 * math.ceil(math.ceil((end - start) * 8) / 2)
               for row in rows for _, start, end in views(source_duration(row)))


def copy_video_metadata(entry):
    """Protect source evidence from HF's in-place odd-frame index padding."""
    result = dict(entry)
    if 'multi_modal_data' in entry:
        result['multi_modal_data'] = dict(entry['multi_modal_data'])
        if 'video' in entry['multi_modal_data']:
            result['multi_modal_data']['video'] = [
                (frames, copy.deepcopy(metadata))
                for frames, metadata in entry['multi_modal_data']['video']]
    if 'mm_processor_kwargs' in entry:
        result['mm_processor_kwargs'] = copy.deepcopy(entry['mm_processor_kwargs'])
    return result


def complete_engine_inputs(renderer, prepared, engine_inputs):
    """Finish a partially rendered batch without changing its input order.

    Graceful stop can interrupt background pre-rendering. Reuse completed
    inputs and render only missing entries, protecting source metadata as usual.
    """
    if len(prepared) != len(engine_inputs):
        raise ValueError('Prepared/rendered input lengths differ')
    inputs = list(engine_inputs)
    missing = [i for i, entry in enumerate(inputs) if entry is None]
    if missing:
        if renderer is None:
            raise ValueError('Cannot mix rendered and unrendered inputs without a renderer')
        rendered = renderer.render_cmpl([copy_video_metadata(prepared[i][0]) for i in missing])
        if len(rendered) != len(missing):
            raise ValueError('Renderer returned incomplete inputs')
        for index, entry in zip(missing, rendered):
            inputs[index] = entry
    return inputs


def check_vllm_version():
    actual = version('vllm')
    if actual != '0.19.0':
        raise RuntimeError(f'Pre-rendered input adapter requires validated vLLM 0.19.0, got {actual}')


class EngineInputRenderer:
    def __init__(self, renderer):
        check_vllm_version()
        config = renderer.model_config
        if config.multimodal_config.mm_processor_cache_gb != 0:
            raise ValueError('Parallel renderer requires mm_processor_cache_gb=0')
        if config.renderer_num_workers != 2:
            raise ValueError('This rollout validates exactly two renderer workers')
        self.renderer = renderer
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       name='listen-input-renderer', daemon=True)
        self.thread.start()
        self.closed = False

    def submit(self, prepared):
        if self.closed:
            raise RuntimeError('Renderer is closed')
        slots = tuple(prepared.values())
        entries = [copy_video_metadata(slot['input'][0]) for slot in slots]

        async def render():
            started = time.monotonic()
            output = await self.renderer.render_cmpl_async(entries)
            if len(output) != len(slots):
                raise RuntimeError('Incomplete pre-rendered input batch')
            return output, time.monotonic() - started

        return slots, asyncio.run_coroutine_threadsafe(render(), self.loop)

    @staticmethod
    def attach(pending):
        slots, future = pending
        output, wall_s = future.result()
        for slot, engine_input in zip(slots, output):
            if not isinstance(engine_input, dict) or 'type' not in engine_input:
                raise ValueError('Expected processed EngineInput')
            slot['engine_input'] = engine_input
        return wall_s

    def close(self):
        if self.closed:
            return
        self.closed = True

        async def drain():
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() if task is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(drain(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


def generate_engine_inputs(llm, inputs, params):
    """Same vLLM request insertion and result ordering, without rendering twice."""
    check_vllm_version()
    from vllm.outputs import RequestOutput
    if len(inputs) != len(params):
        raise ValueError('Pre-rendered inputs/parameters length mismatch')
    return llm._render_and_run_requests(prompts=iter(inputs), params=params,
                                       output_type=RequestOutput, use_tqdm=False)
