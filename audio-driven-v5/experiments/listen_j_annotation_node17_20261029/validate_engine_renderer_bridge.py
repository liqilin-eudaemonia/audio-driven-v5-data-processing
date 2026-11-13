"""Real CPU renderer bridge regression, without constructing a GPU engine."""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import json
import copy
import hashlib
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parents[1] / 'src'))
from benchmark_renderer_parallel import compare


def main():
    os.nice(10)
    import torch
    from transformers import AutoProcessor
    from vllm.config import VllmConfig, ModelConfig, DeviceConfig, CacheConfig
    from vllm.renderers import renderer_from_config
    from listen_benchmark.visual_native8 import MODEL, MM_SIZE, prepare_media
    from listen_benchmark.frame_crop import crop_original_frame
    from listen_benchmark.shared_video_inputs import prepare_shared_inputs
    from listen_benchmark.engine_input_renderer import EngineInputRenderer, copy_video_metadata
    from vllm.multimodal.processing.inputs import ProcessorInputs
    torch.set_num_threads(2)
    def make_renderer(workers):
        model = ModelConfig(model=MODEL, dtype='bfloat16', max_model_len=49152,
            trust_remote_code=True, mm_processor_cache_gb=4 if workers == 1 else 0,
            renderer_num_workers=workers,
            mm_processor_kwargs={'size': MM_SIZE}, limit_mm_per_prompt={'image': 0, 'video': 1})
        return renderer_from_config(VllmConfig(model_config=model, device_config=DeviceConfig(device='cpu'),
                                              cache_config=CacheConfig(enable_prefix_caching=True)))
    baseline_renderer, background_renderer = make_renderer(1), make_renderer(2)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    row = json.loads((WORK / 'jobs/naturalistic_V02_S5078_I00000281.json').read_text())['rows'][0]
    media = prepare_media(row['source'], frame_crop=crop_original_frame)
    prepared = prepare_shared_inputs(processor, media)
    def fingerprints():
        output = []
        for slot in prepared.values():
            entry = slot['input'][0]
            array, metadata = entry['multi_modal_data']['video'][0]
            output.append({'array_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                'metadata': copy.deepcopy(metadata), 'kwargs': copy.deepcopy(entry['mm_processor_kwargs']),
                'prompt': entry['prompt']})
        return output
    original_inputs = fingerprints()
    expected_hashes = []
    for slot in prepared.values():
        entry = slot['input'][0]
        original = copy_video_metadata(entry)
        padded = copy_video_metadata(entry)
        for _, metadata in padded['multi_modal_data']['video']:
            indices = metadata['frames_indices']
            if len(indices) % 2:
                indices.append(indices[-1])
        processor = baseline_renderer.get_mm_processor()
        def hashes(value):
            items = processor.info.parse_mm_data(value['multi_modal_data'])
            return ProcessorInputs(prompt=[], mm_data_items=items,
                hf_processor_mm_kwargs=value['mm_processor_kwargs']).get_mm_hashes(processor.info.model_id)
        expected_hashes.append((hashes(original), hashes(padded)))
    baseline = [baseline_renderer.render_cmpl([copy_video_metadata(slot['input'][0])])[0]
                for slot in prepared.values()]
    after_baseline = fingerprints()
    bridge = EngineInputRenderer(background_renderer)
    try:
        render_s = bridge.attach(bridge.submit(prepared))
        trial = [slot['engine_input'] for slot in prepared.values()]
        compare([{k:v for k,v in item.items() if k != 'mm_hashes'} for item in baseline],
                [{k:v for k,v in item.items() if k != 'mm_hashes'} for item in trial])
        assert all(a['mm_hashes'] == original and b['mm_hashes'] == padded
                   for a,b,(original,padded) in zip(baseline,trial,expected_hashes))
    finally:
        bridge.close()
    assert not bridge.thread.is_alive() and bridge.loop.is_closed()
    report = {'at': time.time(), 'sample_id': row['sample_id'], 'windows': len(prepared),
        'scope': 'Real background-loop renderer bridge; CPU only; production-style cache4 serial versus cache0 parallel on unique windows',
        'engine_content_equal_except_arrival_and_cache_keys': True, 'background_render_s': render_s,
        'mm_hashes_equal': all(a['mm_hashes'] == b['mm_hashes'] for a,b in zip(baseline,trial)),
        'cache_key_comparison': [{'view':view,'serial':a['mm_hashes'],'parallel':b['mm_hashes']}
                                for view,a,b in zip(prepared,baseline,trial)],
        'raw_inputs_unmodified_by_baseline': original_inputs == after_baseline,
        'raw_inputs_unmodified_by_bridge': after_baseline == fingerprints(),
        'cache_key_differences_exactly_explained_by_odd_frame_index_padding': True,
        'actual_prefix_settings': [baseline_renderer.config.cache_config.enable_prefix_caching,
                                   background_renderer.config.cache_config.enable_prefix_caching],
        'prefix_caching': True,
        'thread_exited': True, 'event_loop_closed': True}
    (WORK / 'engine_renderer_bridge_cpu_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    baseline_renderer._executor.shutdown(wait=True)
    background_renderer._executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
