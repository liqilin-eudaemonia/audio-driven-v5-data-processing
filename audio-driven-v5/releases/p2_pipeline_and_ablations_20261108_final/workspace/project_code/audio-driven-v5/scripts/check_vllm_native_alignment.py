#!/usr/bin/env python3
"""CPU-only regression for vLLM 0.19 Qwen3-Omni cached AV token ordering.

Run in the benchmark's vLLM environment; no model weights or GPU inference
are loaded. Synthetic media only tests preprocessing, not event accuracy.
"""
import argparse
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/private/ckpts/Qwen3-Omni/Instruct')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import numpy as np
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext

    config = ModelConfig(model=args.model, trust_remote_code=True,
                         dtype='bfloat16', max_model_len=49152)
    processor = MULTIMODAL_REGISTRY.create_processor(
        config, cache=MultiModalProcessorOnlyCache(config))
    hf = processor.info.get_hf_processor()
    prompt = hf.apply_chat_template([dict(role='user', content=[
        dict(type='text', text='Describe.'), dict(type='video', video='decoded')])],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    items = processor.info.parse_mm_data(dict(
        video=[np.zeros((16, 56, 56, 3), dtype=np.uint8)],
        audio=[np.zeros(32000, dtype=np.float32)]))
    cfg = processor.info.get_hf_config()
    tokens = {cfg.video_token_id: 'video', cfg.audio_token_id: 'audio'}
    sequences = {}
    report = dict(synthetic=True, gpu_inference=False, frames=16, fps=8,
                  audio_seconds=2, temporal_patch_size=hf.image_processor.temporal_patch_size,
                  cases={})
    for name, extra in [('old', {}), ('fixed', {
            'second_per_grid_ts': [hf.image_processor.temporal_patch_size / 8]})]:
        opts = dict(use_audio_in_video=True, fps=8, do_sample_frames=False, **extra)
        inputs = ProcessorInputs(prompt, items, hf_processor_mm_kwargs=opts)
        # Direct HF ordering is the reference, before vLLM cache replacements.
        hf_ids, info, _ = processor._apply_hf_processor(inputs, TimingContext(enabled=False))
        hf_sequence = [i for i in hf_ids if i in tokens]
        result = processor.apply(inputs, TimingContext(enabled=False))
        seq = [i for i in result['prompt_token_ids'] if i in tokens]
        sequences[name] = seq
        report['cases'][name] = dict(
            kwargs=opts, matches_hf_order=seq == hf_sequence,
            processed_seconds_per_grid=info.kwargs.get_data()['second_per_grid_ts'].tolist(),
            token_runs=[dict(modality=tokens[k], count=len(list(v)))
                        for k, v in itertools.groupby(seq)])
    assert report['cases']['fixed']['matches_hf_order'], 'Fixed AV ordering differs from HF'
    assert not report['cases']['old']['matches_hf_order'], 'Old mismatch no longer reproduces; reassess version'
    assert sorted(sequences['old']) == sorted(sequences['fixed']), 'Unexpected token count change'
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('PASS: old cached order differs; explicit grid time matches HF; token counts unchanged')


if __name__ == '__main__':
    main()
