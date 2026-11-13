"""Free generation from source inputs, with a saved planning adapter or base model."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    import torch
    from p2_listen.contracts import read, atomic_json
    from p2_planning_sft.data import PlanningDataset, PlanningCollator, SYSTEM, USER
    from p2_planning_sft.model import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--split', choices=['train','dev'], default='dev')
    parser.add_argument('--limit', type=int, default=24, help='Reference-data generation limit; never loads the full dataset into RAM')
    parser.add_argument('--audio-ablation', choices=['none','silence'], default='none')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing generation results')
    c = read(args.config)
    torch.set_num_threads(c.get('cpu_threads', 2))
    torch.cuda.set_device(0)
    if os.environ.get('PLANNING_MEMORY_GIB'):
        torch.cuda.set_per_process_memory_fraction(float(os.environ['PLANNING_MEMORY_GIB'])*2**30 /
                                                  torch.cuda.get_device_properties(0).total_memory)
    if args.adapter:
        model = build_model(c, adapter=args.adapter, trainable=False)
    else:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(c['model'],
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.cuda().eval().requires_grad_(False)
    collator = PlanningCollator(c['model'], c['max_length'], c.get('max_pixels',50176))
    processor = collator.get_processor()
    processor.tokenizer.padding_side = 'left'
    rows = []
    if c.get('data',{}).get('mode') == 'reference_jsonl':
        from p2_planning_sft.reference_data import ReferencePlanningDataset
        ds = ReferencePlanningDataset(c['data']['seal'], args.split, limit=args.limit, seed=c['seed'])
        rows = [ds[i] for i in range(len(ds))]
    else:
        for split in ('train','dev'):
            ds = PlanningDataset(c['review_cache'], split)
            rows.extend(ds[i] for i in range(len(ds)))
    if args.audio_ablation == 'silence':
        import numpy as np
        rows = [dict(row, audio=np.zeros_like(row['audio'])) for row in rows]
    eos = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    results = []
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        selected = rows[offset:offset+args.batch_size]
        inputs = collator.source_inputs(selected).to('cuda', dtype=torch.bfloat16)
        assert 'labels' not in inputs
        base = model.get_base_model() if args.adapter else model
        base.rope_deltas = None
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                use_cache=True, use_audio_in_video=False, eos_token_id=eos,
                pad_token_id=processor.tokenizer.pad_token_id)
        for row, ids in zip(selected, generated[:, inputs.input_ids.shape[1]:].tolist()):
            ended = eos in ids
            ids = ids[:ids.index(eos)+1] if ended else ids
            results.append(dict(sample_id=row['sample_id'], split=row['split'],
                text=processor.tokenizer.decode(ids, skip_special_tokens=True),
                token_ids=ids, ended_with_eos=ended))
    atomic_json(output, dict(model=c['model'], adapter=args.adapter, system=SYSTEM, user=USER,
        seconds=time.monotonic()-started, results=results, target_in_prompt=False,
        audio_ablation=args.audio_ablation))


if __name__ == '__main__':
    main()
