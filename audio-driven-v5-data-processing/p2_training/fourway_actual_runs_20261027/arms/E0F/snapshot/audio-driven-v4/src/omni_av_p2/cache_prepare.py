"""Resumable cache preparation with bounded concurrent batched Gemma extraction."""
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch

from omni_av.data import Observation, TextPlan, file_hash, gt_text_plan
from omni_av.ltx_runtime import NativeTextConditioner, audit_target_window, load_av_targets
from .checkpoint import atomic_json_new, atomic_save
from .connector import ConditionInputs
from .data import contract, extract_condition, extract_teacher, load_omni, unload, validate_cached_example


def read_complete(path):
    """Preserve an interrupted writer's file for audit; never silently accept a partial tensor."""
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        interrupted = path.with_name(path.name + f'.interrupted.{time.time_ns()}')
        path.rename(interrupted)
        print(json.dumps(dict(event='quarantined_interrupted_file', path=str(interrupted))), flush=True)
        return None


@contextmanager
def text_slot(root, gpu, slots=2):
    directory = root/'gpu_slots'/f'{socket.gethostname()}_gpu{gpu}'
    directory.mkdir(parents=True, exist_ok=True)
    acquired = None
    while acquired is None:
        for i in range(slots):
            stream = (directory/f'text{i}.lock').open('a')
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = stream
                break
            except BlockingIOError:
                stream.close()
        if acquired is None:
            time.sleep(.5)
    try:
        yield
    finally:
        fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
        acquired.close()


def prepare(args, cfg, batch_size=4):
    torch.set_num_threads(4)
    torch.manual_seed(45)
    random.seed(45)
    np.random.seed(45)
    examples = json.loads(args.manifest.read_text())
    if not examples or any(e['split'] != 'train' for e in examples):
        raise ValueError('Expected nonempty Train shard')
    expected_contract = contract(cfg)
    if len({e['observation']['sample_id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate sample IDs')
    entries = []
    runtime = processor = condition = targets = None
    started = time.monotonic()
    for i, example in enumerate(examples):
        path = args.output/f'features_{i:04d}.pt'
        obs = Observation(**example['observation'])
        existing = read_complete(path)
        if existing is not None:
            if existing['contract'] != expected_contract or existing['example'] != example:
                raise ValueError('Existing feature contract or manifest changed')
            validate_cached_example(existing)
            ConditionInputs(**existing['condition']).validate()
            if not all(torch.isfinite(v).all() for v in existing['targets'].values()):
                raise ValueError('Nonfinite existing targets')
            del existing
            event = 'reused_features'
        else:
            if not math.isfinite(example['duration_s']) or not 0 < example['duration_s'] <= obs.output_duration_s:
                raise ValueError('GT must fit the external output budget')
            plan = TextPlan(**example['text_plan'])
            if plan != gt_text_plan(json.loads(Path(example['gt_path']).read_text()), obs.sample_id):
                raise ValueError('Text differs from paired GT')
            if runtime is None:
                runtime, processor = load_omni(cfg)
            av, metadata = load_av_targets(args.data_root, obs.sample_id, device='cpu', split='train')
            del av
            window = audit_target_window(dict(example=example), metadata)
            condition = extract_condition(runtime, processor, obs, plan)
            targets = extract_teacher(runtime, processor, condition, example, window['renderer_target_duration_s'])
            if not all(torch.isfinite(v).all() for v in targets.values()):
                raise ValueError('Nonfinite extracted targets')
            atomic_save(path, dict(contract=expected_contract, split='train', sample_id=obs.sample_id,
                example=example, condition={k:v.cpu() for k,v in vars(condition).items()},
                targets={k:v.cpu() for k,v in targets.items()}, window=window,
                gt_sha256=file_hash(example['gt_path']), source='paired_gt_text_frozen_omni'))
            event = 'p2_features'
        entries.append(dict(features=str(path), sample_id=obs.sample_id, split='train',
                            text=str(args.output/f'text_{i:04d}.pt')))
        print(json.dumps(dict(event=event, index=i, sample_id=obs.sample_id,
                              elapsed_s=time.monotonic()-started)), flush=True)
    del runtime, processor, condition, targets
    unload()
    pending = []
    for i, (example, entry) in enumerate(zip(examples, entries)):
        cached = read_complete(Path(entry['text']))
        if cached is None:
            pending.append(i)
        else:
            if (cached['prompt'] != TextPlan(**example['text_plan']).render_prompt()
                    or cached['video'].shape != (1,1024,4096) or cached['audio'].shape != (1,1024,2048)
                    or not all(torch.isfinite(cached[k]).all() for k in ('video','audio'))):
                raise ValueError('Existing text cache mismatch')
            print(json.dumps(dict(event='reused_text', index=i)), flush=True)
    if pending:
        # Omni is unloaded before waiting: idle queues hold no frozen model on GPU.
        with text_slot(args.output.parents[1], args.gpu):
            text_runtime = NativeTextConditioner(cfg)
            try:
                for start in range(0,len(pending),batch_size):
                    selected = pending[start:start+batch_size]
                    plans = [TextPlan(**examples[i]['text_plan']) for i in selected]
                    texts = text_runtime.encode_batch(plans)
                    for i, text in zip(selected,texts):
                        if not all(torch.isfinite(text[k]).all() for k in ('video','audio')):
                            raise ValueError('Nonfinite batched text cache')
                        atomic_save(Path(entries[i]['text']),
                                    {k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in text.items()})
                        print(json.dumps(dict(event='p2_text', index=i, batch_size=len(selected))), flush=True)
                    del texts, text
            finally:
                text_runtime.close()
    index = dict(contract=expected_contract, split='train', entries=entries)
    atomic_json_new(args.output/'cache_index.json',index)
    atomic_json_new(args.output/'completed.json',dict(prepared=len(entries),text_batch_size=batch_size,
        max_gemma_processes_per_gpu=2,quality_pass=None))
    return index
