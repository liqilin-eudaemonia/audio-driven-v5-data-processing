"""Replace prompt-dependent conditions while preserving verified teacher/text tensors."""
import json
import os
from pathlib import Path
import time

import torch

from omni_av.data import Observation, TextPlan
from .checkpoint import atomic_json_new, atomic_save
from .connector import ConditionInputs
from .data import contract, extract_condition, load_omni, validate_cached_example


def legacy_contract(current):
    return {k: v for k, v in current.items() if k != 'observation_prompt'}


def validate_payload(payload, example, expected):
    split = example['split']
    if (payload['contract'] != expected or payload['example'] != example
            or payload['split'] != split):
        raise ValueError('Refresh payload identity/contract mismatch')
    validate_cached_example(payload, split)
    condition = ConditionInputs(**payload['condition'])
    condition.validate()
    for mod in ('video', 'audio'):
        target, valid = payload['targets'][mod], payload['targets'][mod + '_valid']
        mask = getattr(condition, mod + '_valid')
        if (target.shape != (*mask.shape, expected['config']['target_dim'])
                or valid.shape != mask.shape or valid.dtype != torch.bool or not valid.any()
                or not torch.isfinite(target).all()):
            raise ValueError('Invalid reusable teacher targets')


def checked_text(source, destination, plan):
    text = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    if (text['prompt'] != plan.render_prompt() or text['video'].shape != (1, 1024, 4096)
            or text['audio'].shape != (1, 1024, 2048)
            or text['mask'].shape != (1, 1024) or text['mask'].dtype != torch.bool
            or not text['mask'].any()
            or not all(torch.isfinite(text[k]).all() for k in ('video', 'audio'))):
        raise ValueError('Invalid reusable Gemma cache')
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise ValueError('Text destination is not the verified source hardlink')
    else:
        os.link(source, destination)


def delete_verified_source(source, source_root, verified_stat):
    source = Path(source)
    if (not source.resolve().is_relative_to(source_root.resolve())
            or source.is_symlink() or not source.name.startswith('features_') or source.suffix != '.pt'):
        raise ValueError('Refuse deletion outside the explicitly selected feature cache')
    now = source.stat()
    if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
            verified_stat.st_dev, verified_stat.st_ino, verified_stat.st_size, verified_stat.st_mtime_ns):
        raise ValueError('Source changed during refresh; refusing deletion')
    source.unlink()


def read_refreshed(path, source, example, source_contract):
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (OSError, RuntimeError, EOFError):
        # Regenerate only a damaged new file, and only if its old source is intact.
        previous = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        validate_payload(previous, example, source_contract)
        damaged = path.with_name(path.name + f'.invalid.{time.time_ns()}')
        path.rename(damaged)
        print(json.dumps(dict(event='quarantined_refresh', path=str(damaged))), flush=True)
        return None


def refresh_job(job, destination, source_root, cfg, delete_old=False):
    torch.set_num_threads(4)
    torch.manual_seed(45)
    destination.mkdir(parents=True, exist_ok=True)
    expected = contract(cfg)
    if job['source_contract'] != legacy_contract(expected):
        raise ValueError('Only the verified legacy prompt contract can be migrated')
    runtime = processor = None
    entries, started = [], time.monotonic()
    for i, item in enumerate(job['items']):
        old, example = item['entry'], item['example']
        obs, plan = Observation(**example['observation']), TextPlan(**example['text_plan'])
        source, text_source = Path(old['features']), Path(old['text'])
        if (old['sample_id'] != obs.sample_id or old['split'] != example['split']
                or not source.resolve().is_relative_to(source_root.resolve())
                or not text_source.resolve().is_relative_to(source_root.resolve())):
            raise ValueError('Source index identity/path mismatch')
        path, text_path = destination / f'features_{i:04d}.pt', destination / f'text_{i:04d}.pt'
        checked_text(text_source, text_path, plan)
        source_stat = source.stat() if source.exists() else None
        previous = None
        result = read_refreshed(path, source, example, job['source_contract'])
        if result is not None:
            validate_payload(result, example, expected)
            if (result.get('cache_refresh', {}).get('source_features') != str(source)
                    or not result['cache_refresh'].get('teacher_tensors_equal_verified')):
                raise ValueError('Resume provenance mismatch')
            event = 'reused_refreshed_condition'
        else:
            if source_stat is None:
                raise ValueError('Neither old nor refreshed feature exists')
            previous = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
            validate_payload(previous, example, job['source_contract'])
            if runtime is None:
                runtime, processor = load_omni(cfg)
            condition = extract_condition(runtime, processor, obs, plan)
            values = {k: v.cpu() for k, v in vars(condition).items()}
            for k in ('video_coordinates', 'audio_coordinates', 'video_valid', 'audio_valid', 'layer_ids'):
                if not torch.equal(values[k], previous['condition'][k]):
                    raise ValueError('Output layout changed; teacher reuse is unsafe')
            result = dict(previous, contract=expected, condition=values, cache_refresh=dict(
                source_features=str(source), source_contract=job['source_contract'],
                source_feature_bytes=source_stat.st_size, teacher_tensors_equal_verified=True,
                text_reused=True, user_message_time_information=False))
            pending = result
            for attempt in range(2):
                atomic_save(path, pending)
                result = read_refreshed(path, source, example, job['source_contract'])
                if result is not None:
                    break
            if result is None:
                raise RuntimeError('Repeated refresh readback failure; intact source retained')
            validate_payload(result, example, expected)
            if (not all(torch.equal(result['condition'][k], v) for k, v in values.items())
                    or not all(torch.equal(result['targets'][k], v) for k, v in previous['targets'].items())
                    or result['window'] != previous['window']):
                # Never leave a failed readback looking like a resumable completed file.
                path.rename(path.with_name(path.name + f'.invalid.{time.time_ns()}'))
                raise ValueError('Refresh readback differs from extracted condition or source targets')
            del condition, values, pending
            event = 'refreshed_condition'
        if source_stat is not None and delete_old:
            if previous is None:
                previous = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
                validate_payload(previous, example, job['source_contract'])
                if not all(torch.equal(result['targets'][k], v) for k, v in previous['targets'].items()):
                    raise ValueError('Resume teacher mismatch; keep source')
            delete_verified_source(source, source_root, source_stat)
        entries.append(dict(features=str(path), text=str(text_path), sample_id=obs.sample_id,
                            split=example['split']))
        print(json.dumps(dict(event=event, index=i, sample_id=obs.sample_id,
            old_deleted=delete_old and not source.exists(), hidden_shape=list(result['condition']['hidden'].shape),
            elapsed_s=time.monotonic() - started)), flush=True)
        del result, previous
    index = dict(contract=expected, split=job['split'], entries=entries)
    index_path = destination / 'cache_index.json'
    if index_path.exists():
        if json.loads(index_path.read_text()) != index:
            raise ValueError('Completed refresh index mismatch')
    else:
        atomic_json_new(index_path, index)
    return index
