"""Sharded, resumable cache writers; models are loaded only for missing samples."""
from pathlib import Path
import os
import time
import traceback
import torch

from p2_listen.cache import load_part, save_part
from p2_listen.config import activate, extractor_config
from p2_listen.contracts import ROOT, atomic_json, digest, input_identity, observation, sha
from p2_listen.features import unload, weights_identity
from p3_listen.prompts import contract as prompt_contract
from .protocol import open_protocol, sample, select, lock


def asset_identity(path):
    """Match native weight metadata, also bind tokenizer/template/config files."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and
              p.suffix in ('.json', '.safetensors', '.model', '.txt', '.jinja'))
    if not files:
        raise FileNotFoundError(path)
    return {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
                        sha256=sha(p) if p.suffix != '.safetensors' else None) for p in files}


def dependencies(c, part):
    old = activate(c); cfg = extractor_config(c)
    if part not in ('text', 'input_plan'):
        raise ValueError('Unknown cache part')
    common = [ROOT/'src/p2_listen/config.py', ROOT/'src/p2_listen/contracts.py',
              ROOT/'src/p2_listen/features.py', ROOT/'src/p2_listen/cache.py',
              old/'src/omni_av_p2/config.py']
    if part == 'text':
        import ltx_core.text_encoders.gemma.embeddings_processor as embeddings_processor
        import ltx_core.text_encoders.gemma.embeddings_connector as embeddings_connector
        import ltx_core.text_encoders.gemma.tokenizer as gemma_tokenizer
        paths = common+[old/'src/omni_av/ltx_runtime.py', Path(embeddings_processor.__file__),
                        Path(embeddings_connector.__file__), Path(gemma_tokenizer.__file__)]
        assets = dict(gemma=asset_identity(cfg.gemma_path), ltx=weights_identity(c, 'ltx'))
    else:
        paths = common+[ROOT/'src/p3_listen/features.py', ROOT/'src/p2_listen/layout.py',
                        old/'src/omni_av_p2/data.py', old/'src/omni_av_p2/connector.py',
                        old/'src/omni_av/omni.py', old/'src/omni_av/data.py']
        assets = dict(qwen=asset_identity(cfg.omni_path))
    import transformers
    return dict(assets=assets, code={str(p): sha(p) for p in paths},
                torch=torch.__version__, transformers=transformers.__version__,
                extractor=cfg.to_dict(), dtype='bfloat16', qwen_attention='sdpa')


def identity(p, item, part, dep):
    g = item['guidance']
    result = dict(schema='p2-plan-part-v1', part=part, code=p['code'], dependencies=dep,
                  prompt=g['prompt'], prompt_sha256=g['prompt_sha256'],
                  plan_source=p['plan_source'], guidance_sha256=digest(g),
                  max_tokens=p['max_text_tokens' if part == 'text' else 'max_qwen_tokens'])
    if part == 'input_plan':
        result.update(input=input_identity(item['observation']),
                      listening_prompt=prompt_contract(),
                      append_policy='system_task+user_image_audio+assistant_exact_prompt+im_end-v2',
                      hidden_scope='all_input_and_plan_tokens', qwen_frozen=True)
    return result


def validate_payload(payload, item, part):
    g = item['guidance']
    if payload.get('prompt_sha256') != g['prompt_sha256'] or payload.get('truncated') is not False:
        raise ValueError('Cache prompt identity/truncation mismatch')
    if part == 'text':
        pos = payload['positive']; length = payload['sequence_length']
        if length not in (1024, 2048, 4096, 8192) or not 0 < payload['tokens'] <= length:
            raise ValueError('Invalid Gemma length policy')
        if pos['prompt'] != g['prompt'] or pos['video'].shape != (1, length, 4096) or pos['audio'].shape != (1, length, 2048):
            raise ValueError('Gemma payload shape/prompt mismatch')
        # LTX's native text connectors may replace padding with learned registers.
        # Their OUTPUT mask is then all-valid, unlike Gemma's INPUT token mask.
        # encode_guidance checks the exact input token count before that conversion.
        if pos['mask'].shape != (1, length) or pos['mask'].dtype != torch.bool or not pos['mask'].any():
            raise ValueError('LTX text condition mask mismatch')
        if not all(torch.isfinite(pos[k]).all() for k in ('video', 'audio')):
            raise ValueError('Nonfinite Gemma payload')
    else:
        from omni_av_p2.connector import ConditionInputs
        x = ConditionInputs(**payload['condition']); x.validate()
        n, m = payload['prefix_tokens'], payload['plan_tokens']; length = n+m+1
        if payload['prompt'] != g['prompt'] or payload['qwen_frozen'] is not True or n <= 0 or m <= 0:
            raise ValueError('Invalid planned input metadata')
        if x.hidden.shape[2] != length or payload['terminal_tokens'] != 1:
            raise ValueError('Plan hidden missing or truncated')
        mask = payload['plan_mask']; expected = torch.zeros_like(x.hidden_valid)
        expected[:, n:n+m] = True
        if mask.dtype != torch.bool or not torch.equal(mask, expected) or not x.hidden_valid.all():
            raise ValueError('Connector does not receive all plan tokens')
        if payload['position_ids'].shape != (3, 1, length) or payload['plan_token_ids'].shape != (1, m):
            raise ValueError('Invalid plan token/position shape')
        if not torch.isfinite(x.hidden).all():
            raise ValueError('Nonfinite Qwen payload')


def cache(work, part, *, sample_ids=None, shard_index=0, shards=1, device='cuda:0'):
    from .encoding import extract_planned_input, encode_text
    work = Path(work); p = open_protocol(work); c = p['config']
    chosen = select(p, sample_ids, shard_index, shards)
    dep = dependencies(c, part); cfg = extractor_config(c)
    cuda = str(device).startswith('cuda')
    if cuda:
        limit = os.environ.get('P2_PLAN_GPU_MEMORY_GIB')
        if limit is not None:
            gib = float(limit); total = torch.cuda.get_device_properties(device).total_memory
            if not 0 < gib*1024**3 < total: raise ValueError('Invalid per-process GPU memory limit')
            torch.cuda.set_per_process_memory_fraction(gib*1024**3/total, device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = None; processor = None; results = []
    try:
        for sid in chosen:
            sample_started = time.monotonic()
            try:
                with lock(work/'locks'/part/(sid+'.lock')):
                    item = sample(work, p, sid); expected = identity(p, item, part, dep)
                    folder = work/'cache'/part/sid
                    value = load_part(folder, expected, missing_ok=True)
                    if value is not None:
                        validate_payload(value, item, part)
                        status = 'reused'
                    else:
                        if model is None:
                            if part == 'text':
                                from omni_av.ltx_runtime import NativeTextConditioner
                                model = NativeTextConditioner(cfg, device)
                            else:
                                from omni_av_p2.data import load_omni
                                model, processor = load_omni(cfg, device)
                        g = item['guidance']
                        if part == 'text':
                            positive, meta = encode_text(model, g['prompt'], p['max_text_tokens'])
                            value = dict(positive=positive, prompt_sha256=g['prompt_sha256'], **meta)
                        else:
                            # Hash the actual speaker/image files before encoding, not listener GT.
                            obs = observation(item['observation'], verify=True)
                            value = extract_planned_input(c, obs, g['prompt'], model, processor, p['max_qwen_tokens'])
                        validate_payload(value, item, part)
                        save_part(folder, expected, value); status = 'cached'
                    result = dict(sample_id=sid, status=status, part=part,
                                  prompt_sha256=item['guidance']['prompt_sha256'])
                    del value
            except Exception as exc:
                result = dict(sample_id=sid, status='failed', part=part,
                              error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
                atomic_json(work/'errors'/part/(sid+'.json'), result)
            results.append(result)
            result['elapsed_s'] = time.monotonic()-sample_started
            print(__import__('json').dumps({k:v for k,v in result.items() if k != 'traceback'}, ensure_ascii=False), flush=True)
    finally:
        if model is not None and part == 'text':
            model.close()
        del model, processor
        unload()
    report = dict(success=all(r['status'] in ('cached', 'reused') for r in results),
                  part=part, samples=len(chosen), shard_index=shard_index, shards=shards,
                  protocol_sha256=digest(p), results=results, elapsed_s=time.monotonic()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if cuda else None,
                  peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if cuda else None)
    atomic_json(work/'reports'/f'{part}.{shard_index}-of-{shards}.{digest(chosen)[:12]}.json', report)
    return report
