"""Paired GT/full-Omni-plan generation; no paired media or training-cache inputs."""
from dataclasses import asdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import traceback

from PIL import Image
import torch

from omni_av.data import Observation, TextPlan, file_hash, gt_text_plan
from omni_av.ltx_runtime import NativeTextConditioner, with_text
from omni_av.omni import generate_reply, prepare_observation
from . import VERSION
from .checkpoint import atomic_json_new, atomic_save
from .config import Config
from .connector import ConditionInputs, FusionConnector
from .data import contract, extract_condition, load_omni, unload
from .inference import reference_latents
from .ltx import append_conditions, decode_and_save, load_renderer, sample, unpack_latents

MODES = ('gt', 'omni')
PROTOCOL = 'p2_dev_full_plan_pair_v1'


def geometry_from_metadata(example, metadata):
    obs = Observation(**example['observation'])
    if metadata['sample_id'] != obs.sample_id or metadata['split'] != 'dev':
        raise ValueError('Dev geometry identity mismatch')
    frames, height, width, fps = (metadata[k] for k in ('target_frames', 'target_height', 'target_width', 'fps'))
    if (type(frames) is not int or frames < 9 or (frames - 1) % 8
            or any(type(x) is not int or x < 64 or x % 32 for x in (height, width))
            or not math.isfinite(fps) or fps <= 0):
        raise ValueError('Invalid native training geometry')
    if (height, width) != (example['target_height'], example['target_width']):
        raise ValueError('Manifest and training geometry differ')
    if frames / fps > obs.output_duration_s + 1e-6:
        raise ValueError('Rendered duration exceeds condition budget')
    return dict(frames=frames, fps=fps, height=height, width=width, duration_s=frames / fps,
                external_budget_s=obs.output_duration_s, source='training_dev_metadata')


def prepare_suite(manifest, metadata_path, output, cfg, seed=1045, sampling_steps=30, max_plan_tokens=512):
    """CPU-only plan snapshots; GT video/audio paths are deliberately not exported."""
    if sampling_steps < 1 or max_plan_tokens < 1 or not 0 <= seed < 2**32 - 100000:
        raise ValueError('Invalid sampling configuration')
    examples = json.loads(manifest.read_text())
    ids = [e['observation']['sample_id'] for e in examples]
    if (not examples or len(set(ids)) != len(ids) or any(e['split'] != 'dev' for e in examples)
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', sid) or sid in ('.', '..') for sid in ids)):
        raise ValueError('Need unique safe sample IDs from Dev only')
    metadata = {v['sample_id']: v for v in map(json.loads, metadata_path.open())}
    cases = []
    for i, example in enumerate(examples):
        obs = Observation(**example['observation'])
        plan = gt_text_plan(json.loads(Path(example['gt_path']).read_text()), obs.sample_id)
        if plan != TextPlan(**example['text_plan']):
            raise ValueError('Manifest plan differs from training GT conversion')
        geometry = geometry_from_metadata(example, metadata[obs.sample_id])
        with Image.open(obs.listener_image) as image:
            if image.size != (geometry['width'], geometry['height']):
                raise ValueError('Reference would require resize/crop; refusing')
        cases.append(dict(sample_id=obs.sample_id, split='dev', observation=asdict(obs),
            gt_plan=asdict(plan), gt_sha256=file_hash(example['gt_path']), geometry=geometry,
            seed=seed + i, input_sha256={k: file_hash(getattr(obs, k)) for k in ('speaker_audio', 'listener_image')}))
    suite = dict(protocol=PROTOCOL, contract=contract(cfg), config=cfg.to_dict(), modes=list(MODES),
        plan_fields=['reply', 'delivery', 'behavior', 'vocal_events'], sampling_steps=sampling_steps,
        max_plan_tokens=max_plan_tokens, seed=seed, manifest_sha256=file_hash(manifest),
        geometry_metadata_sha256=file_hash(metadata_path), cases=cases, generated=False,
        sampler='existing Euler without CFG; configurable steps, not a quality guarantee')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_new(output / 'suite.json', suite)
    return suite


def validate_suite(suite, cfg):
    if (suite['protocol'] != PROTOCOL or suite['contract'] != contract(cfg)
            or suite['modes'] != list(MODES) or not suite['cases']):
        raise ValueError('Validation suite protocol/config mismatch')
    ids = [c['sample_id'] for c in suite['cases']]
    if (len(ids) != len(set(ids)) or any(c['split'] != 'dev' for c in suite['cases'])
            or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', sid) or sid in ('.', '..') for sid in ids)
            or suite['sampling_steps'] < 1 or suite['max_plan_tokens'] < 1):
        raise ValueError('Invalid Dev identities')


def checkpoint_state(path, suite):
    """Require the trainer's completion marker when reading a periodic checkpoint."""
    match = re.fullmatch(r'(extractor_step_(\d+))(?:_connector)?\.pt', path.name)
    if match:
        marker = path.with_name(match[1] + '.complete.json')
        if not marker.is_file() or json.loads(marker.read_text())['step'] != int(match[2]):
            raise ValueError('Checkpoint is not completely published yet')
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved['version'] != VERSION or saved['identity']['contract'] != suite['contract']:
        raise ValueError('Checkpoint does not use this no-time prompt/config contract')
    train_ids = {e['sample_id'] for e in saved['identity']['entries']}
    if train_ids & {c['sample_id'] for c in suite['cases']}:
        raise ValueError('Dev overlaps checkpoint training data')
    state = {k.removeprefix('connector.'): v for k, v in saved['model'].items() if k.startswith('connector.')}
    if not state or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid Connector checkpoint tensors')
    return state, saved['step']


def plan_conditions(runtime, processor, obs, gt_plan, cfg, max_tokens, recorder=None, allow_rejected=False):
    """Omni sees observation only; both plans then get a fresh training-style forward."""
    prepared = prepare_observation(processor, obs, cfg)
    try:
        predicted, raw = generate_reply(runtime, processor, prepared, max_new_tokens=max_tokens)
    except Exception as exc:
        if recorder:
            recorder(dict(status='rejected' if isinstance(exc, ValueError) else 'error',
                          error=f'{type(exc).__name__}: {exc}', gt_fallback=False))
        if not (allow_rejected and isinstance(exc, ValueError)):
            raise
        predicted, raw = None, ''
    if recorder and predicted is not None:
        recorder(dict(status='valid', plan=asdict(predicted), raw=raw, gt_fallback=False))
    plans = dict(gt=gt_plan)
    if predicted is not None:
        plans['omni'] = predicted
    conditions = {}
    for mode, plan in plans.items():
        value = extract_condition(runtime, processor, obs, plan)
        conditions[mode] = ConditionInputs(**{k: v.cpu() for k, v in vars(value).items()})
    return plans, conditions, raw


def collect_conditions(obs, gt_plan, cfg, max_tokens, plan_output=None):
    runtime = processor = None
    def record(value):
        if plan_output is not None:
            atomic_json_new(plan_output / 'planning_result.json', value)
    if plan_output is not None:
        atomic_json_new(plan_output / 'planning_started.json', dict(sample_id=obs.sample_id))
    try:
        runtime, processor = load_omni(cfg)
        return plan_conditions(runtime, processor, obs, gt_plan, cfg, max_tokens, record, allow_rejected=True)
    except Exception as exc:
        if plan_output is not None and not (plan_output / 'planning_result.json').exists():
            record(dict(status='error', error=f'{type(exc).__name__}: {exc}', gt_fallback=False))
        raise
    finally:
        del runtime, processor
        unload()


def tensor_digest(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write_comparison_report(output, suite):
    rows = []
    for case in suite['cases']:
        sid = case['sample_id']
        cells = [f'<th>{escape(sid)}</th>']
        for mode in MODES:
            folder = output / sid / mode
            content = '<p>尚无生成结果；检查任务状态或失败记录。</p>'
            if (folder / 'generated.mp4').is_file():
                content = (f'<video controls preload="none" src="{escape(sid)}/{mode}/generated.mp4"></video>')
            if (folder / 'text_plan.json').exists():
                plan = json.loads((folder / 'text_plan.json').read_text())
                content += '<pre>' + escape(TextPlan(**plan).serialize()) + '</pre>'
            cells.append('<td>' + content + '</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    html = ('<!doctype html><meta charset="utf-8"><title>GT / Omni 验证对照</title>'
        '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #ccc;padding:12px;vertical-align:top}th{overflow-wrap:anywhere}'
        'td{width:42%}video{width:100%;max-height:420px}pre{white-space:pre-wrap}</style>'
        '<h1>GT / Omni 完整计划生成对照</h1>'
        '<p>每行使用相同参考图、生成几何和随机种子；两组切换整份计划。此页面不代表质量通过。</p>'
        '<table><thead><tr><th>样本</th><th>GT 计划</th><th>Omni 计划</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>')
    with (output / 'comparison.html').open('x', encoding='utf8') as stream:
        stream.write(html)


def sample_branch(renderer, strategy, sampler, av, text, u, seed, steps):
    # Reset before BOTH strategy preparation and actual noise creation. Planning
    # token count and Gemma loading must never shift a branch's sampling RNG.
    torch.manual_seed(seed)
    prepared = strategy.prepare_training_inputs(with_text(av, text), sampler)
    inputs = append_conditions(prepared, u)
    torch.manual_seed(seed)
    sampled = sample(renderer, inputs, steps)
    video, audio = unpack_latents(sampled, av)
    return video.cpu(), audio.cpu()


@torch.no_grad()
def run_case(case, suite, cfg, connector, output, device='cuda:0'):
    obs = Observation(**case['observation'])
    if obs.sample_id != case['sample_id']:
        raise ValueError('Case/observation mismatch')
    for key, expected in case['input_sha256'].items():
        if file_hash(getattr(obs, key)) != expected:
            raise ValueError('Observation media changed after preparation')
    geometry = case['geometry']
    with Image.open(obs.listener_image) as image:
        if image.size != (geometry['width'], geometry['height']):
            raise ValueError('Refusing reference resize/crop')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json_new(output / 'case.json', case)
    plans, conditions, raw = collect_conditions(obs, TextPlan(**case['gt_plan']), cfg, suite['max_plan_tokens'], output)
    (output / 'omni_raw.txt').write_text(raw, encoding='utf8')
    controls, records = {}, {}
    for mode in plans:
        folder = output / mode
        folder.mkdir()
        plan, condition = plans[mode], conditions[mode]
        condition.validate()
        atomic_json_new(folder / 'text_plan.json', asdict(plan))
        atomic_json_new(folder / 'geometry.json', geometry)
        atomic_save(folder / 'hidden.pt', dict(contract=suite['contract'], condition=vars(condition),
            text_plan=asdict(plan), source=mode, observation=asdict(obs)))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            controls[mode] = connector(condition.to(device))
        atomic_save(folder / 'condition.pt', dict(source=mode, text_plan=asdict(plan),
            u={k: v.cpu() for k, v in vars(controls[mode]).items()}))
        records[mode] = dict(plan=asdict(plan), hidden_shape=list(condition.hidden.shape),
            hidden_sha256=tensor_digest(condition.hidden), seed=case['seed'],
            gt_plan_used=mode == 'gt', paired_target_media_used=False,
            hidden_layers=condition.layer_ids.tolist(), auxiliary_decoder_loaded=False)
    del conditions
    unload()
    texts = {}
    text_runtime = NativeTextConditioner(cfg)
    try:
        for mode in plans:
            texts[mode] = text_runtime.encode(plans[mode])
            records[mode]['gemma_prompt'] = texts[mode]['prompt']
            if texts[mode]['prompt'] != plans[mode].render_prompt():
                raise ValueError('Gemma and hidden plan differ')
    finally:
        text_runtime.close()
    # First frame only; zeros for video tail/audio. No target VAE cache is loaded.
    av = reference_latents(cfg, obs, geometry, device=device)
    renderer, strategy, sampler = load_renderer(cfg, device=device)
    latents = {}
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for mode in plans:
                video, audio = sample_branch(renderer, strategy, sampler, av, texts[mode], controls[mode],
                                             case['seed'], suite['sampling_steps'])
                latents[mode] = (video, audio)
                atomic_save(output / mode / 'sampled_latents.pt', dict(video=video, audio=audio, source=mode))
    finally:
        del renderer, strategy, sampler, av, texts, controls
        unload()
    for mode in plans:
        video, audio = latents.pop(mode)
        decoded = decode_and_save(cfg, video, audio, geometry['fps'], output / mode / 'generated.mp4', device=device)
        if any(decoded[k] != geometry[k] for k in ('frames', 'height', 'width')):
            raise ValueError('Decoded output differs from training geometry')
        records[mode]['decoded'] = decoded
        atomic_json_new(output / mode / 'completed.json', records[mode])
    result = dict(sample_id=obs.sample_id, modes=records, pair_complete=set(records) == set(MODES), same_seed_and_geometry=True,
        protocol=PROTOCOL, hidden_extractor='omni_av_p2.data.extract_condition', quality_pass=None)
    atomic_json_new(output / 'completed.json', result)
    return result


def run_shard(suite_path, checkpoint, output, shard_index, shards):
    torch.set_num_threads(4)
    suite = json.loads(suite_path.read_text())
    cfg = Config(**suite['config'])
    validate_suite(suite, cfg)
    if shards < 1 or not 0 <= shard_index < shards or shard_index >= len(suite['cases']):
        raise ValueError('Invalid or empty validation shard')
    state, step = checkpoint_state(checkpoint, suite)
    connector = FusionConnector(cfg).requires_grad_(False).eval()
    connector.load_state_dict(state, strict=True)
    del state
    connector.cuda()
    output.mkdir(parents=True, exist_ok=True)
    # One immutable run identity per worker; output paths never overwrite old results.
    atomic_json_new(output / f'worker_{shard_index:03d}.json', dict(protocol=PROTOCOL,
        suite_sha256=file_hash(suite_path), checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
        step=step, shard_index=shard_index, shards=shards, contract=suite['contract']))
    results = []
    for case in suite['cases'][shard_index::shards]:
        destination = output / case['sample_id']
        try:
            completed = run_case(case, suite, cfg, connector, destination)
            results.append(dict(sample_id=case['sample_id'],
                status='complete' if completed['pair_complete'] else 'partial',
                available_modes=list(completed['modes']), output=str(destination)))
        except Exception:
            error = dict(sample_id=case['sample_id'], status='failed', traceback=traceback.format_exc())
            atomic_json_new(output / f"{case['sample_id']}.failed.json", error)
            results.append(error)
            unload()
        print(json.dumps(results[-1]), flush=True)
    result = dict(step=step, cases=results, passed=all(r['status'] == 'complete' for r in results), quality_pass=None)
    atomic_json_new(output / f'worker_{shard_index:03d}.completed.json', result)
    if shards == 1:
        write_comparison_report(output, suite)
    return result
