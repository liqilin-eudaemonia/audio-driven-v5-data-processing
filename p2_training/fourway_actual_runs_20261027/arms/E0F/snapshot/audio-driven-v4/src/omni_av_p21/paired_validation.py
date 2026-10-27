"""P2 full-plan paired generation with mandatory P2.1 text-LoRA restoration."""
from dataclasses import asdict
import json
import traceback

from PIL import Image
import torch

from omni_av.data import Observation, TextPlan, file_hash
from omni_av.ltx_runtime import NativeTextConditioner
from omni_av_p2.checkpoint import atomic_json_new, atomic_save
from omni_av_p2.config import Config
from omni_av_p2.connector import FusionConnector
from omni_av_p2.data import unload
from omni_av_p2.inference import reference_latents
from omni_av_p2.ltx import decode_and_save, load_renderer
from omni_av_p2.paired_validation import (MODES, collect_conditions, sample_branch,
                                         tensor_digest, validate_suite, write_comparison_report)
from omni_av_p2.paired_validation import prepare_suite as prepare_suite
from . import VERSION
from .checkpoint import require_complete, validate_state
from .config import LoRAConfig
from .lora import inject_text_lora

PROTOCOL = 'p21_dev_full_plan_pair_v1'


def checkpoint_state(path, suite):
    require_complete(path)
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (saved.get('version') != VERSION or saved['identity']['contract'] != suite['contract']
            or saved['architecture']['model'] != suite['config']):
        raise ValueError('P2.1 checkpoint/suite contract mismatch')
    if {e['sample_id'] for e in saved['identity']['entries']} & {c['sample_id'] for c in suite['cases']}:
        raise ValueError('Dev overlaps checkpoint training data')
    connector = {k.removeprefix('extractor.connector.'): v for k, v in saved['model'].items()
                 if k.startswith('extractor.connector.')}
    adapters = {k.removeprefix('adapters.'): v for k, v in saved['model'].items() if k.startswith('adapters.')}
    if not connector or not adapters or any(not torch.isfinite(v).all() for v in [*connector.values(), *adapters.values()]):
        raise ValueError('P2.1 inference requires finite Connector AND text LoRA')
    return dict(connector=connector, adapters=adapters, architecture=saved['architecture']), saved['step']


def load_adapted_renderer(cfg, deployment, device='cuda:0'):
    renderer, strategy, sampler = load_renderer(cfg, device=device)
    updates, targets = inject_text_lora(renderer, LoRAConfig(**deployment['architecture']['lora']))
    if targets != deployment['architecture']['targets']:
        raise ValueError('P2.1 renderer LoRA target mismatch')
    validate_state(updates, deployment['adapters'])
    updates.load_state_dict(deployment['adapters'], strict=True)
    renderer.requires_grad_(False).eval()
    return renderer, strategy, sampler


@torch.no_grad()
def run_case(case, suite, cfg, connector, output, deployment, device='cuda:0'):
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
            hidden_layers=condition.layer_ids.tolist(), auxiliary_decoder_loaded=False, text_attention_lora_loaded=True)
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
    renderer, strategy, sampler = load_adapted_renderer(cfg, deployment, device=device)
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
    deployment, step = checkpoint_state(checkpoint, suite)
    connector = FusionConnector(cfg).requires_grad_(False).eval()
    validate_state(connector, deployment["connector"])
    connector.load_state_dict(deployment["connector"], strict=True)
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
            completed = run_case(case, suite, cfg, connector, destination, deployment)
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
