"""Residual stage2/3 full-plan pairs with native schedule and text+Qwen CFG."""
from pathlib import Path
from dataclasses import asdict
import json
import fcntl
import traceback

from PIL import Image
import torch

from omni_av.data import Observation, TextPlan, file_hash
from omni_av.ltx_runtime import NativeTextConditioner
from omni_av_p2.checkpoint import atomic_json_new, atomic_save
from omni_av_p2.config import Config
from omni_av_p2.connector import FusionConnector, ConditionInputs
from omni_av_p2.data import unload
from omni_av_p2.inference import reference_latents
from omni_av_p2.ltx import load_renderer
from .residual_media import decode_and_save
from omni_av_p2.paired_validation import (MODES, collect_conditions,
                                         tensor_digest, validate_suite, write_comparison_report)
from omni_av_p2.paired_validation import prepare_suite as _prepare_suite
from .checkpoint import validate_state
from .config import LoRAConfig

PROTOCOL = 'p21_residual_native_cfg_full_plan_pair_v1'


def prepare_suite(manifest, metadata, output, cfg, seed=1045, sampling_steps=30, max_plan_tokens=512):
    suite = _prepare_suite(manifest, metadata, output, cfg, seed, sampling_steps, max_plan_tokens)
    suite.update(generation_protocol=PROTOCOL, sampler='native LTX schedule / Euler / video CFG3 / audio CFG7 / unconditional text and Qwen removed; no STG, rescale or upscaler')
    (output / 'suite.json').write_text(json.dumps(suite, ensure_ascii=False, indent=2))
    return suite


def checkpoint_state(path, suite):
    from .residual_checkpoint import read, sha
    saved, _ = read(path)
    if saved['stage'] < 2:
        raise ValueError('Stage1 has no trained AV residual branch')
    if (saved['identity']['contract'] != suite['contract']
            or saved['architecture']['model'] != suite['config']):
        raise ValueError('Residual checkpoint/suite contract mismatch')
    cache = Path(saved['identity']['cache'])
    if sha(cache) != saved['identity']['cache_sha256']:
        raise ValueError('Training cache index changed')
    train_ids = {e['sample_id'] for e in json.loads(cache.read_text())['entries']}
    if train_ids & {c['sample_id'] for c in suite['cases']}:
        raise ValueError('Train/Dev overlap')
    connector = {k.removeprefix('extractor.connector.'): v for k,v in saved['model'].items() if k.startswith('extractor.connector.')}
    qwen = {k.removeprefix('qwen.'): v for k,v in saved['model'].items() if k.startswith('qwen.')}
    adapters = {k.removeprefix('adapters.'): v for k,v in saved['model'].items() if k.startswith('adapters.')}
    if not connector or not qwen or (saved['stage'] == 3 and not adapters):
        raise ValueError('Missing residual deployment components')
    return dict(connector=connector,qwen=qwen,adapters=adapters,architecture=saved['architecture']),saved['step']


def load_adapted_renderer(cfg, deployment, device='cuda:0'):
    from .residual_config import ResidualConfig
    from .residual_model import ResidualTrainingModel
    renderer,strategy,sampler=load_renderer(cfg,device=device)
    arch=deployment['architecture']
    raw=ResidualTrainingModel(cfg,LoRAConfig(**arch['lora']),ResidualConfig(**arch['residual']),
                             arch['stage'],renderer,strategy,auxiliary=False)
    if raw.targets != arch['targets'] or raw.qwen_targets != arch['qwen_targets']:
        raise ValueError('Residual target topology mismatch')
    for name in ('qwen','adapters'):
        module=getattr(raw,name);validate_state(module,deployment[name]);module.load_state_dict(deployment[name],strict=True)
    renderer.requires_grad_(False).eval()
    # Inference has no optimizer/master-weight requirement.
    renderer.to(dtype=torch.bfloat16)
    return renderer,strategy,sampler


def sample_branch(renderer,strategy,sampler,av,text,u,seed,steps,negative):
    from omni_av.ltx_runtime import with_text
    from omni_av_p2.ltx import unpack_latents
    from .residual_model import attach_hidden
    from .residual_sampling import guided_sample
    torch.manual_seed(seed)
    prepared=strategy.prepare_training_inputs(with_text(av,text),sampler)
    prepared=attach_hidden(prepared,u)
    torch.manual_seed(seed)
    sampled=guided_sample(renderer,prepared,negative,steps=steps)
    video,audio=unpack_latents(sampled,av)
    return video.cpu(),audio.cpu()


def publish_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Resume identity changed: {path}')
    else:
        atomic_json_new(path, value)


@torch.no_grad()
def run_case(case, suite, cfg, connector, output, deployment, device='cuda:0'):
    obs = Observation(**case['observation'])
    if obs.sample_id != case['sample_id']:
        raise ValueError('Case/observation mismatch')
    for key, expected in case['input_sha256'].items():
        if file_hash(getattr(obs,key)) != expected:
            raise ValueError('Observation changed')
    geometry=case['geometry']
    with Image.open(obs.listener_image) as im:
        if im.size != (geometry['width'],geometry['height']):
            raise ValueError('Refusing reference resize/crop')
    output.mkdir(parents=True,exist_ok=True)
    publish_json(output/'case.json',case)
    done=output/'completed.json'
    if done.exists():
        result=json.loads(done.read_text())
        for mode,record in result['modes'].items():
            if file_hash(output/mode/'generated.mp4') != record['video_sha256']:
                raise ValueError('Completed video changed')
        return result
    bundle=output/'planning_conditions.pt'
    if bundle.exists():
        saved=torch.load(bundle,map_location='cpu',weights_only=True)
        if saved['contract'] != suite['contract'] or saved['case'] != case:
            raise ValueError('Cached planning identity changed')
        plans={k:TextPlan(**v) for k,v in saved['plans'].items()}
        conditions={k:ConditionInputs(**v) for k,v in saved['conditions'].items()}
        raw=saved['raw']
    else:
        # A failed planner may leave diagnostic files; keep each attempt separately.
        attempts=output/'planning_attempts';attempts.mkdir(exist_ok=True)
        attempt=attempts/f'{len(list(attempts.iterdir())):03d}';attempt.mkdir()
        plans,conditions,raw=collect_conditions(obs,TextPlan(**case['gt_plan']),cfg,suite['max_plan_tokens'],attempt)
        atomic_save(bundle,dict(contract=suite['contract'],case=case,plans={k:asdict(v) for k,v in plans.items()},
                               conditions={k:vars(v.to('cpu')) for k,v in conditions.items()},raw=raw))
    (output/'omni_raw.txt').write_text(raw,encoding='utf8')
    controls,records={},{}
    for mode,plan in plans.items():
        folder=output/mode;folder.mkdir(exist_ok=True)
        condition=conditions[mode];condition.validate()
        publish_json(folder/'text_plan.json',asdict(plan));publish_json(folder/'geometry.json',geometry)
        if not (folder/'hidden.pt').exists():
            atomic_save(folder/'hidden.pt',dict(contract=suite['contract'],condition=vars(condition.to('cpu')),
                text_plan=asdict(plan),source=mode,observation=asdict(obs)))
        if not (folder/'sampled_latents.pt').exists():
            with torch.autocast('cuda',dtype=torch.bfloat16):controls[mode]=connector(condition.to(device))
        records[mode]=dict(plan=asdict(plan),hidden_shape=list(condition.hidden.shape),
            hidden_sha256=tensor_digest(condition.hidden),seed=case['seed'],gt_plan_used=mode=='gt',
            paired_target_media_used=False,hidden_layers=condition.layer_ids.tolist(),
            auxiliary_decoder_loaded=False,text_attention_lora_loaded=deployment['architecture']['stage']==3,
            qwen_residual_loaded=True,gemma_prompt=plan.render_prompt())
    del conditions
    unload()
    if controls:
        runtime=NativeTextConditioner(cfg,device)
        try:
            class EmptyPlan:
                def render_prompt(self):return ''
            negative=runtime.encode(EmptyPlan())
            texts={mode:runtime.encode(plans[mode]) for mode in controls}
        finally:runtime.close()
        for mode,text in texts.items():
            if text['prompt'] != plans[mode].render_prompt():raise ValueError('Gemma/hidden plan mismatch')
        av=reference_latents(cfg,obs,geometry,device=device)
        renderer,strategy,sampler=load_adapted_renderer(cfg,deployment,device=device)
        try:
            for mode,u in controls.items():
                video,audio=sample_branch(renderer,strategy,sampler,av,texts[mode],u,
                                         case['seed'],suite['sampling_steps'],negative)
                atomic_save(output/mode/'sampled_latents.pt',dict(video=video,audio=audio,source=mode,
                    seed=case['seed'],sampling_steps=suite['sampling_steps'],protocol=PROTOCOL))
        finally:
            del renderer,strategy,sampler,av,texts,negative,controls
            unload()
    for mode in plans:
        folder=output/mode;completed=folder/'completed.json';video_path=folder/'generated.mp4'
        if completed.exists():
            record=json.loads(completed.read_text())
            if file_hash(video_path)!=record['video_sha256']:raise ValueError('Completed video changed')
            records[mode]=record;continue
        saved=torch.load(folder/'sampled_latents.pt',map_location='cpu',weights_only=True)
        if saved['source']!=mode or saved['seed']!=case['seed'] or saved['sampling_steps']!=suite['sampling_steps'] or saved['protocol']!=PROTOCOL:
            raise ValueError('Sampling identity changed')
        if not video_path.exists():
            temp=folder/'generated.pending.mp4';temp.unlink(missing_ok=True)
            decoded=decode_and_save(cfg,saved['video'],saved['audio'],geometry['fps'],temp,device=device)
            if any(decoded[k]!=geometry[k] for k in ('frames','height','width')):
                raise ValueError('Output geometry changed')
            publish_json(folder/'decoded.json',decoded);temp.replace(video_path)
        else:decoded=json.loads((folder/'decoded.json').read_text())
        records[mode].update(decoded=decoded,video_sha256=file_hash(video_path))
        atomic_json_new(completed,records[mode]);del saved
    result=dict(sample_id=obs.sample_id,modes=records,pair_complete=set(records)==set(MODES),
                same_seed_and_geometry=True,protocol=PROTOCOL,
                hidden_extractor='omni_av_p2.data.extract_condition',quality_pass=None)
    atomic_json_new(done,result)
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
    worker_lock=(output/f'worker_{shard_index:03d}.lock').open('a')
    fcntl.flock(worker_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # One immutable worker identity; completed videos and latents are reusable.
    publish_json(output / f'worker_{shard_index:03d}.json', dict(protocol=PROTOCOL,
        suite_sha256=file_hash(suite_path), checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
        step=step, shard_index=shard_index, shards=shards, contract=suite['contract']))
    results = []
    for case in suite['cases'][shard_index::shards]:
        destination = output / case['sample_id']
        try:
            completed = run_case(case, suite, cfg, connector, destination, deployment)
            if completed['pair_complete']:
                (output / f"{case['sample_id']}.failed.json").unlink(missing_ok=True)
            results.append(dict(sample_id=case['sample_id'],
                status='complete' if completed['pair_complete'] else 'partial',
                available_modes=list(completed['modes']), output=str(destination)))
        except Exception:
            error = dict(sample_id=case['sample_id'], status='failed', traceback=traceback.format_exc())
            error_path = output / f"{case['sample_id']}.failed.json"
            error_path.write_text(json.dumps(error, ensure_ascii=False, indent=2))
            results.append(error)
            unload()
        print(json.dumps(results[-1]), flush=True)
    result = dict(step=step, cases=results, passed=all(r['status'] == 'complete' for r in results), quality_pass=None)
    worker_result = output / f'worker_{shard_index:03d}.completed.json'
    pending = worker_result.with_suffix('.pending.json')
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2)); pending.replace(worker_result)
    if shards == 1 and not (output/'comparison.html').exists():
        write_comparison_report(output, suite)
    return result
