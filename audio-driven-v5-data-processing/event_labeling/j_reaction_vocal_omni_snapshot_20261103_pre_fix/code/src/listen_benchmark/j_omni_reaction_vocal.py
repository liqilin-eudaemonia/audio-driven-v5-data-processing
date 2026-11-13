"""Full-scale J-pair Omni captioning with final bounded crop, purified listener audio,
25fps nod + gated visual reactions (head_shaking, frowning, surprised), and v4 vocal cues.

Strictly operates on quality-retained clips (decision == 'retain', excluding no_detected_face),
applies quality-bounded-crop-v1-20261030 to both 25fps reaction/nod detection and 8fps Omni
video input, uses official word/sentence/vad union purified listener audio, and enforces
cue-completeness checks with automatic retry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import threading
import time

from .j_annotation import JobQueue, acquire, bounded_claim_groups, locked, read, result_path, write
from .j_omni import (
    EvidenceUnavailable,
    audio_recipe,
    fingerprint,
    immutable,
    official_metadata,
    read_audio,
)
from .j_omni_crop import REVISION, effective_row
from .native_caption_quality import generation_failures
from listen_visual_quality.io import file_hash

PIPELINE_ID = 'listen-j-nod-vocal-reaction-omni-caption-v1-20261103'
PROMPT_VERSION = 'native-av-first-plus-nod-vocal-reaction-v1'
_reaction_services = {}
_reaction_services_lock = threading.Lock()


def _measure_reaction(request, settings, cfg, resources, cache):
    """One persistent reaction+nod service per preparer process."""
    fer_gpu = os.environ.get('J_OMNI_FER_GPU', '')
    key = (settings['runtime_python'], settings['fer_python'], str(resources), fer_gpu)
    with _reaction_services_lock:
        service = _reaction_services.get(key)
        if service is None or service['process'].poll() is not None:
            log = cache / 'runtime_logs' / os.uname().nodename / f'{os.getpid()}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES='',
                OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',
                GLOG_minloglevel='2',
            )
            cmd = [
                settings['runtime_python'],
                str(Path(cfg['project_root']) / 'scripts/listen_j_corrected_reaction.py'),
                '--serve',
                '--fer-python',
                str(settings['fer_python']),
            ]
            if fer_gpu != '':
                cmd.extend(['--fer-gpu', str(fer_gpu)])
            with log.open('a') as output:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=output,
                    text=True,
                    bufsize=1,
                )
            service = {'process': process, 'lock': threading.Lock(), 'log': str(log)}
            _reaction_services[key] = service
    with service['lock']:
        process = service['process']
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        if not select.select([process.stdout], [], [], 300)[0]:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TimeoutError('Corrected reaction service timed out; see ' + service['log'])
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('Corrected reaction service exited; see ' + service['log'])
        reply = json.loads(line)
        if not reply.get('ok') or reply.get('output') != request['output']:
            raise RuntimeError(
                'Corrected reaction service failed: ' + str(reply) + '; see ' + service['log']
            )


def corrected_reaction_and_nod(
    original, applied, old_work, work, resources, cfg, media_provider=None
):
    """Compute or load 25fps nod + gated reaction cues on the final bounded crop."""
    settings = cfg['reaction_input']
    cache = Path(settings['reaction_cache'])
    sid = applied['sample_id']
    target = result_path(cache, 'reaction', sid)
    with locked(str(target) + '.lock'):
        if not target.exists():
            from listen_visual_native8_nod import decode_slot

            request_path = result_path(cache, 'requests', sid)
            write(request_path, applied)
            request = {
                'row': str(request_path),
                'output': str(target),
                'resources': str(resources),
            }
            if media_provider is not None:
                decoded = media_provider()
                if decoded is not None:
                    request['decoded_dir'] = str(decoded)
            slot_n = int(
                os.environ.get('J_OMNI_REACTION_SLOTS')
                or settings.get('reaction_slots_per_host', 24)
            )
            with decode_slot(Path(resources) / 'resources_reaction', slot_n):
                _measure_reaction(request, settings, cfg, resources, cache)

    n = read(target)
    if (
        n['sample_id'] != sid
        or n['source'] != applied['source']
        or n['fps'] != 25
        or n['profile'] != 'recall'
    ):
        raise ValueError('Reaction/nod identity, crop, FPS, or profile mismatch')
    if abs(n['duration_s'] - applied['source']['duration_s']) > 1e-8:
        raise ValueError('Reaction/nod duration mismatch')
    if n['status'] != 'ok':
        raise EvidenceUnavailable(
            'Reaction/nod evidence unavailable; not an empty detection: ' + str(n.get('reason'))
        )
    nod_bundle = {
        'sample_id': sid,
        'fps': 25,
        'profile': 'recall',
        'events': [
            {'id': e['event_id'], 'start': e['start'], 'end': e['end']} for e in n['events']
        ],
        'detector_file': str(target),
        'detector_sha256': file_hash(target),
        'crop_reuse': 'corrected_reaction_cache',
    }
    reaction_bundle = {
        'sample_id': sid,
        'fps': 25,
        'fer_status': n.get('fer_status'),
        'cues': list(n.get('reaction_cues') or []),
        'detector_file': str(target),
        'detector_sha256': nod_bundle['detector_sha256'],
    }
    return nod_bundle, reaction_bundle


def vocal_cues_for_row(applied, work, duration_s: float, silent: bool) -> list[dict]:
    """Build v4 normalized, deduplicated vocal cues inside the official listener gate."""
    if silent:
        return []
    from compare_listener_official_six import official
    from run_glare_reaction_omni_pilot import compact_vocal_cues
    from run_listen_native_caption_vocal_pilot import build_vocal_cues

    root = Path(cfg_root := work).resolve()
    while root.name != 'audio-driven-v5' and root.parent != root:
        root = root.parent
    rec_path = (
        root
        / 'experiments/p2_full_audio_postprocess_crop384_20261026/records'
        / f"{applied['sample_id']}.json"
    )
    if rec_path.exists():
        rec = read(rec_path)
        events = rec.get('audio_events') or []
        off = rec.get('official_listener') or {}
    else:
        events = []
        meta_Work = root / 'experiments/listen_j_omni_20261030'
        if not (meta_Work / 'official_metadata').exists():
            meta_Work = Path(work)
        raw_meta = official_metadata(applied['source']['source'], meta_Work)
        off = official(
            raw_meta['raw'],
            'listener',
            float(applied['source']['source_start_s']),
            float(duration_s),
        )
    return compact_vocal_cues(build_vocal_cues(float(duration_s), False, events, off))


def detect_clip(row, work, resources):
    """Run the five CPU detectors and write their caches. No Omni frames or GPU work.

    Nod, head shake, frown, and brow-raise land in the reaction cache. Vocal cues,
    after the same silence and overlap filter used at caption time, land beside it.
    An existing measurement is kept. A clip with no detected face is a finished
    detection, not a failure.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['J_OMNI_FER_GPU'] = ''
    cfg = read(Path(work) / 'config.json')
    quality = row.get('quality')
    if quality is not None and quality.get('decision') != 'retain':
        raise ValueError(f"Non-retained clip cannot be detected: {row['sample_id']}")
    applied, _crop = effective_row(row, cfg, decision=quality)
    if applied['source']['crop_proposal']['revision'] != REVISION:
        raise ValueError('Final bounded crop was not applied')

    reaction_path = result_path(Path(cfg['reaction_input']['reaction_cache']), 'reaction', applied['sample_id'])
    try:
        nod_bundle, reaction_bundle = corrected_reaction_and_nod(
            row, applied, work, work, resources, cfg
        )
    except EvidenceUnavailable as exc:
        if not reaction_path.exists():
            raise
        saved = read(reaction_path)
        nod_bundle = {
            'sample_id': applied['sample_id'],
            'events': saved.get('events') or [],
            'status': saved.get('status'),
            'reason': saved.get('reason'),
        }
        reaction_bundle = {
            'sample_id': applied['sample_id'],
            'fer_status': saved.get('fer_status'),
            'cues': saved.get('reaction_cues') or [],
            'reason': str(exc),
        }

    vocal_path = result_path(work, 'vocal', applied['sample_id'])
    if not vocal_path.exists():
        existing_recipe_work = Path(cfg['project_root']) / 'experiments/listen_j_omni_20261030'
        existing_recipe = result_path(existing_recipe_work, 'audio_recipes', applied['sample_id'])
        recipe = read(existing_recipe) if existing_recipe.exists() else audio_recipe(applied, work)
        _wave, _sr, identity = read_audio(recipe)
        silent = bool(identity['full_silence'])
        duration_s = float(applied['source']['duration_s'])
        cues = vocal_cues_for_row(applied, work, duration_s, silent)
        write(
            vocal_path,
            {
                'schema': 'listen-j-vocal-cue-v1',
                'sample_id': applied['sample_id'],
                'duration_s': duration_s,
                'listener_audio_is_silent': silent,
                'audio_origin': 'virtual_official_listener_union',
                'vocal_cues': cues,
                'training_admitted': False,
            },
        )
    else:
        saved_vocal = read(vocal_path)
        silent = bool(saved_vocal['listener_audio_is_silent'])
        cues = saved_vocal['vocal_cues']

    return {
        'sample_id': applied['sample_id'],
        'nod_events': len(nod_bundle.get('events') or []),
        'reaction_cues': len(reaction_bundle.get('cues') or []),
        'vocal_cues': len(cues),
        'silent': silent,
        'fer_status': reaction_bundle.get('fer_status'),
        'reaction_status': nod_bundle.get('status', 'ok'),
    }


def prepare_input(row, old_work, work, resource_work):
    optim = json.loads(os.environ.get('J_OMNI_INPUT_OPTIMIZATIONS', '{}'))
    if optim.get('shared_decode'):
        import tempfile

        with tempfile.TemporaryDirectory(prefix='j-omni-react-shared-', dir='/dev/shm') as d:
            return _prepare_input(row, old_work, work, resource_work, optim, Path(d))
    return _prepare_input(row, old_work, work, resource_work, optim, None)


def _prepare_input(row, old_work, work, resource_work, optim, native_dir):
    import numpy as np
    from scipy.signal import resample_poly
    from listen_benchmark.full_raw import prepare
    from listen_benchmark.media import uniform_video
    from listen_visual_native8_nod import decode_slot
    from run_glare_reaction_omni_pilot import build_prompt, check_prompt

    cfg = read(Path(work) / 'config.json')
    original = row
    quality_decision = row.get('quality')
    if quality_decision is not None and quality_decision.get('decision') != 'retain':
        raise ValueError(f"Non-retained clip cannot enter pipeline: {row['sample_id']}")

    applied, crop_application = effective_row(original, cfg, decision=quality_decision)
    if applied['source']['crop_proposal']['revision'] != REVISION:
        raise ValueError('Final bounded crop was not applied')

    media = None
    shared_used = False
    shared_format = None

    def shared_media():
        nonlocal media, shared_used, shared_format
        with decode_slot(Path(resource_work) / 'resources', cfg['decode_slots']):
            media = prepare(
                applied['source'],
                'visual',
                fast_crop=optim.get('fast_crop', True),
                native_dump=native_dir,
                raw_native_dump=optim.get('shared_raw_decode', False),
                roi_crop=optim.get('roi_crop', True),
            )
        shared_used = native_dir is not None and (native_dir / 'manifest.json').exists()
        if shared_used:
            shared_format = read(native_dir / 'manifest.json')['schema']
        return native_dir if shared_used else None

    nod_bundle, reaction_bundle = corrected_reaction_and_nod(
        original,
        applied,
        old_work,
        work,
        resource_work,
        cfg,
        media_provider=shared_media if native_dir is not None else None,
    )

    # Reuse existing verified audio_recipe when present, otherwise create in work.
    existing_recipe_Work = Path(cfg['project_root']) / 'experiments/listen_j_omni_20261030'
    existing_recipe_path = result_path(existing_recipe_Work, 'audio_recipes', applied['sample_id'])
    if existing_recipe_path.exists():
        recipe = read(existing_recipe_path)
    else:
        recipe = audio_recipe(applied, work)
    wave, sr, identity = read_audio(recipe)
    silent = bool(identity['full_silence'])

    if media is None:
        with decode_slot(Path(resource_work) / 'resources', cfg['decode_slots']):
            media = prepare(
                applied['source'],
                'visual',
                fast_crop=optim.get('fast_crop', True),
                roi_crop=optim.get('roi_crop', True),
            )
    if abs(media['meta']['video']['source_time_origin_s'] - recipe['source_time_origin_s']) > 1e-9:
        raise ValueError('Audio/video origins differ')
    frames, evidence = uniform_video(media, 8)
    audio = wave[:, 0]
    if sr != 16000:
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

    duration_s = float(applied['source']['duration_s'])
    vocal_cues = vocal_cues_for_row(applied, work, duration_s, silent)
    reaction_cues = reaction_bundle['cues']
    nod_events = nod_bundle['events']

    prompt = build_prompt(
        applied['sample_id'],
        duration_s,
        silent,
        nod_events,
        vocal_cues,
        reaction_cues,
    )
    check_prompt(
        {
            'short': applied['sample_id'],
            'nod_events': nod_events,
            'listener_audio_is_silent': silent,
            'vocal_cues': vocal_cues,
            'reaction_cues': reaction_cues,
        },
        prompt,
    )

    evidence.update(
        audio_identity=identity,
        listener_audio_is_silent=silent,
        audio_encoder_rate=16000,
        audio_encoder_samples=len(audio),
        audio_origin='virtual_official_listener_union',
        video_evidence=media['meta']['video'],
        nod_supplement=nod_bundle,
        vocal_supplement=vocal_cues,
        reaction_supplement=reaction_bundle,
        crop_application=crop_application,
        video_frames_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
        audio_encoder_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
    )
    return {
        'frames': frames,
        'audio': audio,
        'prompt': prompt,
        'evidence': evidence,
        'recipe': recipe,
        'effective_source': applied['source'],
        'nod_events': nod_events,
        'vocal_cues': vocal_cues,
        'reaction_cues': reaction_cues,
        '_input_execution': {
            'fast_crop': optim.get('fast_crop', True),
            'shared_decode_used': shared_used,
            'shared_decode_format': shared_format,
            'roi_crop': optim.get('roi_crop', True),
        },
    }


def _build_request(processor, prepared, prompt_text, tag):
    ev = prepared['evidence']
    ev['second_per_grid_ts'] = [float(processor.image_processor.temporal_patch_size) / 8.0]
    rendered = processor.apply_chat_template(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt_text},
                    {'type': 'video', 'video': 'listener://decoded'},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    out = {
        'prompt': rendered,
        'multi_modal_data': {'video': [prepared['frames']], 'audio': [prepared['audio']]},
        'mm_processor_kwargs': {
            'use_audio_in_video': True,
            'fps': 8,
            'do_sample_frames': False,
            'second_per_grid_ts': ev['second_per_grid_ts'],
        },
    }
    if tag is not None:
        out['multi_modal_uuids'] = {'audio': [tag], 'video': [tag]}
    return out


def generate_with_cue_checks(llm, processor, records, cfg):
    """Batch-generate natural-language captions (up to 4 ready requests per call)
    and retry only the items missing supplied cues."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from run_glare_reaction_omni_pilot import cue_gaps, retry_note

    if not records:
        return []
    run_id = str(time.time_ns())
    attempts = [[] for _ in records]
    best = [None for _ in records]
    missing = [None for _ in records]
    pending = list(range(len(records)))

    def run_batch(indices, prompts, rep_penalty, step):
        params = SamplingParams(
            temperature=0,
            max_tokens=4096,
            repetition_penalty=rep_penalty,
            structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][ -~]*'),
        )
        reqs = [
            _build_request(processor, records[i]['prepared'], prompts[i], f'{run_id}-{i}-{step}')
            for i in indices
        ]
        llm.reset_mm_cache()
        outputs = llm.generate(reqs, params, use_tqdm=False)
        for i, req, out in zip(indices, reqs, outputs):
            prepared = records[i]['prepared']
            gen = out.outputs[0]
            clean = re.sub(r'^Assistant:\s*', '', gen.text).strip()
            gaps = cue_gaps(
                prepared['vocal_cues'],
                prepared['reaction_cues'],
                prepared['nod_events'],
                clean,
            )
            flags = list(generation_failures(clean, gen.finish_reason))
            if gaps['vocal']:
                flags.append('missing_vocal_cue')
            if gaps['reaction']:
                flags.append('missing_reaction_cue')
            if gaps['nod']:
                flags.append('missing_nod_cue')
            cand = {
                'caption': clean,
                'raw_caption': gen.text,
                'finish_reason': gen.finish_reason,
                'completion_tokens': len(gen.token_ids),
                'prompt_tokens': len(out.prompt_token_ids),
                'prompt': prompts[i],
                'rendered_prompt': req['prompt'],
                'decoding': {'temperature': 0, 'max_tokens': 4096, 'repetition_penalty': rep_penalty},
                'gaps': gaps,
                'failures': flags,
            }
            attempts[i].append(cand)
            if missing[i] is None:
                missing[i] = {
                    'vocal': list(gaps['vocal']),
                    'reaction': list(gaps['reaction']),
                    'nod': list(gaps['nod']),
                }
            else:
                for v in gaps['vocal']:
                    if v not in missing[i]['vocal']:
                        missing[i]['vocal'].append(v)
                for r in gaps['reaction']:
                    if r not in missing[i]['reaction']:
                        missing[i]['reaction'].append(r)
                if gaps['nod'] and not missing[i]['nod']:
                    missing[i]['nod'] = list(gaps['nod'])
            if best[i] is None or len(cand['failures']) <= len(best[i]['failures']):
                best[i] = cand

    base_prompts = {i: records[i]['prepared']['prompt'] for i in pending}
    run_batch(pending, base_prompts, 1.05, 0)
    pending = [i for i in pending if best[i]['failures']]

    for step, rep in enumerate((1.05, 1.10), 1):
        if not pending:
            break
        retry_prompts = {}
        for i in pending:
            note = retry_note(missing[i])
            retry_prompts[i] = base_prompts[i] + ('\n\n' + note if note else '')
        run_batch(pending, retry_prompts, rep, step)
        pending = [i for i in pending if best[i]['failures']]

    return [{'best': best[i], 'attempts': attempts[i]} for i in range(len(records))]
