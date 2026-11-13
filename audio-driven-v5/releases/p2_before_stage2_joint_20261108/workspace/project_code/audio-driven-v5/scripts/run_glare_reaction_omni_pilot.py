#!/usr/bin/env python3
"""Test nod, v4 vocal, and gated reaction cues in one Qwen3-Omni caption.

The paragraph is the finalized v4 vocal instruction. Nod intervals stay the
existing caption supplement. Vocal intervals use the same three-tier ASR
normalization. A reaction class is named only when this clip has a kept
interval. Original captions and detector files are not modified.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

from run_listen_native_caption_vocal_pilot import (  # noqa: E402
    INSTRUCTION, build_vocal_cues, prompt_vocal_cues,
)

RUN = ROOT / 'experiments/listen_glare_gate30_20261103'
OUT = RUN / 'omni_reaction_vocal_v3'
RECORDS = ROOT / 'experiments/p2_full_audio_postprocess_crop384_20261026/records'
OBSERVATION = {'head_shaking': 'head shaking', 'frowning': 'brow lowering', 'surprised': 'brows raised'}
MENTION = {
    'head_shaking': re.compile(r'\b(?:head[- ]shak\w*|shak\w* (?:his|her|their|the)? ?head|side[- ]to[- ]side)\b', re.I),
    'frowning': re.compile(r'\b(?:frown\w*|furrow\w*|lowers? (?:\w+ ){0,3}(?:eye)?brows?|(?:eye)?brows? (?:lower|down|drawn|knit)\w*|brow lowers)\b', re.I),
    'surprised': re.compile(r'\b(?:raises?|raising|raised) (?:\w+ ){0,3}(?:eye)?brows?\b|\b(?:eye)?brows? raise\w*\b|\beyes widen\w*\b', re.I),
    'nodding': re.compile(r'\b(?:nods?|nodding)\b', re.I),
}
MULTIPLIERS = {'twice': 2, 'three times': 3, 'four times': 4, 'five times': 5}
CLASS_NOTE = {
    'head shaking': (
        'Each supplied "head shaking" interval must be written as a side-to-side head shake. '
        'Do not rewrite it as a nod. Keep every supplied nodding interval as nodding, including when a nod and a head shake overlap.'
    ),
    'brow lowering': (
        'Include each supplied "brow lowering" interval at its position in the action sequence. '
        'Do not add a count of one unless a single movement is clearly visible.'
    ),
    'brows raised': (
        'Include each supplied "brows raised" interval at its position in the action sequence. '
        'Do not add a count of one unless a single movement is clearly visible.'
    ),
}
ORDER_NOTE = (
    'Use the supplied intervals to order the paragraph with words such as initially, then, and toward the end. '
    'Mention each supplied interval in its own place. An interval may contain one movement or several continuous movements; do not force a count of one. '
    'Do not stretch a brief movement across a long part of the clip, and do not attach it to a different action.'
)


def read_json(path):
    return json.loads(Path(path).read_text())


def short(sample_id):
    return sample_id.split('_S', 1)[1].split('_', 1)[0]


def vocal_inputs(caption, duration, silent):
    if silent:
        return [], {'words': [], 'vad': []}
    record_path = RECORDS / f"{caption['sample_id']}.json"
    if record_path.exists():
        record = read_json(record_path)
        return record.get('audio_events') or [], record.get('official_listener') or {}
    from compare_listener_official_six import load_metadata, official
    source = caption['source']
    raw, _provenance = load_metadata(source['source'])
    return [], official(raw, 'listener', float(source['source_start_s']), float(duration))


def _token_key(cue):
    raw = (cue.get('asr_text') or cue.get('official_text') or '').strip()
    return ' '.join(re.findall(r"[a-z][a-z'-]*", raw.lower()))


def compact_vocal_cues(cues):
    """Drop textless pure-VAD windows and merge overlapping duplicate-word cues."""
    items = [
        dict(cue) for cue in cues
        if cue.get('asr_text') or cue.get('official_text')
        or (cue.get('observation') != 'audible vocalization / voice activity' and cue['end'] - cue['start'] >= 0.05)
    ]
    kept = []
    for cue in items:
        key = _token_key(cue)
        merged = False
        for prev in kept:
            prev_key = _token_key(prev)
            if max(prev['start'], cue['start']) > min(prev['end'], cue['end']) + 0.15:
                continue
            if key and prev_key and key == prev_key:
                prev['start'] = min(prev['start'], cue['start'])
                prev['end'] = max(prev['end'], cue['end'])
                prev['asr_text'] = prev.get('asr_text') or cue.get('asr_text')
                prev['official_text'] = prev.get('official_text') or cue.get('official_text')
                merged = True
                break
            if key and prev_key and (' ' in prev_key and key in prev_key.split()):
                prev['start'] = min(prev['start'], cue['start'])
                prev['end'] = max(prev['end'], cue['end'])
                merged = True
                break
            if key and prev_key and (' ' in key and prev_key in key.split()):
                start = min(prev['start'], cue['start'])
                end = max(prev['end'], cue['end'])
                prev.update(cue)
                prev['start'], prev['end'] = start, end
                merged = True
                break
        if not merged:
            kept.append(cue)
    for index, cue in enumerate(kept, 1):
        cue['id'] = f'vocal-{index:04d}'
    return kept


def _phrase_options(cue):
    options = []
    for field in ('asr_text', 'official_text'):
        raw = (cue.get(field) or '').strip().lower()
        if raw in ('ok.', 'ok', 'okay.', 'okay'):
            for item in ('okay', 'ok'):
                if item not in options:
                    options.append(item)
            continue
        words = []
        for word in re.findall(r"[a-z][a-z'-]*", raw):
            if word not in words:
                words.append(word)
        if words:
            phrase = ' '.join(words)
            if phrase not in options:
                options.append(phrase)
            if phrase in ('okay', 'ok'):
                for item in ('okay', 'ok'):
                    if item not in options:
                        options.append(item)
    return options


def _fold_match_text(text):
    low = (text or '').lower().replace('uh huh', 'uh-huh').replace('mm-mm', 'mmmm')
    low = low.replace("'", '').replace('’', '')
    low = re.sub(r'[,.;:!?"()]+', ' ', low)
    return re.sub(r'\s+', ' ', low).strip()


def _written_count(phrase, text):
    low = _fold_match_text(text)
    phrase = _fold_match_text(phrase)
    if phrase == 'uh huh':
        phrase = 'uh-huh'
    elif phrase == 'mm-mm':
        phrase = 'mmmm'
    elif phrase == 'right':
        low = re.sub(r'\b(?:his|her|their|the) right\b|\bright (?:hand|side|arm|ear|eye|shoulder|cheek)\b', ' ', low)
    spans = list(re.finditer(r'\b' + re.escape(phrase) + r'\b', low))
    hits = 0
    previous = -1
    for match in spans:
        gap = low[previous:match.start()] if previous >= 0 else ' '
        if previous >= 0 and re.fullmatch(r'\s*', gap or ''):
            previous = match.end()
            continue
        hits += 1
        previous = match.end()
    for label, mult in MULTIPLIERS.items():
        if (re.search(re.escape(phrase) + r'[^.]{0,30}\b' + label + r'\b', low)
                or re.search(r'\b' + label + r'\b[^.]{0,30}' + re.escape(phrase), low)):
            hits = max(hits, mult)
    return hits


_NEGATION_BEFORE = re.compile(r"\b(?:no|not|never|without)\b|\b\w+n't\b", re.I)
DISTANT_GAP_S = 1.5
_VOCAL_CLAIM = re.compile(
    r'\b(?:says?|said|speaks?|speaking|laugh\w*|hum\w*|chuckl\w*|giggl\w*|vocaliz\w*|yeah|yes|okay|ok|mhm|uh-huh|hmm+|ha+)\b|"[^"]+"',
    re.I,
)


def _mention_state(key, text):
    """Count affirmative mentions. A mention inside a negation does not count."""
    positive = 0
    negated = False
    for sentence in re.split(r'[.!?]+', text or ''):
        for match in MENTION[key].finditer(sentence):
            prefix = sentence[max(0, match.start() - 48):match.start()]
            if _NEGATION_BEFORE.search(prefix):
                negated = True
                continue
            positive += 1
            for label, mult in MULTIPLIERS.items():
                if re.search(r'\b' + label + r'\b', sentence, re.I):
                    positive += mult - 1
                    break
    return positive, negated


def _clusters(intervals):
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    groups = []
    for start, end in ordered:
        if groups and start - groups[-1][1] <= DISTANT_GAP_S:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    return groups


def _interval_coverage(key, intervals, text):
    """Missing means the action was omitted. Uncertain means the wording cannot cover every distant interval."""
    if not intervals:
        return [], []
    positive, negated = _mention_state(key, text)
    rounded = [(round(float(start), 2), round(float(end), 2)) for start, end in intervals]
    if positive == 0 and negated:
        return [], [{'kind': key, 'reason': 'negated_mention', 'intervals': rounded}]
    if positive == 0:
        return rounded, []
    clusters = _clusters(intervals)
    if len(clusters) <= 1 or positive >= len(clusters):
        return [], []
    return [], [{
        'kind': key,
        'reason': 'distant_intervals_not_separated',
        'intervals': [(round(start, 2), round(end, 2)) for start, end in clusters],
        'mentions': positive,
    }]


def _silent_clauses(text):
    parts = re.split(r'[.!?;,]+|\b(?:but|then|and|while|however)\b', text or '', flags=re.I)
    return [part.strip() for part in parts if part and part.strip()]


def _silent_violations(text):
    """A vocal claim counts unless the same clause negates that claim."""
    hits = []
    text = re.sub(r"\b(?:doesn['’]t|don['’]t|didn['’]t|isn['’]t|wasn['’]t|won['’]t|can['’]t)\b", 'not', text, flags=re.I)
    for clause in _silent_clauses(text):
        for match in _VOCAL_CLAIM.finditer(clause):
            prefix = clause[max(0, match.start() - 40):match.start()]
            if re.search(r'\b(?:no|not|never|without)\b[\w\s\']{0,24}$', prefix, re.I):
                continue
            hit = match.group(0)
            if hit not in hits:
                hits.append(hit)
    return hits


def cue_gaps(vocal_cues, reaction_cues, nod_events, text, silent=False):
    groups = []
    for cue in vocal_cues:
        opts = tuple(_phrase_options(cue))
        if not opts:
            continue
        label = cue.get('official_text') or cue.get('asr_text') or opts[0]
        if groups and groups[-1]['opts'] == opts and cue['start'] - groups[-1]['end'] < DISTANT_GAP_S:
            groups[-1]['end'] = cue['end']
        else:
            groups.append({'opts': opts, 'label': label.strip('.,'), 'start': cue['start'], 'end': cue['end']})
    by_opts = {}
    for group in groups:
        by_opts.setdefault(group['opts'], []).append(group)
    missing_vocal = []
    for opts, items in by_opts.items():
        got = max(_written_count(option, text) for option in opts)
        if got < len(items):
            missing_vocal.append({'word': items[0]['label'], 'needed': len(items), 'written': got,
                                  'starts': [round(item['start'], 2) for item in items]})
    missing_reaction = []
    uncertain = []
    for key, obs in OBSERVATION.items():
        matching = [cue for cue in reaction_cues if cue['observation'] == obs]
        if not matching:
            continue
        missing, unsure = _interval_coverage(
            key, [(cue['start'], cue['end']) for cue in matching], text
        )
        if missing:
            missing_reaction.append({'observation': obs, 'intervals': missing})
        uncertain.extend(unsure)
    missing_nod = []
    if nod_events:
        missing_nod, unsure = _interval_coverage(
            'nodding', [(event['start'], event['end']) for event in nod_events], text
        )
        uncertain.extend(unsure)
    out = {'vocal': missing_vocal, 'reaction': missing_reaction, 'nod': missing_nod}
    violations = _silent_violations(text) if silent else []
    if violations:
        out['silent_vocal'] = violations
    if uncertain:
        out['uncertain'] = uncertain
    return out


def retry_note(gaps):
    """Ask only for omitted cues. Uncertain coverage is not turned into an instruction to invent an action."""
    lines = []
    if gaps.get('silent_vocal'):
        lines.append(
            'The listener audio is silent. Remove claims of speech, laughter, or other audible vocalization, and do not add any.'
        )
    for item in gaps.get('vocal') or []:
        times = ', '.join(f'{t:.2f}s' for t in item['starts'])
        if item['needed'] == 1:
            lines.append(f'Include the audible spoken response "{item["word"]}" around {times} in its position in the clip.')
        else:
            lines.append(
                f'The listener says "{item["word"]}" {item["needed"]} separate times (around {times}). '
                f'Write all {item["needed"]} occurrences in order.'
            )
    if gaps.get('nod') and len(gaps.get('reaction') or []) == 1 and len(gaps['nod']) == 1:
        r = gaps['reaction'][0]
        if r['observation'] == 'brow lowering':
            nod_span = ', '.join(f'{a:.2f}-{b:.2f}s' for a, b in gaps['nod'])
            brow_span = ', '.join(f'{a:.2f}-{b:.2f}s' for a, b in r['intervals'])
            lines.append(
                f'Toward the end of the clip, explicitly describe both the supplied nodding at {nod_span} '
                f'(using the word "nod" or "nodding") and the brief brow lowering at {brow_span} '
                '(using "lowers her brows" or "brows lower"), while keeping the audible laughter and smile. '
                'Do not write "once" unless a single movement is clearly visible.'
            )
            return '\n'.join(lines)
    if gaps.get('nod'):
        span = ', '.join(f'{a:.2f}-{b:.2f}s' for a, b in gaps['nod'])
        lines.append(
            f'The clip includes head nodding at {span}. '
            'Include that nodding at its position using the word "nod" or "nodding". '
            'Do not write "once" unless a single nod is clearly visible.'
        )
    for item in gaps.get('reaction') or []:
        span = ', '.join(f'{a:.2f}-{b:.2f}s' for a, b in item['intervals'])
        obs = item['observation']
        if obs == 'head shaking':
            lines.append(
                f'The clip includes a side-to-side head shake at {span}. '
                'Describe this side-to-side head shake in its position in the sequence while keeping every supplied nod.'
            )
        elif obs == 'brow lowering':
            lines.append(
                f'At {span}, the listener lowers their brows (furrows their forehead). '
                'Write a clause stating that their brows lower or furrow at this point in the sequence. '
                'Do not add a count of one unless a single movement is clearly visible.'
            )
        else:
            lines.append(
                f'Include the supplied {obs} at {span} in its position in the sequence. '
                'Do not add a count of one unless a single movement is clearly visible.'
            )
    return '\n'.join(lines)


def needs_cue_retry(gaps):
    return bool(gaps.get('vocal') or gaps.get('reaction') or gaps.get('nod') or gaps.get('silent_vocal'))


def coverage_failure_flags(gaps):
    flags = []
    if gaps.get('vocal'):
        flags.append('missing_vocal_cue')
    if gaps.get('reaction'):
        flags.append('missing_reaction_cue')
    if gaps.get('nod'):
        flags.append('missing_nod_cue')
    if gaps.get('silent_vocal'):
        flags.append('invented_vocal_on_silence')
    if gaps.get('uncertain'):
        flags.append('coverage_uncertain')
    return flags


def reaction_block(cues, nod_events=None):
    if not cues:
        return ''
    present = []
    for cue in cues:
        if cue['observation'] not in present:
            present.append(cue['observation'])
    lines = [CLASS_NOTE[name] for name in present]
    if nod_events and 'head shaking' not in present:
        lines.append('Keep every supplied nodding interval as nodding in the action sequence.')
    lines.append(ORDER_NOTE)
    lines.append('Do not describe this metadata.')
    return '\n' + '\n'.join(lines) + '\n\nSUPPLEMENTARY VISUAL REACTION INTERVALS:\n' + json.dumps(cues, ensure_ascii=False)


def cues_for(sample_id, shake, expression):
    cues = []
    for item in shake:
        if item['sample_id'] == sample_id and item['retain']:
            event = item['event']
            cues.append({'start': event['start'], 'end': event['end'], 'observation': OBSERVATION['head_shaking']})
    for item in expression:
        if item['sample_id'] == sample_id and item['retain']:
            cues.append({'start': item['start'], 'end': item['end'], 'observation': OBSERVATION[item['type']]})
    cues.sort(key=lambda item: (item['start'], item['observation']))
    for index, cue in enumerate(cues, 1):
        cue['id'] = f'reaction-{index:04d}'
    return cues


def prepare(out):
    selection = read_json(RUN / 'selection.json')
    shake = read_json(RUN / 'reactions/headshake_direction_v1/diagnostics.json')
    expression = read_json(RUN / 'reactions/expression_gate_v3/diagnostics.json')
    samples = []
    for index, row in enumerate(selection['rows'], 1):
        caption = read_json(row['caption_file'])
        nod = caption['input_evidence']['nod_supplement']
        silent = bool(caption['input_evidence']['listener_audio_is_silent'])
        events, official_listener = vocal_inputs(caption, row['duration_s'], silent)
        vocal_cues = compact_vocal_cues(build_vocal_cues(row['duration_s'], silent, events, official_listener))
        samples.append({
            'index': index, 'sample_id': row['sample_id'], 'short': short(row['sample_id']),
            'duration_s': row['duration_s'], 'caption_file': row['caption_file'],
            'prior_caption': caption['caption'], 'nod_events': nod['events'],
            'listener_audio_is_silent': silent, 'vocal_cues': vocal_cues,
            'reaction_cues': cues_for(row['sample_id'], shake, expression),
        })
    for sample in samples:
        text = build_prompt(sample['sample_id'], sample['duration_s'], sample['listener_audio_is_silent'],
                            sample['nod_events'], sample['vocal_cues'], sample['reaction_cues'])
        sample['prompt'] = text
        check_prompt(sample, text)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'samples': len(samples),
        'with_reactions': sum(bool(s['reaction_cues']) for s in samples),
        'reaction_cues': sum(len(s['reaction_cues']) for s in samples),
        'with_vocal': sum(bool(s['vocal_cues']) for s in samples),
        'vocal_cues': sum(len(s['vocal_cues']) for s in samples),
        'silent': sum(s['listener_audio_is_silent'] for s in samples),
    }))


def build_prompt(sample_id, duration, silent, nod_events, vocal_cues, reaction_cues):
    nod_cues = [
        {'id': event.get('id') or event.get('event_id'), 'start': round(float(event['start']), 3),
         'end': round(float(event['end']), 3), 'observation': 'head nodding'}
        for event in nod_events
    ]
    return (
        INSTRUCTION.format(duration=duration)
        + '\n\nlistener_audio_is_silent: ' + json.dumps(bool(silent))
        + '\n\nSUPPLEMENTARY NODDING INTERVALS:\n' + json.dumps(nod_cues, ensure_ascii=False)
        + '\n\nSUPPLEMENTARY VOCAL RESPONSE INTERVALS:\n' + json.dumps(prompt_vocal_cues(vocal_cues), ensure_ascii=False)
        + reaction_block(reaction_cues, nod_events)
    )


def check_prompt(sample, text):
    present = {cue['observation'] for cue in sample['reaction_cues']}
    for name in CLASS_NOTE:
        if name not in present and name in text:
            raise ValueError(f"{sample['short']} names absent reaction class {name}")
    if 'head shaking' in present and 'side-to-side head shake' not in text:
        raise ValueError(f"{sample['short']} shake cue missing the side-to-side instruction")
    if sample['nod_events'] and present and 'Keep every supplied nodding interval as nodding' not in text:
        raise ValueError(f"{sample['short']} reaction+nod clip missing the nod-retention instruction")
    if 'SUPPLEMENTARY VOCAL RESPONSE INTERVALS:' not in text or 'SUPPLEMENTARY NODDING INTERVALS:' not in text:
        raise ValueError(f"{sample['short']} missing nod or vocal section")
    if sample['listener_audio_is_silent'] and sample['vocal_cues']:
        raise ValueError(f"{sample['short']} silent clip still has vocal cues")
    observations = {cue['observation'] for cue in sample['reaction_cues']}
    if not observations <= set(CLASS_NOTE):
        raise ValueError(f"{sample['short']} has a reaction class outside head shake, brow lowering, and brows raised")


def mentions(text):
    return {name: bool(pattern.search(text or '')) for name, pattern in MENTION.items()}


def run(out):
    import numpy as np
    from scipy.signal import resample_poly
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from listen_benchmark.full_raw import prepare as prepare_media
    from listen_benchmark.media import uniform_video
    from listen_benchmark.j_omni import read_audio
    from listen_benchmark.native_caption_quality import generation_failures

    samples = read_json(out / 'manifest.json')
    model = '/private/ckpts/Qwen3-Omni/Instruct'
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=model, trust_remote_code=True, dtype='bfloat16', tensor_parallel_size=2,
        gpu_memory_utilization=float(os.environ.get('VLLM_GPU_MEM', '0.30')),
        max_model_len=16384, max_num_seqs=1, max_num_batched_tokens=16384,
        enable_chunked_prefill=False, limit_mm_per_prompt={'image': 128, 'audio': 1, 'video': 1},
        skip_mm_profiling=True, enforce_eager=True,
    )
    rows = []
    for index, item in enumerate(samples, 1):
        dest = out / 'results' / f"{item['sample_id']}.json"
        if dest.exists():
            rows.append(read_json(dest))
            print(f"[{index}/{len(samples)}] resume {item['short']}", flush=True)
            continue
        caption = read_json(item['caption_file'])
        recipe = caption['audio_recipe']
        wave, sr, audio_identity = read_audio(recipe)
        silent = bool(audio_identity['full_silence'])
        media = prepare_media(caption['source'], 'visual', fast_crop=True, roi_crop=True)
        audio = wave[:, 0]
        if sr != 16000:
            g = math.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        frames, _evidence = uniform_video(media, 8)
        vocal_cues = [] if silent else item['vocal_cues']
        text = build_prompt(item['sample_id'], item['duration_s'], silent, item['nod_events'],
                            vocal_cues, item['reaction_cues'])
        second_per_grid_ts = [float(processor.image_processor.temporal_patch_size) / 8.0]
        params = SamplingParams(temperature=0, max_tokens=4096, repetition_penalty=1.05,
                                structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][ -~]*'))

        def generate_once(prompt_text):
            rendered = processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text}, {'type': 'video', 'video': 'listener://decoded'}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tag = f"reaction-pilot-{item['sample_id']}-{time.time_ns()}"
            request = {
                'prompt': rendered,
                'multi_modal_data': {'video': [frames], 'audio': [audio]},
                'mm_processor_kwargs': {'use_audio_in_video': True, 'fps': 8, 'do_sample_frames': False,
                                        'second_per_grid_ts': second_per_grid_ts},
                'multi_modal_uuids': {'audio': [tag], 'video': [tag]},
            }
            llm.reset_mm_cache()
            return llm.generate([request], params, use_tqdm=False)[0].outputs[0]

        started = time.monotonic()
        base_text = text
        output = generate_once(base_text)
        clean_text = re.sub(r'^Assistant:\s*', '', output.text).strip()
        attempts = [clean_text]
        gaps = cue_gaps(vocal_cues, item['reaction_cues'], item['nod_events'], clean_text, silent=silent)
        missing = {
            'vocal': list(gaps['vocal']),
            'reaction': list(gaps['reaction']),
            'nod': list(gaps['nod']),
            'silent_vocal': list(gaps.get('silent_vocal') or []),
        }
        for rep in (1.05, 1.10):
            if not needs_cue_retry(gaps):
                break
            retry_prompt = base_text + '\n\n' + retry_note(missing)
            if rep != 1.05:
                params = SamplingParams(temperature=0, max_tokens=4096, repetition_penalty=rep,
                                        structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][ -~]*'))
            second = generate_once(retry_prompt)
            second_text = re.sub(r'^Assistant:\s*', '', second.text).strip()
            attempts.append(second_text)
            second_gaps = cue_gaps(
                vocal_cues, item['reaction_cues'], item['nod_events'], second_text, silent=silent
            )
            for v in second_gaps['vocal']:
                if v not in missing['vocal']:
                    missing['vocal'].append(v)
            for r in second_gaps['reaction']:
                if r not in missing['reaction']:
                    missing['reaction'].append(r)
            if second_gaps['nod'] and not missing['nod']:
                missing['nod'] = list(second_gaps['nod'])
            for item_hit in second_gaps.get('silent_vocal') or []:
                if item_hit not in missing['silent_vocal']:
                    missing['silent_vocal'].append(item_hit)
            old_total = len(gaps['vocal']) + len(gaps['reaction']) + bool(gaps['nod']) + len(gaps.get('silent_vocal') or [])
            new_total = (
                len(second_gaps['vocal']) + len(second_gaps['reaction']) + bool(second_gaps['nod'])
                + len(second_gaps.get('silent_vocal') or [])
            )
            if new_total <= old_total:
                output = second
                clean_text = second_text
                gaps = second_gaps
                text = retry_prompt
        flags = list(generation_failures(clean_text, output.finish_reason))
        flags.extend(coverage_failure_flags(gaps))
        result = {
            'sample_id': item['sample_id'], 'short': item['short'], 'index': item['index'],
            'vocal_cues': vocal_cues, 'reaction_cues': item['reaction_cues'], 'nod_events': item['nod_events'],
            'prior_caption': item['prior_caption'], 'caption': clean_text, 'attempts': attempts,
            'remaining_gaps': gaps,
            'prior_mentions': mentions(item['prior_caption']), 'new_mentions': mentions(clean_text),
            'finish_reason': output.finish_reason, 'completion_tokens': len(output.token_ids),
            'seconds': round(time.monotonic() - started, 3), 'failures': flags,
            'listener_audio_is_silent': silent, 'prompt': text,
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        rows.append(result)
        print(json.dumps({'done': index, 'short': item['short'], 'cues': item['reaction_cues'],
                          'prior': result['prior_mentions'], 'new': result['new_mentions'],
                          'caption': output.text}, ensure_ascii=False), flush=True)
    summary = {'samples': len(rows), 'classes': {}}
    for name in OBSERVATION:
        supplied = [row for row in rows if any(cue['observation'] == OBSERVATION[name] for cue in row['reaction_cues'])]
        summary['classes'][name] = {
            'clips_with_cue': len(supplied),
            'old_caption_mentioned': sum(row['prior_mentions'][name] for row in supplied),
            'new_caption_mentioned': sum(row['new_mentions'][name] for row in supplied),
            'mentioned_without_cue': sum(row['new_mentions'][name] for row in rows if row not in supplied),
        }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--run', type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.run.resolve())
    else:
        run(args.run.resolve())


if __name__ == '__main__':
    main()
