#!/usr/bin/env python3
"""Frozen 50-clip Omni natural-language pilot, with virtual purified listener audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from p2_listen.contracts import atomic_json, digest, sha

PROMPT = '''Watch and listen to this {duration:.3f}-second clip of a listener. Write one coherent English paragraph describing the listener's observable visual behavior and audible vocal responses across the whole clip.
Describe how actions unfold in order, which expressions or postures persist, which actions happen together, and which actions genuinely repeat. Use natural temporal language such as initially, then, while, and toward the end when useful. Describe a sustained state once rather than turning it into repeated actions. Give a repetition count only when clearly supported.
Describe directly visible movements of the face, eyes, head, hands, and body, including brief movements when visible. Use action names such as nodding only when the motion supports them. Describe audible laughter or other nonverbal vocalizations, and quote clearly intelligible spoken responses exactly as heard. Do not guess unclear words. Distinguish visible mouth movement from an audible vocal response, and mention periods without vocalization when relevant.
Use only the supplied video and synchronized listener audio. Do not infer thoughts, intentions, feelings, agreement, communicative meaning, or causes; avoid interpretive claims such as indicating, suggesting, or as if. Do not invent actions or sounds to fill a quiet interval. Keep a clear subject and write concrete, connected sentences without redundant descriptions or unrelated appearance details.
Output only the natural-language paragraph. Do not output JSON, a list, individual event intervals, or numeric timestamps.'''

MEDIA = dict(fps=8, max_frames=128, max_edge=768, max_audio_seconds=16)
MODEL = '/private/ckpts/Qwen3-Omni/Instruct'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(run):
    import numpy as np
    import soundfile as sf
    from listen_audio_postprocess.recipes import read_listener_audio, masks, make_recipe
    from listen_audio_postprocess.bulk import derive
    from compare_listener_official_six import load_metadata
    source = ROOT / 'experiments/listen_native_simple50_20261024T064600Z/manifest.json'
    manifest = read(source)
    page = read(ROOT / 'data_previews/listen_benchmark_review/native_simple/comparison.json')
    assert [r['sample_id'] for r in manifest] == [s['id'] for s in page['samples']]
    assert len(manifest) == len({r['sample_id'] for r in manifest}) == 50
    fused_path = ROOT / 'data_previews/listen_benchmark_review/existing_fusion_verbatim/review.json'
    fused = {s['sample_id']: s for s in read(fused_path)['samples']}
    old_recipes = ROOT / 'experiments/p2_full_audio_postprocess_crop384_20261026/view/audio_postprocess/recipes'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'recipes').mkdir(exist_ok=True)
    (run / 'raw_metadata').mkdir(exist_ok=True)
    def one(row):
        sid = row['sample_id']; original = read(row['source_reference']['metadata_file']['path'])
        assert original['sample_id'] == sid
        assert abs(original['start'] - row['source_reference']['source_start_s']) < 1e-6
        prior = old_recipes / (sid + '.json')
        if prior.exists():
            recipe = read(prior)
            assert recipe['sample_id'] == sid
            assert abs(recipe['provenance']['clip_absolute_start'] - original['start']) < 1e-6
            assert abs(recipe['source']['samples']/recipe['source']['sample_rate'] - row['duration_s']) < 1e-6
            a, ar = sf.read(row['target_audio'], dtype='float32', always_2d=True)
            b, br = sf.read(recipe['source']['path'], dtype='float32', always_2d=True)
            assert ar == br and a.shape == b.shape
            difference = float(np.max(np.abs(a-b)))
            assert difference <= 1/32768 + 1e-7, (sid, difference)
            provenance = dict(mode='reuse_completed_recipe', source=str(prior), sha256=sha(prior),
                              package_vs_recipe_max_abs_pcm_difference=difference)
        else:
            raw, prov = load_metadata(original['sources']['listener'])
            atomic_json(run / 'raw_metadata' / (sid + '.json'), dict(raw=raw, provenance=prov), immutable=True)
            job = dict(sample_id=sid, target_audio=row['target_audio'], duration_s=row['duration_s'],
                       source_audio_sha256=row['media_hashes']['target_audio'], start=original['start'],
                       integrated_sha256=digest(fused[sid]), events=fused[sid]['events'])
            record, audit = derive(job, raw, prov)
            recipe = make_recipe(sid, row['target_audio'], row['media_hashes']['target_audio'],
                                 record['official_listener'], record['extra_mute_intervals'], record['provenance'])
            provenance = dict(mode='same_approved_policy_for_historical_sample', audit=audit,
                              integrated_source=str(fused_path))
        recipe_path = run / 'recipes' / (sid + '.json')
        atomic_json(recipe_path, recipe, immutable=True)
        wave, rate, identity = read_listener_audio(recipe)
        gate, extra, info = masks(recipe)
        assert wave.shape == (recipe['source']['samples'], 1)
        assert abs(len(wave)/rate-row['duration_s']) < 1e-6
        assert np.all(wave[(gate*extra)==0] == 0)
        result = dict(row, listener_audio_recipe=str(recipe_path), recipe_sha256=digest(recipe),
                      audio_preflight=dict(identity, duration_s=len(wave)/rate, sample_rate=rate,
                                           samples=len(wave), positive_samples=int(np.count_nonzero(gate*extra)),
                                           full_silence=not bool(np.any(wave)), provenance=provenance))
        print(sid, provenance['mode'], 'silent' if not np.any(wave) else 'audible', flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, manifest))
    atomic_json(run / 'manifest.json', rows, immutable=True)
    protocol = dict(schema='listen-native-caption-pilot-v1', model=MODEL, media=MEDIA, prompt=PROMPT,
                    input_manifest_sha256=sha(source), prepared_manifest_sha256=sha(run/'manifest.json'),
                    temperature=0, max_tokens=4096, max_model_len=49152, batch_size=1,
                    gpus=list(range(8)), single_paragraph_regex=r'[A-Za-z][^\n\r]*',
                    repetition_penalty=1.05, decode_graph=True,
                    native_alignment='explicit_vllm_grid_time_v1', human_ground_truth=False,
                    training_admitted=False, original_results_changed=False, new_wav_files=False,
                    scope='50 historical native_simple samples; natural-language caption, not an audio-effect ablation')
    atomic_json(run / 'protocol.json', protocol, immutable=True)
    (run / 'prompt.txt').write_text(PROMPT + '\n')
    paths = [Path(__file__), ROOT/'src/listen_audio_postprocess/recipes.py', ROOT/'src/listen_audio_postprocess/bulk.py',
             ROOT/'src/listen_benchmark/media.py', ROOT/'src/listen_labeling/media.py',
             ROOT/'src/listen_benchmark/official_audio_gate.py', ROOT/'src/listen_benchmark/chinese_asr_filter.py',
             ROOT/'src/listen_benchmark/official_event_alignment.py', ROOT/'scripts/compare_listener_official_six.py']
    hashes = {}
    for p in paths:
        relative = p.relative_to(ROOT); dest = run/'code'/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest); hashes[str(relative)] = sha(p)
    atomic_json(run/'code_hashes.json', hashes, immutable=True)
    atomic_json(run/'preflight.json', dict(samples=len(rows), passed=True,
                reused_recipes=sum(r['audio_preflight']['provenance']['mode']=='reuse_completed_recipe' for r in rows),
                full_silence=sum(r['audio_preflight']['full_silence'] for r in rows), preserved_lengths=True))


def worker(run, rank):
    import numpy as np
    from scipy.signal import resample_poly
    from listen_benchmark.media import prepare as prepare_media, uniform_video
    from listen_audio_postprocess.recipes import read_listener_audio
    assert rank in range(8)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(rank), 'Worker must use its assigned GPU'
    protocol = read(run/'protocol.json'); all_rows = read(run/'manifest.json')
    assert sha(run/'manifest.json') == protocol['prepared_manifest_sha256']
    for p, expected in read(run/'code_hashes.json').items():
        assert sha(ROOT/p) == expected, f'Code changed: {p}'
    # Greedy duration balancing across eight independent single-request engines.
    bins = [[] for _ in range(8)]; loads = [0.]*8
    pending_rows=all_rows
    if protocol.get('task')=='nod_guided_revision':
        pending_rows=[r for r in all_rows if r.get('nod_supplement',{}).get('events')]
    for row in sorted(pending_rows, key=lambda r:-r['duration_s']):
        i = min(range(8), key=lambda k:loads[k]); bins[i].append(row); loads[i] += row['duration_s']
    rows = bins[rank]
    state = run/'workers'/f'gpu{rank}.json'
    def status(stage, **extra):
        atomic_json(state, dict(stage=stage, rank=rank, gpu=rank, pid=os.getpid(), host=os.uname().nodename,
                                total=len(rows), time=time.time(), **extra))
    status('loading_model')
    if not rows:
        status('complete',completed=0,seconds=0.,model_loaded=False)
        return
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    memory=protocol.get('gpu_memory_by_gpu',{}).get(str(rank),.85)
    llm = LLM(model=MODEL, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=memory,
              max_model_len=49152, max_num_seqs=1, max_num_batched_tokens=49152,
              enable_chunked_prefill=False, limit_mm_per_prompt={'image':128,'audio':1,'video':1},
              skip_mm_profiling=True, enforce_eager=False,
              compilation_config=dict(mode=0,cudagraph_mode='FULL_DECODE_ONLY',
                                      cudagraph_capture_sizes=[1],max_cudagraph_capture_size=1))
    params = SamplingParams(temperature=0, max_tokens=4096,repetition_penalty=1.05,
                            structured_outputs=StructuredOutputsParams(regex=r'[A-Za-z][^\n\r]*'))
    def prepare_input(row):
        assert sha(row['target_video']) == row['media_hashes']['target_video']
        recipe = read(row['listener_audio_recipe']); assert digest(recipe) == row['recipe_sha256']
        media = prepare_media(row, 'visual', MEDIA)
        wave, sr, audio_identity = read_listener_audio(recipe)
        assert audio_identity['processed_waveform_sha256'] == row['audio_preflight']['processed_waveform_sha256']
        audio = wave[:,0]
        if sr != 16000:
            g = math.gcd(sr,16000); audio = resample_poly(audio,16000//g,sr//g).astype(np.float32)
        media['audio'] = audio
        frames, evidence = uniform_video(media, 8)
        evidence.update(audio_origin='virtual_postprocessed_listener_recipe', audio_identity=audio_identity,
                        audio_encoder_rate=16000, audio_encoder_samples=len(audio),
                        second_per_grid_ts=[float(processor.image_processor.temporal_patch_size)/8],
                        video_evidence=media['meta']['video'])
        if protocol.get('task') in ('nod_guided_revision','nod_guided_direct_annotation'):
            from listen_benchmark.native_caption_nod import render
            supplement=row['nod_supplement']
            assert sha(supplement['detector_file'])==supplement['detector_sha256']
            if protocol.get('task')=='nod_guided_revision':
                assert sha(supplement['prior_result_file'])==supplement['prior_result_sha256']
            text=render(row)
            evidence['nod_supplement']=supplement
        else:
            text = PROMPT.format(duration=row['duration_s'])
        rendered = processor.apply_chat_template([dict(role='user', content=[dict(type='text', text=text),
                     dict(type='video', video='listener://decoded')])], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        request = dict(prompt=rendered, multi_modal_data=dict(video=[frames], audio=[audio]),
                       mm_processor_kwargs=dict(use_audio_in_video=True, fps=8, do_sample_frames=False,
                                                second_per_grid_ts=evidence['second_per_grid_ts']))
        return request, evidence, text
    started = time.monotonic(); done = 0
    status('running', completed=0)
    # Two CPU preparation threads, at most two pending clips per GPU.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {}; next_index = 0
        for i, row in enumerate(rows):
            for j in range(next_index, min(len(rows),i+2)):
                pending[j] = pool.submit(prepare_input, rows[j]); next_index = j+1
            request,evidence,text = pending.pop(i).result()
            sid = row['sample_id']; dest = run/'results'/(sid+'.json')
            if dest.exists():
                prior = read(dest)
                assert prior['recipe_sha256']==row['recipe_sha256'] and prior['prompt']==text
                done+=1; continue
            begin=time.monotonic(); out=llm.generate([request],params,use_tqdm=False)[0]; item=out.outputs[0]
            raw=item.text
            flags=[]
            if item.finish_reason!='stop': flags.append('generation_not_finished')
            if not raw.strip(): flags.append('empty_caption')
            if re.search(r'\b(indicat\w*|suggest\w*|as if|seems? to|appears? to)\b',raw,re.I): flags.append('possible_interpretation_review')
            if re.search(r'\b\d+(?:\.\d+)?\s*(?:seconds?|–|—)',raw,re.I): flags.append('numeric_timing_review')
            if (protocol.get('task') in ('nod_guided_revision','nod_guided_direct_annotation')
                    and row['nod_supplement']['events'] and not re.search(r'\bnod\w*\b',raw,re.I)):
                flags.append('nod_cue_not_explicitly_named_review')
            atomic_json(dest,dict(sample_id=sid, prompt=text, rendered_prompt=request['prompt'], caption=raw,
                        finish_reason=item.finish_reason, completion_tokens=len(item.token_ids),
                        prompt_tokens=len(out.prompt_token_ids), seconds=time.monotonic()-begin,
                        recipe_sha256=row['recipe_sha256'], input_evidence=evidence, review_flags=flags,
                        human_ground_truth=False, training_admitted=False, gpu=rank,
                        decoding=dict(single_paragraph_regex=r'[A-Za-z][^\n\r]*',repetition_penalty=1.05,
                                      decode_graph=True)),immutable=True)
            done+=1;status('running',completed=done,seconds=time.monotonic()-started)
            print(sid,len(item.token_ids),item.finish_reason,flags,flush=True)
    status('complete',completed=done,seconds=time.monotonic()-started)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','worker']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--rank',type=int);a=p.parse_args();run=a.run.resolve()
    if a.command=='prepare': prepare(run)
    else: worker(run,a.rank)
