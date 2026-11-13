"""Frozen Qwen3-Omni annotation backend, with blind separate modality views."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .contracts import CLASSES, digest, read, save, sha
from .media import ROOT, verify
from .prompts import ASR_PROMPT, ASR_SCHEMA, VERSION, parse_label_response, prompt, schema


def model_identity(path):
    path = Path(path)
    index = read(path/'model.safetensors.index.json')
    files = sorted(set(index['weight_map'].values()))
    return dict(root=str(path.resolve()), index_sha256=sha(path/'model.safetensors.index.json'),
        config_sha256=sha(path/'config.json'),
        preprocessing_files={p.name:sha(p) for p in sorted([*path.glob('*.json'),*path.glob('*.jinja'),
            *[path/n for n in ['merges.txt','vocab.txt'] if (path/n).exists()]])},
        shards={name: {'bytes': (path/name).stat().st_size, 'mtime_ns': (path/name).stat().st_mtime_ns} for name in files},
        weight_identity_level='index/config SHA256 plus shard size/mtime; no claim of full-shard cryptographic verification')


def request(processor, mod, data, text, opaque, fps=8):
    content = [{'type': 'text', 'text': text},
               {'type': 'video', 'video': 'listener://anonymous'} if mod == 'visual' else {'type': 'audio', 'audio': 'listener://anonymous'}]
    rendered = processor.apply_chat_template([{'role': 'user', 'content': content}],tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
    kind = 'video' if mod == 'visual' else 'audio'
    options = dict(use_audio_in_video=False)
    if mod == 'visual':
        options.update(fps=fps, do_sample_frames=False,
            second_per_grid_ts=[float(processor.image_processor.temporal_patch_size)/fps])
    return dict(prompt=rendered,multi_modal_data={kind:[data]},mm_processor_kwargs=options,
                multi_modal_uuids={kind:[opaque]})


def infer(work):
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    work = Path(work); contract, _ = verify(work); policy = contract['policy']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(policy['physical_gpu']): raise ValueError('Bind only authorized physicalGPU7')
    gpu = subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if gpu[0].strip() != policy['node11_gpu_uuid']: raise ValueError('Wrong physical GPU')
    free, total = [int(v.strip()) for v in gpu[1:]]
    if free < total*policy['gpu_memory_utilization'] + policy['gpu_reserve_gib']*1024:
        raise RuntimeError('Insufficient freeGPU7 memory with reserve; no other job stopped')
    model = model_identity(policy['model'])
    identity = dict(policy_identity=contract['policy_identity'], model=model, prompt_version=VERSION,
                    source=contract['source'], versions={n:importlib.metadata.version(n) for n in ['transformers','vllm','torch','numpy']})
    identity_hash = digest(identity)
    save(work/'detector_contract.json',dict(identity=identity,identity_hash=identity_hash,
        detector_based=True,human_calibrated=False,pilot_only=True,source_pipeline='native-av-first-plus-nod-vocal-reaction-v1',
        adaptations=['separate video-only/audio-only views','explicit class status JSON','own ASR on each waveform','no GT gate for generated audio']))
    processor = AutoProcessor.from_pretrained(policy['model'],trust_remote_code=True,local_files_only=True)
    llm = LLM(model=policy['model'],trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,
        gpu_memory_utilization=policy['gpu_memory_utilization'],cpu_offload_gb=policy['cpu_offload_gb'],
        max_model_len=policy['max_model_len'],max_num_seqs=1,max_num_batched_tokens=policy['max_model_len'],
        enable_chunked_prefill=False,limit_mm_per_prompt={'image':128,'audio':1,'video':1},skip_mm_profiling=True,enforce_eager=True)
    from run_listen_native_caption_vocal_pilot import normalize_asr_cue_for_english_prompt

    def generate(req, structure, dest):
        attempts=[]
        for budget in [policy['max_output_tokens'], policy['max_output_tokens']*2]:
            params=SamplingParams(temperature=0.,seed=policy['seed'],max_tokens=budget,
                structured_outputs=StructuredOutputsParams(json=structure))
            response=llm.generate([req],sampling_params=params,use_tqdm=False)[0].outputs[0]
            attempts.append(dict(text=response.text,finish_reason=response.finish_reason,max_tokens=budget))
            save(dest,dict(detector_identity=identity_hash,attempts=attempts))
            if response.finish_reason != 'stop': continue
            return json.loads(response.text)
        raise ValueError('Structured detection truncated; do not invent missing labels')

    prepared=read(work/'PREPARED.json')
    if sha(work/'prepared_pairs.jsonl') != prepared['prepared_pairs_sha256']: raise ValueError('Prepared manifest changed')
    from .contracts import rows
    unique={m['media_id']:m for row in rows(work/'prepared_pairs.jsonl') for m in row['prepared_media'].values()}
    for opaque, media in unique.items():
        folder=Path(media['folder']); output=folder/'detection.json'
        cuefile=folder/'candidates.json'
        media_id=digest([media,identity_hash,sha(cuefile) if cuefile.exists() else None])
        if output.exists():
            old=read(output)
            if old.get('cache_identity') != media_id: raise ValueError('Detector/media changed; use a new output directory')
            continue
        result={'visual':{},'audio':{}}; errors={}; auxiliary=[]
        candidates=read(cuefile) if cuefile.exists() else {}
        for mod in CLASSES:
            if mod in media['errors']:
                result[mod]={c:dict(state='error',evidence=media['errors'][mod]) for c in CLASSES[mod]};continue
            if mod=='audio' and 'audio' not in media:
                result[mod]={c:dict(state='unknown',evidence='audio_not_available_for_this_media') for c in CLASSES[mod]};continue
            if mod=='visual' and 'visual25' not in media:
                result[mod]={c:dict(state='error',evidence='video_not_available') for c in CLASSES[mod]};continue
            if mod=='audio' and media['full_silence']:
                result[mod]={c:dict(state='absent',evidence='supplied_full_waveform_is_exactly_zero') for c in CLASSES[mod]};continue
            windows=[]
            for wi, window in enumerate(media['windows']):
                try:
                    if mod=='visual':
                        if sha(window['frames'])!=window['frames_sha256']:raise ValueError('Video window changed')
                        data=np.load(window['frames'],allow_pickle=False)
                        hints={k:v for k,v in candidates.items() if k in ['frame_coverage','cues','missing_candidates_are_not_absence']}
                        hints['cues']=[dict(type=c['type'],start=max(0.,c['start']-window['start_s']),
                            end=min(window['end_s'],c['end'])-window['start_s']) for c in candidates.get('cues',[])
                            if max(c['start'],window['start_s'])<min(c['end'],window['end_s'])]
                        asr=None
                    else:
                        if sha(media['audio'])!=media['audio_file_sha256']:raise ValueError('Audio changed')
                        wave,sr=sf.read(media['audio'],dtype='float32');assert sr==16000
                        data=wave[round(window['start_s']*sr):round(window['end_s']*sr)]
                        asr=generate(request(processor,mod,data,ASR_PROMPT,opaque+f'-audio-{wi}'),ASR_SCHEMA,folder/f'audio_{wi:03d}_asr_raw.json')
                        normalized=[normalize_asr_cue_for_english_prompt(text,None,'candidate vocalization')[0]
                                    for text in [asr['transcript'],*asr['nonlexical_syllables']]]
                        asr=dict(asr,normalized_candidate_texts=normalized,official_transcript_used=False);hints=None
                    text=prompt(mod,window['end_s']-window['start_s'],hints,asr)
                    (folder/f'{mod}_{wi:03d}_prompt.txt').write_text(text+'\n')
                    raw=generate(request(processor,mod,data,text,opaque+f'-{mod}-{wi}'),schema(mod),folder/f'{mod}_{wi:03d}_raw.json')
                    parsed=parse_label_response(raw,mod)
                    if mod=='visual' and candidates.get('frame_coverage',0.) < 1:
                        for label in parsed['classes'].values():
                            if label['state']=='absent':label.update(state='unknown',evidence='incomplete_face_detection; '+label['evidence'])
                    windows.append(parsed);auxiliary.append(dict(modality=mod,window=window,asr=asr,response=parsed))
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}';errors[f'{mod}:{wi}']=error
                    windows.append(dict(classes={c:dict(state='error',evidence=error) for c in CLASSES[mod]}))
            for c in CLASSES[mod]:
                states=[w['classes'][c]['state'] for w in windows]
                state='present' if 'present' in states else 'error' if 'error' in states else 'unknown' if 'unknown' in states or not states else 'absent'
                evidence=' | '.join(w['classes'][c]['evidence'] for w in windows) or 'no_complete_observation'
                result[mod][c]=dict(state=state,evidence=evidence)
        save(output,dict(cache_identity=media_id,detector_identity=identity_hash,media_identity=media['identity'],
            labels=result,errors=errors,window_observations=auxiliary,detector_based=True,human_verified=False,pilot_only=True))
        save(work/'progress.json',dict(media_id=opaque,complete_media=sum((Path(m['folder'])/'detection.json').exists() for m in unique.values()),total_media=len(unique)))
        print('Detected anonymous media',opaque[:12],errors,flush=True)
    save(work/'DETECTION_COMPLETE.json',dict(media=len(unique),detector_identity=identity_hash,completed_at=time.time(),pilot_only=True,
        detections={mid:sha(Path(m['folder'])/'detection.json') for mid,m in unique.items()}))
