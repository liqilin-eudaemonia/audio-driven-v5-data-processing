"""Read selected J source clips without generating intermediate audio/video files."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
from functools import lru_cache

import numpy as np
from PIL import Image
import soundfile as sf

from p2_listen.contracts import ROOT, atomic_json, digest, read, safe_id, sha
from p2_listen.config import load as load_config
from p3_listen.prompts import GEMMA, contract as prompt_contract
from listen_benchmark.j_omni import check_file, read_audio
from listen_benchmark.j_omni_crop import effective_row, caption_crop_info, REVISION
from . import VERSION


def code_identity():
    files = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'src/p3_listen/prompts.py', ROOT/'src/p3_listen/features.py',
        ROOT/'src/p2_listen/features.py', ROOT/'src/p2_listen/runtime.py',
        ROOT/'src/p2_plan_cache/encoding.py', ROOT/'src/p2_plan_cache/runtime.py',
        ROOT/'src/listen_benchmark/j_omni.py', ROOT/'src/listen_benchmark/j_omni_crop.py',
        ROOT/'src/listen_benchmark/frame_crop.py']
    return {str(p): sha(p) for p in sorted(files)}


def caption_prompt(caption):
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError('Missing original listener caption')
    if '<|' in caption or '|>' in caption:
        raise ValueError('Model control tokens in caption')
    return GEMMA + '\n\nListener response description:\n' + caption


@lru_cache(maxsize=128)
def paired_result(path, size, mtime_ns):
    return read(path)


def freeze_sample(row, crop_config):
    sid = safe_id(row['sid'])
    with open(row['file'], 'rb') as stream:
        stream.seek(row['offset']); raw = stream.readline()
    rec = json.loads(raw)
    if rec['sample_id'] != sid or rec['split'] != row['split'] or not rec['retained']:
        raise ValueError('Selection/statistics mismatch: '+sid)
    cap_path = Path(rec['caption_file']); cap_bytes = cap_path.read_bytes(); cap = json.loads(cap_bytes)
    cap_hash = hashlib.sha256(cap_bytes).hexdigest()
    if cap['sample_id'] != sid or cap['caption'] != rec['caption'] or cap_hash != rec['caption_file_sha256']:
        raise ValueError('Original caption changed: '+sid)
    measurement = read(rec['quality']['measurement_file'])
    original = cap.get('original_source', cap['source'])
    if original != measurement['source']:
        raise ValueError('Quality measurement/source mismatch')
    applied, provenance = effective_row({'sample_id': sid, 'source': original}, crop_config, decision=rec['quality'])
    source = applied['source']
    if provenance['revision'] != REVISION or source['video_transform']['crop'] != measurement['crop']['new']:
        raise ValueError('Final corrected crop not applied')
    check_file(source['source_result'], source['source_result_sha256'])
    stat = Path(source['source_result']).stat()
    clip = paired_result(source['source_result'],stat.st_size,stat.st_mtime_ns)['clips'][source['source_clip_index']]
    if (clip['sample_id'] != sid or clip['split'] != row['split'] or
        clip['listener_id'] != source['listener_id'] or clip['speaker_id'] != source['speaker_id'] or
        clip['listener_id'] == clip['speaker_id'] or clip['sources']['listener'] != source['source'] or
        abs(clip['start']-source['source_start_s']) > 1e-7 or abs(clip['end']-source['source_end_s']) > 1e-7):
        raise ValueError('Paired source identity/time mismatch')
    speaker = clip['normalized_audio']['speaker']
    if (speaker['contract']['source'] != clip['sources']['speaker'] or
        Path(clip['sources']['speaker']['member']).stem != source['speaker_id'] or
        cap['audio_recipe']['listener_id'] != source['listener_id']):
        raise ValueError('Speaker/listener audio swapped')
    files = {str(cap_path): cap_hash, source['source_result']: source['source_result_sha256'],
             rec['quality']['measurement_file']: sha(rec['quality']['measurement_file'])}
    return dict(schema=VERSION, sample_id=sid, split=row['split'], source=source, speaker=speaker,
        audio_recipe=cap['audio_recipe'], listener_audio_identity=cap['input_evidence']['audio_identity'],
        caption=cap['caption'], prompt=caption_prompt(cap['caption']), caption_sha256=cap_hash,
        caption_origin=caption_crop_info(cap), caption_crop=cap['source']['video_transform']['crop'],
        crop_provenance=provenance, statistics=rec, statistic_source=dict(path=row['file'], offset=row['offset'],
        line_sha256=hashlib.sha256(raw).hexdigest()), selection=row, source_files=files,
        training_admitted=bool(rec.get('training_admitted', False)), human_ground_truth=False)


def prepare(config, sample_list, work, selection=None, review=None):
    work = Path(work).resolve(); config = Path(config).resolve(); sample_list = Path(sample_list).resolve()
    selection = Path(selection or ROOT/'experiments/listen_j_final_quality_selection_20261101/selection.sqlite')
    crop_path = ROOT/'experiments/listen_j_omni_20261030/config.json'
    c = load_config(config)
    if c['data']['fps'] != 25 or c['data']['size'] != 512:
        raise ValueError('J source reader currently uses the verified 25 fps, 512 px grid')
    with sample_list.open(encoding='utf-8-sig', newline='') as stream:
        requested = list(csv.DictReader(stream))
    ids = [safe_id(x['sample_id']) for x in requested]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty/duplicate sample list')
    con = sqlite3.connect('file:'+str(selection)+'?mode=ro', uri=True); con.row_factory = sqlite3.Row
    rows = []
    try:
        for entry in requested:
            found = con.execute('SELECT * FROM samples WHERE sid=?', (entry['sample_id'],)).fetchone()
            if found is None or entry['split'] != found['split'] or abs(float(entry['duration_s'])-found['duration']) > 1e-7:
                raise ValueError('Selected sample/split/duration differs from final selection')
            rows.append(dict(found))
    finally:
        con.close()
    c = dict(c, output=str(work), data=dict(c['data']))
    c['data'].pop('audio_postprocess', None)
    crop_config = read(crop_path)
    reviewed = {} if review is None else {r['id']: r for r in read(review)['samples']}
    if review is not None and set(ids) != set(reviewed):
        raise ValueError('Review sample list must match exactly; do not substitute clips')
    entries = {}
    for row in rows:
        item = freeze_sample(row, crop_config); sid = item['sample_id']
        if sid in reviewed:
            item['review'] = reviewed[sid]
            if item['caption'] != reviewed[sid]['caption'] or item['source']['video_transform']['crop'] != reviewed[sid]['final_crop']:
                raise ValueError('Review caption/final crop mismatch')
        atomic_json(work/'samples'/(sid+'.json'), item, immutable=True)
        entries[sid] = digest(item)
    sources = {str(p): sha(p) for p in (config, sample_list, crop_path)}
    if review is not None:
        sources[str(Path(review).resolve())] = sha(review)
    value = dict(schema=VERSION, config=c, code=code_identity(), prompts=prompt_contract(), sources=sources,
        entries=entries, sample_ids=ids, selection_sqlite=str(selection),
        splits={split:[r['sid'] for r in rows if r['split']==split] for split in ('train','dev','test')},
        guidance='original_whole_clip_caption_verbatim', event_labels='not_mapped_from_open_statistics',
        source_admission_flags_preserved=True, intermediate_wav=False, intermediate_video=False)
    atomic_json(work/'manifest.json', value, immutable=True)
    return value


def open_manifest(work):
    value = read(Path(work)/'manifest.json')
    if value['schema'] != VERSION or value['code'] != code_identity() or value['prompts'] != prompt_contract():
        raise ValueError('J cache code/prompt changed; prepare a new work directory')
    for path, h in value['sources'].items():
        if sha(path) != h:
            raise ValueError('Frozen J input list/config changed: '+path)
    return value


def sample(work, manifest, sid):
    item = read(Path(work)/'samples'/(safe_id(sid)+'.json'))
    if manifest['entries'].get(sid) != digest(item) or item['prompt'] != caption_prompt(item['caption']):
        raise ValueError('Frozen J sample/prompt changed')
    return item


class SourceVideoGap(ValueError):
    """A source frame cannot cover the requested grid without explicit review."""


def decode_frames(source, fps=25, limit=None, gap_overrides=None):
    import av
    from listen_visual_quality.measurement import SourceBytes
    from listen_benchmark.frame_crop import crop_original_frame_roi
    count = math.ceil(source['duration_s']*fps-1e-7)
    if limit is not None:
        count = min(count, limit)
    desired = [source['source_start_s']+i/fps for i in range(count)]
    pixels = []; times = []; previous = None; previous_t = None; cached_pts = None; cropped = None
    with SourceBytes(source['source']) as member, av.open(member) as container:
        stream = container.streams.video[0]; stream.thread_type = 'SLICE'; stream.codec_context.thread_count = 1
        if [stream.width, stream.height] != source['video_transform']['source_size']:
            raise ValueError('Native video geometry changed')
        native_fps = float(stream.average_rate)
        container.seek(int(max(0,source['source_start_s']-1)/float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError('Missing source PTS')
            t = float(frame.pts*frame.time_base)
            while len(pixels)<count and t+1e-8>=desired[len(pixels)]:
                wanted = desired[len(pixels)]
                chosen = previous if previous is not None and abs(previous_t-wanted)<=abs(t-wanted) else frame
                ts = float(chosen.pts*chosen.time_base)
                if abs(ts-wanted)>1/native_fps+1e-5:
                    override = (gap_overrides or {}).get(str(len(pixels)))
                    actual = dict(target_pts_s=wanted, previous_pts_s=previous_t,
                                  next_pts_s=t, selected_pts_s=ts)
                    if (override is None or previous_t is None or set(override)!=set(actual)
                            or any(abs(override[k]-v)>1e-8 for k,v in actual.items())):
                        raise SourceVideoGap('Source video gap')
                if chosen.pts != cached_pts:
                    cropped = np.asarray(crop_original_frame_roi(chosen,source['video_transform']))
                    cached_pts = chosen.pts
                pixels.append(cropped); times.append(ts)
            if len(pixels)==count:
                break
            previous, previous_t = frame, t
        # At EOF there is no following frame to trigger the nearest-PTS loop.
        # The last decoded frame is still the nearest valid frame within one
        # native frame interval. Keep the same grid-error limit as above.
        while (len(pixels)<count and previous is not None and
               abs(previous_t-desired[len(pixels)])<=1/native_fps+1e-5):
            if previous.pts!=cached_pts:
                cropped=np.asarray(crop_original_frame_roi(previous,source['video_transform']))
                cached_pts=previous.pts
            pixels.append(cropped);times.append(previous_t)
    if len(pixels)!=count:
        raise ValueError(f'Incomplete source video: {len(pixels)}/{count}; last PTS={previous_t}, desired={desired[-1]}, fps={native_fps}')
    return np.stack(pixels), times


def read_speaker(item):
    source=item['source'];audio=item['speaker'];check_file(audio['path'],audio['sha256'])
    with sf.SoundFile(audio['path']) as stream:
        rate=stream.samplerate
        if stream.channels!=1 or rate!=audio['sample_rate']:
            raise ValueError('Speaker audio format changed')
        first,last=round(source['source_start_s']*rate),round(source['source_end_s']*rate)
        if not 0<=first<last<=len(stream):
            raise ValueError('Speaker interval outside source')
        stream.seek(first);wave=stream.read(last-first,dtype='float32',always_2d=True)
    return wave,rate


def read_input(item,work):
    """Qwen inputs use only the first source frame and the paired speaker audio."""
    for path,h in item['source_files'].items():
        check_file(path,h)
    speaker,rate=read_speaker(item)
    pixels,times=decode_frames(item['source'],limit=1)
    first_hash=hashlib.sha256(pixels[0].tobytes()).hexdigest()
    speaker_hash=hashlib.sha256(np.asarray(speaker,dtype='<f4',order='C').tobytes()).hexdigest()
    if 'review' in item:
        review=item['review']
        if (first_hash!=review['video']['first_frame_rgb_sha256'] or
            speaker_hash!=review['speaker_audio']['raw_sha256'] or
            times[0]!=review['video']['source_frame_pts_s'][0]):
            raise ValueError('Qwen source inputs differ from reviewed first frame/speaker audio')
    reference=Path(work)/'references'/(item['sample_id']+'.png');reference.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pixels[0]).save(reference)
    return dict(speaker=speaker,rate=rate,reference=str(reference))


def read_media(item, work):
    sid = item['sample_id']; source = item['source']
    for path, h in item['source_files'].items():
        check_file(path, h)
    listener, rate, identity = read_audio(item['audio_recipe'])
    if identity['processed_waveform_sha256'] != item['listener_audio_identity']['processed_waveform_sha256']:
        raise ValueError('Listener recipe waveform changed')
    speaker,speaker_rate=read_speaker(item)
    if speaker_rate!=rate:
        raise ValueError('Paired audio sample rate mismatch')
    if listener.shape!=speaker.shape or len(listener)!=round(source['duration_s']*rate):
        raise ValueError('Paired audio interval mismatch')
    policy_path = Path(work)/'video_sampling_overrides'/(sid+'.json')
    policy = read(policy_path) if policy_path.exists() else None
    if policy is not None and policy['source_digest'] != digest(source):
        raise ValueError('Reviewed video timing source changed')
    try:pixels, times = decode_frames(source, gap_overrides=policy['frames'] if policy else None)
    except SourceVideoGap as exc:raise SourceVideoGap(sid+': '+str(exc)) from exc
    except ValueError as exc:raise ValueError(sid+': '+str(exc)) from exc
    raw_hash = lambda x: hashlib.sha256(np.asarray(x,dtype='<f4',order='C').tobytes()).hexdigest()
    audit = dict(sample_id=sid, source_frame_pts_s=times, frames=len(pixels),
        rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        first_rgb_sha256=hashlib.sha256(pixels[0].tobytes()).hexdigest(),
        listener_raw_sha256=raw_hash(listener), speaker_raw_sha256=raw_hash(speaker),
        listener_full_silence=not bool(np.any(listener)), rate=rate, samples=len(listener),
        final_crop=source['video_transform']['crop'], crop_revision=REVISION,
        caption_crop=item['caption_crop'], caption_sha256=item['caption_sha256'],
        split=item['split'], reference_from_first_frame=True, source_video_used=True)
    if policy is not None:
        audit['video_sampling_override'] = policy
    if 'review' in item:
        review = item['review']
        expected = (review['video']['preencode_rgb_sha256'],review['video']['first_frame_rgb_sha256'],
                    review['listener_audio']['raw_sha256'],review['speaker_audio']['raw_sha256'])
        actual = tuple(audit[k] for k in ('rgb_sha256','first_rgb_sha256','listener_raw_sha256','speaker_raw_sha256'))
        if expected!=actual or times!=review['video']['source_frame_pts_s']:
            raise ValueError('Training source pixels/audio differ from the reviewed sample: '+sid)
        audit['review_pixels_audio_pts_exact'] = True
    reference = Path(work)/'references'/(sid+'.png'); reference.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(pixels[0]).save(reference)
    atomic_json(Path(work)/'media_audit'/(sid+'.json'),audit,immutable=True)
    return dict(pixels=pixels,listener=listener,speaker=speaker,rate=rate,reference=str(reference),audit=audit)


def observation(item, media):
    # The existing native reader accepts file-like objects. This WAV container
    # exists only in RAM; no quantization or extra audio file is introduced.
    wave=media['speaker'].mean(-1);rate=media['rate']
    if rate!=16000:
        from scipy.signal import resample_poly
        common=math.gcd(rate,16000)
        wave=resample_poly(wave,16000//common,rate//common).astype(np.float32)
    count=round(item['source']['duration_s']*16000)
    # scipy returns ceil(N * up/down). The native observation grid uses round;
    # remove only that possible extra endpoint sample, as in the existing
    # repair_p2_audio_sample_grid.py policy. Original 48k audio is untouched.
    if not count<=len(wave)<=count+1:raise ValueError('Speaker resampling does not cover exact window')
    audio = io.BytesIO(); sf.write(audio,wave[:count],16000,format='WAV',subtype='FLOAT'); audio.seek(0)
    return dict(sample_id=item['sample_id'],speaker_audio=audio,listener_image=media['reference'],
        requested_duration_s=item['source']['duration_s'],geometry=dict(fps=25,height=512,width=512))
