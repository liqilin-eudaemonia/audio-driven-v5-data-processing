"""Align original NPZ movement fields to the five retained V00 review clips."""
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from listen_benchmark.j_omni import read_audio

BASE = ROOT/'data_previews/listen_benchmark_review/v00_face_size_20261101'
OUT = BASE/'emotions'
KEYS = ['movement:emotion_scores', 'movement:emotion_valence', 'movement:emotion_arousal',
        'movement:EmotionValenceToken', 'movement:EmotionArousalToken', 'movement:is_valid']
NAMES_EN = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
NAMES_ZH = ['愤怒', '轻蔑', '厌恶', '恐惧', '快乐', '中性', '悲伤', '惊讶']
MAPPING = dict(status='provisional_document_list_order', verified=False,
    names_en=NAMES_EN, names_zh=NAMES_ZH,
    reference='https://github.com/facebookresearch/seamless_interaction/commit/6803583c2694',
    caveat='README lists categories but does not explicitly specify dimension indices. Display hypothesis only; not training ground truth.')


def add_candidate_classes(frames):
    """Names are explicitly provisional; invalid frames never get a class."""
    totals=np.zeros(8,dtype=np.float64); weights=0.; occupied=np.zeros(8,dtype=np.float64)
    for f in frames:
        usable=f['valid'] and f['finite']
        ix=f['top_dimension'] if usable else None
        f['candidate_emotion_zh']=NAMES_ZH[ix] if ix is not None else None
        f['candidate_emotion_en']=NAMES_EN[ix] if ix is not None else None
        if ix is not None:
            dt=f['end']-f['start']; totals+=np.asarray(f['scores'])*dt
            weights+=dt; occupied[ix]+=dt
    means=totals/weights if weights else None
    ix=int(np.argmax(means)) if means is not None else None
    return dict(candidate_emotion_zh=NAMES_ZH[ix] if ix is not None else None,
        candidate_emotion_en=NAMES_EN[ix] if ix is not None else None,
        top_dimension=ix, mean_scores=means.tolist() if means is not None else None,
        dominant_frame_seconds=occupied.tolist(),
        rule='argmax of valid-frame raw scores weighted by clipped frame duration',
        mapping_verified=False, training_admitted=False)


def clean(value):
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def srt_time(seconds):
    ms=round(seconds*1000)
    return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'


def main():
    OUT.mkdir(exist_ok=True)
    sources = json.loads((BASE/'sources.json').read_text())
    views = json.loads((BASE/'data.json').read_text())
    records = []
    for number, (rec, view) in enumerate(zip(sources, views), 1):
        src = rec['source']; tarpath = src['source']['archive']
        member = rec['listener_id'] + '.npz'
        with tarfile.open(tarpath, 'r:') as archive:
            for entry in archive:
                if Path(entry.name).name == member:
                    payload = archive.extractfile(entry).read()
                    offset = entry.offset_data
                    break
            else:
                raise FileNotFoundError(member)
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            arrays = {k:z[k] for k in KEYS}
        n = len(arrays[KEYS[0]])
        assert arrays[KEYS[0]].shape == (n,8)
        assert all(arrays[k].shape == (n,1) for k in KEYS[1:])
        # Original source manifests and movement documentation both use 30 Hz.
        result = json.loads(Path(src['source_result']).read_text())
        original = next(x for x in result['clips'] if x['sample_id'] == rec['sample_id'])
        fps = float(original['fps'])
        assert fps == 30.0
        assert abs(n/fps-result['source_seconds']) < .1
        start, end = rec['source_start_s'], rec['source_end_s']
        lo, hi = math.floor(start*fps), math.ceil(end*fps)
        assert 0 <= lo < hi <= n
        frames = []
        for idx in range(lo,hi):
            scores = arrays[KEYS[0]][idx]
            valid = bool(arrays['movement:is_valid'][idx,0]>.5)
            finite = bool(all(np.isfinite(arrays[k][idx]).all() for k in KEYS))
            frames.append(dict(source_frame=idx, source_time_s=idx/fps,
                start=max(0.,idx/fps-start), end=min(end-start,(idx+1)/fps-start),
                valid=valid, finite=finite, scores=clean(scores.tolist()),
                valence=clean(float(arrays[KEYS[1]][idx,0])),
                arousal=clean(float(arrays[KEYS[2]][idx,0])),
                valence_token=clean(float(arrays[KEYS[3]][idx,0])),
                arousal_token=clean(float(arrays[KEYS[4]][idx,0])),
                top_dimension=int(np.argmax(scores)) if valid and finite else None))
        valid_frames = [f for f in frames if f['valid'] and f['finite']]
        class_summary=add_candidate_classes(frames)
        denom = sum(f['end']-f['start'] for f in valid_frames)
        mean = lambda key: sum(f[key]*(f['end']-f['start']) for f in valid_frames)/denom if denom else None
        cap = json.loads(Path(rec['caption_file']).read_text())
        wave, rate, identity = read_audio(cap['audio_recipe'])
        assert abs(len(wave)/rate-rec['duration_s']) <= 1/rate+.00001
        expected = cap.get('input_evidence', {}).get('audio_identity', {})
        if expected.get('processed_waveform_sha256'):
            assert expected['processed_waveform_sha256'] == identity['processed_waveform_sha256']
        stream = io.BytesIO();sf.write(stream,wave,rate,format='WAV',subtype='FLOAT')
        # Preview AAC only; no new persistent WAV or training audio caches.
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-threads','1',
            '-i',str(BASE/str(number)/'crop512.mp4'),'-f','wav','-i','pipe:0',
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','128k',
            '-t',str(rec['duration_s']),'-movflags','+faststart',str(OUT/f'{number}.mp4')],
            input=stream.getvalue(),check=True,capture_output=True)
        subtitles=[]
        for k, f in enumerate(frames,1):
            label=(f"Candidate: {f['candidate_emotion_en']} (mapping unverified)\nValence {f['valence']:+.3f}  |  Arousal {f['arousal']:+.3f}"
                   if f['valid'] and f['finite'] else 'Invalid emotion frame - no label')
            subtitles.append(f"{k}\n{srt_time(f['start'])} --> {srt_time(f['end'])}\n{label}\n")
        srt=OUT/f'{number}.srt';srt.write_text('\n'.join(subtitles))
        # Put labels below the face, preserving the 512px picture and audio.
        subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-threads','1',
            '-i',str(OUT/f'{number}.mp4'),'-filter_threads','1','-vf',
            f"pad=512:576:0:0:black,subtitles={srt}:force_style='FontName=DejaVu Sans,FontSize=11,Alignment=2,MarginV=8'",
            '-c:v','libx264','-preset','fast','-crf','18','-threads','2','-c:a','copy',
            '-movflags','+faststart',str(OUT/f'{number}_labeled.mp4')],check=True,capture_output=True)
        item = dict(id=rec['sample_id'], number=number, video=f'{number}.mp4',
            labeled_video=f'{number}_labeled.mp4', subtitles=f'{number}.srt',
            face_p10=view['p10'], duration=rec['duration_s'], source_start_s=start,
            source_end_s=end, fps=fps, frames=frames,
            summary=dict(valid_frames=len(valid_frames),total_frames=len(frames),
                valid_duration_s=denom,mean_valence=mean('valence'),mean_arousal=mean('arousal')),
            class_summary=class_summary, category_mapping=MAPPING,
            audio=identity, npz=dict(archive=tarpath,member=member,offset=offset,
                sha256=hashlib.sha256(payload).hexdigest(),total_source_frames=n),
            alignment=dict(rule='NPZ frame i spans [i/30,(i+1)/30); intersect source clip, subtract source start',
                video_boundary_precision_s=1/fps,source_duration_checked=True),
            category_name_mapping_verified=False)
        records.append(item)
        (OUT/f'{number}.json').write_text(json.dumps(item,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        print(number,item['summary'],flush=True)
    data=dict(schema='v00-emotion-synchronized-review-v1',samples=records,
              reference='https://github.com/facebookresearch/seamless_interaction#movementimitator-feature-types',
              category_mapping=MAPPING,category_name_mapping_verified=False, probabilities=False, human_ground_truth=False)
    (OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    (OUT/'index.html').write_text((ROOT/'scripts/v00_emotion_review.html').read_text())
    (OUT/'audit.json').write_text(json.dumps(dict(passed=True,samples=len(records),
        all_fields_present=True,source_duration_and_30hz_checked=True,
        invalid_frames_not_presented_as_valid=True,persistent_wav_exports=0,
        category_name_mapping_verified=False),indent=2)+'\n')
    old=BASE/'index.html'; text=old.read_text()
    link='<p><a href="emotions/"><b>新增：裁剪视频＋逐帧情绪字段同步复核 →</b></a></p>'
    if link not in text:
        text=text.replace('<h1>V00 原始人脸尺寸：直接看实际画面</h1>', '<h1>V00 原始人脸尺寸：直接看实际画面</h1>'+link)
        old.write_text(text)


if __name__ == '__main__':
    main()
