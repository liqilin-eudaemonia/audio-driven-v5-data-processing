#!/usr/bin/env python3
"""Read-only, 24-pair visual QC spike. No admission or segmentation changes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import bisect
import hashlib
import importlib.util
import json
import math
import pickle
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
# Reuse the installed PyAV binary, without changing either existing environment.
sys.path.append('/private/software/conda/envs/qwenomni/lib/python3.10/site-packages')
import av

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WEB = ROOT / 'data_previews/visual_quality_pilot_20261030'
WORK = ROOT / 'experiments/listen_j_annotation_node17_20261029'
sys.path.insert(0, str(ROOT / 'src'))
from listen_benchmark.full_raw import SourceBytes
FACE = ROOT / 'related/visual_reaction_annotation/nod_headshake/face_detection/detect_face_multithread.py'
MODEL = FACE.parent / 'face_landmarker_v2_with_blendshapes.task'
spec = importlib.util.spec_from_file_location('pilot_vendor_face', FACE)
face = importlib.util.module_from_spec(spec)
spec.loader.exec_module(face)
cv2.setNumThreads(1)

PAIRS = [
    'naturalistic_V02_S5228_I00000302', 'naturalistic_V03_S2098_I00000195',
    'improvised_V00_S0862_I00000794', 'naturalistic_V02_S3579_I00000175',
    'naturalistic_V03_S1909_I00000205', 'naturalistic_V00_S0153_I00000384',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def runs(mask, fps, duration):
    intervals, start = [], None
    for i in range(len(mask) + 1):
        active = i < len(mask) and bool(mask[i]) and i / fps < duration
        if active and start is None:
            start = i
        if not active and start is not None:
            intervals.append({'start': start / fps, 'end': min(i / fps, duration),
                              'duration': min(i / fps, duration) - start / fps})
            start = None
    return intervals


def quantiles(values):
    a = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'), np.quantile(a, [0, .05, .5, .95, 1]).tolist())) if len(a) else None


def raw_measure(result, width, height, c, t, source_t):
    r = dict(t=t, source_time=source_t, detected=bool(result.face_landmarks))
    if not r['detected']:
        return r
    points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
    x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
    area = (x1-x0)*(y1-y0)
    inter = max(0, min(x1,c['x']+c['width'])-max(x0,c['x'])) * max(0,min(y1,c['y']+c['height'])-max(y0,c['y']))
    outside = ((points[:,0]<c['x']) | (points[:,0]>c['x']+c['width']) |
               (points[:,1]<c['y']) | (points[:,1]>c['y']+c['height']))
    center_inside = c['x'] <= (x0+x1)/2 <= c['x']+c['width'] and c['y'] <= (y0+y1)/2 <= c['y']+c['height']
    r.update(bbox_native=[x0,y0,x1,y1], face_width_native=x1-x0, face_height_native=y1-y0,
             bbox_retained_fraction=inter/area if area>0 else None,
             landmarks_outside_fraction=float(outside.mean()), center_inside_crop=bool(center_inside))
    return r


def encode(stream, container, rgb, index):
    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
    frame.pts = index; frame.time_base = Fraction(1,25)
    for packet in stream.encode(frame):
        container.mux(packet)


def render(row, directory, temp):
    s = row['source']; transform = s['video_transform']; c = transform['crop']
    width, height = transform['source_size']; rate = s['normalized_audio']['sample_rate']
    origin = round(s['source_start_s']*rate)/rate; duration = s['duration_s']
    n = math.ceil(duration*25-1e-9); raw_width = int(round(width*512/height/2))*2
    raw_rows, source_times = [], []
    crop_path = directory / 'crop.mp4'; silent = temp / 'comparison_silent.mp4'
    with av.open(str(crop_path),'w') as crop_out, av.open(str(silent),'w') as preview_out, \
         face.FaceLandmarker.create_from_options(face.build_options(str(MODEL))) as detector, \
         SourceBytes(s['source']) as source, av.open(source) as inp:
        streams=[]
        for out, w in [(crop_out,512),(preview_out,raw_width+512)]:
            st=out.add_stream('libx264',rate=25);st.width=w;st.height=512;st.pix_fmt='yuv420p'
            st.options={'crf':'18','preset':'fast','threads':'1'};streams.append(st)
        video=inp.streams.video[0];video.codec_context.thread_count=1
        assert [video.width,video.height] == [width,height]
        inp.seek(int(origin/video.time_base),stream=video,backward=True,any_frame=False)
        index=0;last=None;last_t=None
        def write_frame(rgb,t,i):
            crop=np.asarray(Image.fromarray(rgb[c['y']:c['y']+c['height'],c['x']:c['x']+c['width']]).resize((512,512),Image.Resampling.BICUBIC))
            raw_small=cv2.resize(rgb,(raw_width,512),interpolation=cv2.INTER_AREA)
            scale=512/height
            cv2.rectangle(raw_small,(round(c['x']*scale),round(c['y']*scale)),
                          (round((c['x']+c['width'])*scale),round((c['y']+c['height'])*scale)),(40,230,100),2)
            preview=np.concatenate([raw_small,crop],axis=1)
            encode(streams[0],crop_out,crop,i);encode(streams[1],preview_out,preview,i)
            source_times.append(origin+t)
            if i % 5 == 0:
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),i*40)
                measurement=raw_measure(result,width,height,c,i/25,origin+t)
                raw_rows.append(measurement)
                if measurement['detected']:
                    b=measurement['bbox_native'];q=[round(v*scale) for v in b]
                    cv2.rectangle(preview,(q[0],q[1]),(q[2],q[3]),(255,190,30),2)
                cv2.imwrite(str(temp/f'frame_{i:05d}.jpg'),cv2.cvtColor(preview,cv2.COLOR_RGB2BGR))
        for f in inp.decode(video):
            t=float(f.pts*f.time_base)-origin
            if t<0:continue
            if t>=duration:break
            if index>=n:break
            last=f.to_ndarray(format='rgb24');last_t=t
            while index<n and index/25<=t+1e-10:
                write_frame(last,t,index);index+=1
        if last is None or not source_times or source_times[0]-origin>=.1 or duration-last_t>=.1:
            raise ValueError('Incomplete original frame support')
        while index<n:
            write_frame(last,last_t,index);index+=1
        for st,out in zip(streams,[crop_out,preview_out]):
            for packet in st.encode():out.mux(packet)
    # A single comparison video keeps source and crop exactly synchronized.
    original=json.loads(Path(s['source_result']).read_text())['clips'][s['source_clip_index']]
    assert original['sample_id']==row['sample_id']
    speaker=original['normalized_audio']['speaker']['path']
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-threads','1','-i',str(silent),
         '-ss',str(origin),'-i',s['normalized_audio']['path'],'-ss',str(origin),'-i',speaker,
         '-filter_complex_threads','1','-filter_complex','[1:a][2:a]amix=inputs=2:duration=shortest:normalize=1[a]',
         '-map','0:v','-map','[a]','-t',str(duration),'-c:v','copy','-c:a','aac','-b:a','96k','-movflags','+faststart',str(directory/'comparison.mp4')]
    subprocess.run(cmd,check=True,timeout=90)
    return dict(source_start_s=origin,duration_s=duration,frames=n,fps=25,raw_sampling_fps=5,
                source_frame_times_s=source_times,raw_rows=raw_rows,transform=transform,
                sampling='first original frame at/after each 25 Hz timestamp; last frame reused only at clip tail',
                source_result_sha256=sha(s['source_result']))


def summarize(row, media, frames):
    s=row['source'];duration=s['duration_s'];c=s['video_transform']['crop']
    if len(frames)!=media['frames']:raise ValueError('Detection frame count mismatch')
    detected=[bool(f['detected']) for f in frames]
    missing=runs([not v for v in detected],25,duration)
    edge=[];heights=[];widths=[];margins=[]
    for f in frames:
        if not f['detected']:
            edge.append(False);heights.append(None);widths.append(None);margins.append(None);continue
        xy=np.asarray(f['landmarks'])[:,:2]
        x0,y0=xy.min(axis=0);x1,y1=xy.max(axis=0)
        margin=float(min(x0,y0,1-x1,1-y1))
        margins.append(margin);edge.append(margin<.03)
        heights.append(float((y1-y0)*c['height']));widths.append(float((x1-x0)*c['width']))
    raw=media['raw_rows'];rd=[r for r in raw if r['detected']]
    # Detection identity/correspondence remains a manual check, never a hard rejection.
    raw_cut=[r['t'] for r in rd if r['bbox_retained_fraction']<.98]
    raw_small=[r['t'] for r in rd if min(r['face_height_native'],r['face_width_native'])<96]
    near=runs(edge,25,duration)
    small=runs([h is not None and min(h,w)<96 for h,w in zip(heights,widths)],25,duration)
    findings=[]
    for r in missing:
        findings.append(dict(kind='missing_face',label='裁剪画面未检出人脸',**r))
    for r in near:
        findings.append(dict(kind='near_edge',label='面部关键点接近裁剪边缘（<3%）',**r))
    for r in small:
        findings.append(dict(kind='small_face',label='原始面部关键点跨度偏小（短边<96px，试验提示）',**r))
    # Only sampled points: do not invent continuous truncation intervals from 5 fps.
    for r in rd:
        if r['bbox_retained_fraction']<.98:
            findings.append(dict(kind='raw_outside_crop',label='原画面检测框部分落在裁框外（抽帧疑点）',
                                 start=r['t'],end=r['t'],duration=0,sampled_point=True,
                                 retained_fraction=r['bbox_retained_fraction']))
    metrics=dict(face_detection_fraction=sum(detected)/len(detected),frames=len(frames),detected_frames=sum(detected),
                 missing_intervals=missing,longest_missing_s=max([r['duration'] for r in missing],default=0),
                 near_edge_intervals=near,longest_near_edge_s=max([r['duration'] for r in near],default=0),
                 face_width_native_from_crop=quantiles(widths),face_height_native_from_crop=quantiles(heights),
                 minimum_crop_margin_fraction=quantiles(margins),
                 raw_detected_samples=len(rd),raw_samples=len(raw),raw_outside_crop_samples=len(raw_cut),
                 raw_small_face_samples=len(raw_small),
                 raw_face_width_native=quantiles([r['face_width_native'] for r in rd]),
                 raw_face_height_native=quantiles([r['face_height_native'] for r in rd]),
                 raw_bbox_retained_fraction=quantiles([r['bbox_retained_fraction'] for r in rd]))
    cached=WORK/'nod/results'/hashlib.sha256(row['sample_id'].encode()).hexdigest()[:2]/f"{row['sample_id']}.json"
    cache_info={'available':cached.exists()}
    if cached.exists():
        old=json.loads(cached.read_text());v=old.get('pose',{}).get('detected',[])
        cache_info.update(path=str(cached),sha256=sha(cached),frames=len(v),
                          source_matches=old.get('source')==s,
                          face_detection_fraction=sum(v)/len(v) if v else None)
    return dict(sample_id=row['sample_id'],source=s,metrics=metrics,findings=findings,
                status='review' if findings else 'no_rule_alert',existing_nod_cache=cache_info,
                crop_detection=[dict(t=i/25,detected=detected[i],margin=margins[i],face_height_native=heights[i],face_width_native=widths[i]) for i in range(len(frames))],
                raw_measurements=raw,training_admitted=False,automatic_rejection=False)


def select():
    if (OUT/'selection.json').exists():return json.loads((OUT/'selection.json').read_text())
    rng=random.Random(20261030);rows=[];inputs=[]
    for pair in PAIRS:
        path=WORK/'jobs'/f'{pair}.json';job=json.loads(path.read_text())
        chosen=sorted(rng.sample(job['rows'],4),key=lambda r:r['source']['source_start_s'])
        rows.extend(chosen);inputs.append(dict(path=str(path),sha256=sha(path),eligible_pairs=len(job['rows'])))
    selection=dict(seed=20261030,method='4 uniformly sampled existing J annotation-eligible pairs per fixed source; 5 previously reviewed sources + 1 control source; not a corpus prevalence estimate',rows=rows,inputs=inputs)
    save(OUT/'selection.json',selection);return selection


def main():
    WEB.mkdir(parents=True,exist_ok=True)
    selection=select();results=[]
    protocol=dict(version='visual-quality-spike-v1',scope='listener training crop only',
                  thresholds={'near_edge_fraction':.03,'raw_bbox_retained_fraction':.98,'native_face_short_side_px':96},
                  thresholds_calibrated=False,automatic_rejection=False,training_admitted=False,
                  limitations=['Face landmark span excludes hair/scalp; not proof of whole-head completeness.',
                  'Full-frame detector runs at 5 fps: sample points cannot establish continuous truncation durations.',
                  'Missing detections are not confirmed missing faces; detected faces are not proven unoccluded.',
                  'FaceLandmarker uses one face; full-frame/crop correspondence needs review for outside-crop findings.',
                  'No whole-corpus quality rate can be inferred from these fixed-source samples.'],
                  face_code_sha256=sha(FACE),model_sha256=sha(MODEL),runner_sha256=sha(__file__),selection_sha256=sha(OUT/'selection.json'))
    save(OUT/'protocol.json',protocol)
    for i,row in enumerate(selection['rows'],1):
        sid=row['sample_id'];directory=WEB/'media'/sid;directory.mkdir(parents=True,exist_ok=True)
        result_path=OUT/'results'/f'{sid}.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result.get('runner_sha256')!=protocol['runner_sha256']:raise ValueError('Changed spike code: use a new output directory or explicitly invalidate results')
        else:
            with tempfile.TemporaryDirectory(prefix='visual-qc-pilot-') as temp_name:
                temp=Path(temp_name);media=render(row,directory,temp)
                pkl=OUT/'landmarks'/f'{sid}.pkl'
                face.process_video(directory/'crop.mp4',pkl,str(MODEL))
                with pkl.open('rb') as f:frames=pickle.load(f)
                result=summarize(row,media,frames)
                # Select the longest missing/edge event; otherwise the smallest native face.
                prioritized=sorted(result['findings'],key=lambda e:({'raw_outside_crop':4,'missing_face':3,'near_edge':2,'small_face':1}[e['kind']],e['duration']),reverse=True)
                focus=(prioritized[0]['start']+prioritized[0]['end'])/2 if prioritized else min(
                    [r for r in media['raw_rows'] if r['detected']],key=lambda r:r['face_height_native'],default={'t':row['source']['duration_s']/2})['t']
                nearest=min(range(0,media['frames'],5),key=lambda k:abs(k/25-focus))
                import shutil
                shutil.copyfile(temp/f'frame_{nearest:05d}.jpg',directory/'focus.jpg')
                result.update(focus_s=nearest/25,focus_reason=prioritized[0]['label'] if prioritized else '对照：面部高度较小的抽样帧',
                              media=f'media/{sid}/comparison.mp4',poster=f'media/{sid}/focus.jpg',
                              runner_sha256=protocol['runner_sha256'],source_result_sha256=media['source_result_sha256'])
                save(OUT/'media_metadata'/f'{sid}.json',media)
                save(result_path,result)
        results.append(result)
        print(json.dumps(dict(done=i,total=len(selection['rows']),id=sid,status=result['status'],
                              coverage=result['metrics']['face_detection_fraction'],missing=result['metrics']['longest_missing_s'],
                              raw_outside=result['metrics']['raw_outside_crop_samples']),ensure_ascii=False),flush=True)
    summary=dict(clips=len(results),sources=len(PAIRS),review=sum(r['status']=='review' for r in results),
                 no_rule_alert=sum(r['status']=='no_rule_alert' for r in results),
                 any_missing=sum(r['metrics']['face_detection_fraction']<1 for r in results),
                 near_edge=sum(bool(r['metrics']['near_edge_intervals']) for r in results),
                 raw_outside_crop=sum(r['metrics']['raw_outside_crop_samples']>0 for r in results),
                 small_face=sum(any(f['kind']=='small_face' for f in r['findings']) for r in results),
                 cached_nod=sum(r['existing_nod_cache']['available'] for r in results),
                 automatic_rejections=0,training_admitted=False)
    save(OUT/'summary.json',summary)
    save(WEB/'data.json',dict(summary=summary,protocol=protocol,results=results))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
