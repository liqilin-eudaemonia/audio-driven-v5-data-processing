#!/usr/bin/env python3
"""Bounded, seeded pair sampling; heuristic listening cuts; review-only media."""
import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import math
import random
import subprocess
import tarfile
import threading
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_previews/seamless_face_listening_v1'
INDEX = ROOT.parent / 'audio-driven-v4/evaluation/benchmark_alignment/assets/seamless_test_index.json'
SEED = 20261021
LOCK = threading.Lock()
HEADERS = {}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def run(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace')[-2500:])
    return p.stdout


def headers(archive):
    with LOCK:
        if archive not in HEADERS:
            with tarfile.open(archive, 'r:') as t:
                HEADERS[archive] = {m.name: (m.offset_data, m.size) for m in t if m.isfile()}
        return HEADERS[archive]


def entry(record, ext):
    ref = record['files'].get(ext)
    if not ref:
        base = record['files']['.mp4']
        ref = dict(archive=base['archive'], member=str(Path(base['member']).with_suffix(ext)))
    off, size = headers(ref['archive'])[ref['member']]
    return dict(**ref, offset=off, byte_size=size)


def read_entry(ref):
    with open(ref['archive'], 'rb') as f:
        f.seek(ref['offset'])
        return f.read(ref['byte_size'])


def uri(ref):
    return f"subfile,,start,{ref['offset']},end,{ref['offset'] + ref['byte_size']},,:{ref['archive']}"


def overlap(intervals, a, b):
    return sum(max(0, min(b, y) - max(a, x)) for x, y in merge(intervals))


def merge(intervals, gap=0):
    result = []
    for a, b in sorted(intervals, key=lambda x: (x[0], x[1])):
        if b <= a:
            continue
        if result and a <= result[-1][1] + gap:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([float(a), float(b)])
    return result


def word_bouts(record):
    words = []
    for segment in record['transcript']:
        for w in segment.get('words', []):
            if isinstance(w.get('start'), (int,float)) and isinstance(w.get('end'), (int,float)) and w['end'] >= w['start']:
                words.append(dict(start=max(0., float(w['start'])), end=float(w['end']), text=w['word']))
    words.sort(key=lambda x: x['start'])
    bouts = []
    for w in words:
        if bouts and w['start'] <= bouts[-1]['end'] + .55:
            b = bouts[-1]; b['end'] = max(b['end'], w['end']); b['words'].append(w)
        else:
            bouts.append(dict(start=w['start'], end=w['end'], words=[w]))
    for b in bouts:
        b['text'] = ' '.join(w['text'] for w in b['words'])
    return words, bouts


def candidates(records):
    prepared = []
    for r in records:
        w, bouts = word_bouts(r)
        vad = merge([(v['start'], v['end']) for v in r['vad']])
        prepared.append((w, bouts, vad))
    result = []
    for si in [0, 1]:
        li = 1-si
        sw, sb, sv = prepared[si]; lw, lb, lv = prepared[li]
        # A long listener speech bout excludes a window; short speech remains.
        blocked = [(b['start']-.25, b['end']+.25) for b in lb
                   if b['end']-b['start'] > 2.5 or len(b['words']) > 5]
        blocked += [(a-.15, b+.15) for a, b in lv if b-a > 3.0]
        floor = merge(sv + [(b['start'], b['end']) for b in sb], gap=1.5)
        for fa, fb in floor:
            for a in np.arange(max(0, fa), fb-9.9, 1.5):
                b = min(a+12, fb)
                if overlap(blocked, a, b) > 0:
                    continue
                speaker_words = [w for w in sw if a <= w['start'] < b]
                listener_words = [w for w in lw if w['end'] > a and w['start'] < b]
                sfraction = overlap(sv, a, b)/(b-a)
                lfraction = overlap(lv, a, b)/(b-a)
                if len(speaker_words) < 10 or sfraction < .50 or lfraction > .25:
                    continue
                short = [dict(start=x['start'], end=x['end'], text=x['text']) for x in lb
                         if x['end'] > a and x['start'] < b]
                kind = 'short_feedback_candidate' if short else ('vocal_candidate' if lfraction > .01 else 'quiet_candidate')
                result.append(dict(start=round(float(a), 3), end=round(float(b), 3),
                                   speaker_index=si, listener_index=li, kind=kind,
                                   speaker_vad_fraction=round(sfraction, 3), listener_vad_fraction=round(lfraction, 3),
                                   listener_events=short,
                                   speaker_text=' '.join(w['text'] for w in speaker_words),
                                   listener_text=' '.join(w['text'] for w in listener_words),
                                   speaker_vad=[[max(a,x)-a, min(b,y)-a] for x,y in sv if y>a and x<b],
                                   listener_vad=[[max(a,x)-a, min(b,y)-a] for x,y in lv if y>a and x<b]))
    return result


def pick_clips(items, pair):
    rng = random.Random(SEED + int(hashlib.sha1(pair.encode()).hexdigest()[:8], 16))
    rng.shuffle(items)
    chosen = []
    for preferred in ['short_feedback_candidate', 'quiet_candidate']:
        for c in sorted(items, key=lambda x: x['kind'] != preferred):
            if all(min(c['end'], x['end']) <= max(c['start'], x['start']) for x in chosen):
                chosen.append(c); break
    return chosen


def load_record(r):
    mp4, wav, npz = [entry(r, ext) for ext in ['.mp4', '.wav', '.npz']]
    with np.load(io.BytesIO(read_entry(npz)), allow_pickle=False) as z:
        required = ['movement:emotion_scores', 'movement:emotion_valence', 'movement:emotion_arousal']
        if not all(k in z.files for k in required):
            raise ValueError('Missing required emotion feature fields')
        arrays = {k: z[k] for k in required + ['boxes_and_keypoints:keypoints', 'boxes_and_keypoints:is_valid_box', 'movement:is_valid']}
    streams = json.loads(run(['ffprobe','-v','error','-show_streams','-of','json',uri(mp4)]))['streams']
    v = next(x for x in streams if x['codec_type']=='video')
    n, d = map(int, v['avg_frame_rate'].split('/'))
    fps = n/d
    samples, sr = sf.read(io.BytesIO(read_entry(wav)), dtype='float32', always_2d=True)
    duration = float(v['duration'])
    if abs(duration-len(samples)/sr) > .15:
        raise ValueError(f'AV duration mismatch {duration} vs {len(samples)/sr}')
    expected = int(v.get('nb_frames', round(duration*fps)))
    if abs(len(arrays['boxes_and_keypoints:keypoints'])-expected) > 1:
        raise ValueError('Feature/video frame count mismatch')
    return dict(id=r['id'], source=mp4, audio=samples.mean(axis=1), sr=sr, fps=fps,
                duration=min(duration,len(samples)/sr), width=int(v['width']),height=int(v['height']), arrays=arrays)


def crop_box(r, a, b):
    # COCO WholeBody face landmarks occupy 23:91 in the 133-point array.
    k = r['arrays']['boxes_and_keypoints:keypoints'][int(a*r['fps']):math.ceil(b*r['fps']),23:91]
    valid = (k[:,:,2]>.4).sum(axis=1)>=45
    if valid.mean() < .8:
        raise ValueError(f'Low face landmark validity: {valid.mean():.3f}')
    k=k[valid]
    # Robust per-frame face bounds then a fixed window across this clip.
    xlo=np.nanpercentile(np.where(k[:,:,2]>.4,k[:,:,0],np.nan),3,axis=1)
    xhi=np.nanpercentile(np.where(k[:,:,2]>.4,k[:,:,0],np.nan),97,axis=1)
    ylo=np.nanpercentile(np.where(k[:,:,2]>.4,k[:,:,1],np.nan),3,axis=1)
    yhi=np.nanpercentile(np.where(k[:,:,2]>.4,k[:,:,1],np.nan),97,axis=1)
    fw=float(np.median(xhi-xlo));fh=float(np.median(yhi-ylo))
    left=float(np.percentile(xlo,1))-.4*fw;right=float(np.percentile(xhi,99))+.4*fw
    top=float(np.percentile(ylo,1))-.55*fh;bottom=float(np.percentile(yhi,99))+.3*fh
    cx=(left+right)/2;cy=(top+bottom)/2
    side=int(math.ceil(max(right-left,bottom-top,fw*2.2,fh*2.2)/2)*2)
    if side>3.2*max(fw,fh):
        raise ValueError('Large head displacement requires an excessively wide fixed crop')
    side=min(side,r['width']//2*2,r['height']//2*2)
    if side < 60:raise ValueError('Implausibly small face crop')
    x=int(np.clip(cx-side/2,0,r['width']-side))//2*2
    y=int(np.clip(cy-side/2,0,r['height']-side))//2*2
    return dict(x=x,y=y,size=side,valid_fraction=round(float(valid.mean()),4),
                face_width=round(fw,1),face_height=round(fh,1))


def curve(r,a,b,key):
    arr=r['arrays'][key]
    out=[]
    for t in np.arange(a,b,.2):
        ix=min(len(arr)-1,int(round(t*r['fps'])))
        valid=float(r['arrays']['movement:is_valid'][ix].reshape(-1)[0])>.5
        value=float(arr[ix].reshape(-1)[0])
        out.append(round(value,4) if valid and np.isfinite(value) else None)
    return out


def render_face_media(s, l, boxes, a, b, d):
    """Render original source crops to 512 pixels, plus individual AV files."""
    duration=b-a
    inp=[]
    for r in [s,l]:inp+=['-threads','2','-ss',str(a),'-t',str(duration),'-i',uri(r['source'])]
    inp+=['-i',str(d/'speaker.m4a'),'-i',str(d/'listener.m4a')]
    filters=[]
    for i,box in enumerate(boxes):
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,crop={box["size"]}:{box["size"]}:{box["x"]}:{box["y"]},scale=512:512,fps=25,setsar=1,split=2[f{i}][single{i}]')
    filters+=['[f0][f1]hstack=inputs=2[v]','[2:a][3:a]amix=inputs=2:duration=shortest:normalize=0,volume=0.5[a]']
    args=['ffmpeg','-v','error','-nostdin','-y',*inp,'-filter_complex_threads','1','-filter_complex',';'.join(filters)]
    for name,video,audio in [('faces','[v]','[a]'),('speaker_512','[single0]','2:a'),('listener_512','[single1]','3:a')]:
        args+=['-map',video,'-map',audio,'-t',str(duration),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','21','-pix_fmt','yuv420p']
        args+=['-c:a','aac','-b:a','160k'] if name=='faces' else ['-c:a','copy']
        args+=['-movflags','+faststart',str(d/f'{name}.pending.mp4')]
    run(args)
    for name in ['faces','speaker_512','listener_512']:
        (d/f'{name}.pending.mp4').replace(d/f'{name}.mp4')
    run(['ffmpeg','-v','error','-nostdin','-y','-ss','1','-i',str(d/'faces.mp4'),'-frames:v','1',str(d/'poster.pending.jpg')])
    (d/'poster.pending.jpg').replace(d/'poster.jpg')


def make_clip(pair_dir, n, c, people):
    s,l=people[c['speaker_index']],people[c['listener_index']]
    a,b=c['start'],c['end']
    if b>min(s['duration'],l['duration']):raise ValueError('Clip past paired duration')
    boxes=[crop_box(r,a,b) for r in [s,l]]
    d=pair_dir/f'clip_{n:02d}';d.mkdir(parents=True,exist_ok=True)
    duration=b-a
    # Preserve amplitudes, including silence; independent tracks remain unmixed.
    for name,r in [('speaker',s),('listener',l)]:
        audio=r['audio'][round(a*r['sr']):round(b*r['sr'])]
        sf.write(d/f'{name}.wav',audio,r['sr'],subtype='PCM_16')
        run(['ffmpeg','-v','error','-nostdin','-y','-i',str(d/f'{name}.wav'),'-c:a','aac','-b:a','128k',str(d/f'{name}.m4a')])
    render_face_media(s,l,boxes,a,b,d)
    context=[]
    for i,box in enumerate(boxes):
        context.append(f'[{i}:v]setpts=PTS-STARTPTS,drawbox=x={box["x"]}:y={box["y"]}:w={box["size"]}:h={box["size"]}:color=0x58e0b4:t=5,scale=320:360:force_original_aspect_ratio=decrease,pad=320:360:(ow-iw)/2:(oh-ih)/2,fps=15,setsar=1[c{i}]')
    context.append('[c0][c1]hstack=inputs=2[v]')
    video_inputs=[]
    for r in [s,l]:video_inputs+=['-threads','2','-ss',str(a),'-t',str(duration),'-i',uri(r['source'])]
    run(['ffmpeg','-v','error','-nostdin','-y',*video_inputs,'-filter_complex_threads','1','-filter_complex',';'.join(context),
         '-map','[v]','-an','-t',str(duration),'-c:v','libx264','-threads','2','-preset','veryfast','-crf','25',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(d/'context.mp4')])
    run(['ffmpeg','-v','error','-nostdin','-y','-ss','1','-i',str(d/'faces.mp4'),'-frames:v','1',str(d/'poster.jpg')])
    result=dict(c, speaker=s['id'],listener=l['id'],duration=round(duration,3),face_resolution=512,media_revision=3,
                path=str(d.relative_to(OUT)),crop_boxes=dict(speaker=boxes[0],listener=boxes[1]),
                emotion={role:{key.split(':')[-1]:curve(r,a,b,key) for key in ['movement:emotion_valence','movement:emotion_arousal']}
                         for role,r in [('speaker',s),('listener',l)]})
    dump(d/'metadata.json',result)
    for name in ['speaker','listener']:(d/f'{name}.wav').unlink()
    return result


def process_pair(item):
    pair=item['pair'];dest=OUT/'media'/pair;dest.mkdir(parents=True,exist_ok=True)
    manifest=dest/'pair.json'
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing.get('version')==3:return existing
    print('START',pair,flush=True)
    people=[load_record(r) for r in item['records']]
    if abs(people[0]['duration']-people[1]['duration'])>.2:raise ValueError('Pair durations differ')
    clips=[];errors=[]
    for n,c in enumerate(item['clips'],1):
        try:clips.append(make_clip(dest,n,c,people))
        except Exception as e:errors.append(str(e))
    if not clips:raise ValueError('No rendered clips: '+repr(errors))
    result=dict(version=3,pair=pair,subset=pair.split('_')[0],split='test',clips=clips,errors=errors,
                sources=[dict(id=r['id'],fps=r['fps'],duration=r['duration'],width=r['width'],height=r['height'],source=r['source']) for r in people])
    dump(manifest,result)
    print('DONE',pair,len(clips),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=50);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads(INDEX.read_text());groups=collections.defaultdict(list)
    for r in data['records']:
        if all(k in r['files'] for k in ['.json','.mp4','.wav']) and r.get('transcript') and r.get('vad'):
            groups[r['pair']].append(r)
    pools={};selection=[]
    # The existing test index has V00 only under naturalistic. V00 examples
    # provide the requested movement emotion fields; still verify every pair.
    for kind in ['naturalistic']:
        pairs=[(k,v) for k,v in groups.items() if k.startswith(kind+'_V00_') and len(v)==2]
        random.Random(SEED).shuffle(pairs)
        pool=[]
        for key,rs in pairs:
            cs=pick_clips(candidates(rs),key)
            if cs:
                pool.append(dict(pair=key,records=rs,clips=cs))
            if len(pool)>=args.count+30:break
        pools[kind]=iter(pool)
    successes=[];failures=[];counts=collections.Counter();pending={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        def submit(kind):
            try:item=next(pools[kind])
            except StopIteration:return False
            selection.append(item);pending[ex.submit(process_pair,item)]=kind;return True
        for kind in pools:
            for _ in range(min(args.workers,args.count)):submit(kind)
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                kind=pending.pop(future)
                try:
                    result=future.result();successes.append(result);counts[kind]+=1
                except Exception as e:
                    failures.append(dict(subset=kind,error=str(e)));print('SKIP',kind,str(e)[:350],flush=True)
                outstanding=sum(k==kind for k in pending.values())
                if counts[kind]+outstanding<args.count:submit(kind)
                dump(OUT/'manifest.json',dict(seed=SEED,target_pairs=args.count,counts=dict(counts),pairs=sorted(successes,key=lambda x:x['pair']),failures=failures))
                print('PROGRESS',dict(counts),'clips',sum(len(x['clips']) for x in successes),flush=True)
    dump(OUT/'selection.json',dict(seed=SEED,source_index=str(INDEX),attempts=selection))
    print('FINISHED',dict(counts),flush=True)


if __name__=='__main__':main()
