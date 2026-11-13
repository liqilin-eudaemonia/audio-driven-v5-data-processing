#!/usr/bin/env python3
"""Source-bound orientation candidate audit using existing full-pool measurements."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import orjson

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'experiments/listen_j_clean_regen_20261105'
AUTHORITY=ROOT/'experiments/listen_j_final_quality_selection_20261101/retained.jsonl'


def source_key(source):
    media=source['source']
    return hashlib.sha256((media['archive']+'\0'+media['member']).encode()).hexdigest()[:24]


def inspect(row):
    sid=row['sample_id'];q=row['quality'];p=Path(q['measurement_file'])
    try:
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=q['measurement_sha256']:raise ValueError('measurement_hash_mismatch')
        r=orjson.loads(raw)
        if r['id']!=sid or r['source_sha256']!=q['source_sha256'] or r['status']!='ok':raise ValueError('measurement_source_or_status_mismatch')
        samples=r['focus_samples'];rolls=[(float(x['roll'])+180)%360-180 for x in samples if x.get('roll') is not None]
        med=statistics.median(rolls) if rolls else None
        near=sum(60<=abs(x)<=120 for x in rolls)/len(rolls) if rolls else 0
        support=len(rolls)/len(samples) if samples else 0
        candidate=len(rolls)>=3 and support>=.5 and near>=.8
        src=r['source']
        result=dict(sample_id=sid,source_key=source_key(src),split=row['split'],duration_s=row['duration_s'],
            source_start_s=src['source_start_s'],roll_median=med,roll_samples=len(rolls),roll_support=support,
            near_90_fraction=near,candidate_sideways=candidate,status='ok')
        brief=dict(media=src['source'],source_size=src['video_transform']['source_size'])
        if candidate:result['candidate_source']=src
        return result,brief
    except Exception as exc:
        return dict(sample_id=sid,status='error',error=repr(exc),measurement_file=str(p)),None


def scan(workers):
    WORK.mkdir(parents=True,exist_ok=True)
    outputs={name:(WORK/(name+'.jsonl')).open('wb') for name in ['orientation_all_clips','orientation_candidates','orientation_errors']}
    hashes={name:hashlib.sha256() for name in outputs};counts=Counter();groups={};source_hash=hashlib.sha256()
    started=last=time.monotonic()
    def rows():
        with AUTHORITY.open('rb') as f:
            for raw in f:
                source_hash.update(raw);yield orjson.loads(raw)
    iterator=rows()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while batch:=list(itertools.islice(iterator,workers*16)):
                for result,brief in pool.map(inspect,batch,chunksize=8):
                    counts['total']+=1;counts[result['status']]+=1
                    names=['orientation_all_clips']
                    if result['status']=='error':names.append('orientation_errors')
                    else:
                        key=result['source_key'];g=groups.setdefault(key,dict(source_key=key,**brief,clips=0,candidate_clips=0,reliable_non_sideways_clips=0))
                        g['clips']+=1
                        if result['candidate_sideways']:
                            names.append('orientation_candidates');counts['candidate_clips']+=1;g['candidate_clips']+=1
                            if 'representative' not in g:g['representative']=result['candidate_source']
                        elif result['roll_support']>=.5:g['reliable_non_sideways_clips']+=1
                    raw=orjson.dumps(result)+b'\n'
                    for name in names:outputs[name].write(raw);hashes[name].update(raw)
                now=time.monotonic()
                if now-last>=20:
                    print(json.dumps(dict(stage='orientation_scan',counts=dict(counts),sources=len(groups),elapsed_s=round(now-started,1))),flush=True);last=now
        for handle in outputs.values():handle.close()
        candidates=[g for g in groups.values() if g['candidate_clips']]
        candidates.sort(key=lambda g:(g['media']['archive'],g['media']['member']))
        for n,g in enumerate(candidates,1):g['review_id']=f'S{n:04d}'
        (WORK/'orientation_candidate_sources.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
        assert counts['total']==1107323
        result=dict(counts=dict(counts),candidate_sources=len(candidates),sources=len(groups),
            authority_manifest=str(AUTHORITY),authority_sha256=source_hash.hexdigest(),
            policy=dict(stage='candidate_only_requires_source_frame_confirmation',near_90_abs_roll_degrees=[60,120],minimum_near90_fraction=.8,minimum_roll_support=.5),
            outputs_sha256={k:v.hexdigest() for k,v in hashes.items()},elapsed_s=time.monotonic()-started)
        (WORK/'orientation_scan_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        for handle in outputs.values():
            if not handle.closed:handle.close()


def source_image(row,dest):
    sys.path.insert(0,str(ROOT/'src'))
    from listen_visual_quality.measurement import SourceBytes
    import av
    from PIL import Image
    src=row['representative'];t=src['source_start_s']+src['duration_s']/2
    with SourceBytes(src['source']) as member,av.open(member) as inp:
        st=inp.streams.video[0];st.codec_context.thread_count=1
        inp.seek(int(max(0,t-.2)/float(st.time_base)),stream=st,backward=True,any_frame=False)
        for frame in inp.decode(video=0):
            if frame.pts is not None and float(frame.pts*frame.time_base)>=t:
                image=frame.to_image();image.thumbnail((960,960));image.save(dest,quality=92)
                return dict(rotation=int(getattr(frame,'rotation',0)),stream_metadata=st.metadata,
                    source_pts_s=float(frame.pts*frame.time_base),source_frame_file=str(dest))
    raise RuntimeError('No representative frame: '+row['review_id'])


def ensure_source_frame(row):
    dest=WORK/'orientation_source_frames'
    file=dest/(row['review_id']+'.jpg');meta=dest/(row['review_id']+'.json')
    if not meta.exists():meta.write_text(json.dumps(source_image(row,file),ensure_ascii=False,indent=2)+'\n')
    return row['review_id']


def sheets(workers):
    from PIL import Image,ImageDraw,ImageFont
    rows=json.loads((WORK/'orientation_candidate_sources.json').read_text());dest=WORK/'orientation_source_frames';dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,review_id in enumerate(pool.map(ensure_source_frame,rows),1):
            if n%64==0:print('FRAMES',n,len(rows),flush=True)
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),13) if fontpath.exists() else ImageFont.load_default()
    pages=[]
    for page,start in enumerate(range(0,len(rows),64),1):
        canvas=Image.new('RGB',(2080,1920),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for n,row in enumerate(rows[start:start+64]):
            x=(n%8)*260;y=(n//8)*240
            im=Image.open(dest/(row['review_id']+'.jpg'));im.thumbnail((252,202));canvas.paste(im,(x+(260-im.width)//2,y+35))
            draw.text((x+4,y+2),f"{row['review_id']} C{row['candidate_clips']}/{row['clips']} N{row['reliable_non_sideways_clips']}",font=font,fill='black')
            draw.text((x+4,y+18),row['media']['member'][:30],font=font,fill='black')
        p=WORK/f'orientation_sources_page_{page:03d}.jpg';canvas.save(p,quality=92);pages.append(str(p))
    (WORK/'orientation_sheets.json').write_text(json.dumps(pages,indent=2)+'\n')
    print(json.dumps(dict(sources=len(rows),pages=pages)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['scan','sheets']);p.add_argument('--workers',type=int,default=24);args=p.parse_args()
    if args.command=='scan':scan(args.workers)
    else:sheets(args.workers)
