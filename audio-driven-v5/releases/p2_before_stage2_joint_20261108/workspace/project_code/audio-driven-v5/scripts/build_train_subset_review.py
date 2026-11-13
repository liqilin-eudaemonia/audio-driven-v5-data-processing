#!/usr/bin/env python3
"""Frozen random review of an existing train subset; no labels or training media changed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.backchannel_face_stats import CATEGORIES, read, sha, atomic_json


def sample_ids(rows,seed,overall,per_class):
    ids=sorted(r['sample_id'] for r in rows)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate input IDs')
    draws={'overall':random.Random(seed).sample(ids,min(overall,len(ids)))}
    for cat in CATEGORIES:
        pool=sorted(r['sample_id'] for r in rows if int(r[cat+'_count'])>0)
        draws[cat]=random.Random(f'{seed}:{cat}').sample(pool,min(per_class,len(pool)))
    ordered=list(dict.fromkeys(sid for values in draws.values() for sid in values))
    return draws,ordered


def render_html(data,out,font=None):
    template=(ROOT/'scripts/train_subset_review.html').read_text()
    html=template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    font_css=''
    if font:
        from fontTools.ttLib import TTFont
        from fontTools.subset import Subsetter,Options
        face=TTFont(str(font),fontNumber=2);options=Options();options.flavor='woff'
        subset=Subsetter(options=options);subset.populate(text=html);subset.subset(face)
        face.flavor='woff';face.save(out/'review.woff')
        font_css="@font-face{font-family:Review;src:url(review.woff) format('woff');font-display:swap}"
    (out/'index.html').write_text(html.replace('__FONT_CSS__',font_css))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261026);p.add_argument('--overall',type=int,default=50)
    p.add_argument('--per-class',type=int,default=5);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--font',type=Path,help='Optional NotoSansCJK-Regular.ttc; embeds a Chinese font subset')
    a=p.parse_args();dataset=a.dataset.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Review output already exists')
    if min(a.overall,a.per_class,a.workers)<1:raise ValueError('Positive counts/workers required')
    complete=read(dataset/'COMPLETE.json')
    for name in ['samples.csv','manifests/observations.json','manifests/targets.json','summary.json']:
        if sha(dataset/name)!=complete['files'][name]:raise ValueError(f'Dataset changed: {name}')
    rows=list(csv.DictReader((dataset/'samples.csv').open()));draws,ids=sample_ids(rows,a.seed,a.overall,a.per_class)
    byid={r['sample_id']:r for r in rows}
    targets={r['sample_id']:r for r in read(dataset/'manifests/targets.json')}
    observations={r['sample_id']:r for r in read(dataset/'manifests/observations.json')}
    out.mkdir(parents=True);(out/'media').mkdir()
    def build_one(sid):
        row=byid[sid];target=targets[sid];obs=observations[sid]
        if row['split']!='train' or obs['official_split']!='train' or obs['experiment_split']!='train':
            raise ValueError('Non-train review sample')
        with Path(row['label_file']).open('rb') as f:
            f.seek(int(row['label_offset']));raw=f.read(int(row['label_bytes']))
        if hashlib.sha256(raw).hexdigest()!=row['label_sha256']:raise ValueError('Original integrated label changed')
        label=json.loads(raw)
        if label['sample_id']!=sid:raise ValueError('Label/sample mismatch')
        for cat in CATEGORIES:
            if sum(cat in e['categories'] for e in label['events'])!=int(row[cat+'_count']):
                raise ValueError('Category accounting mismatch')
        for name in ['target_video','target_audio']:
            if sha(target[name])!=target['media_hashes'][name]:raise ValueError('Target media changed')
        dest=out/'media'/sid;dest.mkdir()
        # Preview only: copied H.264 pictures, listener WAV encoded to AAC.
        # One HTML video clock keeps sound and video synchronized when seeking.
        subprocess.run(['ffmpeg','-v','error','-nostdin','-threads','1','-i',target['target_video'],
            '-i',target['target_audio'],'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac',
            '-b:a','192k','-threads','1','-t',str(obs['requested_duration_s']),'-movflags','+faststart',
            str(dest/'listener.mp4')],check=True,capture_output=True)
        shutil.copy2(target['target_audio'],dest/'listener.wav')
        shutil.copy2(obs['listener_image'],dest/'poster.png')
        (dest/'label.json').write_bytes(raw)
        sample=dict(id=sid,duration=obs['requested_duration_s'],split='train',
            cohorts=[key for key,values in draws.items() if sid in values],
            selection_reason=row['selection_reason'],pair=row['pair'],
            video=f'media/{sid}/listener.mp4',audio=f'media/{sid}/listener.wav',
            poster=f'media/{sid}/poster.png',label_url=f'media/{sid}/label.json',
            face_p10=float(row['face_short_side_p10']),face_p50=float(row['face_short_side_p50']),
            face_valid=float(row['face_valid_fraction']),
            crop_width=float(row['crop_width']),crop_height=float(row['crop_height']),
            event_count=int(row['event_count']),non_smile_count=int(row['non_smile_event_count']),
            counts={cat:int(row[cat+'_count']) for cat in CATEGORIES},events=label['events'],
            source_label_sha256=row['label_sha256'],target_media_sha256=target['media_hashes'],
            preview_sha256=sha(dest/'listener.mp4'))
        return sample
    with ThreadPoolExecutor(max_workers=a.workers) as pool:samples=list(pool.map(build_one,ids))
    data=dict(schema='p2-train50h-random-review-v1',seed=a.seed,draws=draws,
        summary=read(dataset/'summary.json'),samples=samples,
        policy=dict(overall_random_count=a.overall,per_class_random_count=a.per_class,
            overall_sampling='uniform_without_replacement_from_sorted_sample_ids',
            class_sampling='uniform_without_replacement_with_independent_per_class_seeds',
            stratified_samples_are_not_population_proportions=True,raw_events_preserved=True,
            audio='listener only; original WAV available',video='existing training 512x512 crop; picture stream copied',
            weak_labels=True),dataset=str(dataset),dataset_completion_sha256=sha(dataset/'COMPLETE.json'))
    atomic_json(out/'review.json',data)
    render_html(data,out,a.font)
    atomic_json(out/'build_check.json',dict(samples=len(samples),overall=len(draws['overall']),
        class_draws={c:len(draws[c]) for c in CATEGORIES},media_hashes_checked=True,original_event_counts_checked=True,
        dataset_completion_sha256=data['dataset_completion_sha256'],
        code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/train_subset_review.html']}))
    print(json.dumps(dict(output=str(out),samples=len(samples),seed=a.seed,draws={k:len(v) for k,v in draws.items()})))


if __name__=='__main__':main()
