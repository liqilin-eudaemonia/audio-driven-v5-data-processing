#!/usr/bin/env python3
"""Playable blind review packet; previews NEVER enter the detector or scorer."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p2_event_presence.contracts import CLASSES,digest,read,rows,save,sha
from p2_event_presence.reporting import review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-packet',type=Path)
    p.add_argument('--human-labels',type=Path)
    args=p.parse_args();work=args.work.resolve();out=args.output.resolve()
    # Rebuilding a presentation must not rerun detection verification or reset review status.
    if not (work/'blind_review_media.jsonl').exists() or not (work/'human_labels.template.csv').exists():
        review(work)
    entries=rows(work/'blind_review_media.jsonl');identity=digest(entries)
    if (out/'review_packet_contract.json').exists() and read(out/'review_packet_contract.json')['media_index_identity']!=identity:
        raise ValueError('Existing review packet belongs to another input')
    if (out/'review_packet_contract.json').exists() and read(out/'review_packet_contract.json').get('superseded_by'):
        raise ValueError('This packet has been superseded; build the new packet instead of overwriting its redirect')
    out.mkdir(parents=True,exist_ok=True);(out/'preview').mkdir(exist_ok=True);(out/'audio').mkdir(exist_ok=True)
    if not (out/'human_labels.template.csv').exists():
        shutil.copyfile(work/'human_labels.template.csv',out/'human_labels.template.csv')
    pages={}
    style='body{font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 16px;color:#222}article{border:1px solid #ccc;padding:16px;margin:18px 0;border-radius:8px}video{max-width:100%;width:512px}audio{width:min(100%,500px)}code{overflow-wrap:anywhere}li{margin:8px 0}'
    for mod in CLASSES:
        public_entries=[]
        for entry in entries:
            mid=entry['media_id'];src=''
            if mod=='visual' and entry.get('video'):
                target=out/'preview'/(mid+'.mp4')
                if not target.exists():subprocess.run(['ffmpeg','-v','error','-threads','1','-i',entry['video'],'-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)],check=True)
                src=f'preview/{mid}.mp4'
            elif mod=='audio' and entry.get('audio'):
                target=out/'audio'/(mid+'.wav')
                if not target.exists():target.symlink_to(entry['audio'])
                src=f'audio/{mid}.wav'
            public_entries.append(dict(media_id=mid,duration_s=entry['duration_s'],src=src))
        packet=dict(identity=identity,modality=mod,classes=CLASSES[mod],entries=public_entries)
        if args.previous_packet:
            packet['previous_identity']=read(args.previous_packet/'review_packet_contract.json')['media_index_identity']
        if args.human_labels:
            from p2_event_presence.reporting import human_labels
            human_labels(args.human_labels)
            with args.human_labels.open(newline='') as stream:
                submitted=[r for r in csv.DictReader(stream) if r.get('state')]
            raters={r['rater_id'] for r in submitted}
            if len(raters)!=1:raise ValueError('A UI seed must belong to exactly one rater')
            ids={e['media_id'] for e in public_entries}
            packet['seed']=dict(rater=next(iter(raters)),revision=sha(args.human_labels),
                rows=[r for r in submitted if r['modality']==mod and r['media_id'] in ids])
        template=Path(__file__).with_name('p2_event_presence_review_ui.html').read_text()
        payload=json.dumps(packet,ensure_ascii=False).replace('<', '\\u003c')
        name=f'review_{mod}.html'
        (out/name).write_text(template.replace('__PACKET_JSON__',payload))
        pages[mod]=name
    (out/'index.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>整段事件匿名核验</title><style>{style}</style><h1>整段事件匿名核验</h1><p>人工标注只核验事件类别；语境适当性及视听协调性需另行评价。请先独立标注，再查看检测结果，避免共享错误。</p><p><a href="review_visual.html">视觉四类核验</a> · <a href="review_audio.html">音频三类核验</a> · <a href="human_labels.template.csv">空白标签表</a></p></html>')
    save(out/'review_packet_contract.json',dict(media_index_identity=identity,media=len(entries),pages=pages,
        builder_sha256=sha(Path(__file__)),ui_template_sha256=sha(Path(__file__).with_name("p2_event_presence_review_ui.html")),ui_version=2,previews_used_for_detection=False,method_and_side_blinded=True,
        original_media_index_sha256=sha(work/'blind_review_media.jsonl'),human_validation_complete=False))
    print(out)


if __name__=='__main__':main()
