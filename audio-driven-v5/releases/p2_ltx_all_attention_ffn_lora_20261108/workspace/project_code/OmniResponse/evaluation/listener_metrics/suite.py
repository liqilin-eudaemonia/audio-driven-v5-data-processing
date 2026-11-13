"""Bind prepared test data, run eight video + reaction/audio metrics, collect reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from ..responsenet_pairs.bind import read_rows
from ..responsenet_pairs.prepare import write_json,write_jsonl
from ..video.paired_metrics.media import sha256


def enrich(rows,references,mapping,mapping_root):
    refs={r['id']:r for r in references};maps={r['id']:r for r in mapping};out=[]
    for row in rows:
        ref=refs[row['id']];item=maps[row['id']];row=dict(row)
        row['pair_duration_s']=ref['duration_s']
        if 'gt_reference' not in ref:
            row.update(gt_audio=ref['listener']['audio'],gt_asr=ref['listener']['asr'])
        if item.get('generated_audio'):
            path=(mapping_root/Path(item['generated_audio'])).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            if 'gt_audio' in row and path==Path(row['gt_audio']).resolve():raise ValueError('GT audio cannot be supplied as generated audio')
            row['generated_audio']=str(path)
        # Without an explicit WAV, extraction is from generated video only.
        out.append(row)
    return out


def execute(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prepared=args.prepared.resolve();mapping=args.generated_map.resolve()
    stages=args.stages.split(',')
    if not stages or not set(stages)<={'video','reaction','audio'}:raise ValueError('Unknown stage')
    module='seamless' if args.dataset=='seamless' else 'responsenet_pairs'
    command=[sys.executable,'-m',f'evaluation.{module}.bind','--prepared',str(prepared),'--generated-map',str(mapping),'--output',str(out/'bound')]
    if args.size is not None:
        if args.dataset!='responsenet':raise ValueError('--size only supported by ResponseNet explicit resize binding')
        command+=['--size',str(args.size)]
    with (out/'binding.log').open('w') as log:status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if status:raise RuntimeError(f'Binding failed ({status}); see binding.log and bound/binding_audit.json')
    rows=enrich(read_rows(out/'bound/pairs.jsonl'),read_rows(prepared/'references.jsonl'),read_rows(mapping),mapping.parent)
    for row in rows:row['dataset']=args.dataset
    write_jsonl(out/'pairs.jsonl',rows)
    write_json(out/'contract.json',dict(dataset=args.dataset,requested_pairs=len(rows),stages=stages,
        references_sha256=sha256(prepared/'references.jsonl'),generation_map_sha256=sha256(mapping),
        enriched_manifest_sha256=sha256(out/'pairs.jsonl'),size=args.size))
    results={};statuses={}
    for stage in ('video','extensions'):
        if stage=='video':
            if 'video' not in stages:continue
            cmd=[sys.executable,'-m','evaluation.video.paired_metrics.run','--manifest',str(out/'pairs.jsonl'),
                '--assets',str(args.assets.resolve()),'--output',str(out/'video'),'--device',args.device,'--threads',str(args.threads),
                '--batch-size',str(args.batch_size),'--face-padding','64' if args.dataset=='responsenet' else '0',
                '--face-device',args.face_device,'--emotion-device',args.emotion_device,'--pixel-workers',str(args.pixel_workers)]
            if args.gt_cache:cmd+=['--gt-cache',str(args.gt_cache.resolve())]
        else:
            extra=[s for s in stages if s!='video']
            if not extra:continue
            cmd=[sys.executable,'-m','evaluation.listener_metrics.run','--manifest',str(out/'pairs.jsonl'),
                '--assets',str(args.assets.resolve()),'--output',str(out/'extensions'),'--device',args.device,
                '--batch-size',str(args.batch_size),'--stages',','.join(extra)]
        with (out/(stage+'.log')).open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        statuses[stage]=code
        file=out/stage/'summary.json'
        results[stage]=json.loads(file.read_text()) if file.exists() else dict(status='failed',exit_code=code)
    summary=dict(dataset=args.dataset,requested_pairs=len(rows),status='complete' if all(c==0 for c in statuses.values()) else 'partial',
        stage_exit_codes=statuses,results=results,interpretation='See coverage and not_applicable reasons; null is not a successful score')
    write_json(out/'summary.json',summary)
    return 0 if summary['status']=='complete' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['responsenet','seamless'],required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--generated-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path(__file__).parents[1]/'video/paired_metrics/assets.local.json')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--pixel-workers',type=int,default=1)
    p.add_argument('--face-device',default='cpu');p.add_argument('--emotion-device',default='cpu')
    p.add_argument('--gt-cache',type=Path)
    p.add_argument('--stages',default='video,reaction,audio');p.add_argument('--size',type=int)
    raise SystemExit(execute(p.parse_args()))


if __name__=='__main__':main()
