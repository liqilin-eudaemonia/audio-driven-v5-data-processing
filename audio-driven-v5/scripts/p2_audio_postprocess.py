#!/usr/bin/env python3
"""Virtual listener audio and audio-only VAE cache preparation (never starts training)."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--base-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--review',type=Path);source.add_argument('--records',type=Path,help='JSONL of explicit per-sample metadata and final events')
    p=sub.add_parser('verify-waveforms');p.add_argument('--config',type=Path,required=True);p.add_argument('--review',type=Path,help='Compare bitwise with already reviewed WAVs');p.add_argument('--output',type=Path)
    p=sub.add_parser('cache');p.add_argument('--config',type=Path,required=True);p.add_argument('--part',choices=['audio','events','all'],default='all');p.add_argument('--device',default='cuda:0');p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--limit',type=int);p.add_argument('--cpu-threads',type=int,default=2)
    a=parser.parse_args()
    from p2_listen.contracts import read,atomic_json,path_at
    if a.command=='prepare':
        from listen_audio_postprocess.prepare import prepare_view,review_records
        def records():
            with a.records.open() as f:
                for line in f:
                    if line.strip():yield json.loads(line)
        result=prepare_view(a.base_config,review_records(a.review) if a.review else records(),a.output)
    else:
        from p2_listen.config import load,activate
        from listen_audio_postprocess.overlay import settings
        c=load(a.config);activate(c);root,base=settings(c)
        observations=read(root/'manifests/observations.json');targets={r['sample_id']:r for r in read(root/'manifests/targets.json')}
        if a.command=='verify-waveforms':
            from listen_audio_postprocess.recipes import read_listener_audio
            import numpy as np
            import soundfile as sf
            cases={x['sample_id']:x for x in read(a.review)['cases']} if a.review else {}
            report=[]
            for o in observations:
                recipe=read(root/'audio_postprocess/recipes'/f'{o["sample_id"]}.json');wave,rate,meta=read_listener_audio(recipe)
                if cases:
                    case=cases[o['sample_id']];preview=ROOT/'data_previews/listen_benchmark_review/audio_official_chinese100'
                    physical,sr=sf.read(preview/case['media']['processed'],dtype='float32',always_2d=True)
                    if sr!=rate:raise ValueError('Sample rate mismatch')
                    np.testing.assert_array_equal(wave,physical)
                report.append(dict(sample_id=o['sample_id'],**meta))
            result=dict(samples=len(report),physical_comparisons=len(report) if cases else 0,wav_files_written=0,records=report)
            if a.output:atomic_json(a.output,result)
        else:
            if a.shards<1 or not 0<=a.shard_index<a.shards or a.cpu_threads<1 or a.limit is not None and a.limit<1:raise ValueError('Invalid worker limits')
            import torch
            torch.set_num_threads(a.cpu_threads)
            selected=observations[a.shard_index::a.shards]
            if a.limit:selected=selected[:a.limit]
            from listen_audio_postprocess.overlay import build_one
            from listen_audio_postprocess.encoder import NativeAudioEncoder
            from p2_listen import features
            from p2_listen.annotations import rasterize,validate,validate_contract
            from p2_listen.cache import save_part
            session=None
            def encoder():
                nonlocal session
                if session is None:session=NativeAudioEncoder(c,a.device)
                return session
            results=[]
            for o in selected:
                sid=o['sample_id'];r=dict(sample_id=sid)
                if a.part in ('audio','all'):r.update(build_one(c,o,targets[sid],encoder))
                if a.part in ('events','all'):
                    label=read(root/'labels/canonical'/f'{sid}.json');validate(label,o);validate_contract(label,c)
                    _,layout=features.load_input(c,o,'cpu')
                    save_part(features.folder(c,'events',sid),features.identity(c,'events',o,label=label),rasterize(label,layout,c['events']['K_train']))
                    r['events_ready']=True
                results.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
            result=dict(processed=len(results),part=a.part,wav_files_written=0,results=results)
            atomic_json(root/'audio_postprocess'/f'worker-{a.part}-{a.shard_index}.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
