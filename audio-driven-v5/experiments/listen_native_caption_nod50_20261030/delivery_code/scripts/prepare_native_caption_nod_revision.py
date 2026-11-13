#!/usr/bin/env python3
"""Freeze 25 Hz nod evidence and prior captions for the Omni revision experiment."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,digest,sha
from listen_benchmark.native_caption_nod import INSTRUCTION,render


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    baseline=ROOT/'experiments/listen_native_caption_clean50_20261030/text8';output=run/'omni'
    rows=json.loads((baseline/'manifest.json').read_text());cfg=json.loads((ROOT/'configs/listen_visual_native8_nod_v1.json').read_text())
    nod_protocol=json.loads((run/'nod/protocol.json').read_text())
    expected={k:v for k,v in cfg['nodding'].items() if k not in ('input_fps','profile')}
    assert nod_protocol['profiles']=={'recall':expected}
    manifest=[];counts=dict(samples=0,nod_clips=0,nod_intervals=0,reused_no_nod=0)
    for row in rows:
        row=copy.deepcopy(row);sid=row['sample_id'];np=run/'nod/results'/(sid+'.json');nr=json.loads(np.read_text())
        prior_file=baseline/'results'/(sid+'.json');prior=json.loads(prior_file.read_text())
        meta=json.loads((Path(nr['video']).parent/'media.json').read_text())
        assert nr['status']=='ok' and nr['id']==sid and nr['fps']==25
        assert abs(nr['duration']-row['duration_s'])<1e-6
        assert meta['video_transform']==row['video_transform'] and abs(meta['source_start_s']-row['source_reference']['source_start_s'])<1e-6
        assert prior['sample_id']==sid and prior['recipe_sha256']==row['recipe_sha256']
        events=[dict(e,id=f'nod-{i+1:04d}') for i,e in enumerate(nr['profiles']['recall']['events'])]
        row['nod_supplement']=dict(sample_id=sid,fps=25,profile='recall',events=events,detector_file=str(np),
                                  detector_sha256=sha(np),detector_protocol_sha256=sha(run/'nod/protocol.json'),
                                  prior_result_file=str(prior_file),prior_result_sha256=sha(prior_file),prior_caption=prior['caption'])
        counts['samples']+=1;counts['nod_intervals']+=len(events)
        if events:
            counts['nod_clips']+=1
            text=render(row)
            (output/'prompts').mkdir(parents=True,exist_ok=True)
            (output/'prompts'/(sid+'.txt')).write_text(text+'\n')
        else:
            counts['reused_no_nod']+=1
            result=dict(prior,reused_no_nod=True,reused_from=str(prior_file),gpu=None,
                        reuse_source_gpu=prior['gpu'],nod_supplement=row['nod_supplement'])
            atomic_json(output/'results'/(sid+'.json'),result,immutable=True)
        manifest.append(row)
    atomic_json(output/'manifest.json',manifest,immutable=True)
    protocol=json.loads((baseline/'protocol.json').read_text())
    protocol.update(task='nod_guided_revision',nod_instruction=INSTRUCTION,baseline=str(baseline),
                    prepared_manifest_sha256=sha(output/'manifest.json'),
                    scope='same 50 clips; revise only clips with GLARE recall candidates; reuse other captions verbatim',
                    nod_protocol_sha256=sha(run/'nod/protocol.json'),counts=counts)
    atomic_json(output/'protocol.json',protocol,immutable=True)
    (output/'prompt.txt').write_text(INSTRUCTION+'\n')
    paths=list(json.loads((baseline/'code_hashes.json').read_text()))+[
        'src/listen_benchmark/native_caption_nod.py','scripts/run_glare_nodding_pilot.py',
        'scripts/prepare_native_caption_nods.py','scripts/prepare_native_caption_nod_revision.py',
        'configs/listen_visual_native8_nod_v1.json']
    hashes={}
    for rel in paths:
        f=ROOT/rel;dest=output/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[rel]=sha(f)
    atomic_json(output/'code_hashes.json',hashes,immutable=True);atomic_json(run/'plan.json',counts,immutable=True)
    print(json.dumps(counts))


if __name__=='__main__':main()
