#!/usr/bin/env python3
"""Freeze all 50 media-only annotation requests with nod cues and no caption history."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha
from listen_benchmark.native_caption_nod import INSTRUCTION,PROMPT_VERSION,render


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    source=ROOT/'experiments/listen_native_caption_clean50_20261030/text8'
    nod_run=ROOT/'experiments/listen_native_caption_nod50_20261030/nod'
    rows=json.loads((source/'manifest.json').read_text());assert len(rows)==50
    cfg=json.loads((ROOT/'configs/listen_visual_native8_nod_v1.json').read_text())
    ndp=json.loads((nod_run/'protocol.json').read_text())
    assert ndp['profiles']=={'recall':{k:v for k,v in cfg['nodding'].items() if k not in ('input_fps','profile')}}
    out=[]
    for row in rows:
        row=copy.deepcopy(row);sid=row['sample_id'];file=nod_run/'results'/(sid+'.json');nr=json.loads(file.read_text())
        meta=json.loads((Path(nr['video']).parent/'media.json').read_text())
        assert nr['id']==sid and nr['status']=='ok' and nr['fps']==25
        assert abs(nr['duration']-row['duration_s'])<1e-6 and meta['video_transform']==row['video_transform']
        assert abs(meta['source_start_s']-row['source_reference']['source_start_s'])<1e-6
        row['nod_supplement']=dict(sample_id=sid,fps=25,profile='recall',
            events=[dict(e,id=f'nod-{i+1:04d}') for i,e in enumerate(nr['profiles']['recall']['events'])],
            detector_file=str(file),detector_sha256=sha(file),detector_protocol_sha256=sha(nod_run/'protocol.json'))
        text=render(row);assert 'PRIOR PARAGRAPH' not in text and '25 fps' not in text and '8 fps' not in text
        cues=json.loads(text.split('SUPPLEMENTARY NODDING INTERVALS:\n',1)[1])
        assert all(set(e)=={'id','start','end','observation'} for e in cues)
        (run/'prompts').mkdir(parents=True,exist_ok=True);(run/'prompts'/(sid+'.txt')).write_text(text+'\n');out.append(row)
    atomic_json(run/'manifest.json',out,immutable=True)
    protocol=json.loads((source/'protocol.json').read_text())
    for key in ('initial_attempt','baseline'):protocol.pop(key,None)
    protocol.update(task='nod_guided_direct_annotation',prompt=INSTRUCTION,nod_instruction=INSTRUCTION,
                    prompt_version=PROMPT_VERSION,
                    decoding_attempts=[dict(temperature=0,max_tokens=4096,repetition_penalty=1.05),
                                       dict(temperature=0,max_tokens=4096,repetition_penalty=1.12)],
                    retry_policy='At most one retry for empty, unfinished or verbatim-loop output; retain every raw attempt; never truncate or rewrite captions',
                    source_manifest=str(source/'manifest.json'),prepared_manifest_sha256=sha(run/'manifest.json'),
                    scope='fresh independent annotation for every one of the 50 clips; no old paragraph input or output reuse',
                    old_caption_input=False,reuse_previous_outputs=False,
                    nod_protocol_sha256=sha(nod_run/'protocol.json'),gpu_memory_by_gpu={str(g):.85 for g in range(8)})
    atomic_json(run/'protocol.json',protocol,immutable=True);(run/'prompt.txt').write_text(INSTRUCTION+'\n')
    paths=list(json.loads((source/'code_hashes.json').read_text()))+[
        'src/listen_benchmark/native_caption_nod.py','scripts/prepare_native_caption_nod_direct.py',
        'src/listen_benchmark/native_caption_quality.py',
        'configs/listen_visual_native8_nod_v1.json']
    hashes={}
    for rel in paths:
        f=ROOT/rel;dest=run/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[rel]=sha(f)
    atomic_json(run/'code_hashes.json',hashes,immutable=True)
    stats=dict(samples=50,with_nod_cues=sum(bool(r['nod_supplement']['events']) for r in out),
               nod_intervals=sum(len(r['nod_supplement']['events']) for r in out),fresh_requests=50,
               prompt_cue_fields=['id','start','end','observation'],prior_caption_inputs=0,reused_outputs=0)
    atomic_json(run/'preflight.json',stats,immutable=True);print(json.dumps(stats))


if __name__=='__main__':main()
