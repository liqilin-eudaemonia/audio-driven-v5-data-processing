#!/usr/bin/env python3
"""Audit confirmed-scheme exports; optionally check one real source decode, CPU only."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from listen_visual_native8_nod import ROOT, CONFIG, sha, save
from listen_benchmark.visual_native8 import observation_prompt

RUN=ROOT/'experiments/listen_visual_native8_nod_20261029'
FROZEN=ROOT/'experiments/listen_visual_input50_20261029'

def main():
    p=argparse.ArgumentParser();p.add_argument('--decode-one',action='store_true');args=p.parse_args()
    cfg=json.loads(CONFIG.read_text());assert cfg['supplement_enabled']==['nodding']
    assert cfg['fps']==8 and 'head_shaking' in cfg['supplement_disabled']
    protocol=json.loads((FROZEN/'protocol.json').read_text())
    assert observation_prompt(6)==protocol['prompt_example']
    rows=[json.loads(x) for x in (RUN/'bundles.jsonl').read_text().splitlines()]
    assert len(rows)==15
    for row in rows:
        raw=json.loads(Path(row['sources']['vlm_file']).read_text())
        nod=json.loads(Path(row['sources']['nod_file']).read_text())
        assert row['vlm_observations']==[r for r in raw['records'] if r['arm']=='video8']
        full=next(r for r in row['vlm_observations'] if r['view']=='full')
        assert row['vlm_full_events']==full['parsed']['events']
        assert len(row['nod_candidates'])==len(nod['profiles']['recall']['events'])
        for actual,original in zip(row['nod_candidates'],nod['profiles']['recall']['events']):
            assert all(actual[k]==v for k,v in original.items())
            assert actual['type']=='nodding'
        assert row['training_admitted'] is False and row['config_sha256']==sha(CONFIG)
    report={'passed':True,'samples':len(rows),'checks':['exact C prompt','only video8 observations; invalid windows preserved',
            'verbatim full-clip descriptions and times','exact recalled nod intervals; no shake/expression supplements',
            'candidate separation; no automatic training admission'],'new_gpu_inference_run':False}
    if args.decode_one:
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
        import numpy as np
        from transformers import AutoProcessor
        from listen_benchmark.visual_native8 import prepare_media,prepare_input
        spec=importlib.util.spec_from_file_location('historical_native_run',FROZEN/'run.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        samples=json.loads((FROZEN/'samples.json').read_text())
        sample=min(samples,key=lambda s:s['source']['duration_s'])
        processor=AutoProcessor.from_pretrained(cfg['model'],trust_remote_code=True,local_files_only=True)
        first=prepare_media(sample['source']);second=old.prepare_media(sample['source'])
        assert set(first)=={8} and first[8]['meta']==second[8]['meta']
        assert all(np.array_equal(a,b) for a,b in zip(first[8]['images'],second[8]['images']))
        for view,start,end in old.views(first[8]['duration_s']):
            new,evidence=prepare_input(processor,first,start,end,view)
            old_inputs,old_evidence=old.prepared_inputs(processor,second,start,end,view)
            assert evidence==old_evidence[1]
            assert new['prompt']==old_inputs[1]['prompt'] and new['mm_processor_kwargs']==old_inputs[1]['mm_processor_kwargs']
            nv,nm=new['multi_modal_data']['video'][0];ov,om=old_inputs[1]['multi_modal_data']['video'][0]
            assert nm==om and np.array_equal(nv,ov)
        report['real_decode_check']={'sample_id':sample['sample_id'],'result':'8fps frames/JPEG metadata, video tensor, rendered prompt and all windows exactly match historical video8 preparation'}
    save(RUN/'audit.json',report);print(json.dumps(report))

if __name__=='__main__':main()
