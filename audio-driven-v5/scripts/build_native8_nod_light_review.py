#!/usr/bin/env python3
"""Publish CPU-only time alignment/exact-dedup results; no semantic result implied."""
import copy
import json
import shutil
from pathlib import Path
from prepare_native8_nod_semantic_pilot import ROOT,OUT
from listen_visual_native8_nod import save

def main():
    parent=ROOT/'data_previews/listen_benchmark_review/prompt_versions_20261029'
    media={s['id']:s for s in json.loads((parent/'data.json').read_text())['samples']}
    prep=json.loads((OUT/'preprocessing.json').read_text());samples=[]
    for stat in prep['samples']:
        sid=stat['sample_id'];s=copy.deepcopy(media[sid]);s.pop('series',None)
        s['video']='../../'+s['video']
        for k in ('original','speaker'):s['audio'][k]='../../'+s['audio'][k]
        record=json.loads((OUT/'inputs'/f'{sid}.json').read_text())
        s.update(record=record,stats=stat)
        s['originals']=sorted([o for e in record['events'] for o in e['original_observations']],key=lambda e:(e['start'],e['end'],e['observation_id']))
        assert len(s['originals'])==stat['original'] and len(record['events'])==stat['light_dedup']
        for e in record['events']:
            assert all(o['description']==e['description'] and abs(o['start']-e['start'])<1e-9 and abs(o['end']-e['end'])<1e-9 for o in e['original_observations'])
        samples.append(s)
    target=parent/'glare_nodding/semantic_light'
    save(target/'data.json',{'schema':'native8-nod-light-review-v1','llm_executed':False,'training_admitted':False,
                            'samples':samples,'raw_count':prep['raw_count'],'light_count':prep['light_count']})
    shutil.copyfile(ROOT/'scripts/native8_nod_light_review.html',target/'index.html')
    shutil.copyfile(ROOT/'scripts/listen_visual_prompt_review_audio.js',target/'audio.js')
    print(target)

if __name__=='__main__':main()
