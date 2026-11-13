#!/usr/bin/env python3
"""Replay all six channels from immutable inputs using the vendor functions."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from run_glare_nodding_pilot import OUT as NOD_OUT, VENDOR, SOURCE, module, sha, intervals, save
from run_glare_reaction_pilot import OUT, TYPES

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    ids=json.loads((OUT/'samples.json').read_text())['ids']
    summary=json.loads((OUT/'summary.json').read_text())
    assert len(ids)==15 and summary['ok']==15 and not summary['errors']
    assert sha(SOURCE)==protocol['source_sha256']
    mods={k:module('verify_'+k,VENDOR/f'reaction_detector/{k}/run_{k}_intensity_mt.py') for k in TYPES if k!='nodding'}
    overlaps=0;total=0;fer_missing=0
    for sid in ids:
        r=json.loads((OUT/'results'/f'{sid}.json').read_text());nod=json.loads((NOD_OUT/'results'/f'{sid}.json').read_text())
        assert r['status']=='ok' and r['channels']['nodding']==nod['profiles']['recall']
        pkl=NOD_OUT/'landmarks'/f'{sid}.pkl';f=OUT/'fer'/f'{sid}.csv'
        assert sha(pkl)==r['input_hashes']['pose'] and sha(Path(nod['video']))==r['input_hashes']['video']
        rows=list(csv.DictReader(f.open()));raw=json.loads((OUT/'fer_raw'/f'{sid}.json').read_text())
        assert len(rows)==len(raw)==r['frames']
        for i,(row,item) in enumerate(zip(rows,raw)):
            assert int(row['frame_index'])==item['frame_index']==i
            assert (item['selected'] is not None)==bool(r['fer_detected'][i])
            for k in ('angry','disgust','fear','happy','sad','surprise','neutral'):
                val=item['faces'][item['selected']]['emotions'][k] if item['selected'] is not None else 0
                assert float(row[k])==val and 0<=val<=1
        fer_missing+=r['frames']-sum(r['fer_detected'])
        for name,mod in mods.items():
            assert sha(Path(mod.__file__))==protocol['vendor_sha256'][name]
            cfg=protocol['parameters'][name]
            if name=='head_shaking':expected=mod.compute_head_shaking_intensity(*mod.load_pose_from_pkl(pkl),argparse.Namespace(**cfg))
            elif name=='smiling':expected=mod.compute_smiling_intensity(mod.load_fer_happy_scores(f),*mod.load_smile_blendshape_scores(pkl),**cfg)
            elif name=='laughing':expected=mod.compute_laughing_intensity(mod.load_fer_happy_scores(f),*mod.load_blendshape_signals(pkl),**cfg)
            elif name=='frowning':expected=mod.compute_frowning_intensity(*mod.load_fer_emotion_scores(f),*mod.load_blendshape_signals(pkl),**cfg)
            else:expected=mod.compute_surprised_intensity(mod.load_fer_surprise_scores(f),*mod.load_blendshape_signals(pkl),**cfg)
            np.testing.assert_array_equal(expected,r['channels'][name]['scores'])
            derived=intervals(expected,nod['pose']['detected'],r['fps'],r['duration'])
            events=r['channels'][name]['events'];assert len(events)==len(derived)
            for event,ref in zip(events,derived):
                for k in ref:assert event[k]==ref[k]
                assert 0<=event['start']<event['end']<=r['duration']
        active=np.array([r['channels'][k]['scores'] for k in TYPES])>0
        overlaps+=int((active.sum(axis=0)>1).sum())
        assert len(r['supplementary_events'])==sum(len(r['channels'][k]['events']) for k in TYPES)
        for e in r['supplementary_events']:
            assert e['modality']=='visual' and e['status']=='candidate' and e['training_admitted'] is False
        total+=len(r['supplementary_events'])
    assert overlaps>0,'Multilabel coexistence should remain intact'
    report={'passed':True,'samples':len(ids),'supplementary_events':total,'multilabel_frames_retained':overlaps,'fer_missing_frames':fer_missing,
            'checks':['raw FER probabilities and zero-based indices match every CSV row','all five vendor detector outputs exactly replayed',
                      'approved nodding scores and intervals unchanged','independent overlapping channels preserved; no argmax',
                      'time bounds, video/landmark/code hashes, candidate provenance and immutable VLM source verified']}
    save(OUT/'audit_results.json',report);print(json.dumps(report))

if __name__=='__main__':main()
