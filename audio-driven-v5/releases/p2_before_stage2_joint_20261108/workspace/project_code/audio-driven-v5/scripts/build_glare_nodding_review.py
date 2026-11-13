#!/usr/bin/env python3
"""Publish the independent GLARE pilot next to the unchanged VLM comparison."""
import copy
import argparse
import json
import shutil
from pathlib import Path
from run_glare_nodding_pilot import ROOT, SOURCE, OUT, PROFILES, save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-reactions', action='store_true')
    parser.add_argument('--headshake-direction-check', action='store_true')
    args=parser.parse_args()
    reaction_root=ROOT/'experiments/listen_glare_reactions15_20261029'
    base_reaction_root=reaction_root
    if args.headshake_direction_check:
        assert args.with_reactions, '--headshake-direction-check requires --with-reactions'
        reaction_root=reaction_root/'headshake_direction_v1'
    data = json.loads(SOURCE.read_text())
    samples = []
    by_id = {s['id']: s for s in data['samples']}
    for sid in json.loads((OUT/'samples.json').read_text())['ids']:
        original = by_id[sid]
        sample = copy.deepcopy(original)
        sample['video'] = '../' + sample['video']
        for key in ('original', 'speaker'):
            sample['audio'][key] = '../' + sample['audio'][key]
        sample['series'] = {k: original['series'][k] for k in ('r4:video8', 'r4:video16')}
        sample['detector'] = json.loads((OUT/'results'/f"{sample['id']}.json").read_text())
        if args.with_reactions:
            sample['reactions']=json.loads((reaction_root/'results'/f"{sample['id']}.json").read_text())
            if args.headshake_direction_check:
                from run_glare_nodding_pilot import sha
                check=sample['reactions']['headshake_direction_review']
                assert check['baseline_sha256']==sha(base_reaction_root/'results'/f"{sample['id']}.json")
        samples.append(sample)
    target = SOURCE.parent/'glare_nodding'
    published={'schema': 'glare-nodding-review-v1', 'human_ground_truth': False,
                             'training_admitted': False, 'profiles': PROFILES, 'samples': samples,
                             'summary': json.loads((OUT/'summary.json').read_text())}
    selected_config=ROOT/'configs/listen_visual_native8_nod_v1.json'
    if selected_config.exists():published['selected_scheme']=json.loads(selected_config.read_text())
    if args.with_reactions:
        published['reaction_summary']=json.loads((reaction_root/'summary.json').read_text())
        published['reaction_protocol']=json.loads((base_reaction_root/'protocol.json').read_text())
        if args.headshake_direction_check:
            published['headshake_direction_protocol']=json.loads((reaction_root/'protocol.json').read_text())
        assert published['reaction_summary']['ok']==len(samples)
    save(target/'data.json',published)
    for src, dst in [('glare_nodding_review.html', 'index.html'), ('glare_nodding_review.js', 'review.js'),
                     ('listen_visual_prompt_review_audio.js', 'audio.js')]:
        shutil.copyfile(ROOT/'scripts'/src, target/dst)
    print(target)

if __name__ == '__main__':
    main()
