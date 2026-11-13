#!/usr/bin/env python3
"""Check supplementary nod provenance and caption reuse, not detector accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha
from listen_benchmark.native_caption_nod import render


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();base=a.run.resolve();run=base/'omni'
    rows=read(run/'manifest.json');protocol=read(run/'protocol.json');assert len(rows)==50
    assert sha(run/'manifest.json')==protocol['prepared_manifest_sha256']
    ids={r['sample_id'] for r in rows};assert len(ids)==50 and {f.stem for f in (run/'results').glob('*.json')}==ids
    counts=Counter();cases=[];gpus=Counter();total_new_seconds=0.
    for row in rows:
        sid=row['sample_id'];supp=row['nod_supplement'];r=read(run/'results'/(sid+'.json'));prior=read(supp['prior_result_file']);nod=read(supp['detector_file'])
        assert sha(supp['prior_result_file'])==supp['prior_result_sha256'] and sha(supp['detector_file'])==supp['detector_sha256']
        assert sha(base/'nod/protocol.json')==supp['detector_protocol_sha256']==protocol['nod_protocol_sha256']
        assert nod['status']=='ok' and nod['id']==sid and nod['fps']==25
        assert nod['video_sha256']==sha(nod['video'])
        assert nod['media_metadata_sha256']==sha(Path(nod['video']).parent/'media.json')
        assert r['sample_id']==sid and r['recipe_sha256']==prior['recipe_sha256']==row['recipe_sha256']
        assert r['input_evidence']['audio_identity']==prior['input_evidence']['audio_identity']
        assert r['input_evidence']['video_evidence']==prior['input_evidence']['video_evidence']
        assert r['finish_reason']=='stop' and len(r['caption'].split())>=10
        assert re.fullmatch(r'[A-Za-z][^\n\r]*',r['caption'])
        counts['samples']+=1;counts['candidate_intervals']+=len(supp['events'])
        if not supp['events']:
            assert r['reused_no_nod'] and r['caption']==prior['caption'] and r['gpu'] is None
            counts['reused_no_nod']+=1;continue
        assert r['prompt']==render(row) and r['input_evidence']['nod_supplement']==supp
        assert not r.get('reused_no_nod');gpus[r['gpu']]+=1;total_new_seconds+=r['seconds'];counts['fresh_omni_revisions']+=1
        named=bool(re.search(r'\bnod\w*\b',r['caption'],re.I));unchanged=r['caption']==prior['caption']
        kind='nod_word_present' if named else 'unchanged_despite_candidates' if unchanged else 'changed_without_nod_word'
        counts[kind]+=1;cases.append(dict(sample_id=sid,candidate_intervals=supp['events'],text_outcome=kind))
    workers=[read(f) for f in (run/'workers').glob('*.json')]
    assert len(workers)==8 and set(gpus)==set(range(8)) and sum(gpus.values())==counts['fresh_omni_revisions']==38
    assert all(w['stage']=='complete' and w['completed']==w['total']==gpus[w['gpu']] for w in workers)
    assert counts['reused_no_nod']==12
    audit=dict(passed=True,counts=dict(counts),per_gpu=dict(sorted(gpus.items())),
               prior_captions_preserved=True,no_nod_captions_reused_verbatim=True,
               frame_and_audio_identity_unchanged=True,nod_source_identity_verified=True,
               new_inference_seconds=total_new_seconds,human_ground_truth=False,training_admitted=False,
               accuracy_or_temporal_placement_verified=False,
               note='Text counts do not establish true nod counts, detector precision, or correctly integrated timing.')
    atomic_json(run/'audit.json',audit);atomic_json(run/'integration_cases.json',cases)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
