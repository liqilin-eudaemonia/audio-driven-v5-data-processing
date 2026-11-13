#!/usr/bin/env python3
"""Verify all 50 captions were newly inferred without previous caption inputs."""
import argparse
from collections import Counter
import json
import importlib.util
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import atomic_json,sha


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    rows=read(run/'manifest.json');protocol=read(run/'protocol.json');assert len(rows)==50
    # Historical runs must remain auditable after the live prompt changes.
    spec=importlib.util.spec_from_file_location('frozen_caption_prompt',run/'code/src/listen_benchmark/native_caption_nod.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);render=module.render
    assert protocol['task']=='nod_guided_direct_annotation' and not protocol['old_caption_input']
    assert sha(run/'manifest.json')==protocol['prepared_manifest_sha256']
    for rel,expected in read(run/'code_hashes.json').items():
        assert sha(run/'code'/rel)==expected, f'Frozen source changed: {rel}'
    ids={r['sample_id'] for r in rows};assert len(ids)==50 and {f.stem for f in (run/'results').glob('*.json')}==ids
    counts=Counter();gpus=Counter();cases=[]
    prior_dir=ROOT/'experiments/listen_native_caption_clean50_20261030/text8/results'
    for row in rows:
        sid=row['sample_id'];r=read(run/'results'/(sid+'.json'));supp=row['nod_supplement'];text=render(row)
        assert not any(k in supp for k in ('prior_caption','prior_result_file','prior_result_sha256'))
        assert r['prompt']==text and text in r['rendered_prompt']
        assert not any(s in text for s in ('PRIOR PARAGRAPH','prior paragraph','25 fps','8 fps','Revise'))
        cues=json.loads(text.split('SUPPLEMENTARY NODDING INTERVALS:\n',1)[1])
        assert all(set(c)=={'id','start','end','observation'} for c in cues)
        assert [(c['id'],c['start'],c['end']) for c in cues]==[(e['id'],e['start'],e['end']) for e in supp['events']]
        assert sha(supp['detector_file'])==supp['detector_sha256'] and r['input_evidence']['nod_supplement']==supp
        assert r['sample_id']==sid and r['gpu'] in range(8) and not r.get('reused_no_nod') and 'reused_from' not in r
        if protocol.get('prompt_version'):
            silent=row['audio_preflight']['full_silence']
            assert r['input_evidence']['listener_audio_is_silent'] is silent
            assert 'listener_audio_is_silent: '+json.dumps(silent) in text
            attempts=r['attempts'];assert 1<=len(attempts)<=2
            assert r['selected_attempt']==len(attempts)
            assert attempts[-1]['caption']==r['caption'] and attempts[-1]['finish_reason']==r['finish_reason']
            for i,attempt in enumerate(attempts):
                saved=read(run/'attempts'/sid/f'{i+1}.json')
                assert {k:v for k,v in saved.items() if k!='failures'}=={k:v for k,v in attempt.items() if k!='failures'}
                # Early worker versions shared the list with later semantic review flags.
                # The immediately persisted attempt is authoritative for generation checks.
                assert attempt['failures'][:len(saved['failures'])]==saved['failures']
                assert all(f in r['review_flags'] for f in attempt['failures'][len(saved['failures']):])
                assert attempt['decoding']==protocol['decoding_attempts'][i]
                if i<len(attempts)-1:assert saved['failures']
            assert r['generation_checks_passed'] is (not bool(saved['failures']))
            counts['retried_samples']+=len(attempts)>1
            counts['unresolved_generation_failures']+=not r['generation_checks_passed']
            counts['silence_flags_verified']+=1
        assert r['finish_reason'] in ('stop','length') and re.fullmatch(r'[A-Za-z][^\n\r]*',r['caption'])
        assert len(r['caption'].split())>=10
        prior=read(prior_dir/(sid+'.json'))
        # Historical results are read only by this post-inference audit, not by the request builder.
        assert r['input_evidence']['audio_identity']==prior['input_evidence']['audio_identity']
        assert r['input_evidence']['video_evidence']==prior['input_evidence']['video_evidence']
        counts['fresh_inferences']+=1;counts['with_nod_cues']+=bool(cues);counts['without_nod_cues']+=not cues
        counts['normal_finish']+=r['finish_reason']=='stop';counts['truncated']+=r['finish_reason']=='length'
        named=bool(re.search(r'\bnod\w*\b',r['caption'],re.I));counts['nod_word_mentions']+=named
        counts['with_cues_and_nod_word']+=bool(cues) and named
        gpus[r['gpu']]+=1
        cases.append(dict(sample_id=sid,nod_intervals=cues,nod_word_mention=named,
                          review_flags=r['review_flags'],finish_reason=r['finish_reason'],full_silence=row['audio_preflight']['full_silence']))
    workers=[read(f) for f in (run/'workers').glob('*.json')]
    assert len(workers)==8 and set(gpus)==set(range(8)) and sum(gpus.values())==50
    assert all(w['stage']=='complete' and w['completed']==w['total']==gpus[w['gpu']] for w in workers)
    audit=dict(passed=counts['truncated']==0 and counts['unresolved_generation_failures']==0,input_identity_and_fresh_inference_checks_passed=True,
               counts=dict(counts),per_gpu=dict(sorted(gpus.items())),
               previous_caption_input_count=0,reused_output_count=0,four_cue_fields_only=True,
               all_media_identity_checks_passed=True,all_captions_nonempty=True,all_finished_normally=counts['truncated']==0,
               human_ground_truth=False,training_admitted=False,semantic_quality_verified=False)
    atomic_json(run/'audit.json',audit);atomic_json(run/'review_cases.json',cases)
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
