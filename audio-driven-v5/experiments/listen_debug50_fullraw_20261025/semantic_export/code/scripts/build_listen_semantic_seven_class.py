"""Preserve all natural-language annotations and append model semantic categories."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.semantic_mapping import PROMPT, AUDIO_PROMPT, request_prompt, digest, enrich_sample, policy, validate_answer
from listen_benchmark.seven_class import TAXONOMY


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_answers(run):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['policy']['prompt']!=PROMPT or manifest['policy'].get('audio_prompt')!=AUDIO_PROMPT:raise ValueError('Semantic prompt/version mismatch')
    for name,sha in manifest['input_sha256'].items():
        if hashlib.sha256((run/'input'/name).read_bytes()).hexdigest()!=sha:raise ValueError('Changed semantic input: '+name)
    items=json.loads((run/'items.json').read_text());by_id={r['item_id']:r for r in items}
    if len(by_id)!=len(items):raise ValueError('Duplicate semantic item IDs')
    answers={};seen=set();response_hashes={}
    for job in json.loads((run/'jobs.json').read_text()):
        path=run/'responses'/(job['job_id']+'.json');record=json.loads(path.read_text())
        if (digest(job['payloads'])!=job['payload_sha256'] or record['payload_sha256']!=job['payload_sha256']
                or record['prompt_sha256']!=digest(request_prompt(job['payloads'])) or record['finish_reason']!='stop'
                or record['job_id']!=job['job_id']):raise ValueError('Incomplete/stale response: '+path.name)
        parsed=json.loads(record['raw'])
        if set(parsed)!=set(job['payloads']):raise ValueError('Missing/extra response keys')
        response_hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key,value in parsed.items():
            if key in seen or key not in by_id:raise ValueError('Repeated/unknown response item')
            seen.add(key);item=by_id[key]
            if item['payload']!=job['payloads'][key]:raise ValueError('Payload changed')
            validate_answer(item['payload'],value)
            sample=answers.setdefault(item['sample_id'],{})
            if item['open_event_id'] in sample:raise ValueError('Repeated sample event')
            sample[item['open_event_id']]=dict(payload=item['payload'],answer=value,
                response_file=str(path.relative_to(run)),item_id=key)
    if seen!=set(by_id):raise ValueError('Unclassified input events')
    return answers,manifest,response_hashes


def export_semantic(semantic_run,output,preview=None,baseline=None):
    run=Path(semantic_run).resolve();output=Path(output).resolve();preview=Path(preview).resolve() if preview else None
    targets=[output]+([preview] if preview else [])
    for target in targets:
        if target==run or run in target.parents or target in run.parents:raise ValueError('Use separate export directory')
        if target.exists() and any(target.iterdir()):raise ValueError('Use empty new output: '+str(target))
    if preview and (preview==output or preview in output.parents or output in preview.parents):raise ValueError('Overlapping output directories')
    answers,manifest,response_hashes=load_answers(run)
    labels=[json.loads(s) for s in (run/'input/labels.jsonl').read_text().splitlines() if s.strip()]
    provenance=json.loads((run/'input/provenance.json').read_text())['samples']
    media=json.loads((run/'input/input_review.json').read_text());media={s['id']:s for s in media['samples']}
    ids=[s['sample_id'] for s in labels]
    if len(ids)!=len(set(ids)) or set(ids)!=set(provenance) or set(ids)!=set(media):raise ValueError('Sample identity mismatch')
    if set(answers)-set(ids):raise ValueError('Unknown classified samples')
    results=[enrich_sample(s,provenance[s['sample_id']],answers.get(s['sample_id'],{})) for s in labels]
    baseline_decisions={}
    if baseline:
        baseline=Path(baseline)
        prior_manifest=json.loads((baseline/'manifest.json').read_text())
        if prior_manifest['input_sha256']!=manifest['input_sha256']:raise ValueError('Baseline has different source data')
        baseline_decisions={sid:{d['open_event_id']:d for d in value['decisions']}
                            for sid,value in json.loads((baseline/'provenance.json').read_text())['samples'].items()}
    statuses=Counter();counts=Counter();pending=Counter();changes=Counter()
    for r in results:
        for e in r['events']:
            statuses[e['mapping_status']]+=1
            if e['mapping_status']=='mapped_candidate':counts.update(e['categories'])
            if e['mapping_status']=='needs_review':pending.update(e['categories'])
        for d in r['decisions']:
            previous=baseline_decisions.get(r['sample_id'],{}).get(d['open_event_id'])
            if previous:
                d['previous_rule_status']=previous['status'];d['previous_rule_types']=previous['candidate_types']
                changes[previous['status']+' -> '+d['status']]+=1
    summary=dict(run_id=output.name,source_run=Path(manifest['source_run']).name,semantic_run=run.name,
        samples=len(results),open_events=sum(len(r['events']) for r in results),status_counts=dict(statuses),
        mapped_by_class={k:counts[k] for k in TAXONOMY},review_candidates_by_class={k:pending[k] for k in TAXONOMY},
        word_anchors_preserved=sum(e['event_unit']=='word' for r in results for e in r['events']),
        taxonomy=list(TAXONOMY),all_natural_language_events_preserved=True,model_requests=len(response_hashes),
        previous_rule_transitions=dict(changes),human_ground_truth=False,training_admitted=False,K_train=[],
        media_inference_rerun=False,text_model_inference=True,model='Qwen3-VL-8B-Instruct')
    exports=[{k:v for k,v in r.items() if k not in ('decisions','upstream_exclusions')} for r in results]
    candidates=[dict(r,events=[e for e in r['events'] if e['mapping_status']=='mapped_candidate']) for r in exports]
    samples=[]
    for r in results:
        m=media[r['sample_id']]
        if m['duration']!=r['duration_s']:raise ValueError('Media duration mismatch')
        samples.append(dict(id=r['sample_id'],duration=r['duration_s'],video=m['video'],audio=m['audio'],
                            events=[e for e in r['events'] if e['mapping_status']=='mapped_candidate'],
                            decisions=r['decisions']))
    review=dict(summary=summary,policy=policy(),samples=samples)
    html=(ROOT/'scripts/listen_seven_class_review.html').read_text().replace('__DATA__',json.dumps(review,ensure_ascii=False).replace('</','<\\/'))
    trace=dict(run_id=output.name,policy=policy(),samples={r['sample_id']:{k:r[k] for k in ('decisions','upstream_exclusions')} for r in results})
    for target in targets:
        target.mkdir(parents=True,exist_ok=True)
        for name,rows in [('labels.jsonl',exports),('category_candidates.jsonl',candidates)]:
            (target/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        dump(target/'labels.json',dict(schema='listen-open-semantic-dataset-v1',samples=exports))
        for name,value in [('provenance.json',trace),('summary.json',summary),('mapping_policy.json',policy()),('review.json',review)]:dump(target/name,value)
        (target/'index.html').write_text(html)
    shutil.copytree(run/'input',output/'input')
    shutil.copytree(run/'responses',output/'model_responses')
    for name in ['jobs.json','items.json','manifest.json']:shutil.copy2(run/name,output/('semantic_'+name))
    code_paths=['src/listen_benchmark/semantic_mapping.py','src/listen_benchmark/seven_class.py',
                'scripts/build_listen_semantic_seven_class.py','scripts/build_listen_seven_class.py',
                'scripts/run_listen_semantic_mapping.py','scripts/prepare_listen_semantic_mapping.py',
                'scripts/listen_seven_class_review.html','tests/test_listen_semantic_mapping.py']
    for rel in code_paths:
        dest=output/'code'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    dump(output/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_run=manifest['source_run'],
         semantic_run=str(run),input_sha256=manifest['input_sha256'],response_sha256=response_hashes,
         code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths}))
    print(json.dumps(summary,ensure_ascii=False))
    return summary
