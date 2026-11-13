"""Export open event labels, separate provenance, and before/after browser review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.open_merge import atomic_consolidate

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--seven-class-output',help='Optional separate, new directory for seven-class candidates')
    p.add_argument('--semantic-run',help='Completed semantic classification run for the optional seven-class export')
    a=p.parse_args();run=Path(a.run).resolve()
    if a.seven_class_output and not a.semantic_run:p.error('--seven-class-output requires --semantic-run; keyword rules are no longer the default')
    out=ROOT/'data_previews/listen_benchmark_review/open_merge';out.mkdir(parents=True,exist_ok=True)
    data=json.loads((run/'input_review.json').read_text());descriptions=json.loads((run/'description_index.json').read_text());normalizations={};normalization_errors=[]
    for path in sorted((run/'normalized').glob('*.json')):
        r=json.loads(path.read_text());v=r.get('parsed');expected=r['job']['descriptions']
        if r['finish_reason']!='stop' or not isinstance(v,dict) or set(v)!=set(expected):normalization_errors.append(path.name);continue
        for key,value in v.items():
            assert descriptions[key]==expected[key]
            normalizations[descriptions[key]]={'id':key,'value':value,'raw_file':str(path.relative_to(run))}
    rules=json.loads((run/'merge_rules.json').read_text()) if (run/'merge_rules.json').exists() else {}
    samples=[];labels=[];trace={};counts=Counter();flags=Counter()
    for sample in data['samples']:
        original=sample['integrated']['events'];result=atomic_consolidate(original,normalizations,sample['duration'],rules=rules)
        counts.update(input_candidates=len(original),output_events=len(result['events']),atomic_observations=result['atomic_observations'],proposed_exclusions=len(result['excluded']),fallbacks=len(result['audit']))
        for event in result['events']:flags.update(event['review_flags'])
        label={'schema':'listen-open-events-v1','version':'atomic-semantic-merge-v2','sample_id':sample['id'],'duration_s':sample['duration'],
               'time_origin':'clip_start_seconds','events':result['events'],'human_ground_truth':False,'training_admitted':False,
               'unlabeled_time_policy':'unknown','negative_coverage':result['negative_coverage']}
        labels.append(label)
        trace[sample['id']]={'source_candidates':original,'events':result['trace'],'excluded':result['excluded'],'audit':result['audit']}
        samples.append({k:sample[k] for k in ['id','duration','video','audio','note','head_pitch']}|{'before':original,'after':result['events'],'trace':result['trace'],'excluded':result['excluded'],'audit':result['audit']})
    summary={'samples':len(samples),**counts,'review_flags':flags,'normalized_descriptions':len(normalizations),'normalization_errors':normalization_errors,
             'human_ground_truth':False,'training_admitted':False,'source_run':data['run'],'source_sha256':hashlib.sha256((run/'input_review.json').read_bytes()).hexdigest()}
    for directory in [run,out]:
        (directory/'labels.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in labels))
        (directory/'labels.json').write_text(json.dumps({'schema':'listen-open-events-dataset-v1','run_id':run.name,'human_ground_truth':False,'samples':labels},ensure_ascii=False,indent=2))
        (directory/'provenance.json').write_text(json.dumps({'run_id':run.name,'normalizations':normalizations,'samples':trace},ensure_ascii=False))
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    review={'run':run.name,'summary':summary,'samples':samples}
    (out/'review.json').write_text(json.dumps(review,ensure_ascii=False))
    template=ROOT/'scripts/listen_open_merge_review.html'
    if template.exists():
        html=template.read_text().replace('__DATA__',json.dumps(review,ensure_ascii=False).replace('</','<\\/'))
        (out/'index.html').write_text(html)
    print(json.dumps(summary,ensure_ascii=False))
    if a.seven_class_output:
        from build_listen_semantic_seven_class import export_semantic
        semantic=Path(a.semantic_run)
        recorded=json.loads((semantic/'manifest.json').read_text())['input_sha256']
        if any(hashlib.sha256((run/name).read_bytes()).hexdigest()!=sha for name,sha in recorded.items()):
            raise ValueError('Semantic run was built from different open labels')
        export_semantic(semantic,a.seven_class_output)

if __name__=='__main__':main()
