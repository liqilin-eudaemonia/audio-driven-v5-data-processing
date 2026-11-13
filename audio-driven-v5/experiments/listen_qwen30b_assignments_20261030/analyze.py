"""Reproduce text-only pilot metrics; accepted output is not human ground truth."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

WORK=Path(__file__).resolve().parent
EXPERIMENTS=WORK.parent
spec=importlib.util.spec_from_file_location('base_analysis',EXPERIMENTS/'listen_qwen30b_fusion4_20261030/analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

RUNS={
    'baseline':('listen_qwen30b_fusion4_20261030',['prior15_direct','prior15_light','j50_direct','j50_light']),
    'groups_v1':('listen_qwen30b_grouped4_20261030',['prior15_direct','prior15_light','j50_direct','j50_light']),
    'fixed_v2':('listen_qwen30b_fixed4_20261030',['prior15_eager','prior15_graph','j50_eager','j50_graph']),
    'constrained_v3':('listen_qwen30b_constrained65_20261030',['all65']),
    'constrained_v3_checks':('listen_qwen30b_constrained_checks_20261030',['direct65','regression8','repeat15']),
    'assignments_v4':('listen_qwen30b_assignments_20261030',['all65','regression8','repeat15']),
}


def compare(left,right):
    common=left.keys()&right.keys()
    return {'both_accepted':len(common),
        'identical_visual_outputs':sum(analysis.output_key(left[k])==analysis.output_key(right[k]) for k in common),
        'left_only_accepted':sorted(left.keys()-right.keys()),
        'right_only_accepted':sorted(right.keys()-left.keys()),
        'left_events_common':sum(len(left[k]['events']) for k in common),
        'right_events_common':sum(len(right[k]['events']) for k in common)}


def main():
    report={'scope':'Text-only proposal, model self-review and CPU replay. No media recheck; no accuracy claim. Each speed arm ran once concurrently, includes startup and retries.',
        'runs':{},'paired':{}}
    outputs={}
    for version,(directory,arms) in RUNS.items():
        analysis.WORK=EXPERIMENTS/directory
        finished=json.loads((analysis.WORK/'finished.json').read_text())
        report['runs'][version]={};outputs[version]={}
        for arm in arms:
            state=finished['arms'][arm]
            assert state['exit_code']==state['audit_exit_code']==0,(version,arm,state)
            metrics,results=analysis.analyze_arm(arm)
            metrics['program_errors']=dict(Counter(
                x['error'].split(':',1)[0].split(' targets ',1)[0] if not 'explicit keepers' in x['error'] else 'target_not_a_kept_source'
                for x in metrics['rejections'] if x['stage']=='program'))
            report['runs'][version][arm]=metrics;outputs[version][arm]=results
    for cohort in ['prior15','j50']:
        report['paired']['v2_'+cohort+'_eager_graph']=compare(outputs['fixed_v2'][cohort+'_eager'],outputs['fixed_v2'][cohort+'_graph'])
    report['paired']['v3_direct_vs_light65']=compare(outputs['constrained_v3_checks']['direct65'],outputs['constrained_v3']['all65'])
    report['paired']['v3_vs_v4_light65']=compare(outputs['constrained_v3']['all65'],outputs['assignments_v4']['all65'])
    report['paired']['v4_same15_batch_sensitivity']=compare(
        {k:v for k,v in outputs['assignments_v4']['all65'].items() if k in {json.loads(l)['job_id'] for l in (WORK/'repeat15.jsonl').read_text().splitlines()}},
        outputs['assignments_v4']['repeat15'])
    (WORK/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    compact={version:{arm:{k:v for k,v in metrics.items() if k not in ('samples_detail','rejections','program_errors')}
        for arm,metrics in arms.items()} for version,arms in report['runs'].items()}
    (WORK/'metrics.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
