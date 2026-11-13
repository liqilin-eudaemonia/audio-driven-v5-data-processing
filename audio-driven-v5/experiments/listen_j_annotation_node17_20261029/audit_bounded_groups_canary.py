"""Read only the captured 50 canary IDs; never scan the results tree."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write,result_path
from listen_benchmark.full_raw import views
from listen_benchmark.visual_native8 import observation_prompt

work=Path(__file__).resolve().parent
inputs=read(work/'monitor_snapshots/bounded_groups_canary_inputs.json')
expected={r['sample_id']:r for j in inputs['jobs']
          for r in read(work/'jobs'/f'{j}.json')['rows']}
report={'at':time.time(),'pid':inputs['worker']['pid'],'expected_clips':50,
        'counts':{'clips':0,'requests':0,'valid':0,'truncated':0},
        'missing':[],'errors':[],'invalid':[],'group_metrics':[],'source_receipts':[]}
for sid in inputs['sample_ids']:
    path=result_path(work,'vlm',sid)
    if not path.exists():report['missing'].append(sid);continue
    d=read(path);report['counts']['clips']+=1
    try:
        assert d['sample_id']==sid and d['source']==expected[sid]['source']
        assert d['completed_at']>=inputs['worker']['started_at']
        assert d['training_admitted'] is False
        assert d['execution']['bounded_source_groups'] is True
        assert d['execution']['renderer_workers']==2
        assert d['execution']['clip_batch_size']==50
        windows=views(d['duration_s'])
        assert sorted((r['view'],r['start'],r['end']) for r in d['records'])==sorted(windows)
        for r in d['records']:
            report['counts']['requests']+=1
            report['counts']['valid']+=r['status']=='valid'
            report['counts']['truncated']+=r['finish_reason']=='length'
            if r['status']!='valid':report['invalid'].append({'sample_id':sid,'view':r['view'],'status':r['status'],'error':r.get('error')})
            assert r['arm']=='video8' and r['native']['audio_input'] is False
            assert r['prompt']==observation_prompt(r['end']-r['start'])
            assert r['media']['video']['requested_fps']==8
        metrics=d['prefetch_group_performance']
        if metrics not in report['group_metrics']:report['group_metrics'].append(metrics)
    except (AssertionError,KeyError,TypeError) as e:
        report['errors'].append({'sample_id':sid,'error':repr(e)})
for job in inputs['jobs']:
    data=read(work/'jobs'/f'{job}.json')
    saved=sum(result_path(work,'vlm',r['sample_id']).exists() for r in data['rows'])
    receipt=work/'vlm/completed'/f'{job}.json'
    report['source_receipts'].append({'job':job,'saved':saved,'total':len(data['rows']),'receipt_exists':receipt.exists()})
    if receipt.exists() and saved!=len(data['rows']):report['errors'].append({'job':job,'error':'premature source receipt'})
report['complete']=not report['missing']
write(work/'bounded_groups_canary_result.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
