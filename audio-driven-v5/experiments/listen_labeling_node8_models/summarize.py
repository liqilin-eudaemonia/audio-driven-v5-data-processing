"""Snapshot actual smoke artifacts; schema success is not annotation accuracy."""
import json
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def effective(reviewed,unknown):
    points=sorted(set(t for span in reviewed+unknown for t in span))
    return sum(b-a for a,b in zip(points,points[1:])
        if any(x<=a and b<=y for x,y in reviewed) and not any(x<b and a<y for x,y in unknown))

runs=[]
for path in sorted(OUT.glob('*/runs/*/summary.json')):
    s=json.loads(path.read_text());items=[]
    for item in s['items']:
        stages=[]
        for branch,results in item['stage_results'].items():
            for stage,filename in results.items():
                r=json.loads(Path(filename).read_text());v=r['parsed']
                stages.append(dict(branch=branch,stage=stage,path=str(Path(filename).relative_to(ROOT)),
                    parsed=v,runtime=r['raw_response'].get('annotation_runtime'),
                    inspected_minus_unknown_seconds={k:effective(spans,v['unknown'][k]) for k,spans in v['reviewed'].items()}))
        items.append(dict(sample_id=item['sample_id'],status=item['status'],errors=item['errors'],
                          training_admitted=item['training_admitted'],stages=stages))
    runs.append(dict(name=path.parents[2].name,mode=s['mode'],summary=str(path.relative_to(ROOT)),
                     succeeded=s['succeeded'],failed=s['failed'],items=items))
attempts=[]
for path in sorted(OUT.glob('*/branches/**/attempt-*.json')):
    a=json.loads(path.read_text());body=json.loads(a['body']) if 'body' in a else {}
    choice=(body.get('choices') or [{}])[0]
    attempts.append(dict(path=str(path.relative_to(ROOT)),status=a.get('status'),transport_error=a.get('transport_error'),
        finish_reason=choice.get('finish_reason'),runtime=body.get('annotation_runtime'),
        response_characters=len(choice.get('message',{}).get('content',''))))
result=dict(generated_at=datetime.now(timezone.utc).isoformat(),host='node8 / 172.29.104.220',
    real_models_tested=True,human_event_accuracy_evaluated=False,training_started=False,
    inputs='Original-pixel ROI crop then bicubic 512x512 JPEG; isolated listener audio; timestamped image list, not native video sync',
    interpretation='succeeded means strict schema and pipeline draft only; inspected_minus_unknown_seconds is model-claimed support before human review and frame-bin censoring, NOT verified annotation quality',
    runs=runs,attempts=attempts)
(OUT/'model_smoke_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for r in runs:print(r['name'],r['succeeded'],r['failed'])
