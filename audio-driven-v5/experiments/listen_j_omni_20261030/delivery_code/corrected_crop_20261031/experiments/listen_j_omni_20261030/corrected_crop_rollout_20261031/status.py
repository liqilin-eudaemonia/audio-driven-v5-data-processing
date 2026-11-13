import json,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent
states=[]
for p in sorted((W/'workers').glob('*-gpu-*.json')):
 s=json.loads(p.read_text());states.append(dict(name=p.stem,pid=s['pid'],stage=s['stage'],updated='+bounded-crop-v1' in s.get('execution_revision',''),completed=s.get('completed_clips',0),error=s.get('last_error'),heartbeat_age=round(time.time()-s['heartbeat'],1)))
print(json.dumps(dict(at=time.time(),updated=sum(x['updated'] for x in states),producing_new=sum(x['updated'] and x['completed']>0 for x in states),states=states),ensure_ascii=False))
