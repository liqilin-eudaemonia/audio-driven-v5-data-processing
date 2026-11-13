"""Read-only live wait for a scheduled report; surface anomalies for inspection."""
import argparse,json,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=range(21),required=True);a=p.parse_args()
folder=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
state=read(folder/'state.json');pid=state['pid'];ticks=state['start_ticks'];next_update=0
print('LIVE_WAIT',pid,'REPORT',a.slot,'UTC',time.strftime('%H:%M:%S',time.gmtime()),flush=True)
while True:
    proc=Path(f'/proc/{pid}/stat')
    if not proc.exists():raise RuntimeError('Guardian PID missing; inspect before recovery')
    identity=proc.read_text().split(') ')[1].split()
    if identity[0]=='Z' or identity[19]!=ticks:raise RuntimeError('Guardian identity no longer live')
    state=read(folder/'state.json');now=time.time()
    if state['pid']!=pid or state['start_ticks']!=ticks:raise RuntimeError('Guardian changed; revalidate new identity')
    if now-state['heartbeat']>180:raise RuntimeError('Guardian heartbeat stale; inspect before recovery')
    latest=read(folder/'latest.json')
    if latest.get('failed_sources'):raise RuntimeError('Failed source queue requires inspection')
    problems={n:{k:v.get(k) for k in ['observation_error','supervisor_error','blocked_gpus','actions']} for n,v in latest['nodes'].items() if any(v.get(k) for k in ['observation_error','supervisor_error','blocked_gpus','actions'])}
    if problems:
        print('INSPECT',json.dumps(problems),flush=True);break
    if now>=next_update:
        print('HEALTH',time.strftime('%H:%M:%S',time.gmtime(now)),'resolved',latest.get('resolved_clips'),'live',[sum(x.get('alive',False) for x in v.get('workers',[])) for v in latest['nodes'].values()],'latest_age',round(now-latest['at']),flush=True)
        next_update=now+60
    report=folder/f'report-{a.slot:02d}.json'
    if report.exists():
        print('REPORT_READY',a.slot,json.dumps({k:v for k,v in read(report).items() if k!='nodes'}),flush=True);break
    time.sleep(10)
