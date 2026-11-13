"""Summarize the live trial without changing production settings."""
import json
from pathlib import Path
import statistics
BASE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())

def rate(rows,key,timekey='at'):
 dt=rows[-1][timekey]-rows[0][timekey]
 return (rows[-1][key]-rows[0][key])*60/dt if dt else 0

def main():
 baseline=read(BASE/'baseline.json');samples=read(BASE/'samples.json')
 if not (BASE/'stable.json').exists():print('Workers still draining');return
 stable=read(BASE/'stable.json');samples=samples[stable['first_sample']:];report={}
 for n in ('node17','node8'):
  a=[r for r in baseline[n]['annotation'] if 'vlm' in (r.get('progress') or {})];q=baseline[n]['quality']
  original={branch:rate([{'at':r['at'],branch:r['progress'][branch]['completed_clips']} for r in a],branch) for branch in ('vlm','nod')}
  original['quality']=rate([{'at':r['heartbeat'],'quality':r['completed_clips']} for r in q],'quality')
  original['gpu_util']=statistics.mean(g['utilization'] for r in a for g in r['gpus'])
  trial=[r[n] for r in samples];current={branch:rate(trial,branch) for branch in ('vlm','nod','quality')}
  current['gpu_util']=statistics.mean(u for r in trial for u in r['gpu_util'].values())
  stats=lambda s:{k:int(v) for k,v in (line.split() for line in s['cpu_stat'].splitlines())}
  dt=trial[-1]['at']-trial[0]['at'];cpu=(stats(trial[-1])['usage_usec']-stats(trial[0])['usage_usec'])/1e6/dt if dt else 0
  report[n]={'measurement_s':dt,'baseline_per_min':original,'trial_per_min':current,'cpu_cores':cpu,
    'change_percent':{k:(current[k]/original[k]-1)*100 for k in ('vlm','nod','quality')},
    'gpu_pids_unchanged':trial[0]['gpu_pids']==trial[-1]['gpu_pids'],'last_failures':trial[-1]['failures'],'last_quality_failures':trial[-1]['quality_failures']}
 report['control_adjusted_percent']={k:((report['node17']['trial_per_min'][k]/report['node17']['baseline_per_min'][k])/(report['node8']['trial_per_min'][k]/report['node8']['baseline_per_min'][k])-1)*100 if report['node8']['trial_per_min'][k] else None for k in ('vlm','nod','quality')}
 report['finished']=(BASE/'finished.json').exists();report['restored']=(BASE/'restored.json').exists()
 (BASE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
