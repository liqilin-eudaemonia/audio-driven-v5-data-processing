"""Report measured ABBA results without treating warmup as control."""
from pathlib import Path
import json,statistics
W=Path(__file__).resolve().parent
report={}
for label in ['single_exact','eight_exact']:
 p=W/label/'RESULTS.json'
 if not p.exists():continue
 rows=json.loads(p.read_text())['results'];valid=[r for r in rows if r['kind']!='warmup']
 groups={}
 for kind in ['baseline','combined']:
  rr=[r for r in valid if r['kind']==kind]
  if rr:groups[kind]=dict(rounds=len(rr),round_means=[r['mean_seconds']for r in rr],mean_seconds=statistics.mean(r['mean_seconds']for r in rr),max_peak_gib=max(r['peak_gib']for r in rr),bitwise_equal=all(all(x['bitwise_equal']for x in r['comparison'].values())for r in rr))
 if len(groups)==2:groups['time_reduction_percent']=100*(1-groups['combined']['mean_seconds']/groups['baseline']['mean_seconds'])
 report[label]=dict(complete=len(rows)==5,groups=groups)
print(json.dumps(report,indent=2))
