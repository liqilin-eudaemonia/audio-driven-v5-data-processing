from pathlib import Path
import subprocess,json,time,statistics
root=Path(__file__).resolve().parents[1];w=root/'experiments/p2_vico256_combined8000_20261106';rows=[]
p=subprocess.Popen(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=subprocess.PIPE,text=True)
try:
 with (w/'gpu7.jsonl').open('a',buffering=1) as log:
  for line in p.stdout:
   try:mem,util,power=map(float,line.strip().split(','));state=json.loads((w/'status.json').read_text());r=dict(at=time.time(),stage=state['stage'],memory_mib=mem,utilization_pct=util,power_w=power);rows.append(r);log.write(json.dumps(r)+'\n')
   except (ValueError,json.JSONDecodeError,FileNotFoundError):continue
   if state['stage'] in ['complete','failed']:break
finally:
 p.terminate();p.wait();p.stdout.close()
 report={}
 for stage in sorted({r['stage'] for r in rows}):
  s=[r for r in rows if r['stage']==stage];report[stage]=dict(samples=len(s),mean_utilization_pct=statistics.mean(r['utilization_pct'] for r in s),median_utilization_pct=statistics.median(r['utilization_pct'] for r in s),peak_memory_mib=max(r['memory_mib'] for r in s),peak_power_w=max(r['power_w'] for r in s))
 (w/'GPU_USAGE.json').write_text(json.dumps(report,indent=2))
