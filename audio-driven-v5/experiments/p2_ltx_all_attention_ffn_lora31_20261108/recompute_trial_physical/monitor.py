from pathlib import Path
import json,os,time,statistics,math,fcntl
OUT=Path('/private/liqi_workspace/project_code/audio-driven-v5/experiments/p2_ltx_all_attention_ffn_lora31_20261108');HERE=Path(__file__).resolve().parent
with(HERE/'monitor.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not(HERE/'MONITOR_STOP').exists():
  p=OUT/'metrics.jsonl'
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-250000));lines=f.read().splitlines()
  rows=[]
  for line in lines:
   try:rows.append(json.loads(line))
   except ValueError:pass
  rows=rows[-30:]
  if rows:
   value=dict(at=time.time(),pid=os.getpid(),step=rows[-1]['step'],policy='all',mean_step_seconds=statistics.mean(x['seconds']for x in rows),input_wait_seconds=statistics.mean(x['input_wait_seconds']for x in rows),peak_rank0_reserved_gib=max(x['peak_reserved_gib']for x in rows),metrics_age_seconds=time.time()-p.stat().st_mtime,finite=all(math.isfinite(x['weighted_total'])for x in rows))
   tmp=HERE/'PERFORMANCE.tmp';tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,HERE/'PERFORMANCE.json')
   if json.loads((OUT/'status.json').read_text()).get('complete'):break
  time.sleep(30)
