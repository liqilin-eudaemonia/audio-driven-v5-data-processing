from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent
jobs=[('bench_local_1',1,24),('bench_node17_8',8,24),('bench_local_4_batch8',4,8),('bench_node17_8_batch8',8,8),('bench_node4_long',1,1),('bench_node4_maxlength',1,1),('bench_24_fixed',24,24),('bench_prod8_batch24',8,24),('bench_prod_maxlength',1,1)]
rows=[]
for name,world,batch in jobs:
 folder=p/f'{name}_results' if world==24 else p/name/'results'
 for f in sorted(folder.glob('profile_*_rank0.json')):
  d=json.loads(f.read_text());r=d['records'][1:];seconds=statistics.mean(x['seconds'] for x in r)
  all_profiles=[json.loads(q.read_text()) for q in folder.glob(f"profile_{d['profile']}_rank*.json")]
  peak=max(x['peak_allocated_mib'] for q in all_profiles for x in q['records'])/1024
  row=dict(job=name,world=world,batch=batch,profile=d['profile'],measured_steps=len(r),mean_seconds=seconds,
           samples_per_second=batch/seconds,media_seconds_per_second=sum(x['media_seconds'] for x in r)/sum(x['seconds'] for x in r),
           peak_allocated_gib=peak,parameter_error_vs_baseline=max(q['parameter_error_vs_baseline'] for q in all_profiles),
           rank_count=len(all_profiles),rank_parameters_identical=all(q.get('rank_parameters_identical',True) for q in all_profiles))
  if 'time' in r[-1]:
   start=d['records'][0]['time'];end=r[-1]['time'];row['utilization_window']='recorded_after_warmup'
  else:
   end=f.stat().st_mtime-1;start=end-sum(x['seconds'] for x in r);row['utilization_window']='approximate_from_profile_file_time'
  nodes=[p/f'{name}_node{i}' for i in range(3)] if world==24 else [p/name]
  util=[];powers=[]
  for n in nodes:
   spec=json.loads((n/'spec.json').read_text());visible={int(x) for x in spec['env']['CUDA_VISIBLE_DEVICES'].split(',')}
   for line in (n/'gpu.jsonl').read_text().splitlines():
    t=json.loads(line)
    if not start<=t['time']<=end:continue
    for gpu in t['gpu'].splitlines():
     cells=[x.strip() for x in gpu.split(',')]
     if len(cells)==4 and int(cells[0]) in visible:
      util.append(float(cells[1]));powers.append(float(cells[3]))
  if util:row.update(mean_gpu_utilization=statistics.mean(util),mean_power_w=statistics.mean(powers),telemetry_samples=len(util))
  rows.append(row)
(p/'benchmark_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 print(r['job'],r['profile'],f"{r['mean_seconds']:.3f}s {r['samples_per_second']:.3f}samples/s {r['peak_allocated_gib']:.1f}GiB",f"util={r.get('mean_gpu_utilization',0):.1f}%",'error',r['parameter_error_vs_baseline'])
