#!/usr/bin/env python3
"""Comparable independent requests on the same listener media; no synthetic labels."""
import argparse,concurrent.futures,json,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_labeling.pipeline import load_manifest,branch_inputs,code_hash
from listen_labeling.client import call,prompt
from listen_labeling.guided import response_format
from p2_listen.contracts import atomic_json,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=50);args=p.parse_args()
 cfg=json.loads(Path(args.config).read_text());out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 samples=load_manifest(args.manifest)[:args.limit];code=code_hash();inputs={}
 def prepare(row):
  visual,vm=branch_inputs(row,'visual',cfg['media']);audio,am=branch_inputs(row,'audio',cfg['media'])
  return row['sample_id'],{'visual':(visual,vm),'audio':(audio,am),'omni':(visual+audio,dict(vm,**am))}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for sid,parts in pool.map(prepare,samples):inputs[sid]=parts
 atomic_json(out/'inputs.json',dict(samples=samples,config=cfg,code_hash=code))
 locks={s['endpoint']['base_url']:threading.Semaphore(cfg.get('concurrency_per_server',4)) for s in cfg['systems']}
 def task(row,system,stage):
  branch=system['branch'];ep=system['endpoint'];sid=row['sample_id'];content,meta=inputs[sid][branch]
  fmt=response_format(branch,row['duration_s'],stage) if cfg.get('guided_json',False) else None
  identity=dict(response_format=fmt,code=code,chat_template_kwargs=dict(enable_thinking=False),endpoint=ep,branch=branch,stage=stage,row=row,prompt=prompt(branch,row['duration_s'],stage))
  fp=digest(identity);folder=out/'requests'/system['name']/sid/stage/fp
  result=dict(system=system['name'],branch=branch,stage=stage,sample_id=sid)
  began=time.monotonic()
  try:
   if (folder/'result.json').exists():value=json.loads((folder/'result.json').read_text())
   else:
    with locks[ep['base_url']]:value=call(ep,branch,row['duration_s'],content,folder,fp,meta,stage,chat_template_kwargs=dict(enable_thinking=False),response_format=fmt)
    atomic_json(folder/'result.json',value,immutable=True)
   result.update(status='schema_pass',result=str(folder/'result.json'),parsed=value['parsed'])
  except Exception as exc:result.update(status='failed',error_type=type(exc).__name__,error=str(exc))
  result['wall_seconds_including_queue']=time.monotonic()-began
  atomic_json(folder/'outcome.json',result);return result
 results=[]
 jobs=[(row,s,stage) for row in samples for stage in ['discovery','audit'] for s in cfg['systems']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  pending=[pool.submit(task,*job) for job in jobs]
  for f in concurrent.futures.as_completed(pending):
   r=f.result();results.append(r)
   atomic_json(out/'summary.json',dict(total_requests=len(jobs),completed=len(results),complete=len(results)==len(jobs),human_ground_truth=False,results=results))
   print(json.dumps(dict(completed=len(results),total=len(jobs),system=r['system'],stage=r['stage'],status=r['status']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
