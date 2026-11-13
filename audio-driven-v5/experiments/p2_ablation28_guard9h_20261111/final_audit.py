"""Requirement-by-requirement final audit; never stops healthy training."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,statistics
G=Path(__file__).resolve().parent;sys.path.insert(0,str(G))
import watch as M
import recover as C

def run():
 import torch
 torch.set_num_threads(2)
 protocol=M.read(G/'protocol.json');assert time.time()>=protocol['deadline'],'Nine-hour window has not elapsed'
 reports=[M.read(G/f'hour{i:02d}.json')for i in range(10)]
 assert (G/'WINDOW_FINISHED.json').exists()
 assert all(r['contract']['success']and r['cluster']['observed']and r['cluster']['healthy28']and r['controller_live']for r in reports),'Hourly coverage/health not fully proven'
 assert all(abs(r['health']['at']-(protocol['start']+i*3600))<600 for i,r in enumerate(reports)),'Hourly cadence gap'
 owner=M.read(G/'CURRENT_OWNER.json');assert M.alive(owner)
 h=M.health();s=M.probe(h['arm']);co=M.contract(h['arm']);v=M.speed(h)
 assert s['observed']and s['healthy28']and not h['flags']and h['age']<120
 assert v['normalized_ratio']<=1.3,'Unresolved slowdown'
 assert all(not n['foreign']for n in s['nodes']),'Selected-card competition remains'
 out=M.R.output(h['arm']);cp=C.latest(out);m=M.R.noevent_runtime().load()if h['arm']=='no_event'else M.R.load()
 payload=m.checkpoint.checked(cp);assert payload['identity']==M.read(out/'run.json') and len(payload['rank_rng'])==28
 checkpoint=dict(path=str(cp),step=payload['step'],sha256=M.read(cp.with_suffix('.complete.json'))['sha256'],full_model_adam_rng_verified=True);del payload
 losses=h['rows'][-100:]
 assert all(all(__import__('math').isfinite(r[k])for k in ['av','event','feature_video','feature_audio','weighted_total'])for r in losses)
 result=dict(success=True,at=time.time(),elapsed_seconds=time.time()-protocol['start'],window=protocol,hourly_reports=10,latest_health={k:v for k,v in h.items()if k!='rows'},latest_cluster=s,contract=co,speed=v,checkpoint=checkpoint,training_continues=True,reserved_gpu='node11:4-7',notes='9h observation verified; this is not a claim that all ablation training has completed')
 M.atomic_json(G/'FINAL_ACCEPTED.json',result);print(json.dumps({k:result[k]for k in ['success','at','elapsed_seconds','hourly_reports','checkpoint']}),flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:
  M.atomic_json(G/'FINAL_INCOMPLETE.json',dict(success=False,at=time.time(),error=repr(e)));raise
