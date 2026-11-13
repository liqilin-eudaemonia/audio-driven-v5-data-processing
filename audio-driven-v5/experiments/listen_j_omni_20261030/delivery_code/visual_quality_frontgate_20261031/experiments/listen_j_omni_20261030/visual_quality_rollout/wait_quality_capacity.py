"""Observe a verified live replacement controller before timing its throughput."""
import argparse,json,subprocess,time,socket
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--node',required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-wait',type=int,default=900);a=p.parse_args()
w=Path(__file__).resolve().parents[1];cfg=json.loads((w/'config.json').read_text());assert socket.gethostname()==cfg['nodes'][a.node]['hostname']
t0=time.monotonic();last=0
while time.monotonic()-t0<a.max_wait:
 h=json.loads((w/'visual_quality/nodes'/a.node/'controller.json').read_text());live=False
 try:
  stat=Path('/proc')/str(h['pid'])/'stat';fields=stat.read_text().split(') ')[1].split();live=fields[0]!='Z' and fields[19]==h['start_ticks']
 except FileNotFoundError:pass
 productive=sum(h.get('decisions_this_process',{}).values())
 if time.monotonic()-last>30:
  print(json.dumps({'live':live,'pid':h['pid'],'workers':h['workers'],'stage':h['stage'],'active':h['active_sources'],'completed':productive}),flush=True);last=time.monotonic()
 if live and h['workers']==a.workers and h['stage']=='running' and h['active_sources']>=a.workers*.8 and productive>=a.workers and time.time()-h['heartbeat']<30:
  assert not h.get('failed_sources') and not h.get('execution_errors_this_process')
  a.output.mkdir(parents=True,exist_ok=True);(a.output/'start_controller.json').write_text(json.dumps(h,indent=2)+'\n')
  subprocess.run(['python3',str(Path(__file__).with_name('observe.py')),a.node,str(a.output),'180'],check=True);break
 time.sleep(5)
else:raise RuntimeError('Verified controller did not reach requested productive capacity within the observation window; inspect before changing processes')
