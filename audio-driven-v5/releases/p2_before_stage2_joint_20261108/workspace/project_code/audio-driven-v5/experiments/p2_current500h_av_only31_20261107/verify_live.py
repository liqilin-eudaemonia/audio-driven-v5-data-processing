from pathlib import Path
import sys,json,subprocess,concurrent.futures,time,statistics,math
R=Path('/private/liqi_workspace/project_code/audio-driven-v5');P=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_cluster import transport
from p2_listen.contracts import read,atomic_json
c=read(P/'cluster.json')
code='''from pathlib import Path
import json,subprocess
p=Path(OUT);v=json.loads((p/'rank_logs'/FILE).read_text());out=[]
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu','--format=csv,noheader,nounits'],text=True)
u={x.split(',')[1].strip():int(x.split(',')[0]) for x in g.splitlines()}
a=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
a={int(x.split(',')[0]):u[x.split(',')[1].strip()] for x in a.splitlines()if x.split(',')[0].strip().isdigit()}
for rank,pid in zip(v['ranks'],v['pids']):
 q=Path('/proc',str(pid));s=(q/'stat').read_text().split();args=(q/'cmdline').read_bytes().split(b'\\0');env=dict(x.split(b'=',1)for x in (q/'environ').read_bytes().split(b'\\0')if b'='in x)
 assert s[2]!='Z'and str(p/'config.json').encode()in args
 assert int(env[b'RANK'])==rank and int(env[b'WORLD_SIZE'])==31 and a[pid]==int(env[b'LOCAL_RANK'])
 if rank>=24:assert env[b'CUDA_VISIBLE_DEVICES']==b'0,1,2,3,4,5,6'and a[pid]<7
 out.append(dict(rank=rank,pid=pid,gpu=a[pid],start_ticks=s[21]))
print(json.dumps(dict(node=v['node'],workers=out,gpu_utilization=g.splitlines())))
'''.replace('OUT',repr(str(P)))
def one(n):
 for attempt in range(4):
  try:
   v=subprocess.run(transport(c,n,[c['python'],'-c',code.replace('FILE',repr(n['name']+'_workers.json'))]),capture_output=True,text=True,timeout=30)
   if v.returncode==0:return json.loads(v.stdout)
   error=v.stderr[-2000:]
  except Exception as e:error=repr(e)
  time.sleep(1)
 raise RuntimeError(n['name']+error)
with concurrent.futures.ThreadPoolExecutor(4)as pool:nodes=list(pool.map(one,c['nodes']))
assert sorted(w['rank']for n in nodes for w in n['workers'])==list(range(31))
rows=[json.loads(x)for x in (P/'metrics.jsonl').read_text().splitlines()];assert rows[-1]['step']>=20
for row in rows:
 for key in ('av','av_video','av_audio','weighted_total'):assert math.isfinite(row[key])
 assert abs(row['weighted_total']-row['av'])<1e-9
 if row['step']>=3:
  for key in ('connector','text_lora'):assert row['grad_norm'][key]>0 and math.isfinite(row['grad_norm'][key])
record=dict(success=True,at=time.time(),world=31,step=rows[-1]['step'],nodes=nodes,all_losses_finite=True,
 objective_is_pure_av=True,two_parameter_groups_receive_gradients=True,reserved_gpu='node11:7',
 mean_warm_seconds=statistics.mean(x['seconds']for x in rows[3:][-30:]),peak_reserved_gib=max(x['peak_reserved_gib']for x in rows))
atomic_json(P/'LIVE_ACCEPTANCE.json',record);print(json.dumps({k:v for k,v in record.items()if k!='nodes'},indent=2))
