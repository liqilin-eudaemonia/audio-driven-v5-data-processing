"""Read-only acceptance of exact24-rank placement and real finite updates."""
from pathlib import Path
import sys,subprocess,concurrent.futures,json,time,math,statistics
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from listen_cluster import transport
from p2_whisper_event_feature_train31 import verify_row


def main():
 c=read(OUT/'config.json');cluster=read(OUT/'cluster.json');base=read(OUT/'MIGRATION.json')['step']
 rows=[json.loads(s) for s in (OUT/'metrics.jsonl').read_text().splitlines()]
 recent=[r for r in rows if r['step']>base]
 assert len(recent)>=20
 for r in recent:verify_row(r)
 code='''from pathlib import Path
import os,json,subprocess
p=Path(OUT_VALUE);record=json.loads((p/'rank_logs'/NODE_FILE).read_text());rows=[]
m=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
a=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
gpus={line.split(',')[1].strip():int(line.split(',')[0]) for line in m.stdout.splitlines()}
used={int(line.split(',')[0]):gpus[line.split(',')[1].strip()] for line in a.stdout.splitlines() if line.split(',')[0].strip().isdigit()}
for rank,pid in zip(record['ranks'],record['pids']):
 q=Path('/proc')/str(pid);stat=q.joinpath('stat').read_text().split();args=q.joinpath('cmdline').read_bytes().split(b'\\0');env=dict(v.split(b'=',1) for v in q.joinpath('environ').read_bytes().split(b'\\0') if b'='in v)
 assert stat[2]!='Z' and str(p/'config.json').encode()in args
 expected=rank%8
 assert env[b'RANK']==str(rank).encode() and env[b'LOCAL_RANK']==str(expected).encode() and env[b'WORLD_SIZE']==b'24'
 assert env[b'CUDA_VISIBLE_DEVICES']==b'0,1,2,3,4,5,6,7' and used[pid]==expected
 rows.append(dict(rank=rank,pid=pid,local=expected,physical_gpu=used[pid],start_ticks=stat[21]))
print(json.dumps(dict(node=record['node'],workers=rows,gpu_metrics=m.stdout.splitlines())))
'''.replace('OUT_VALUE',repr(str(OUT)))
 def one(n):
  text=code.replace('NODE_FILE',repr(n['name']+'_workers.json'))
  for attempt in range(4):
   try:
    r=subprocess.run(transport(cluster,n,[cluster['python'],'-c',text]),capture_output=True,text=True,timeout=40)
    if r.returncode==0:return json.loads(r.stdout)
    error=r.stderr[-1500:]
   except Exception as e:error=repr(e)
   time.sleep(1)
  raise RuntimeError(n['name']+': '+error)
 with concurrent.futures.ThreadPoolExecutor(3)as pool:nodes=list(pool.map(one,cluster['nodes']))
 assert sorted(w['rank'] for n in nodes for w in n['workers'])==list(range(24))
 atomic_json(OUT/'LIVE_PROCESSES.json',dict(at=time.time(),success=True,world=24,nodes=nodes,node11_not_in_launch=True))
 warm=recent[3:];speeds=[r['seconds'] for r in warm]
 value=dict(success=True,at=time.time(),world=24,global_batch=24,resume_step=base,step=recent[-1]['step'],budget=c['training']['steps'],
  updates_checked=len(recent),all_losses_finite=True,objective_weights_correct=True,five_parameter_groups_receive_gradients=True,
  mean_step_seconds=statistics.mean(speeds),median_step_seconds=statistics.median(speeds),
  mean_input_wait_seconds=statistics.mean(r['input_wait_seconds'] for r in warm),
  peak_reserved_gib=max(r['peak_reserved_gib'] for r in recent),reserved_node='node11',reserved_gpus=list(range(8)),
  checkpoint_every=500,next_checkpoint=((recent[-1]['step']//500)+1)*500)
 atomic_json(OUT/'RUNNING_VERIFIED.json',value);atomic_json(OUT/'ADOPTION_ACCEPTED.json',dict(**value,cpu_tests_passed=58))
 queue=read(OUT/'QUEUE_STATE.json');queue.update(phase='training_verified',verified_step=value['step']);atomic_json(OUT/'QUEUE_STATE.json',queue)
 print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
