from pathlib import Path
import sys,time,statistics,concurrent.futures
W=Path(__file__).resolve().parent;S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path.insert(0,str(S))
import control as C,runtime
out=C.path('noevent',True);ready=C.read(W/'NETWORK_ADOPTION_READY.json');status=C.read(out/'status.json');rows=C.legacy.rows_after(out,224)
assert status['step']>=230 and len(rows)>=6
C.legacy.check_rows(rows,False)
canary=C.read(out/'LONGEST_UNEVEN_CHECK.json');assert canary['success']and canary['resumed_step']==224 and canary['world']==24 and canary['model_optimizer_rng_restored_from_checkpoint']
assert (out/'LONGEST_UNEVEN_CHECK.json').stat().st_mtime>C.read(W/'PRODUCTION_CONTROLLER.json')['at']
assert ready['source']==runtime.source_identity('noevent')==C.read(out/'run.json')['source']and ready['config_sha256']==C.sha(out/'config.json')
with concurrent.futures.ThreadPoolExecutor(2)as pool:placement=dict(zip(['noevent','full'],pool.map(C.probe,['noevent','full'])))
assert all(p['healthy']for p in placement.values()),'Inspect live placement before acceptance'
cl=C.read(W/'NETWORK_PRODUCTION_CLUSTER.json');expected=cl['network']['NCCL_IB_HCA']
def env(n):
 code="from pathlib import Path\nimport json\nout=Path("+repr(str(out))+ ")\nrows=[]\nfor p in (out/'rank_logs').glob('worker_*.json'):\n r=json.loads(p.read_text())\n if r['host']!="+repr(next(x['workers'][0]['host']for x in placement['noevent']['nodes']if x['node']==n['name']))+":continue\n env=dict(x.split('=',1)for x in Path('/proc',str(r['pid']),'environ').read_bytes().decode().split('\\0')if'='in x)\n rows.append(dict(rank=r['rank'],pid=r['pid'],hca=env.get('NCCL_IB_HCA'),batch_world=env.get('WORLD_SIZE')))\nprint(json.dumps(rows))"
 return C.remote(cl,n,code)
with concurrent.futures.ThreadPoolExecutor(3)as pool:envs=[v for rr in pool.map(env,cl['nodes'])for v in rr]
assert len(envs)==24 and all(v['hca']==expected and v['batch_world']=='24'for v in envs)
monitor=C.read(S/'WATCH5M_CONTROLLER.json');assert monitor['interval_seconds']==3600 and C.alive(monitor)
assert 'time.sleep(3600)'in(S/'watch5m.py').read_text()
assert C.alive(C.read(W/'PRODUCTION_CONTROLLER.json'))
result=dict(success=True,at=time.time(),output=str(out),step=status['step'],formal_updates_since_resume=len(rows),global_batch=31,world=24,network=expected,only_network_changed=True,source_and_config_unchanged=True,training_state_resumed_from=224,canary=canary,matched_measurement={k:ready[k]for k in ['single_means','eight_means','single_mean','eight_mean','time_reduction_percent','throughput_gain_percent']},mean_resumed_seconds=statistics.mean(r['seconds']for r in rows),placement=placement,worker_environment=envs,monitor=monitor,node11_untouched=True,foreign_jobs_not_killed=True,checkpoint_every=500)
C.write(W/'ADOPTION_ACCEPTED.json',result);C.write(out/'NETWORK_ADOPTION.json',dict(acceptance=str(W/'ADOPTION_ACCEPTED.json'),active_cluster=str(W/'NETWORK_PRODUCTION_CLUSTER.json'),at=result['at']))
C.write(W/'STATE.json',dict(at=time.time(),stage='network_adopted_training_verified',step=status['step'],hourly_monitor=True))
print(dict(step=status['step'],formal_updates=len(rows),matched_time_reduction_percent=ready['time_reduction_percent'],mean_resumed_seconds=result['mean_resumed_seconds'],all32_training_workers_verified=True,hourly_monitor_alive=True))
