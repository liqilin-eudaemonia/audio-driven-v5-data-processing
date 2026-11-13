from pathlib import Path
import os,time,signal,subprocess,statistics
import manager as X
import importlib.util
_spec=importlib.util.spec_from_file_location('watch_noevent28',Path(__file__).with_name('watch_hourly.py'));M=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(M)
C=X.C;W=X.W;OUT=X.OUT;S=X.S
ready=C.read(W/'READY.json');rows=C.legacy.rows_after(OUT,ready['step']);assert len(rows)>=8
C.legacy.check_rows(rows,False);assert C.alive(C.read(OUT/'CONTROLLER.json'))
assert ready['source']==X.R.source()==C.read(OUT/'run.json')['source']
assert C.sha(OUT/'config.json')==ready['config_sha256']
canary=C.read(OUT/'LONGEST_UNEVEN_CHECK.json');assert canary['success']and canary['world']==28 and canary['global_batch']==28 and canary['resumed_step']==ready['step']
assert canary['model_optimizer_rng_restored_from_checkpoint']
report=M.sample();assert report['selected_healthy']and not report['flags']
plan=C.read(W/'PLAN.json')
for r in rows:
 assert r['effective_batch']==28
 assert r['consumed_pairs']==plan['prefix_pairs']+(r['step']-plan['start_step'])*28
with(W/'hourly.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'watch_hourly.py')],cwd=X.ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
C.write(W/'MONITOR_HANDOFF.json',dict(at=time.time(),new_pid=p.pid,interval_seconds=3600,trainers_not_signaled=True))
result=dict(success=True,at=time.time(),output=str(OUT),step=report['status']['step'],resumed_from=ready['step'],verified_updates=len(rows),mean_resumed_seconds=statistics.mean(r['seconds']for r in rows),pipeline='LTX all attention/FFN LoRA + speaker Whisper + AV + dual Feature; no Event; cached Qwen hidden',world=28,global_batch=28,reserved_gpu='node11:4-7',checkpoint_every=500,network='single mlx5_0',source_config_verified=True,migration=C.read(OUT/'MIGRATION.json'),canary=canary,placement=report,monitor_pid=p.pid,resumed_noevent_from660=True,qwen_joint_not_started=True)
C.write(W/'ACCEPTED.json',result);C.write(W/'STATE.json',dict(at=time.time(),stage='full28_training_verified',step=result['step']))
print(dict(step=result['step'],updates=len(rows),mean_seconds=result['mean_resumed_seconds'],monitor_pid=p.pid))
