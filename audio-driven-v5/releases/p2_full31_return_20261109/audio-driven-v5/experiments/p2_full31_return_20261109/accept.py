from pathlib import Path
import os,time,signal,subprocess,statistics
import control as X
import watch_hourly as M
C=X.C;W=X.W;OUT=X.OUT;S=X.S
ready=C.read(W/'READY.json');rows=C.legacy.rows_after(OUT,ready['step']);assert len(rows)>=8
C.legacy.check_rows(rows,True);assert C.alive(C.read(OUT/'CONTROLLER.json'))
assert ready['source']==X.R.source()==C.read(OUT/'run.json')['source']
assert C.sha(OUT/'config.json')==ready['config_sha256']
canary=C.read(OUT/'LONGEST_UNEVEN_CHECK.json');assert canary['success']and canary['world']==31 and canary['global_batch']==31 and canary['resumed_step']==ready['step']
assert canary['model_optimizer_rng_restored_from_checkpoint']
report=M.sample();assert report['selected_healthy']and not report['flags']
# Stop only the old observer after verifying its process identity.
old=C.read(S/'WATCH5M_CONTROLLER.json')
if C.alive(old):
 assert str(S/'watch5m.py')in Path('/proc',str(old['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
 os.kill(old['pid'],signal.SIGTERM)
with(W/'hourly.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'watch_hourly.py')],cwd=X.ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
C.write(W/'MONITOR_HANDOFF.json',dict(at=time.time(),old=old,new_pid=p.pid,interval_seconds=3600,trainers_not_signaled=True))
result=dict(success=True,at=time.time(),output=str(OUT),step=report['status']['step'],resumed_from=ready['step'],verified_updates=len(rows),mean_resumed_seconds=statistics.mean(r['seconds']for r in rows),pipeline='LTX all attention/FFN LoRA + speaker Whisper + AV + clip Event + dual Feature; cached Qwen hidden',world=31,global_batch=31,reserved_gpu='node11:0',checkpoint_every=500,network='single mlx5_0',source_config_verified=True,migration=C.read(OUT/'MIGRATION.json'),canary=canary,placement=report,monitor_pid=p.pid,noevent_stopped_at660=True,qwen_joint_not_started=True)
C.write(W/'ACCEPTED.json',result);C.write(W/'STATE.json',dict(at=time.time(),stage='full31_training_verified',step=result['step']))
print(dict(step=result['step'],updates=len(rows),mean_seconds=result['mean_resumed_seconds'],monitor_pid=p.pid))
