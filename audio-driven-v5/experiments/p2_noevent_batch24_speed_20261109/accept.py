from pathlib import Path
import sys,time,statistics,os,signal,subprocess
W=Path(__file__).resolve().parent;ROOT=W.parents[1];S=W.parent/'p2_ltx_split24_8_batch31_20261109';sys.path[:0]=[str(W),str(S),str(ROOT/'src'),str(ROOT/'scripts')]
import control as C,batch_runtime as R
import watch_hourly as monitor
OUT=monitor.OUT
b=C.read(W/'BOUNDARY.json');status=C.read(OUT/'status.json');rows=C.legacy.rows_after(OUT,b['step']);assert len(rows)>=8
C.legacy.check_rows(rows,False);assert all(r['effective_batch']==24 and r['consumed_pairs']==b['step']*31+(r['step']-b['step'])*24 for r in rows)
assert C.alive(C.read(OUT/'CONTROLLER.json'))
assert R.source()==C.read(OUT/'run.json')['source']==C.read(W/'READY.json')['source']
canary=C.read(OUT/'LONGEST_UNEVEN_CHECK.json');assert canary['success']and canary['global_batch']==24 and canary['resumed_step']==b['step']
assert C.read(OUT/'MIGRATION.json')['model_adam_all24_rng_equal']
report=monitor.sample();assert all(not v['flags']for v in report['arms'].values())
old=C.read(S/'WATCH5M_CONTROLLER.json');assert C.alive(old)and old['interval_seconds']==3600
assert str(S/'watch5m.py')in Path('/proc',str(old['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
os.kill(old['pid'],signal.SIGTERM)
with(W/'hourly.log').open('a')as f:p=subprocess.Popen([C.PY,'-u',str(W/'watch_hourly.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
C.write(W/'MONITOR_HANDOFF.json',dict(at=time.time(),old=old,new_pid=p.pid,interval_seconds=3600,trainers_not_signaled=True))
result=dict(success=True,at=time.time(),output=str(OUT),step=status['step'],original_boundary=b['step'],verified_updates=len(rows),mean_resumed_seconds=statistics.mean(r['seconds']for r in rows),global_batch=24,world=24,plan=C.read(W/'PLAN.json'),canary=canary,placement=report,benchmark=C.read(W/'matched744/RESULTS.json'),checkpoint_every=500,source_and_migration_verified=True,node11_unchanged=True)
C.write(W/'ADOPTION_ACCEPTED.json',result);C.write(W/'STATE.json',dict(stage='batch24_adopted_verified',step=status['step'],at=time.time()))
print(dict(step=status['step'],updates=len(rows),mean_seconds=result['mean_resumed_seconds'],monitor_pid=p.pid))
