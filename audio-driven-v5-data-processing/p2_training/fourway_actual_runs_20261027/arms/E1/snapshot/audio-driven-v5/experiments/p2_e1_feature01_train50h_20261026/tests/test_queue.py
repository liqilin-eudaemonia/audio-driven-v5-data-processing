import copy,json,sys,time,subprocess,os
from pathlib import Path
import pytest
OUT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(OUT))
import queue_common as common
import queue_watchdog as q
import queue_node as node


def record(path,value):
 if path.suffix==".jsonl":
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+"\n")
 else:common.write(path,value)

def ckpt(run,step,valid=True):
 run.mkdir(parents=True,exist_ok=True);p=run/f'step_{step:06d}.pt';p.write_bytes(b'cpu-checkpoint-fixture'+str(step).encode())
 record(p.with_suffix('.complete.json'),dict(step=step,sha256=common.sha(p)))
 if not valid:p.write_bytes(b'broken')
 return p


def nodes(active=False,finished=False,idle=True):
 return [dict(rank=r,owned=([dict(kind='torchrun',pid=r+100)] if active else []),locks_free=not active,
  idle=idle,node_state=dict(state='finished' if finished else ('running' if active else 'failed'),returncode=0 if finished else None)) for r in range(3)]


@pytest.fixture
def sandbox(tmp_path,monkeypatch):
 jobs={}
 for stage in ('E0','E1'):
  mgmt=tmp_path/stage;run=mgmt/'run';mgmt.mkdir();jobs[stage]=dict(management=mgmt,run=run,config=mgmt/'config.json')
 monkeypatch.setattr(q,'JOBS',jobs);monkeypatch.setattr(q,'QUEUE',tmp_path/'queue')
 monkeypatch.setattr(q,'verify_pins',lambda stage:None)
 monkeypatch.setattr(q,'stopped',lambda stage: any(p.exists() for p in (q.QUEUE/'STOP',jobs[stage]['management']/'STOP',jobs[stage]['run']/'STOP')))
 return jobs


def test_completion_requires_budget_all_nodes_and_hash(sandbox):
 run=sandbox['E0']['run'];ckpt(run,q.BUDGET)
 record(run/'status.json',dict(step=q.BUDGET,budget=q.BUDGET,stopped=False))
 assert q.finished('E0',nodes(finished=True))['step']==q.BUDGET
 assert q.finished('E0',nodes(finished=True)[:2]) is None
 assert q.finished('E0',nodes(active=True,finished=True)) is None
 record(run/'status.json',dict(step=100,budget=q.BUDGET,stopped=True))
 assert q.finished('E0',nodes(finished=True)) is None
 record(run/'status.json',dict(step=q.BUDGET,budget=q.BUDGET,stopped=False));ckpt(run,q.BUDGET,valid=False)
 with pytest.raises(ValueError):q.finished('E0',nodes(finished=True))


def test_corrupt_latest_checkpoint_falls_back_and_preserves_artifacts(sandbox):
 job=sandbox['E1'];p=ckpt(job['run'],100);bad=ckpt(job['run'],200,valid=False)
 record(job['run']/'run.json',{'arm':'E1'});record(job['run']/'status.json',{'step':200})
 record(job['management']/'node0.json',{'state':'failed'})
 plan=q.preparation('E1','test_attempt')
 assert plan['resume']==str(p) and len(plan['rejected_checkpoints'])==1
 assert (q.QUEUE/'attempts/test_attempt'/bad.name).exists()
 assert (q.QUEUE/'attempts/test_attempt/rolled_back_run_status.json').exists()
 assert not (job['management']/'node0.json').exists()


def test_no_checkpoint_preserves_failed_run_before_fresh_restart(sandbox):
 run=sandbox['E1']['run'];run.mkdir();record(run/'run.json',{'arm':'E1'});(run/'metrics.jsonl').write_text('uncommitted')
 plan=q.preparation('E1','no_checkpoint')
 assert plan['resume'] is None and not run.exists()
 assert len(list((q.QUEUE/'attempts/no_checkpoint').glob('uncommitted_run_*/metrics.jsonl')))==1


def test_cross_arm_recovery_rejected(sandbox):
 run=sandbox['E1']['run'];record(run/'run.json',{'arm':'E0'})
 with pytest.raises(ValueError,match='cross-arm'):q.preparation('E1','wrong')
 assert (run/'run.json').exists()


def test_healthy_e0_is_only_observed(sandbox,monkeypatch):
 record(sandbox['E0']['management']/'status.json',{'state':'training'})
 record(sandbox['E0']['run']/'metrics.jsonl',{'step':300,'time':time.time()})
 calls=[]
 def cluster(action,stage):calls.append((action,stage));return nodes(active=True,idle=False)
 monkeypatch.setattr(q,'cluster',cluster)
 queue=q.DurableQueue(0);queue.tick()
 assert calls==[('status','E0')] and queue.state['phase']=='waiting_predecessor'
 assert queue.state['progress']['step']==300 and queue.state.get('recovery') is None


def test_final_drain_does_not_restart_healthy_completed_training(sandbox,monkeypatch):
 job=sandbox['E0'];record(job['run']/'status.json',dict(step=q.BUDGET,budget=q.BUDGET,stopped=False))
 monkeypatch.setattr(q,'cluster',lambda *a:nodes(active=True,finished=True))
 queue=q.DurableQueue(0);queue.state['stage_observed_epoch']=time.time()-10000;queue.tick()
 assert queue.state['phase']=='finishing' and not queue.state.get('recovery')


def test_predecessor_completion_queues_fresh_e1_not_e0_resume(sandbox,monkeypatch):
 job=sandbox['E0'];ckpt(job['run'],q.BUDGET);record(job['run']/'status.json',dict(step=q.BUDGET,budget=q.BUDGET,stopped=False))
 monkeypatch.setattr(q,'cluster',lambda *a:nodes(finished=True))
 queue=q.DurableQueue(0);queue.tick()
 assert queue.state['stage']=='E1' and queue.state['recovery']['phase']=='stopping'
 launches=[]
 monkeypatch.setattr(q,'remote',lambda *a,**kw:launches.append((a,kw)) or {'launched':True})
 queue.tick()
 assert len(launches)==1 and launches[0][0]==(0,'launch','E1')
 assert launches[0][1]['resume'] is None


def test_recovery_adopts_lost_launch_ack_without_duplicate(sandbox,monkeypatch):
 queue=q.DurableQueue(0);queue.state['stage']='E1';queue.recover('simulated launch failure')
 monkeypatch.setattr(q,'cluster',lambda *a:nodes())
 count=[]
 def launch(*a,**kw):count.append(1);raise TimeoutError('SSH acknowledgment lost after remote launch')
 monkeypatch.setattr(q,'remote',launch)
 with pytest.raises(TimeoutError):queue.tick()
 assert queue.state['recovery']['phase']=='launching'
 # Simulate another leader after the former process died.
 takeover=q.DurableQueue(2)
 monkeypatch.setattr(q,'cluster',lambda *a:nodes(active=True,idle=False))
 takeover.tick()
 assert takeover.state['phase']=='startup' and takeover.state['recovery'] is None and len(count)==1


def test_stop_never_cleared_or_launched(sandbox,monkeypatch):
 q.QUEUE.mkdir();(q.QUEUE/'STOP').touch()
 monkeypatch.setattr(q,'cluster',lambda *a:pytest.fail('Must not touch training while paused'))
 queue=q.DurableQueue(0);queue.tick();assert queue.state['phase']=='paused' and (q.QUEUE/'STOP').exists()


def test_partial_launch_and_stall_trigger_recovery(sandbox,monkeypatch):
 queue=q.DurableQueue(0);queue.state['stage_observed_epoch']=time.time()-130
 partial=nodes(active=True);partial[1]['owned']=[];partial[2]['owned']=[]
 monkeypatch.setattr(q,'cluster',lambda *a:partial)
 queue.tick();assert 'Partial' in queue.state['recovery']['reason']
 queue.state['recovery']=None;queue.state['stage_observed_epoch']=time.time()-2000
 record(sandbox['E0']['run']/'metrics.jsonl',{'step':12,'time':time.time()-1000})
 monkeypatch.setattr(q,'cluster',lambda *a:nodes(active=True))
 queue.tick();assert 'progress' in queue.state['recovery']['reason']


def test_corrupt_final_goes_to_recovery_not_idle_error_loop(sandbox,monkeypatch):
 job=sandbox['E0'];ckpt(job['run'],q.BUDGET,valid=False);record(job['run']/'status.json',dict(step=q.BUDGET,budget=q.BUDGET,stopped=False))
 monkeypatch.setattr(q,'cluster',lambda *a:nodes(finished=True))
 queue=q.DurableQueue(0);queue.tick();assert 'checkpoint verification' in queue.state['recovery']['reason']


def test_ownership_matches_config_arm_and_birth(sandbox,monkeypatch):
 monkeypatch.setattr(node,'JOBS',sandbox)
 argv=[str(common.PYTHON),str(common.ROOT/'releases/p2_feature_v1/scripts/p2_listen.py'),'train','--config',str(sandbox['E1']['config']),'--arm','E1']
 assert node.classify(argv,'E1',0)=='worker'
 assert node.classify(argv,'E0',0) is None
 assert node.classify([*argv[:-1],'E0'],'E1',0) is None
 monkeypatch.setattr(node,'tokens',lambda pid:(argv,'new_birth'))
 monkeypatch.setattr(node.os,'kill',lambda *a:pytest.fail('PID reused; no signal allowed'))
 monkeypatch.setattr(node.os,'killpg',lambda *a:pytest.fail('PID reused; no signal allowed'))
 node.signal_owned(dict(pid=123,birth='old_birth',kind='worker',pgid=123),'E1',0,15)


def test_real_cpu_orphan_discovery_and_owned_stop_preserves_other_job(sandbox,monkeypatch):
 monkeypatch.setattr(node,'JOBS',sandbox)
 marker=str(common.ROOT/'releases/p2_feature_v1/scripts/p2_listen.py')
 args=[sys.executable,'-c','import time;time.sleep(60)',marker,'--config',str(sandbox['E1']['config']),'--arm','E1']
 owner=subprocess.Popen(args,start_new_session=True);other=subprocess.Popen([*args[:-3],str(sandbox['E0']['config']),'--arm','E1'],start_new_session=True)
 try:
  time.sleep(.1);found=node.owned('E1',0);assert owner.pid in [x['pid'] for x in found] and other.pid not in [x['pid'] for x in found]
  monkeypatch.setattr(node,'snapshot',lambda stage,rank:dict(owned=node.owned(stage,rank),locks_free=True))
  node.stop('E1',0);owner.wait(timeout=2);assert other.poll() is None
 finally:
  for proc in (owner,other):
   if proc.poll() is None:proc.terminate()
   proc.wait(timeout=5)


def test_late_launch_attempt_is_fenced_before_subprocess(sandbox,monkeypatch):
 monkeypatch.setattr(node,'JOBS',sandbox);monkeypatch.setattr(node,'QUEUE',q.QUEUE)
 monkeypatch.setattr(node,'stopped',lambda stage:False)
 monkeypatch.setattr(node,'owned',lambda *a:[])
 monkeypatch.setattr(node.subprocess,'run',lambda *a,**kw:pytest.fail('A stale attempt must not launch any process'))
 record(q.QUEUE/'state.json',dict(stage='E1',recovery=dict(attempt='current',phase='launching')))
 with pytest.raises(RuntimeError,match='Stale'):node.launch('E1',0,None,'old')
 record(q.QUEUE/'state.json',dict(stage='E1',recovery=None))
 with pytest.raises(RuntimeError,match='Stale'):node.launch('E1',0,None,'current')
