from pathlib import Path
import json,runpy,subprocess,sys,time,os
import pytest
P=Path(__file__).resolve().parent
G=runpy.run_path(str(P/'guardian.py'));R=runpy.run_path(str(P/'recovery.py'))

def state(flags=(),ratio=None,elapsed=0,steps=0):return dict(flags=list(flags),work='/example',duration_adjusted_speed_ratio=ratio,slowdown_persist_seconds=elapsed,slowdown_new_steps=steps)

def test_policy_does_not_restart_for_one_slow_sample_or_query_failure():
 choose=G['repair_action']
 assert choose(state(['duration_adjusted_slowdown'],1.5,600,200),1000)is None
 assert choose(state(['duration_adjusted_slowdown'],1.9,10,3),1000)is None
 assert choose(state(['query_timeout']),1000)is None
 assert choose(state(['stalled_or_startup_timeout']),1000,1200)is None
 assert choose(state(['stalled_or_startup_timeout']),1300,1200)==['recover','/example','stalled']
 assert choose(state(['handoff_controller_dead']),1000)==['queue']
 assert choose(state([],1.9,180,20),1000)==['recover','/example','sustained_slowdown']

def test_unknown_or_busy_cards_never_pass_relaunch_gate():
 cluster={'nodes':[{'name':'node11','gpus':[0,1,2,3,4,5,6]}]}
 good={'nodes':[{'node':'node11','cards':[dict(gpu=i,used_mib=0,util=0)for i in range(7)],'workers':[],'foreign':[]}]}
 assert R['idle'](good,cluster)
 for bad in [{'nodes':[]},{'nodes':[{'node':'node11','error':'SSH timeout'}]}]:assert not R['idle'](bad,cluster)
 good['nodes'][0]['foreign']=[{'pid':123}];assert not R['idle'](good,cluster)
 good['nodes'][0]['foreign']=[];good['nodes'][0]['cards'].append(dict(gpu=7,used_mib=0,util=0));assert not R['idle'](good,cluster)

def test_user_stop_is_preserved(tmp_path,monkeypatch):
 g=R['explicit_stop'].__globals__;monkeypatch.setitem(g,'F',tmp_path)
 (tmp_path/'STOP').write_text(json.dumps({'owner':'user-manual'}));assert R['explicit_stop'](tmp_path)
 (tmp_path/'STOP').write_text(json.dumps({'owner':'user-av-feature6000-to-av-whisper31'}));assert not R['explicit_stop'](tmp_path)
 (tmp_path/'step_006000.complete.json').write_text('{}');assert R['explicit_stop'](tmp_path)

def test_deadline_blocks_new_repair_but_does_not_abandon_existing_cleanup(tmp_path,monkeypatch):
 g=R['cancelled'].__globals__;monkeypatch.setitem(g,'WORK',tmp_path)
 (tmp_path/'protocol.json').write_text(json.dumps({'deadline':time.time()-1}))
 assert R['cancelled']()
 assert not R['cancelled'](finishing=True)
 (tmp_path/'STOP').write_text('{}')
 assert R['cancelled']()
 assert not R['cancelled'](finishing=True)

def test_signal_rechecks_pid_start_time_and_only_signals_owned_rank(tmp_path):
 script=tmp_path/'worker.py';script.write_text('import time\ntime.sleep(30)\n');config=tmp_path/'config.json';config.write_text('{}')
 env=dict(os.environ,RANK='0',WORLD_SIZE='31',CUDA_VISIBLE_DEVICES='0')
 owned=subprocess.Popen([sys.executable,str(script),'train','--config',str(config)],env=env)
 foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 try:
  time.sleep(.05);ticks=Path('/proc',str(owned.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
  payload=dict(workers=[dict(pid=owned.pid,rank=0,start_ticks='not-the-original-start')],script=str(script),config=str(config),mask='0')
  def signal():return subprocess.run([sys.executable,'-c',R['KILL'].replace('PAYLOAD',repr(json.dumps(payload)))],text=True,capture_output=True)
  assert signal().returncode!=0 and owned.poll()is None and foreign.poll()is None
  payload['workers'][0]['start_ticks']=ticks
  r=signal();assert r.returncode==0,r.stderr
  owned.wait(5);assert owned.returncode!=0 and foreign.poll()is None
 finally:
  for p in (owned,foreign):
   if p.poll()is None:p.terminate()
   p.wait(5)
