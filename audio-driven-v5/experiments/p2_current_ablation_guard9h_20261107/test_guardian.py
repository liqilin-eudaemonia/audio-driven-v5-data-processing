from pathlib import Path
import runpy,json,time,os
M=runpy.run_path(str(Path(__file__).with_name('guardian.py')))
def test_current_process_identity():
 s=Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()
 assert M['alive']({'pid':os.getpid(),'start_ticks':s[19]})
 assert not M['alive']({'pid':os.getpid(),'start_ticks':'bad'})
def test_save_grace_and_real_stall(tmp_path):
 g=M['health'].__globals__;prior=g['F'];g['F']=tmp_path
 try:
  row=dict(step=500,av=1.,av_video=.9,av_audio=.1,weighted_total=1.05,feature_video=1.,feature_audio=1.,seconds=4.,grad_norm={'connector':1.,'text_lora':1.,'features':1.})
  f=tmp_path/'metrics.jsonl';f.write_text(json.dumps(row)+'\n');now=time.time();os.utime(f,(now-300,now-300))
  assert not M['health'](tmp_path,now,now)['flags']
  (tmp_path/'step_000500.complete.json').write_text('{}')
  assert 'stalled_or_startup_timeout'in M['health'](tmp_path,now,now)['flags']
 finally:g['F']=prior
def test_remote_probe_uses_nul_separator():
 assert "split(b'\\0')"in M['PROBE']
 compile(M['PROBE'].replace('PAYLOAD',repr('{}')),'probe','exec')

def test_failed6000_save_is_not_misclassified_as_expected_exit(tmp_path,monkeypatch):
 g=M['transition_flags'].__globals__;monkeypatch.setitem(g,'F',tmp_path)
 flags=M['transition_flags'](tmp_path,6000,['stalled_or_startup_timeout'],False)
 assert set(flags)=={'stalled_or_startup_timeout','training_controller_dead'}
 assert M['repair_action']({'flags':flags,'work':str(tmp_path)},1000)==['recover',str(tmp_path),'confirmed_dead']
 (tmp_path/'step_006000.complete.json').write_text('{}')
 assert M['transition_flags'](tmp_path,6000,['stalled_or_startup_timeout'],False)==[]

def test_speed_adds_missing_duration_bins_without_absorbing_later_slowdown(tmp_path,monkeypatch):
 g=M['speed'].__globals__;key=str(tmp_path)
 pairs=[(step%2,0)for step in range(80)for _ in range(31)]
 monkeypatch.setitem(g,'WORK',tmp_path)
 monkeypatch.setitem(g,'DURATIONS',{key:([2.,10.],pairs)})
 monkeypatch.setitem(g,'BASELINES',{key:{1:2.}})
 rows=[dict(step=step,seconds=4. if(step-1)%2==0 else 8.)for step in range(40,80)]
 assert M['speed'](tmp_path,rows)==1.5
 assert g['BASELINES'][key]=={1:2.,5:8.}
 assert json.loads((tmp_path/(tmp_path.name+'_SPEED_BASELINE.json')).read_text())['bins']=={'1':2.,'5':8.}
 slower=[dict(r,seconds=r['seconds']*2)for r in rows]
 assert M['speed'](tmp_path,slower)==3.
 assert g['BASELINES'][key]=={1:2.,5:8.}
