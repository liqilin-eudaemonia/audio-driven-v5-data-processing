import runpy,time,threading,hashlib,json,subprocess,os
from pathlib import Path
M=runpy.run_path(str(Path(__file__).with_name('coordinator.py')))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def test_no_stop_or_launch_before6000_and_no_launch_before_receipt(tmp_path,monkeypatch):
 g=M['main'].__globals__;out=tmp_path/'out';par=tmp_path/'parent';out.mkdir();par.mkdir()
 real=Path(__file__).resolve().parent
 for n in ('config.json','cluster.json','preflight.json','RECHECK_ACCEPTED.json'):(out/n).write_bytes((real/n).read_bytes())
 sleeper=subprocess.Popen(['sleep','30'])
 ticks=Path('/proc',str(sleeper.pid),'stat').read_text().split()[21]
 write(par/'status.json',{'step':5999});write(par/'CONTROLLER.json',{'pid':sleeper.pid,'start_ticks':ticks})
 monkeypatch.setitem(g,'OUT',out);monkeypatch.setitem(g,'PARENT',par);monkeypatch.setitem(g,'validate_ready',lambda:None)
 monkeypatch.setitem(g,'selected_resources',lambda _: [{'node':'all','idle':True}])
 called=[];errors=[]
 def launched(*args):called.append(True);write(out/'status.json',{'step':1})
 monkeypatch.setitem(g,'launch',launched)
 native_sleep=time.sleep
 monkeypatch.setattr(g['time'],'sleep',lambda n:native_sleep(min(n,.02)))
 def run():
  try:g['main']()
  except Exception as e:errors.append(e)
 t=threading.Thread(target=run);t.start();native_sleep(.08)
 assert not (par/'STOP').exists() and not called
 write(par/'status.json',{'step':6000})
 deadline=time.monotonic()+3
 while not (par/'STOP').exists() and time.monotonic()<deadline:native_sleep(.02)
 assert (par/'STOP').exists() and not called
 target=par/'step_006000.pt';target.write_bytes(b'complete test checkpoint')
 write(par/'step_006000.complete.json',{'step':6000,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 native_sleep(.1);assert not called
 sleeper.terminate();sleeper.wait(timeout=5)
 t.join(20);assert not t.is_alive() and not errors and called==[True]
 assert json.loads((out/'PARENT_SAVED.json').read_text())['step']==6000
