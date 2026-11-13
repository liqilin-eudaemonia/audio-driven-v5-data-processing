from pathlib import Path
import sys,torch
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path[:0]=[str(W),str(ROOT/'src'),str(ROOT/'scripts')]
import runtime28 as R
from p2_listen.contracts import read

def test_remaining_samples_and_noise_exact():
 c=read(W.parent/'p2_ltx_all_lora_noevent28_batch28_20261109/config.json');s=R.Schedule(c);p=s.plan;flat=[]
 for step in range(661,10082):
  groups=[s.indices(s.size,step,28,s.seed,r,28)for r in range(28)]
  assert all(len(g)<=1 for g in groups)
  pairs=[x for g in groups for x in g];assert len(pairs)==s.count(step);flat.extend(pairs)
 assert flat==s.pairs[20460:]and len(flat)==263779 and s.count(10081)==19
 assert s.bounds(7000)[0]==20460+(7000-660-1)*28

def test_noevent_objective_and_noevent_readers():
 m=R.load();c=read(W.parent/'p2_ltx_all_lora_noevent28_batch28_20261109/config.json')
 from p2_ltx_events.heads import objective,names
 assert names(c)==()and m._event_counts([],[],'cpu',1)=={}
 for n in [28,19]:
  p=torch.tensor(2.,requires_grad=True);v=dict(av=p,features=dict(video=p*3,audio=p*5),event_sums={})
  loss,parts,event=objective(v,c,{},n,1);torch.testing.assert_close(loss,1.2*p/n);assert event==0
  loss.backward();torch.testing.assert_close(p.grad,torch.tensor(1.2/n))

def test_handoff_waits_for_target_before_requesting_stop(tmp_path,monkeypatch):
 import importlib.util
 spec=importlib.util.spec_from_file_location('review_handoff',W/'handoff.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 monkeypatch.setattr(h,'W',tmp_path)
 calls=[];steps=iter([7499,7499])
 def read(p):
  if p.name=='READY.json':return {'source':h.X.R.source()}
  if p.name=='status.json':return {'step':next(steps)}
  return {}
 monkeypatch.setattr(h.C,'read',read);monkeypatch.setattr(h.C,'sha',lambda p:'sha');monkeypatch.setattr(h.C,'alive',lambda p:True);monkeypatch.setattr(h.C,'write',lambda *a,**kw:None);monkeypatch.setattr(h.C,'request_stop',lambda *a,**kw:calls.append((a,kw)))
 class Done(Exception):pass
 monkeypatch.setattr(h.time,'sleep',lambda t:(_ for _ in()).throw(Done()))
 import pytest
 with pytest.raises(Done):h.main()
 assert not calls

def test_handoff_requests_exact7500_stop_then_waits_save(tmp_path,monkeypatch):
 import importlib.util,pytest
 spec=importlib.util.spec_from_file_location('review_handoff_target',W/'handoff.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 monkeypatch.setattr(h,'W',tmp_path);calls=[];steps=iter([7499,7500])
 def read(p):
  if p.name=='READY.json':return {'source':h.X.R.source()}
  if p.name=='status.json':return {'step':next(steps)}
  return {}
 monkeypatch.setattr(h.C,'read',read);monkeypatch.setattr(h.C,'sha',lambda p:'sha');monkeypatch.setattr(h.C,'alive',lambda p:True);monkeypatch.setattr(h.C,'write',lambda *a,**kw:None);monkeypatch.setattr(h.C,'request_stop',lambda *a,**kw:calls.append((a,kw)))
 class Done(Exception):pass
 monkeypatch.setattr(h.time,'sleep',lambda t:(_ for _ in()).throw(Done()))
 with pytest.raises(Done):h.main()
 assert len(calls)==1 and calls[0][1]['target_step']==7500
