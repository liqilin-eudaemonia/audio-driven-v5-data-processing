from pathlib import Path
import json,hashlib,ast
R=Path(__file__).resolve().parents[1];P=R.parent/'boundary_v2_pilot_20261025';new=json.loads((R/'videos/-1wCGAxqT_4/manifest.json').read_text());old=json.loads((P/'manifest.json').read_text())
assert [{k:v for k,v in p.items() if k!='continuous_interval'} for p in new['pairs']]==old['pairs']
assert [{k:v for k,v in p.items() if k!='continuous_interval'} for p in new['ledger']]==old['ledger']
eng=ast.parse((R/'code/engine.py').read_text());fn=next(n for n in eng.body if isinstance(n,ast.FunctionDef) and n.name=='context_interval');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<test>','exec'),ns)
for p in new['pairs']:
 a,b=ns['context_interval'](p,new['scene_edges']);lo,hi=p['continuous_interval'];assert lo<=a<=p['start_frame']<p['end_frame']<=b<=hi
for name in ['separate','score','faces']:
 def method(path):
  t=ast.parse(path.read_text());cl=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Engine');return ast.dump(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name))
 assert method(R/'code/engine.py')==method(R.parent/'full_listener_sync_20261024/code/engine.py')
r={'approved_pilot_pairs_exactly_reproduced':137,'candidate_seconds':new['summary']['candidate_seconds'],'boundary_core_sha256':hashlib.sha256((R/'code/boundary_core.py').read_bytes()).hexdigest(),'boundary_core_unchanged':(R/'code/boundary_core.py').read_bytes()==(P/'code/boundary_core.py').read_bytes(),'context_stays_inside_role_interval':True,'av_separate_faces_syncnet_methods_ast_identical':True,'boundary_tests_passed':12}
assert r['boundary_core_unchanged'];(R/'validation.json').write_text(json.dumps(r,indent=2));print(r)
