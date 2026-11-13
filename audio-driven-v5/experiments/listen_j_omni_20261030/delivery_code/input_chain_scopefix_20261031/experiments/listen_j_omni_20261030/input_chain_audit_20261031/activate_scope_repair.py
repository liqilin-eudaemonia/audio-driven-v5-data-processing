import hashlib,json,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent.parent;R=W.parents[1];F=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'src'),str(W/'nod_cpu_share')]
from listen_benchmark.j_annotation import read,write,locked
from probe import remote
assert read(F/'scope_reconciliation.json')['passed'] and read(F/'real_source_dryrun.json')['passed']
pid=read(W/'producer.json')['pid'];owners=[]
for n in read(W/'config.json')['nodes']:
 sup=read(W/'nodes'/n/'supervisor.json');child=sup['children'].get('producer')
 if child and child['pid']==pid:owners.append((n,child))
assert len(owners)==1,owners
node,child=owners[0]
code=f'''import os,sys,signal,time,json
from pathlib import Path
sys.path.insert(0,{str(R/'scripts')!r});sys.path.insert(0,{str(R/'src')!r})
from listen_j_annotation_supervisor import alive
record={child!r};assert alive(record)
cmd=Path('/proc/'+str(record['pid'])+'/cmdline').read_bytes().decode();assert 'listen_j_omni.py' in cmd and 'producer' in cmd and {str(W)!r} in cmd
os.kill(record['pid'],signal.SIGTERM)
for _ in range(30):
 if not alive(record):break
 time.sleep(.5)
assert not alive(record),'Producer did not stop gracefully'
print(json.dumps(dict(stopped=record['pid'])))'''
stop=remote(node,code)
with locked(W/'producer-step.lock'),locked(W/'config-edit.lock'):
 cfg=read(W/'config.json');write(F/'config_before_repair.json',cfg);write(F/'producer_state_before_repair.json',read(W/'producer_state.json'))
 cfg['source_ownership']=dict(path=str(F/'source_ownership.json'),sha256=read(F/'scope_reconciliation.json')['plan_sha256'])
 cfg['legacy_scope_file']=cfg['scope_file'];cfg['scope_file']=str(F/'source_ownership.json')
 for path in ['src/listen_benchmark/j_omni.py','src/listen_benchmark/j_omni_sources.py']:cfg['code_hashes'][path]=hashlib.sha256((R/path).read_bytes()).hexdigest()
 write(W/'config.json',cfg)
code=f'''import sys,json
from pathlib import Path
w=Path({str(W)!r});sys.path[:0]=[str(w.parents[1]/'src'),str(w.parents[1]/'scripts')]
from listen_j_annotation_supervisor import spawn
from listen_j_omni import environment,PYTHON
print(json.dumps(spawn(w/'nodes'/{node!r},'producer-scope-repair',[PYTHON,str(w.parents[1]/'scripts/listen_j_omni.py'),'producer','--work',str(w)],environment())))'''
started=remote(node,code);record=dict(at=time.time(),node=node,old=child,stop=stop,new=started,scope=cfg['source_ownership'],gpu_restarts=0)
write(F/'activation.json',record);print(json.dumps(record))
