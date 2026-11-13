import sys,json,time,concurrent.futures
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent
sys.path.insert(0,str(W/'nod_cpu_share'));from probe import remote,C
R=Path(C['project_root']);sys.path.insert(0,str(R/'src'))
from listen_benchmark.j_annotation import read,write,locked
source=Path(C['nodes']['node17']['old_work'])
with locked(source/'config-edit.lock'):
 cfg=read(source/'config.json')
 if not (D/'upstream_config.before.json').exists():write(D/'upstream_config.before.json',cfg)
 cfg.pop('corrected_crop_omni_work',None);cfg.update(corrected_crop_omni_work_v2=str(W),nod_worker_revision=6);write(source/'config.json',cfg)
code='''import json,os,signal,time
from pathlib import Path
W=Path(WORK);source=Path(SOURCE);actions=[]
if NODE=='node17':
 supervisor=json.loads((source/'supervisor.json').read_text());children={k:v for k,v in supervisor['children'].items() if k.startswith('nod-')}
else:
 supervisor=json.loads((W/'nod_cpu_share'/NODE/'supervisor.json').read_text());children=supervisor['children']
for key,ch in children.items():
 slot=key.removeprefix('nod-');hp=source/'workers'/f'nod-{slot}.json';h=json.loads(hp.read_text()) if hp.exists() else {}
 if h.get('pid')==ch['pid'] and h.get('revision',0)>=6:continue
 p=Path('/proc')/str(ch['pid'])
 if not p.exists():continue
 stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 if stat[0]=='Z' or stat[19]!=str(ch['start_ticks']):continue
 cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 if NODE=='node17':assert 'listen_j_annotation_nod.py' in cmd and str(source) in cmd
 else:assert 'listen_j_nod_cpu_share.py worker' in cmd and str(W) in cmd and '--node '+NODE in cmd
 os.kill(ch['pid'],signal.SIGTERM);actions.append(dict(slot=slot,pid=ch['pid'],start_ticks=stat[19]))
print(json.dumps(dict(node=NODE,at=time.time(),graceful_replacements=actions)))
'''
def run(n):return remote(n,'WORK='+repr(str(W))+'; SOURCE='+repr(str(source))+'; NODE='+repr(n)+'\n'+code)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 fs={n:pool.submit(run,n) for n in C['nodes']};results={n:f.result() for n,f in fs.items()}
write(D/'nod_roll_v2.json',results);print({n:len(r['graceful_replacements']) for n,r in results.items()})
