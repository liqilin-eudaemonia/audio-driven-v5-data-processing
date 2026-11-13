"""Install verified candidate and gracefully roll only node4 GPU7."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
W=Path(__file__).resolve().parents[1];ROOT=W.parents[1];sys.path.insert(0,str(ROOT/'src'))
from listen_benchmark.j_annotation import read,write
cfg=read(W/'config.json');report=read(W/'performance_check/preparation_benchmark.json');assert report['complete'] and all(r['all_hashes_equal'] for r in report['reports'])
base=next(r for r in report['reports'] if r['backend']=='thread' and r['repeat']==1);fast=next(r for r in report['reports'] if r['backend']=='process' and r['workers']==4 and r['repeat']==1);assert fast['clips_per_second']>base['clips_per_second']*1.1
rev=W/'revisions/process_prefetch_pilot';rev.mkdir(parents=True,exist_ok=True);assert not (rev/'installed.json').exists()
write(rev/'config-before.json',cfg)
core=ROOT/'src/listen_benchmark/j_omni.py';(rev/'j_omni-before.py').write_bytes(core.read_bytes())
candidate=W/'performance_check/j_omni_candidate.py';compile(candidate.read_text(),str(candidate),'exec');tmp=core.with_suffix('.candidate.tmp');tmp.write_bytes(candidate.read_bytes());tmp.replace(core)
for rel in ['src/listen_benchmark/j_omni_prefetch.py','src/listen_benchmark/j_omni_e2e.py']:cfg['code_hashes'][rel]=''
cfg['code_hashes']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in cfg['code_hashes']}
cfg.setdefault('preparation_profiles',{})['node4-gpu-7']=dict(backend='process',workers=4,prefetch=8,ready_first=True)
cfg.setdefault('startup_benchmarks',{})['node4-gpu-7']=str(W/'performance_check/e2e')
write(W/'config.json',cfg);write(rev/'config-after.json',cfg);write(rev/'installed.json',dict(at=time.time(),runtime_scope='node4 GPU7 only; other workers retain existing processes',pipeline_id=cfg['pipeline_id']))
code=f'ROOT={str(ROOT)!r}\nWORK={str(W)!r}\n'+r'''
import sys,os,signal,time,socket
from pathlib import Path
sys.path.insert(0,ROOT+'/scripts');sys.path.insert(0,ROOT+'/src')
from listen_j_annotation_supervisor import alive
from listen_benchmark.j_annotation import read,write
w=Path(WORK);assert socket.gethostname()==read(w/'config.json')['nodes']['node4']['hostname'];s=read(w/'nodes/node4/supervisor.json');c=s['children']['gpu-7'];assert alive(c)
cmd=Path(f"/proc/{c['pid']}/cmdline").read_bytes().decode().replace('\x00',' ');assert 'listen_j_omni.py worker' in cmd and WORK in cmd
os.kill(c['pid'],signal.SIGTERM);write(w/'revisions/process_prefetch_pilot/signaled.json',dict(at=time.time(),child=c));print('node4 GPU7 graceful roll',c['pid'])
'''
r=subprocess.run(['ssh','-i','/private/ssh_keys/root_dedicated-developjob-lq-dev-machine-1nkpf_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/jomni-guard-%C','root@172.29.225.51','python3 -'],input=code,text=True,capture_output=True,timeout=30);print(r.stdout,r.stderr);r.check_returncode()
