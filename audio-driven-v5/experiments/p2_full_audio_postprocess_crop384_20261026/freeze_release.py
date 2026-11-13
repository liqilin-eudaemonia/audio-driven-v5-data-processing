from pathlib import Path
import sys,json,tarfile,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from prepare_validation import manifest
from p2_listen.contracts import atomic_json,sha,read
OUT=Path(__file__).resolve().parent;files=manifest();atomic_json(OUT/'code_manifest.json',files)
with tarfile.open(OUT/'code_snapshot.tar.gz','w:gz') as tar:
 for p in files:tar.add(p,arcname=Path(p).relative_to(ROOT))
reports={}
for name in ['validation100_cpu','native_cpu_validation','integration_cpu_validation']:
 v=read(OUT/(name+'.json'));assert v['status']=='passed';reports[name]={'path':str(OUT/(name+'.json')),'sha256':sha(OUT/(name+'.json')),'samples':v['samples']}
assert '30 passed' in (OUT/'unit_tests.log').read_text()
for job in ['p2_feature01_train50h_20261026','p2_e1_feature01_train50h_20261026']:
 for p,h in read(ROOT/'experiments'/job/'release.json')['files'].items():assert sha(p)==h,p
atomic_json(OUT/'verification.json',{'status':'passed','source_files':len(files),'code_manifest_sha256':sha(OUT/'code_manifest.json'),'snapshot_sha256':sha(OUT/'code_snapshot.tar.gz'),'cpu_tests':30,'real_data_tests':reports,'training_releases_unchanged':True,'full_gpu_pipeline_executed':False,'full_scope_samples':722706,'created':datetime.datetime.now(datetime.timezone.utc).isoformat()})
print({'pinned_sources':len(files),'training_releases_unchanged':True,'reports':reports})
