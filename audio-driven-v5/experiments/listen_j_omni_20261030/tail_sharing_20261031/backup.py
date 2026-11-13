"""Copy only explicit source, configuration and acceptance artifacts."""
import hashlib,json,shutil,time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
out=W/'delivery_code/tail_sharing_20261031'
cfg=json.loads((W/'config.json').read_text())
files=[R/p for p in cfg['code_hashes']]
files.extend([R/'README.md',R/'related/P2_Omni尾部跨卡协作调度_20261031.md',W/'config.json'])
for name in ['test_j_omni_tail.py','test_j_omni_stream.py','test_j_omni_queued.py','test_j_omni_pipeline.py',
             'test_j_omni_prefetch.py','test_j_omni_visual_gate.py','test_j_annotation_queue.py',
             'test_j_omni_sources.py','test_j_omni_crop.py']:
    files.append(R/'tests'/name)
for pattern in ('*.json','*.py','*.md'):files.extend(D.glob(pattern))
manifest={}
for p in sorted(set(files)):
    rel=p.relative_to(R);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==h
    manifest[str(rel)]=h
(out/'MANIFEST.json').write_text(json.dumps(dict(at=time.time(),files=manifest),indent=2)+'\n')
print(json.dumps(dict(path=str(out),files=len(manifest),verified=True)))
