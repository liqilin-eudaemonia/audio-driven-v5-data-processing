import json,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports';index=json.loads((ROOT/'index.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
s=json.loads((OUT/'summary.json').read_text());assert s['complete'] and s['completed_videos']==692;assert json.loads((OUT/'independent_validation.json').read_text())['videos']==692
with tarfile.open(OUT/'manifests.tar.gz','w:gz') as archive:
 for f in sorted((ROOT/'manifests').glob('*.json')):archive.add(f,arcname='manifests/'+f.name)
 for name in ['policy.json','index.json']:archive.add(ROOT/name,arcname=name)
archives={}
for r in index:
 p=Path(r['archive']);stat=p.stat();archives[str(p)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
asrhash={r['video_id']:sha(Path(r['asr'])) for r in index}
provenance=dict(policy=sha(ROOT/'policy.json'),index=sha(ROOT/'index.json'),source_archive_metadata=archives,source_archive_full_hash_not_computed=True,source_asr_sha256=asrhash,code_sha256={p.name:sha(p) for p in sorted((ROOT/'code').glob('*.py'))},report_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['provenance.json']})
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n');print('packaged',len(index),'video manifests',flush=True)
