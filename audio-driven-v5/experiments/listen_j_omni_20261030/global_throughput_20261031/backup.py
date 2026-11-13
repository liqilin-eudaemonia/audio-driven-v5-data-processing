"""Snapshot only named code/config/evidence files, never media or result trees."""
from pathlib import Path
import hashlib,json,shutil,time
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
target=W/'delivery_code/global_throughput_20261031';target.mkdir(parents=True,exist_ok=True)
c=json.loads((W/'config.json').read_text())
files=[R/p for p in c['code_hashes']]
files += [R/'README.md',R/'related/P2_四机32卡整体吞吐优化_20261031.md',W/'config.json']
for pattern in ['test_j_omni*.py','test_j_nod_cpu_share.py','test_shared_visual_decode.py',
                'test_shared_decode_format_guard.py','test_shared_raw_decode.py','test_roi_frame_crop.py','test_exact_frame_crop.py','test_reused_frame_crop.py']:
    files.extend((R/'tests').glob(pattern))
files.extend(D.glob('*.py'));files.extend(D.glob('*.json'))
for label in ['A1_slots16','B1_slots24','A2_slots16','B2_slots24','C_before_roi','D1_after_roi','D2_after_roi']:
    files.extend((D/label).glob('*.json'))
hashes={}
for p in sorted(set(files)):
    rel=p.relative_to(R);dst=target/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    digest=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
    hashes[str(rel)]=digest
(target/'MANIFEST.json').write_text(json.dumps(dict(at=time.time(),files=hashes),indent=2))
print(json.dumps(dict(path=str(target),files=len(hashes),verified=True)))
