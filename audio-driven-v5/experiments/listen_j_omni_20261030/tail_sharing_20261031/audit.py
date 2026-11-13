"""Verify the named post-rollout outputs; no recursive result scan."""
import hashlib
import sys
import time
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,write
from listen_benchmark.j_omni_crop import effective_row
from listen_benchmark.j_omni import fingerprint

cfg=read(W/'config.json');accepted=read(D/'deployment_acceptance.json');checked=[]
for sample in accepted['outputs']:
    r=read(sample['path']);s=read(W/'workers'/f"{sample['node']}-gpu-{sample['gpu']}.json")
    assert s['tail_protocol']=='clip-lock-tail-v1'
    assert s['tail_lock_release']=='consumer-close-after-persist'
    assert s['tail_packet_lifecycle']=='finish-before-source-claim'
    assert time.time()-s['heartbeat']<60 and s['completed_clips']>=3
    assert not s.get('last_error'),s
    row=dict(sample_id=r['sample_id'],source=r['original_source'])
    applied,provenance=effective_row(row,cfg)
    assert r['source']==applied['source'] and r['crop_provenance']==provenance
    assert r['input_evidence']['video_evidence']['source']==r['source']['source']
    assert r['input_evidence']['video_evidence']['video_transform']==r['source']['video_transform']
    assert r['input_evidence']['audio_identity']['recipe_sha256']==fingerprint(r['audio_recipe'])
    nod=Path(r['input_evidence']['nod_supplement']['detector_file'])
    assert hashlib.sha256(nod.read_bytes()).hexdigest()==r['input_evidence']['nod_supplement']['detector_sha256']
    n=read(nod)
    for key in ('video_transform','source_start_s','source_end_s'):
        assert n['source'][key]==r['source'][key]
    assert r['scheduling']['protocol']=='clip-lock-tail-v1'
    checked.append(dict(node=sample['node'],gpu=sample['gpu'],sample_id=r['sample_id'],passed=True))
q=read(W/'queues/caption/state.json')
out=dict(at=time.time(),passed=True,checked=checked,failed_sources=len(q['failed']),
         active_workers=len(checked),helpers_enabled=(W/'tail_sharing/ENABLED.json').exists())
write(D/'input_and_runtime_acceptance.json',out)
print(out)
