"""Bounded read-only acceptance: 32 worker files and at most one result per worker."""
import sys,json,time,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;R=W.parents[1]
sys.path[:0]=[str(R/'src'),str(R/'scripts')]
from listen_benchmark.j_annotation import read,result_path
from listen_benchmark.j_omni_crop import effective_row
from listen_benchmark.j_omni import fingerprint

cfg=read(W/'config.json');workers=[];results=[];errors=[]
for node in cfg['nodes']:
    for gpu in range(8):
        name=f'{node}-gpu-{gpu}';s=read(W/'workers'/f'{name}.json')
        enabled='+decode-before-nod-slot-v1' in s.get('execution_revision','')
        active=enabled and s.get('cross_source_prefetch') and s['completed_clips']>0 and time.time()-s['heartbeat']<60
        workers.append(dict(name=name,active=bool(active),pid=s['pid'],stage=s['stage'],
                            completed_clips=s['completed_clips'],active_sources=s.get('active_sources'),
                            last_error=s.get('last_error')))
        if s.get('last_error'):errors.append([name,s['last_error']])
        if not s.get('sample_id'):continue
        p=result_path(W,'caption',s['sample_id'])
        if not p.exists():continue
        r=read(p)
        if '+decode-before-nod-slot-v1' not in r.get('execution_revision',''):continue
        assert r['node']==node and r['gpu']==gpu
        row=dict(sample_id=r['sample_id'],source=r['original_source'])
        applied,provenance=effective_row(row,cfg)
        assert r['source']==applied['source'] and r['crop_provenance']==provenance,r['sample_id']
        assert r['input_evidence']['video_evidence']['source']==r['source']['source'],r['sample_id']
        assert r['input_evidence']['video_evidence']['video_transform']==r['source']['video_transform'],r['sample_id']
        assert r['input_evidence']['audio_identity']['recipe_sha256']==fingerprint(r['audio_recipe'])
        assert r['preparation_metrics']['fast_crop'] is True
        nod_path=Path(r['input_evidence']['nod_supplement']['detector_file'])
        assert hashlib.sha256(nod_path.read_bytes()).hexdigest()==r['input_evidence']['nod_supplement']['detector_sha256']
        n=read(nod_path)
        assert n['source']['video_transform']==r['source']['video_transform']
        assert n['source']['source_start_s']==r['source']['source_start_s']
        assert n['source']['source_end_s']==r['source']['source_end_s']
        results.append(dict(sample_id=r['sample_id'],node=node,gpu=gpu,passed=True,
                            shared_decode_used=r['preparation_metrics']['shared_decode_used']))
out=dict(at=time.time(),all_active=all(x['active'] for x in workers),workers=workers,
         checked_results=results,errors=errors)
if len(sys.argv)>1:(D/sys.argv[1]).write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(dict(at=out['at'],active=sum(x['active'] for x in workers),
                     checked_results=len(results),errors=errors,workers=workers)))
