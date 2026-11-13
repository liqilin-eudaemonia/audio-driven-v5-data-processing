"""Count finished video seconds using bounded known-source paths, never a tree scan."""
import json,sys,hashlib,time,collections
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parent;label=sys.argv[1];folder=D/label
rows=[json.loads((folder/f'{n}.json').read_text()) for n in ['node17','node8','node4','node12']]
limits={r['node']:r for r in rows};lower=min(r['start'] for r in rows);upper=max(r['end'] for r in rows)
sources=set();full_queue=(folder/'queue_before.json').exists()
if full_queue:
    before=json.loads((folder/'queue_before.json').read_text());after=json.loads((folder/'queue_after.json').read_text())
    for state in [before,after]:sources.update(state['inflight']);sources.update(state.get('deferred',{}))
    with (W/'ready.jsonl').open() as f:
        f.seek(before['offset'])
        while f.tell()<after['offset']:
            line=f.readline()
            if not line:raise ValueError('Journal truncated')
            sources.add(json.loads(line)['job_id'])
else:
    # Earlier windows recorded sample IDs only; expose any coverage shortfall.
    for r in rows:
        for x in r['sampled_outputs']:
            assert '_local_' in x['sample_id']
            sources.add(x['sample_id'].rsplit('_local_',1)[0])
totals={n:dict(clips=0,video_seconds=0.,source_pixel_seconds=0.,changed=0,shared=0,failed=0) for n in limits};found=[];seen=set();paths=0
for jid in sorted(sources):
    job=json.loads((W/'jobs'/f'{jid}.json').read_text())
    for row in job['rows']:
        sid=row['sample_id']
        if sid in seen:continue
        seen.add(sid);p=W/'caption/results'/hashlib.sha256(sid.encode()).hexdigest()[:2]/f'{sid}.json';paths+=1
        try:mtime=p.stat().st_mtime
        except FileNotFoundError:continue
        if not lower-2<=mtime<=upper+2:continue
        r=json.loads(p.read_text());n=r['node'];bound=limits[n]
        if not bound['start']<=r['completed_at']<=bound['end']:continue
        dur=r['source']['duration_s'];width,height=r['source']['video_transform']['source_size'];t=totals[n]
        t['clips']+=1;t['video_seconds']+=dur;t['source_pixel_seconds']+=width*height*dur
        t['changed']+=bool(r.get('crop_provenance',{}).get('coordinates_changed'));t['shared']+=bool(r['preparation_metrics']['shared_decode_used']);t['failed']+=not r['generation_checks_passed']
        found.append(dict(sample_id=sid,node=n,gpu=r['gpu'],completed_at=r['completed_at'],duration=dur,
            source_size=[width,height],shared_format=r['preparation_metrics'].get('shared_decode_format'),
            preparation_seconds=r['preparation_metrics']['seconds'],
            inference_seconds_per_item=r['seconds']/r.get('submitted_requests',1)))
count=sum(t['clips'] for t in totals.values());state_count=sum(r['clips'] for r in rows)
out=dict(label=label,at=time.time(),known_sources=len(sources),named_paths_checked=paths,full_queue_tracking=full_queue,
    state_clips=state_count,matched_clips=count,coverage_ratio=count/state_count,
    clips_per_second=sum(t['clips']/limits[n]['elapsed'] for n,t in totals.items()),
    video_seconds_per_second=sum(t['video_seconds']/limits[n]['elapsed'] for n,t in totals.items()),
    source_megapixel_seconds_per_second=sum(t['source_pixel_seconds']/1e6/limits[n]['elapsed'] for n,t in totals.items()),
    mean_duration=sum(t['video_seconds'] for t in totals.values())/count if count else None,
    changed_fraction=sum(t['changed'] for t in totals.values())/count if count else None,nodes=totals)
(folder/'exact_work.json').write_text(json.dumps(out,indent=2));(folder/'finished_ids.json').write_text(json.dumps(found))
print(json.dumps(out))
