"""Frozen representative inputs and pre-clipping SUM-reduced Qwen gradient diagnostics."""
from pathlib import Path
from collections import defaultdict
import csv,random,math,time
import torch
import torch.distributed as dist
from p2_listen.contracts import read,sha,digest,atomic_json


def select(c):
    """Choose approved training IDs only; existing nonzero waveform metadata defines voice strata."""
    from p2_ltx_events.current_data import SourceReader
    with open(c['generator']['sample_list'])as f:rows=list(csv.DictReader(f))
    approved=set(read(c['planning_training_ids']));hashes=read(c['planning_caption_hashes'])
    eligible=[i for i,r in enumerate(rows)if r['sample_id']in approved]
    rng=random.Random(c['calibration']['seed']);rng.shuffle(eligible)
    source=SourceReader(c['generator']['current_source'],verify=False);groups=defaultdict(list)
    wanted={(voice,d)for voice in (False,True)for d in range(3)}
    def entry(i):
        r=rows[i];v=source.row(r['sample_id']);duration=float(r['duration_s'])
        if v['statistics']['caption_file_sha256']!=hashes[r['sample_id']]:raise ValueError('Calibration caption differs from approved SFT')
        voice=not v['listener_audio_identity']['full_silence'];key=(voice,0 if duration<4 else 1 if duration<8 else 2)
        return key,dict(index=i,sample_id=r['sample_id'],split=r['split'],duration_s=duration,
            voiced=voice,duration_bin=key[1],caption_sha256=hashes[r['sample_id']],label_source_sha256=digest(v))
    try:
        for i in eligible:
            key,value=entry(i)
            if len(groups[key])<3:groups[key].append(value)
            if all(len(groups[k])>=3 for k in wanted):break
        if set(groups)!=wanted or any(len(groups[k])<2 for k in wanted):raise ValueError('Need voice/silence in all three duration bins')
        chosen=[]
        for round_index in range(3):
            for key in sorted(wanted):
                if len(chosen)<c['calibration']['samples']and len(groups[key])>round_index:chosen.append(groups[key][round_index])
        longest=max(eligible,key=lambda i:float(rows[i]['duration_s']));key,value=entry(longest)
        if longest not in {r['index']for r in chosen}:
            # Preserve that stratum's other member and ensure a genuine longest input is represented.
            candidates=[i for i,r in enumerate(chosen)if(r['voiced'],r['duration_bin'])==key]
            if not candidates:raise ValueError('Longest stratum missing')
            chosen[candidates[-1]]=value
    finally:source.conn.close()
    if len(chosen)!=c['calibration']['samples']or len({r['sample_id']for r in chosen})!=len(chosen):raise ValueError('Calibration membership mismatch')
    return dict(schema='joint-v2-calibration-selection',samples=chosen,seed=c['calibration']['seed'],
        strata='existing processed listener full_silence x duration<4/<8/>=8',
        sample_list_sha256=sha(c['generator']['sample_list']),planning_ids_sha256=c['planning_training_ids_sha256'],
        training_only=True,test_used=False)


def pair_statistics(g,p,weight):
    """The same Qwen parameter order for both objectives; no optimizer or clipping here."""
    if g.shape!=p.shape or g.ndim!=1 or not torch.isfinite(g).all()or not torch.isfinite(p).all():raise ValueError('Invalid comparable gradient vectors')
    gn=float(torch.linalg.vector_norm(g.double()));pn=float(torch.linalg.vector_norm(p.double()))
    dot=float(torch.dot(g.double(),p.double()))
    return dict(generation_norm=gn,planning_unweighted_norm=pn,planning_weighted_norm=weight*pn,
        weighted_planning_to_generation=weight*pn/gn if gn>0 else None,
        cosine=max(-1.,min(1.,dot/(gn*pn)))if gn>0 and pn>0 else None)


def sum_reduce_vectors(g,p,device,chunk_elements):
    """Bounded GPU staging; norm/cosine are computed AFTER vector SUM, not by summing norms."""
    if not(dist.is_available()and dist.is_initialized()):return g,p
    for vector in (g,p):
        for start in range(0,vector.numel(),chunk_elements):
            staging=vector[start:start+chunk_elements].to(device)
            dist.all_reduce(staging,op=dist.ReduceOp.SUM)
            vector[start:start+chunk_elements].copy_(staging.cpu())
    return g,p


def flatten(grads,params):
    return torch.cat([(torch.zeros(p.numel(),dtype=torch.float64)if v is None else v.detach().reshape(-1).to(device='cpu',dtype=torch.float64))for v,p in zip(grads,params,strict=True)])


def verify_receipt(c):
    from .config import source_identity
    p=Path(c['output'])/'CALIBRATION.json';v=read(p)
    expected=dict(configuration_sha256=digest(c),source_sha256=digest(source_identity()),manifest_sha256=c['calibration_manifest_sha256'],planning_weight=c['planning_weight'])
    if not v['passed']or any(v.get(k)!=value for k,value in expected.items()):raise ValueError('Current exact gradient calibration required before training')
    if v['world']not in (1,c['global_batch']):raise ValueError('Unsupported calibration reduction layout')
    return v


def run(model,c,dataset,strategy,sampler,pos,effective,device,rank=0,world=1):
    from .config import source_identity
    from .data import to_device
    from .training import global_counts
    from p2_clip_events.model import mask_classes
    from p2_ltx_events.heads import objective as generation_objective
    if world not in (1,c['global_batch']):raise ValueError('Unsupported calibration world size')
    manifest=read(c['calibration_manifest']);records=manifest['samples'];n=len(records)
    if sha(c['calibration_manifest'])!=c['calibration_manifest_sha256']or n!=c['calibration']['samples']:raise ValueError('Calibration selection changed')
    local_records=records[rank::world];samples=[]
    for r in local_records:
        if dataset.ids[r['index']]!=r['sample_id']or r['sample_id']not in dataset.planning_ids:raise ValueError('Calibration outside approved training membership')
        s=dataset.sample(r['index']);s['events']=mask_classes(s['events'],effective)if pos else s.get('events',{})
        if not pos:s.pop('events',None)
        if not s['planning_valid']or s['planning_tokens']<=0:raise ValueError('Calibration requires supported planning targets')
        samples.append(s)
    counts,tokens=global_counts(samples,list(pos),device,world)
    if tokens<=0:raise ValueError('No global planning tokens')
    params=list(model.qwen.adapters);size=sum(p.numel()for p in params)
    gsum=torch.zeros(size,dtype=torch.float64);psum=gsum.clone();diagnostics=[]
    cuda_devices=[torch.device(device).index or 0]if str(device).startswith('cuda')else[]
    with torch.random.fork_rng(devices=cuda_devices):
        for r,sample in zip(local_records,samples,strict=True):
            torch.manual_seed(c['calibration']['seed']+r['index'])
            rng=torch.Generator(device=device).manual_seed(c['calibration']['seed']+1000000+r['index'])
            sample=to_device(sample,device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                v=model(sample,pos,rng,strategy,sampler)
                generation=generation_objective(v,c['generator'],counts,1,n)[0]
                planning=v['planning_sum']/tokens
            grads=torch.autograd.grad(generation,params,retain_graph=True,allow_unused=True);g=flatten(grads,params);del grads
            grads=torch.autograd.grad(planning,params,allow_unused=True);p=flatten(grads,params);del grads
            gsum+=g;psum+=p;diagnostics.append(dict(sample_id=r['sample_id'],**pair_statistics(g,p,c['planning_weight'])))
            del v,generation,planning,g,p,sample
    if world>1:
        all_rows=[None]*world;dist.all_gather_object(all_rows,diagnostics);diagnostics=[r for group in all_rows for r in group]
    if {r['sample_id']for r in diagnostics}!={r['sample_id']for r in records}:raise ValueError('Incomplete distributed calibration coverage')
    gsum,psum=sum_reduce_vectors(gsum,psum,device,c['calibration']['chunk_elements'])
    stats=pair_statistics(gsum,psum,c['planning_weight']);gn=stats['generation_norm'];pn=stats['planning_unweighted_norm']
    ratio=stats['weighted_planning_to_generation']
    passed=gn>0 and pn>0 and ratio is not None and ratio<=c['calibration']['max_global_ratio']
    model.zero_grad(set_to_none=True);model.assert_ownership()
    return dict(success=True,passed=passed,at=time.time(),world=world,samples=n,global_planning_tokens=tokens,
        configuration_sha256=digest(c),source_sha256=digest(source_identity()),manifest_sha256=c['calibration_manifest_sha256'],
        planning_weight=c['planning_weight'],aggregate=stats,per_sample=diagnostics,
        coefficient_sweep={str(w):w*pn/gn if gn>0 else None for w in (.1,.01,.001,.0001)},
        suggested_weight=c['calibration']['target_ratio']*gn/pn if pn>0 else None,
        reference_band=c['calibration']['reference_band'],reference_band_is_not_an_optimum=True,
        recommendation_only_no_config_mutation=True,optimizer_updates=0,rng_restored=True,
        scope='All calibration samples, vector SUM after global sample/token normalization, before clipping; per-sample diagnostics use the same global denominators')
