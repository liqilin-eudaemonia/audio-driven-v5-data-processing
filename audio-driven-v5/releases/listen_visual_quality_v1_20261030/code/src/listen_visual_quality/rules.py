"""Pure measured-data rules, extracted from the verified 120-pair experiment."""
import math
import numpy as np


def runs(mask, fps, duration):
    intervals, start = [], None
    for i in range(len(mask) + 1):
        active = i < len(mask) and bool(mask[i]) and i / fps < duration
        if active and start is None:
            start = i
        if not active and start is not None:
            intervals.append({'start': start / fps, 'end': min(i / fps, duration),
                              'duration': min(i / fps, duration) - start / fps})
            start = None
    return intervals


def retained(box,c):
    x0,y0,x1,y1=box
    return max(0,min(x1,c['x']+c['width'])-max(x0,c['x']))*max(0,min(y1,c['y']+c['height'])-max(y0,c['y']))/max((x1-x0)*(y1-y0),1e-9)


def analyze_focus(samples,duration,policy):
    valid=[x for x in samples if x['measured']]
    lap=float(np.percentile([x['laplacian_variance'] for x in valid],policy["reference_percentile"])) if valid else None
    grad=float(np.percentile([x['gradient_energy'] for x in valid],policy["reference_percentile"])) if valid else None
    for q in samples:
        q['lap_ratio']=q['laplacian_variance']/max(lap,1e-9) if q['measured'] else None
        q['gradient_ratio']=q['gradient_energy']/max(grad,1e-9) if q['measured'] else None
    mask=[]
    for i,q in enumerate(samples):
        vs=[v for v in samples[max(0,i-1):i+2] if v['measured']]
        mask.append(bool(q['measured'] and len(vs)>=2 and np.median([v['lap_ratio'] for v in vs])<policy["lap_ratio_below"] and np.median([v['gradient_ratio'] for v in vs])<policy["gradient_ratio_below"]))
    intervals=[r for r in runs(mask,policy["sampling_fps"],duration) if r['duration']>=policy["min_counted_run_s"]-1e-8]
    longest=max([r['duration'] for r in intervals],default=0)
    total=sum(r['duration'] for r in intervals);fraction=total/duration
    exclude=longest>=policy["min_continuous_low_s"]-1e-9 or fraction>=policy["min_low_fraction"]-1e-9
    support=len(valid)/len(samples) if samples else 0
    status='exclusion_candidate' if exclude else ('measurement_incomplete' if support<policy["minimum_measurement_support"] else 'not_triggered')
    return {'status':status,'candidate_exclusion':exclude,'longest_low_s':longest,'low_s':total,'low_fraction':fraction,
            'valid_samples':len(valid),'total_samples':len(samples),'measured_fraction':support,
            'intervals':intervals,'laplacian_reference_p80':lap,'gradient_reference_p80':grad,
            'measurement_note':'under 80% support is a reporting flag only; it does not change the frozen exclusion condition; no detection is unknown, not zero sharpness',
            'diagnosis':'low-clarity candidate, not confirmed camera defocus'}


def propose_crop(raw,s,policy):
    old=s['video_transform']['crop'];w,h=s['video_transform']['source_size']
    valid=[q for q in raw if q['detected']]
    result={'old':old,'new':dict(old),'changed':False,'scale_ratio':1.,'head_top_estimated':True}
    if len(valid)<max(3,policy["minimum_measurement_support"]*len(raw)):
        return dict(result,status='insufficient_geometry_keep_old')
    b=np.array([q['bbox_native'] for q in valid]);fw=float(np.median(b[:,2]-b[:,0]));fh=float(np.median(b[:,3]-b[:,1]))
    envelope=[float(b[:,0].min()-.12*fw),float(b[:,1].min()-.35*fh),float(b[:,2].max()+.12*fw),float(b[:,3].max()+.12*fh)]
    required=max(old['width'],math.ceil(max(envelope[2]-envelope[0],envelope[3]-envelope[1])/.92))
    result.update(estimated_head_envelope=envelope,required_side=required,max_side=math.floor(old['width']*policy["crop_max_side_growth"]))
    if required>min(w,h,result['max_side']):return dict(result,status='exceeds_growth_cap_keep_old')
    side=required;pad=.04*side
    xlo=max(0,math.ceil(envelope[2]+pad-side));xhi=min(w-side,math.floor(envelope[0]-pad))
    ylo=max(0,math.ceil(envelope[3]+pad-side));yhi=min(h-side,math.floor(envelope[1]-pad))
    if xlo>xhi or ylo>yhi:return dict(result,status='source_edge_or_margin_limit_keep_old')
    x=min(max(old['x'],xlo),xhi);y=min(max(old['y'],ylo),yhi)
    new=dict(x=x,y=y,width=side,height=side)
    old_ret=min(retained(q['bbox_native'],old) for q in valid)
    new_ret=min(retained(q['bbox_native'],new) for q in valid)
    if new_ret<old_ret-1e-9:return dict(result,status='face_coverage_regression_keep_old')
    return dict(result,new=new,changed=new!=old,scale_ratio=old['width']/side,
                status='small_expansion_proposal' if side>old['width'] else ('translation_proposal' if new!=old else 'unchanged'),
                min_face_retained_before=old_ret,min_face_retained_after=new_ret)
