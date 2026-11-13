#!/usr/bin/env python3
"""Frozen-threshold prospective batch, independent of the 24 tuning pairs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,json,math,hashlib,random,datetime,time,concurrent.futures,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/visual_quality_pilot_20261030'))
import run as base
spec=importlib.util.spec_from_file_location('frozen_focus_v1',ROOT/'experiments/visual_quality_crop_focus_20261030/run.py')
focus=importlib.util.module_from_spec(spec);spec.loader.exec_module(focus)
np,cv2,av,mp=base.np,base.cv2,base.av,base.mp
OUT=Path(__file__).resolve().parent
WEB=ROOT/'data_previews/visual_quality_holdout120_20261030'
JOBS=ROOT/'experiments/listen_j_annotation_node17_20261029/jobs'
POLICY={'lap_ratio_below':.45,'gradient_ratio_below':.70,'reference_percentile':80,
        'median_samples':3,'sampling_fps':5,'min_counted_run_s':.6,
        'min_continuous_low_s':2.,'min_low_fraction':.25,
        'crop_max_side_growth':1.05,'crop_policy':'minimum translation first; expand only as needed, at most 5%; otherwise preserve current crop and flag'}


def select():
    path=OUT/'selection.json'
    if path.exists():return json.loads(path.read_text())
    prior=json.loads((base.OUT/'selection.json').read_text())
    excluded={r['source']['pair'] for r in prior['rows']}
    names=[x.name for x in os.scandir(JOBS) if x.name.endswith('.json') and x.name[:-5] not in excluded]
    rng=random.Random(2026103003);rows=[];jobs=[]
    for mode in ['naturalistic','improvised']:
        pool=sorted(n for n in names if n.startswith(mode+'_'));rng.shuffle(pool);count=0
        for name in pool:
            p=JOBS/name;raw=p.read_bytes();job=json.loads(raw);eligible=job['rows']
            if len(eligible)<3:continue
            # One per available length band, then fill without replacement.
            selected=[]
            for lo,hi in [(0,4),(4,8),(8,100)]:
                band=[r for r in eligible if lo<=r['source']['duration_s']<hi]
                if band:selected.append(rng.choice(band))
            ids={r['sample_id'] for r in selected}
            if len(selected)<3:selected+=rng.sample([r for r in eligible if r['sample_id'] not in ids],3-len(selected))
            selected.sort(key=lambda r:r['source']['source_start_s'])
            for r in selected:
                s=r['source'];assert base.sha(s['source_result'])==s['source_result_sha256']
            rows.extend(selected);jobs.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'eligible':len(eligible)})
            count+=1
            if count==20:break
        assert count==20
    assert len(rows)==120 and len({r['sample_id'] for r in rows})==120
    assert not ({r['source']['pair'] for r in rows}&excluded)
    value={'seed':2026103003,'selection':'20 uniformly shuffled eligible sources per conversation type; 3 pairs/source with duration coverage; no quality-based selection',
           'excluded_tuning_sources':sorted(excluded),'rows':rows,'jobs':jobs,
           'limitations':'source-stratified duration-enriched sample, not a corpus pair-weighted prevalence estimate; participant identities may overlap other sources'}
    base.save(path,value);return value


def classify(samples,duration):
    valid=[x for x in samples if x['measured']]
    lap=float(np.percentile([x['laplacian_variance'] for x in valid],80)) if valid else None
    grad=float(np.percentile([x['gradient_energy'] for x in valid],80)) if valid else None
    for q in samples:
        q['lap_ratio']=q['laplacian_variance']/max(lap,1e-9) if q['measured'] else None
        q['gradient_ratio']=q['gradient_energy']/max(grad,1e-9) if q['measured'] else None
    mask=[]
    for i,q in enumerate(samples):
        vs=[v for v in samples[max(0,i-1):i+2] if v['measured']]
        mask.append(bool(q['measured'] and len(vs)>=2 and np.median([v['lap_ratio'] for v in vs])<.45 and np.median([v['gradient_ratio'] for v in vs])<.70))
    intervals=[r for r in base.runs(mask,5,duration) if r['duration']>=.6-1e-8]
    longest=max([r['duration'] for r in intervals],default=0)
    total=sum(r['duration'] for r in intervals);fraction=total/duration
    exclude=longest>=2-1e-9 and fraction>=.25-1e-9
    support=len(valid)/len(samples) if samples else 0
    status='exclusion_candidate' if exclude else ('measurement_incomplete' if support<.8 else 'not_triggered')
    return {'status':status,'candidate_exclusion':exclude,'longest_low_s':longest,'low_s':total,'low_fraction':fraction,
            'valid_samples':len(valid),'total_samples':len(samples),'measured_fraction':support,
            'intervals':intervals,'laplacian_reference_p80':lap,'gradient_reference_p80':grad,
            'measurement_note':'under 80% support is a reporting flag only; it does not change the frozen exclusion condition; no detection is unknown, not zero sharpness',
            'diagnosis':'low-clarity candidate, not confirmed camera defocus'}


def bounded_crop(raw,s):
    old=s['video_transform']['crop'];w,h=s['video_transform']['source_size']
    valid=[q for q in raw if q['detected']]
    result={'old':old,'new':dict(old),'changed':False,'scale_ratio':1.,'head_top_estimated':True}
    if len(valid)<max(3,.8*len(raw)):
        return dict(result,status='insufficient_geometry_keep_old')
    b=np.array([q['bbox_native'] for q in valid]);fw=float(np.median(b[:,2]-b[:,0]));fh=float(np.median(b[:,3]-b[:,1]))
    envelope=[float(b[:,0].min()-.12*fw),float(b[:,1].min()-.35*fh),float(b[:,2].max()+.12*fw),float(b[:,3].max()+.12*fh)]
    required=max(old['width'],math.ceil(max(envelope[2]-envelope[0],envelope[3]-envelope[1])/.92))
    result.update(estimated_head_envelope=envelope,required_side=required,max_side=math.floor(old['width']*1.05))
    if required>min(w,h,result['max_side']):return dict(result,status='exceeds_growth_cap_keep_old')
    side=required;pad=.04*side
    xlo=max(0,math.ceil(envelope[2]+pad-side));xhi=min(w-side,math.floor(envelope[0]-pad))
    ylo=max(0,math.ceil(envelope[3]+pad-side));yhi=min(h-side,math.floor(envelope[1]-pad))
    if xlo>xhi or ylo>yhi:return dict(result,status='source_edge_or_margin_limit_keep_old')
    x=min(max(old['x'],xlo),xhi);y=min(max(old['y'],ylo),yhi)
    new=dict(x=x,y=y,width=side,height=side)
    old_ret=min(focus.retained(q['bbox_native'],old) for q in valid)
    new_ret=min(focus.retained(q['bbox_native'],new) for q in valid)
    if new_ret<old_ret-1e-9:return dict(result,status='face_coverage_regression_keep_old')
    return dict(result,new=new,changed=new!=old,scale_ratio=old['width']/side,
                status='small_expansion_proposal' if side>old['width'] else ('translation_proposal' if new!=old else 'unchanged'),
                min_face_retained_before=old_ret,min_face_retained_after=new_ret)


def measure(task):
    i,row,protocol_hash=task;sid=row['sample_id'];path=OUT/'results'/f'{sid}.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('protocol_hash')==protocol_hash:return old
        raise ValueError('Different protocol already exists')
    started=time.monotonic();s=row['source'];c=s['video_transform']['crop'];w,h=s['video_transform']['source_size']
    origin=round(s['source_start_s']*s['normalized_audio']['sample_rate'])/s['normalized_audio']['sample_rate'];duration=s['duration_s']
    context={'x':max(0,math.floor(c['x']-.25*c['width'])),'y':max(0,math.floor(c['y']-.25*c['height']))}
    context['width']=min(w,math.ceil(c['x']+1.25*c['width']))-context['x'];context['height']=min(h,math.ceil(c['y']+1.25*c['height']))-context['y']
    local=dict(c,x=c['x']-context['x'],y=c['y']-context['y']);samples=[];raw=[];n=math.ceil(duration*5-1e-9)
    try:
        with base.face.FaceLandmarker.create_from_options(base.face.build_options(str(base.MODEL))) as full,\
             base.face.FaceLandmarker.create_from_options(base.face.build_options(str(base.MODEL))) as expanded,\
             base.SourceBytes(s['source']) as source,av.open(source) as inp:
            stream=inp.streams.video[0];stream.codec_context.thread_count=1
            assert [stream.width,stream.height]==[w,h]
            inp.seek(int(origin/stream.time_base),stream=stream,backward=True,any_frame=False)
            k=0;last=None;last_t=None
            def sample(rgb,t,k):
                timestamp=k*200
                fr=full.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),timestamp)
                x,y,cw,ch=[context[q] for q in ('x','y','width','height')]
                cr=expanded.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb[y:y+ch,x:x+cw])),timestamp)
                fm=base.raw_measure(fr,w,h,c,k/5,origin+t);cm=base.raw_measure(cr,cw,ch,local,k/5,origin+t)
                if cm['detected']:
                    b=cm['bbox_native'];cm['bbox_native']=[b[0]+x,b[1]+y,b[2]+x,b[3]+y]
                use_context=not fm['detected'] or not fm.get('center_inside_crop',True)
                selected=dict(cm if use_context else fm);chosen=cr if use_context else fr
                selected.update(detector_input='expanded_context' if use_context else 'full_frame',full_frame_result=fm,context_result=cm)
                raw.append(selected)
                metric=focus.focus_metric(rgb,selected['bbox_native']) if selected['detected'] else None
                q={'t':k/5,'source_time':origin+t,'measured':metric is not None,'raw_detected':selected['detected']}
                if metric:q.update(metric)
                if chosen.facial_transformation_matrixes:
                    pitch,roll,yaw=base.face.matrix_to_euler_degrees(chosen.facial_transformation_matrixes[0]);q.update(pitch=pitch,yaw=yaw,roll=roll)
                if selected['detected']:q['bbox_native']=selected['bbox_native']
                samples.append(q)
            for f in inp.decode(stream):
                t=float(f.pts*f.time_base)-origin
                if t<0:continue
                if t>=duration:break
                if k>=n:break
                if k/5>t+1e-10:
                    # Retain a potential last frame only if it could be the last sample.
                    last=f;last_t=t;continue
                rgb=f.to_ndarray(format='rgb24');last=f;last_t=t
                while k<n and k/5<=t+1e-10:sample(rgb,t,k);k+=1
            if last is None or duration-last_t>=.1 and k<n:raise ValueError('Incomplete original support')
            while k<n:sample(last.to_ndarray(format='rgb24'),last_t,k);k+=1
        assert len(samples)==n and samples[0]['source_time']-origin<.1
        result={'id':sid,'ordinal':i,'source':s,'status':'ok','focus_samples':samples,'raw_measurements':raw,
                'focus':classify(samples,duration),'crop':bounded_crop(raw,s),'context_rect':context,
                'protocol_hash':protocol_hash,'elapsed_s':time.monotonic()-started,'production_applied':False,'training_admitted':False}
    except Exception as e:
        result={'id':sid,'ordinal':i,'source':s,'status':'error','error':repr(e),'protocol_hash':protocol_hash,'elapsed_s':time.monotonic()-started,'production_applied':False}
    base.save(path,result);return result


def check_frozen_rule():
    previous=json.loads((ROOT/'experiments/visual_quality_crop_focus_20261030/data.json').read_text())
    selected=[]
    for r in previous['results']:
        # Do not use these data for selection or retuning; check implementation equivalence only.
        out=classify(r['focus_samples'],r['source']['duration_s'])
        if out['candidate_exclusion']:selected.append(r['ordinal'])
        assert abs(out['longest_low_s']-max([v['duration'] for v in r['focus_intervals']],default=0))<1e-6
    assert selected==[13]
    return {'frozen_rule_reproduces_tuning_batch':True,'selected_ordinals':[13],'not_holdout_accuracy':True}


def main():
    selection=select();protocol={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy':POLICY,
        'selection_sha256':base.sha(OUT/'selection.json'),'runner_sha256':base.sha(__file__),
        'face_detector_sha256':base.sha(base.FACE),'model_sha256':base.sha(base.MODEL),'focus_metric_sha256':base.sha(ROOT/'experiments/visual_quality_crop_focus_20261030/run.py'),
        'workers':2,'frozen_rule_check':check_frozen_rule(),'production_applied':False,
        'sampling':'same 5 Hz first original frame at/after target as previous 25 Hz pipeline sub-sampled every fifth frame; no synthetic blur; no candidate crop influence on focus metrics'}
    # Timestamp is excluded from resumption identity.
    identity={k:v for k,v in protocol.items() if k!='created_at'}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();protocol['hash']=fingerprint
    base.save(OUT/'protocol.json',protocol)
    results=[];started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(measure,(i,r,fingerprint)) for i,r in enumerate(selection['rows'],1)]
        for future in concurrent.futures.as_completed(jobs):
            r=future.result();results.append(r)
            progress={'done':len(results),'total':120,'ordinal':r['ordinal'],'status':r['status'],'candidate':r.get('focus',{}).get('candidate_exclusion'),'elapsed_s':time.monotonic()-started}
            print(json.dumps(progress),flush=True);base.save(OUT/'progress.json',progress)
    results.sort(key=lambda r:r['ordinal'])
    ok=[r for r in results if r['status']=='ok'];summary={'clips':len(results),'sources':40,'ok':len(ok),'errors':[r['ordinal'] for r in results if r['status']!='ok'],
        'exclusion_candidates':[r['ordinal'] for r in ok if r['focus']['candidate_exclusion']],
        'measurement_incomplete':[r['ordinal'] for r in ok if r['focus']['status']=='measurement_incomplete'],
        'crop_status_counts':{status:sum(r['crop']['status']==status for r in ok) for status in sorted({r['crop']['status'] for r in ok})},
        'production_applied':0,'automatically_deleted':0,'elapsed_s':time.monotonic()-started}
    base.save(OUT/'data.json',{'summary':summary,'protocol':protocol,'results':results});base.save(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
