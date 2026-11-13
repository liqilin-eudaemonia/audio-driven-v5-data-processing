"""Native face and head/shoulder proposal distributions; no admission thresholds."""
import math
import listen_local_recovery as policy
import listen_v8_face_statistics as faces

REVISION='native-head-shoulder-proposal-distribution-1'
EDGES=[0,64,96,128,160,192,256,320,384,448,512,640,768,896,1024,1280,1536,1920,2048,float('inf')]
AREA_EDGES=[0,.01,.02,.04,.06,.08,.10,.15,.20,.30,.50,1.01]


def measure(geometry,start,end):
    try:
        result=policy.preview_crop(geometry,start,end)
        if result['status']=='unmeasured':return result
        if not all(math.isfinite(result[k]) for k in ('x','y','size','face_width','face_height','face_area_fraction','output_face_height_px')):
            return dict(status='unmeasured',reason='nonfinite_crop_geometry')
        if result['size']<=0 or result['face_width']<=0 or result['face_height']<=0:
            return dict(status='unmeasured',reason='degenerate_crop_geometry')
        result['revision']=REVISION
        return result
    except (ValueError,IndexError,ZeroDivisionError,FloatingPointError) as e:
        return dict(status='unmeasured',reason=type(e).__name__+': '+str(e))


def summarize(rows):
    groups={}
    for r in rows:
        key='/'.join((r['split'],r['admission_pool'],r['audio_final']['pool']))
        target=groups.setdefault(key,{})
        for role,c in r['crop_proposals'].items():
            s=target.setdefault(role,dict(measured_clips=0,unmeasured_clips=0,unmeasured_seconds=0.,
                estimated_head_not_fitting_clips=0,estimated_head_not_fitting_seconds=0.,
                side_counts=[0]*(len(EDGES)-1),side_seconds=[0.]*(len(EDGES)-1),
                output_face_height_counts=[0]*(len(EDGES)-1),output_face_height_seconds=[0.]*(len(EDGES)-1),
                face_area_in_crop_counts=[0]*(len(AREA_EDGES)-1),face_area_in_crop_seconds=[0.]*(len(AREA_EDGES)-1)))
            if c['status']=='unmeasured':
                s['unmeasured_clips']+=1;s['unmeasured_seconds']+=r['duration_s'];continue
            s['measured_clips']+=1
            if not c['fits_estimated_head']:
                s['estimated_head_not_fitting_clips']+=1;s['estimated_head_not_fitting_seconds']+=r['duration_s']
            for prefix,value,edges in [('side',c['size'],EDGES),('output_face_height',c['output_face_height_px'],EDGES),
                                       ('face_area_in_crop',c['face_area_fraction'],AREA_EDGES)]:
                i=faces.bin_index(value,edges);s[prefix+'_counts'][i]+=1;s[prefix+'_seconds'][i]+=r['duration_s']
    return groups
