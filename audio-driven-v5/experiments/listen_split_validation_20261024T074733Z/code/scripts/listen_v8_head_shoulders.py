"""Head/shoulder crop capacity trial, with independent source/output face limits."""
import math
import numpy as np
import listen_turns_v8 as original
BASE_CROP=original.full_turn_crop
REVISION='head-shoulders-capacity-1'

def crop(r,a,b):
    old=BASE_CROP(r,a,b)
    if old['size']>=448:return old
    allk=r['arrays']['boxes_and_keypoints:keypoints'][int(a*r['fps']):math.ceil(b*r['fps'])]
    k=allk[:,23:91];valid=(k[:,:,2]>.4).sum(axis=1)>=45
    if not len(valid) or valid.mean()<.8:raise ValueError('Face landmark validity below 80%')
    k=k[valid];x=np.where(k[:,:,2]>.4,k[:,:,0],np.nan);y=np.where(k[:,:,2]>.4,k[:,:,1],np.nan)
    xl,xh=np.nanpercentile(x,[3,97],axis=1);yl,yh=np.nanpercentile(y,[3,97],axis=1)
    face_short_p10=float(np.percentile(np.minimum(xh-xl,yh-yl),10))
    if face_short_p10<160:raise ValueError('head_shoulders_native_face_below_160')
    fw=float(np.median(xh-xl));fh=float(np.median(yh-yl))
    left=float(np.percentile(xl,1))-.4*fw;right=float(np.percentile(xh,99))+.4*fw
    top=float(np.percentile(yl,1))-.7*fh
    shoulders=allk[:,[5,6]];sv=(shoulders[:,:,2]>.5).all(axis=1)
    if sv.mean()<.6:raise ValueError('head_shoulders_unreliable_shoulders')
    sx=shoulders[sv,:,0];sy=shoulders[sv,:,1]
    left=min(left,float(np.percentile(sx,2))-.1*fw);right=max(right,float(np.percentile(sx,98))+.1*fw)
    bottom=max(float(np.percentile(yh,99))+.8*fh,float(np.percentile(sy,98))+.3*fh)
    if left<0 or top<0 or right>r['width'] or bottom>r['height']:raise ValueError('head_shoulders_content_outside_source')
    size=int(math.ceil(max(512.,right-left,bottom-top)/2)*2)
    if size>min(r['width'],r['height']):raise ValueError('head_shoulders_does_not_fit_source')
    output_face_p10=face_short_p10*512/size
    if output_face_p10<128:raise ValueError('head_shoulders_output_face_below_128')
    xx=int(np.clip((left+right-size)/2,0,r['width']-size))//2*2
    yy=int(np.clip((top+bottom-size)/2,0,r['height']-size))//2*2
    if xx>left or yy>top or xx+size<right or yy+size<bottom:raise ValueError('head_shoulders_rounding_clips_content')
    return dict(x=xx,y=yy,size=size,valid_fraction=old['valid_fraction'],face_width=old['face_width'],face_height=old['face_height'],wide_crop=False,
        crop_revision=REVISION,crop_recovery=True,old_tight_crop=old,native_face_short_p10_px=face_short_p10,
        output_face_short_p10_px=output_face_p10,shoulder_valid_fraction=float(sv.mean()),output_resolution=512,
        head_coverage_method='face landmark envelope with forehead/hair margin; requires visual review')
