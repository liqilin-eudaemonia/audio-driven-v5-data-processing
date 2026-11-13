"""Preview-only fixed crop: preserve face detail and allocate room below the head."""
import math
import numpy as np

REVISION = 'downward-head-shoulders-preview-1'

def crop(g, start, end, old):
    k = g['arrays']['boxes_and_keypoints:keypoints'][int(start*g['fps']):math.ceil(end*g['fps']),23:91]
    valid = (k[:,:,2] > .4).sum(axis=1) >= 45
    if not len(valid) or valid.mean() < .8:
        raise ValueError('unreliable_face_landmarks')
    k = k[valid]
    x = np.where(k[:,:,2] > .4, k[:,:,0], np.nan)
    y = np.where(k[:,:,2] > .4, k[:,:,1], np.nan)
    xl,xh = np.nanpercentile(x,[3,97],axis=1)
    yl,yh = np.nanpercentile(y,[3,97],axis=1)
    fw,fh = float(np.median(xh-xl)),float(np.median(yh-yl))
    short = float(np.percentile(np.minimum(xh-xl,yh-yl),10))
    # Landmarks exclude hair. This is a conservative estimate, not segmentation.
    head_top = float(np.percentile(yl,1))-.55*fh
    left,right = float(np.percentile(xl,1))-.25*fw,float(np.percentile(xh,99))+.25*fw
    chin = float(np.percentile(yh,99))
    size = int(math.ceil(max(512,fh/.42,right-left,(chin-head_top)/.85)/2)*2)
    if size > min(g['width'],g['height']) or head_top < 0:
        raise ValueError('head_does_not_fit_source')
    if short < 160 or short*512/size < 160:
        raise ValueError('face_short_side_below_160')
    xx = int(np.clip(np.median((xl+xh)/2)-size/2,0,g['width']-size))//2*2
    # Anchor the top; do not recenter around a shoulder envelope.
    yy = int(np.clip(head_top-.06*size,0,g['height']-size))//2*2
    if xx > left or xx+size < right or yy > head_top or yy+size < chin:
        raise ValueError('head_motion_exceeds_fixed_crop')
    return dict(x=xx,y=yy,size=size,valid_fraction=float(valid.mean()),face_width=fw,face_height=fh,
        wide_crop=False,crop_revision=REVISION,crop_recovery=True,old_tight_crop=old,
        native_face_short_p10_px=short,output_face_short_p10_px=short*512/size,
        output_face_height_median_px=fh*512/size,estimated_top_space_fraction=(head_top-yy)/size,
        output_resolution=512,head_coverage_method='Estimated from face landmarks; hair coverage needs visual review',
        training_admitted=False)
