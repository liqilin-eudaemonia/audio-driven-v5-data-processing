"""Interval-only metrics. No event names or spoken words are inferred here."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def union(intervals, duration):
    out=[]
    for a,b in sorted(intervals):
        if not math.isfinite(a+b) or b<a:raise ValueError('Invalid interval')
        a,b=max(0.,float(a)),min(float(duration),float(b))
        if b<=a:continue
        if out and a<=out[-1][1]+1e-9:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out


def mask(intervals, duration, hop=.01):
    times=(np.arange(math.ceil(duration/hop))+.5)*hop
    result=np.zeros(len(times),bool)
    for a,b in intervals:result|=(times>=a)&(times<b)
    return result & (times<duration)


def intervals_from_mask(values, hop=.01, duration=None, min_s=.04, gap_s=.03):
    values=np.asarray(values,dtype=bool).copy()
    def runs(x):
        edges=np.diff(np.r_[False,x,False].astype(int))
        return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    for a,b in runs(~values):
        if a>0 and b<len(values) and (b-a)*hop<=gap_s+1e-9:values[a:b]=True
    limit=len(values)*hop if duration is None else duration
    return [[float(a*hop),float(min(b*hop,limit))] for a,b in runs(values) if min(b*hop,limit)-a*hop>=min_s-1e-9]


def match_events(gt, pred, collar=0.):
    if not gt or not pred:return []
    weights=np.zeros((len(gt),len(pred)));ious=weights.copy()
    for i,(a,b) in enumerate(gt):
        for j,(c,d) in enumerate(pred):
            inter=max(0.,min(b,d)-max(a,c));iou=inter/(max(b,d)-min(a,c))
            ious[i,j]=iou
            if iou>=.5-1e-9 or (collar>0 and abs(a-c)<=collar+1e-9 and abs(b-d)<=collar+1e-9):
                weights[i,j]=1+iou/(min(len(gt),len(pred))+1)
    ix,jx=linear_sum_assignment(weights,maximize=True)
    return [dict(gt_index=int(i),generated_index=int(j),tiou=float(ious[i,j]),
        onset_error_s=pred[j][0]-gt[i][0],offset_error_s=pred[j][1]-gt[i][1],
        duration_error_s=(pred[j][1]-pred[j][0])-(gt[i][1]-gt[i][0]))
        for i,j in zip(ix,jx) if weights[i,j]>0]


def event_scores(gt,pred,collar=0.):
    matches=match_events(gt,pred,collar);tp=len(matches);fp=len(pred)-tp;fn=len(gt)-tp
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        matches=matches,matched_tiou=float(np.mean([m['tiou'] for m in matches])) if matches else None,
        onset_mae_s=float(np.mean([abs(m['onset_error_s']) for m in matches])) if matches else None,
        offset_mae_s=float(np.mean([abs(m['offset_error_s']) for m in matches])) if matches else None)


def activity_comparison(gt,pred,duration,collar=.2):
    gt=union(gt,duration);pred=union(pred,duration)
    gm,pm=mask(gt,duration),mask(pred,duration)
    expanded=mask(union([[a-collar,b+collar] for a,b in gt],duration),duration)
    interior=mask([[a+collar,b-collar] for a,b in gt if b-a>2*collar],duration)
    valid=(np.arange(len(gm))+.5)*.01<duration
    silent=valid&~expanded
    return dict(gt_intervals=gt,generated_intervals=pred,strict=event_scores(gt,pred),
        tolerant=event_scores(gt,pred,collar),gt_active_s=sum(b-a for a,b in gt),
        generated_active_s=sum(b-a for a,b in pred),
        duration_abs_error_s=abs(sum(b-a for a,b in gt)-sum(b-a for a,b in pred)),
        extra_active_s=float(np.sum(pm&silent)*.01),reference_silent_s=float(silent.sum()*.01),
        missed_interior_s=float(np.sum(~pm&interior)*.01),reference_interior_s=float(interior.sum()*.01),
        any_reference_activity=bool(gt),missed_entire_response=bool(gt) and not pred,
        false_activity_on_silent_clip=not gt and bool(pred))


def aggregate_activity(records):
    result=dict(pairs=len(records),silent_reference_pairs=sum(not r['any_reference_activity'] for r in records),
        active_reference_pairs=sum(r['any_reference_activity'] for r in records))
    for key in ('gt_active_s','generated_active_s','extra_active_s','reference_silent_s','missed_interior_s','reference_interior_s',
                'missed_entire_response','false_activity_on_silent_clip'):
        result[key]=sum(r[key] for r in records)
    result['duration_mae_s']=float(np.mean([r['duration_abs_error_s'] for r in records])) if records else None
    for name,top,bottom in [('extra_activity_rate','extra_active_s','reference_silent_s'),('missed_interior_rate','missed_interior_s','reference_interior_s'),
                           ('missed_response_rate','missed_entire_response','active_reference_pairs'),('false_sound_clip_rate','false_activity_on_silent_clip','silent_reference_pairs')]:
        result[name]=result[top]/result[bottom] if result[bottom] else None
    for kind in ('strict','tolerant'):
        counts={k:sum(r[kind][k] for r in records) for k in ('tp','fp','fn')};tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
        matches=[m for r in records for m in r[kind]['matches']]
        result[kind]=dict(**counts,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            matched_tiou=float(np.mean([m['tiou'] for m in matches])) if matches else None,
            onset_mae_s=float(np.mean([abs(m['onset_error_s']) for m in matches])) if matches else None,
            offset_mae_s=float(np.mean([abs(m['offset_error_s']) for m in matches])) if matches else None)
    return result
