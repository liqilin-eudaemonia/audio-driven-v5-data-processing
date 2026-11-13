"""Independent generated event annotations; no trained classifier as judge."""
import math
import random
from collections import defaultdict
from .contracts import EVENTS, atomic_json
from .annotations import union, intersection_length


def subtract(spans, holes):
    result=union(spans)
    for a,b in union(holes):
        next_=[]
        for x,y in result:
            if y<=a or x>=b: next_.append([x,y])
            else:
                if x<a: next_.append([x,a])
                if y>b: next_.append([b,y])
        result=next_
    return result


def iou(a,b):
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return inter/(a[1]-a[0]+b[1]-b[0]-inter)


def event_counts(label, predictions, name, threshold=.3):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    support=subtract(label["coverage"][name],label["unknown"][name])
    refs=[e for e in label["events"] if e["type"]==name]
    excluded=[]
    for e in refs:
        if e.get("boundary_censored",False) or intersection_length(support,e["start"],e["end"])<e["end"]-e["start"]-1e-6:
            excluded.append([e["start"],e["end"]])
    support=subtract(support,excluded)
    good=lambda s: intersection_length(support,*s)>=s[1]-s[0]-1e-6
    truth=[[e["start"],e["end"]] for e in refs if good([e["start"],e["end"]]) and not e.get("boundary_censored",False)]
    all_pred=[[e["start"],e["end"]] for e in predictions if e["type"]==name]
    if any(not all(math.isfinite(x) for x in s) or not 0<=s[0]<s[1]<=label["duration_s"]+1e-6 for s in all_pred):
        raise ValueError("Predicted event outside real support")
    pred=[s for s in all_pred if good(s)]
    pairs=[]
    if truth and pred:
        matrix=np.array([[iou(a,b) for b in pred] for a in truth])
        reward=np.where(matrix>=threshold,matrix,0.)
        ri,pi=linear_sum_assignment(-reward)
        pairs=[(i,j) for i,j in zip(ri,pi) if reward[i,j]>0]
    negatives=subtract(support,truth)
    neg_seconds=sum(y-x for x,y in negatives)
    false_seconds=sum(intersection_length(negatives,x,y) for x,y in union(all_pred))
    return dict(tp=len(pairs),fp=len(pred)-len(pairs),fn=len(truth)-len(pairs),reference_count=len(truth),
                onset_error_sum=sum(abs(truth[i][0]-pred[j][0]) for i,j in pairs),
                tiou_sum=sum(iou(truth[i],pred[j]) for i,j in pairs),
                onset_hits_025=sum(any(abs(t[0]-p[0])<=.25 for p in pred) for t in truth),
                onset_hits_050=sum(any(abs(t[0]-p[0])<=.5 for p in pred) for t in truth),
                negative_seconds=neg_seconds,false_seconds=false_seconds,
                false_events=sum(intersection_length(negatives,*p)>0 and not any(iou(p,t)>0 for t in truth) for p in pred),
                evaluated_seconds=sum(y-x for x,y in support),excluded_reference=len(excluded),
                censored_predictions=len(all_pred)-len(pred))


def summarize(counts):
    tp,fp,fn=counts["tp"],counts["fp"],counts["fn"]
    return dict(counts,f1=2*tp/(2*tp+fp+fn) if tp+fn>0 else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                onset_mae=counts["onset_error_sum"]/tp if tp else None,tiou=counts["tiou_sum"]/tp if tp else None,
                false_occupancy=counts["false_seconds"]/counts["negative_seconds"] if counts["negative_seconds"] else None,
                false_events_per_minute=60*counts["false_events"]/counts["negative_seconds"] if counts["negative_seconds"] else None)


def total_metrics(per_clip):
    result={}
    for name in EVENTS:
        values=defaultdict(float)
        for row in per_clip:
            for k,v in row[name].items(): values[k]+=v
        result[name]=summarize(values)
    scores=[result[k]["f1"] for k in EVENTS]
    result["macro_f1"]=sum(scores)/len(scores) if all(x is not None for x in scores) else None
    result["failure_rate"]=sum(r["failure_rate"] for r in per_clip)/len(per_clip)
    return result


def evaluate(protocol, observations, labels, predictions):
    """Require the complete locked matrix; missing outputs != silently discarded failures."""
    lookup={r["sample_id"]:r for r in observations}
    refs={r["sample_id"]:r for r in labels}
    ids=protocol["sample_ids"]; seeds=protocol["sample_seeds"]; trains=protocol["train_seeds"]
    if len(ids)!=len(set(ids)) or not ids or len(seeds)!=len(set(seeds)) or not seeds:
        raise ValueError("Invalid evaluation matrix")
    if not trains or len(trains)!=len(set(trains)) or not 0<protocol.get("tiou_threshold",.3)<=1:
        raise ValueError("Invalid training seeds or IoU threshold")
    if protocol["arms"]!=["E0","E1"]: raise ValueError("Paired E0/E1 required")
    expected={(t,a,s,i) for t in trains for a in protocol["arms"] for s in seeds for i in ids}
    got={}
    for r in predictions:
        key=(r["train_seed"],r["arm"],r["sample_seed"],r["sample_id"])
        if key in got or key not in expected: raise ValueError("Duplicate/unexpected generation")
        if r["status"] not in ("success","failed"): raise ValueError("Unjudged outputs must be resolved before main scoring")
        if r["status"]=="success" and not r.get("independent_verified",False):
            raise ValueError("Successful generation needs independently verified event annotations")
        if r["status"]=="failed" and r.get("events"): raise ValueError("Failed output must have no events")
        if any(e.get("type") not in EVENTS for e in r.get("events",[])):
            raise ValueError("Unknown prediction event type")
        got[key]=r
    if set(got)!=expected: raise ValueError(f"Missing {len(expected-set(got))} planned generation records")
    rows=[];results={}
    for t in trains:
        arm_rows={}
        for arm in protocol["arms"]:
            clips=[]
            for sid in ids:
                obs=lookup[sid];label=refs[sid]
                from .annotations import validate
                validate(label,obs)
                item=dict(sample_id=sid,group_id=obs["group_id"],train_seed=t,arm=arm,failure_rate=0.)
                for name in EVENTS: item[name]=defaultdict(float)
                for seed in seeds:
                    r=got[(t,arm,seed,sid)]
                    item["failure_rate"]+=(r["status"]=="failed")/len(seeds)
                    for name in EVENTS:
                        counts=event_counts(label,r.get("events",[]) if r["status"]=="success" else [],name,protocol.get("tiou_threshold",.3))
                        for k,v in counts.items(): item[name][k]+=v/len(seeds)
                clips.append(item);rows.append(item)
            arm_rows[arm]=clips
        result={a:total_metrics(r) for a,r in arm_rows.items()}
        e0,e1=result["E0"]["macro_f1"],result["E1"]["macro_f1"]
        result["delta_macro_f1"]=None if e0 is None or e1 is None else e1-e0
        groups=sorted({r["group_id"] for r in arm_rows["E0"]})
        rng=random.Random(protocol.get("bootstrap_seed",20261023));deltas=[]
        for _ in range(protocol.get("bootstrap_repeats",2000)):
            selected=[rng.choice(groups) for _ in groups]
            score={a:total_metrics([r for group in selected for r in arm_rows[a] if r["group_id"]==group])["macro_f1"] for a in protocol["arms"]}
            if all(v is not None for v in score.values()): deltas.append(score["E1"]-score["E0"])
        if deltas:
            import numpy as np
            result["paired_group_ci95"]=np.quantile(deltas,[.025,.975]).tolist()
        else: result["paired_group_ci95"]=None
        result["bootstrap_valid_repeats"]=len(deltas);result["independent_groups"]=len(groups)
        results[str(t)]=result
    differences=[r["delta_macro_f1"] for r in results.values()]
    return dict(training_seeds=results,mean_delta_macro_f1=sum(differences)/len(differences) if all(x is not None for x in differences) else None,
                per_clip=rows,quality_assessed=False,note="Event reference consistency only; naturalness and role quality require blind review.")


def report(result,path):
    from pathlib import Path
    lines=["# P2 事件评测结果","","生成事件参考一致性；自然度、角色保持和音质仍需独立盲评。","",
           "| 训练 seed | E0 macro-F1 | E1 macro-F1 | 差值 | 组级 95% CI |","|---|---|---|---|---|"]
    for seed,r in result["training_seeds"].items():
        lines.append(f"| {seed} | {r['E0']['macro_f1']} | {r['E1']['macro_f1']} | {r['delta_macro_f1']} | {r['paired_group_ci95']} |")
    Path(path).write_text("\n".join(lines)+"\n")
