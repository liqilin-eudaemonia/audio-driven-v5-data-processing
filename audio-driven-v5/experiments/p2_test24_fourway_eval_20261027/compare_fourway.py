"""Four-arm common-support statistics; no detector failure converted to a negative."""
import sys,json,collections,random,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json
from p2_omni_eval.scoring import aggregate
from p2_omni_eval.compare import _supported
arms=['E0','E1','E0F','E1F'];obs=read(OUT/'observations.json');groups={o['sample_id']:o['group_id'] for o in obs};seeds=[137,251]
protocol=read(OUT/'E0/omni/protocol.json');rows={a:{(r['sample_id'],r['sample_seed']):r for r in read(OUT/a/'omni/results.json')['rows']} for a in arms};result={}
for metric in protocol['metrics']:
 valid={key for key in rows['E0'] if all(_supported(rows[a][key]['metrics'][metric]) for a in arms)}
 # Same clips across both generation seeds and all four arms.
 ids=sorted({sid for sid,_ in valid if all((sid,s) in valid for s in seeds)})
 values={a:{} for a in arms}
 for a in arms:
  if ids:
   p=dict(protocol,metrics=[metric],sample_count=len(ids));score=aggregate(p,[rows[a][(sid,s)] for sid in ids for s in seeds]);values[a]=score['per_seed']
 result[metric]=dict(common_ids=ids,samples_per_seed=len(ids),planned_samples=24,common_groups=len({groups[sid] for sid in ids}),values=values)
 if metric in ('utmos','sync') and ids:
  # Pairwise group bootstrap: resample connected groups, retain paired arms and both seeds.
  gids=sorted({groups[sid] for sid in ids});bygroup={g:[sid for sid in ids if groups[sid]==g] for g in gids};rng=random.Random(20261027);draws=[rng.choices(gids,k=len(gids)) for _ in range(2000)]
  fields=['utmos'] if metric=='utmos' else ['lse_d','lse_c'];pairs=[('E0','E1'),('E0F','E1F'),('E0','E0F'),('E1','E1F')];boots={}
  def mean(a,chosen,field):
   vals=[]
   for s in seeds:
    items=[]
    for sid in chosen:
     v=rows[a][(sid,s)]['metrics'][metric]['generated']
     if metric=='utmos':items.append(v['value'])
     else:items.extend(t[field] for t in v['tracks'] if t['status']=='success')
    vals.append(float(np.mean(items)))
   return float(np.mean(vals))
  for left,right in pairs:
   boots[left+'→'+right]={}
   for field in fields:
    deltas=[mean(right,[sid for g in draw for sid in bygroup[g]],field)-mean(left,[sid for g in draw for sid in bygroup[g]],field) for draw in draws]
    boots[left+'→'+right][field]=dict(delta=mean(right,ids,field)-mean(left,ids,field),ci95=np.quantile(deltas,[.025,.975]).tolist(),bootstrap_groups=len(gids),replicates=2000)
  result[metric]['paired_group_bootstrap']=boots
# GLARE shared scope uses its own explicit support and public 0.5 matching protocol.
if (OUT/'glare_common_scope.json').exists():
 scope=read(OUT/'glare_common_scope.json');ids=scope['sample_ids'];scores={a:read(OUT/a/'glare_common.json') for a in arms} if ids else {}
 result['glare']=dict(scope=scope,models={a:v['aggregation'] for a,v in scores.items()})
 if ids:
  from p2_glare_eval.upstream import REACTIONS
  counts={a:{} for a in arms}
  for a in arms:
   for s in seeds:
    for clip in scores[a]['per_seed'][str(s)]['per_clip']:
     c={k:dict(tp=0,fp=0,fn=0) for k in REACTIONS}
     for k in REACTIONS:
      tp=sum(m['reaction']==k for m in clip['matches']);c[k]=dict(tp=tp,fp=sum(e['reaction']==k for e in clip['generated_events'])-tp,fn=sum(e['reaction']==k for e in clip['gt_events'])-tp)
     counts[a][(clip['sample_id'],s)]=c
  def macro(a,chosen):
   f=[]
   for s in seeds:
    per=[]
    for k in REACTIONS:
     v={key:sum(counts[a][(sid,s)][k][key] for sid in chosen) for key in ['tp','fp','fn']};den=2*v['tp']+v['fp']+v['fn'];per.append(2*v['tp']/den if den else 0.)
    f.append(float(np.mean(per)))
   return float(np.mean(f))
  gids=sorted({groups[sid] for sid in ids});bg={g:[sid for sid in ids if groups[sid]==g] for g in gids};rng=random.Random(20261027);draws=[[sid for g in rng.choices(gids,k=len(gids)) for sid in bg[g]] for _ in range(2000)]
  result['glare']['paired_group_bootstrap']={left+'→'+right:dict(delta=macro(right,ids)-macro(left,ids),ci95=np.quantile([macro(right,d)-macro(left,d) for d in draws],[.025,.975]).tolist(),bootstrap_groups=len(gids),replicates=2000) for left,right in [('E0','E1'),('E0F','E1F'),('E0','E0F'),('E1','E1F')]}
atomic_json(OUT/'comparison.json',dict(evaluation_split='held_out_test',metrics=result));print(json.dumps({k:{x:v[x] for x in ['samples_per_seed','common_groups'] if x in v} for k,v in result.items()}),flush=True)
