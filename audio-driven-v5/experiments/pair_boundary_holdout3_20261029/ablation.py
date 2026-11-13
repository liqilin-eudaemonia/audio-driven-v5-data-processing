"""Read-only causal attribution; neither variant nor QC results are modified."""
from pathlib import Path
import sys
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(EXP),str(ROOT/'experiments/pair_boundary_real_audit_20261029')]
import compare,rerun_fixed_pairs as run
import listen_local_recovery_gap5 as old
import listen_local_recovery_localbc as new
run.EXP=EXP
pair='improvised_V00_S0862_I00000794';d=compare.read(EXP/(pair+'.json'))
plans={}
for name,mod in [('old',old),('classification_only',new),('ranking_only',new),('both',new)]:
 run.repair=mod;ctx=run.context(pair)
 if name=='classification_only':ctx.people=new.prepare_people(d['people'])
 if name=='ranking_only':ctx.people=d['people']
 ctx.regions,ctx.source_gaps=mod.role_regions(ctx.people,ctx.limit,d['shots'])
 original_priority=new.end_priority
 if name=='classification_only':new.end_priority=lambda kind,natural: 0 if kind is not None else 1 if natural else 2
 try:plans[name]=ctx.propose([])[0]
 finally:new.end_priority=original_priority
result={}
for name,rows in plans.items():
 result[name]=dict(around_okay=[{k:r[k] for k in ['start','end','text']} for r in rows if r['end']>212 and r['start']<222],comparison_to_old=compare.evaluate(d,plans['old'],rows))
compare.write(EXP/'ablation.json',result)
print({k:v['around_okay'] for k,v in result.items()})
