import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'experiments/pair_boundary_holdout3_20261029'),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import listen_local_recovery_semantic_trial as repair
import listen_local_recovery_head_attach as head
import compare
runner.EXP=EXP
out=[]
for item in runner.read(EXP/'selection.json')['items']:
 pair=item['pair'];runner.repair=head;ctx=runner.context(pair);h,_=ctx.propose([])
 runner.repair=repair;ctx=runner.context(pair);j,diag=ctx.propose([])
 diff=compare.evaluate(runner.read(EXP/(pair+'.json')),h,j)
 cases=[dict(start=g['start'],end=g['end'],before=[{k:r[k] for k in ('start','end','text','semantic_start','semantic_end')} for r in g['before']],after=[{k:r[k] for k in ('start','end','text','semantic_start','semantic_end','sentence_end_protection','soft_pause_rebalance') if k in r} for r in g['after']]) for g in compare.changes(h,j)]
 out.append(dict(pair=pair,metrics=diff,cases=cases))
 print(pair,len(h),len(j),diff['unchanged'],'groups',diff['changed_groups'],flush=True)
runner.write(EXP/'initial_inspection.json',out)
