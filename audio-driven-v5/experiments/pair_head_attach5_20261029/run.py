"""Run the approved prefix-attachment trial on all five existing sources."""
import ctypes
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import rerun_fixed_pairs as run
import listen_local_recovery_head_attach as repair
PAIRS=run.PAIRS+[r['pair'] for r in run.read(HOLD/'selection.json')['items']]
for p in PAIRS:
 target=EXP/(p+'.json')
 if not target.exists():target.symlink_to((OLD if p in run.PAIRS else HOLD)/(p+'.json'))
run.EXP=EXP;run.OUT=EXP/'results';run.PAIRS=PAIRS;run.repair=repair
run.EXTRA_CACHE_DIRS=[OLD/'fixed_pairs',OLD/'gap5_pairs',OLD/'localbc_pairs',HOLD/'gap5',HOLD/'localbc',EXP/'results_initial']
original_context=run.context
def context(pair):
 ctx=original_context(pair);d=run.read(EXP/(pair+'.json'))
 ctx.shots=d.get('shots',[]);ctx.shots_by_track=d.get('shots_by_track',[[],[]])
 ctx.regions,ctx.source_gaps=repair.role_regions(ctx.people,ctx.limit,ctx.shots)
 return ctx
run.context=context
if __name__=='__main__':run.main()
