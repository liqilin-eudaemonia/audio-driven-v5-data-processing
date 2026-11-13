"""Run either existing rule on the fixed holdout sample, without changing it."""
import ctypes
# Avoid host transparent-hugepage allocation stalls; numerical code is unchanged.
assert ctypes.CDLL(None,use_errno=True).prctl(41,1,0,0,0)==0
import sys,faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90,repeat=True)
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'experiments/pair_boundary_real_audit_20261029'))
import rerun_fixed_pairs as run
import listen_local_recovery_gap5 as gap5
import listen_local_recovery_localbc as localbc
variant=sys.argv[1]
assert variant in ('gap5','localbc')
run.EXP=EXP;run.OUT=EXP/variant;run.repair=gap5 if variant=='gap5' else localbc
run.PAIRS=[r['pair'] for r in run.read(EXP/'selection.json')['items']]
run.EXTRA_CACHE_DIRS=[] if variant=='gap5' else [EXP/'gap5']
_original_context=run.context
def context(pair):
 ctx=_original_context(pair)
 d=run.read(EXP/(pair+'.json'))
 ctx.shots=d['shots'];ctx.shots_by_track=d['shots_by_track']
 ctx.regions,ctx.source_gaps=run.repair.role_regions(ctx.people,ctx.limit,ctx.shots)
 return ctx
run.context=context
if __name__=='__main__':run.main()
