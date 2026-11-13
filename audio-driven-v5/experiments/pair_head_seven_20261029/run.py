"""Apply the unchanged final H rule to the seven explicitly requested sources."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import sys,json,shutil
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029';HEAD=ROOT/'experiments/pair_head_attach5_20261029'
sys.path[:0]=[str(OLD),str(HOLD)]
import rerun_fixed_pairs as run
import listen_local_recovery_head_attach as repair
PAIRS=['naturalistic_V02_S3610_I00000088','improvised_V01_S0122_I00000125','naturalistic_V02_S5228_I00000302','naturalistic_V03_S2098_I00000195','improvised_V00_S0862_I00000794','naturalistic_V02_S3579_I00000175','naturalistic_V03_S1909_I00000205']
BASELINE=EXP/'baseline_views';BASELINE.mkdir(exist_ok=True)
for pair in PAIRS:
 snapshot=(HOLD if pair in PAIRS[2:5] else OLD)/(pair+'.json')
 target=EXP/(pair+'.json')
 if not target.exists():target.symlink_to(snapshot)
 dest=BASELINE/(pair+'.json')
 if not dest.exists():shutil.copyfile(ROOT/'data_previews/seamless_listen_full_source_recovery'/pair/'review.json',dest)
for pair in PAIRS[:2]:
 pilot=run.read(ROOT/'experiments/p3_listen_full_source_recovery_20261023/results'/(pair+'.json'))
 for row in pilot['attempts']:
  dest=EXP/'pilot_cache'/pair/'qc'/(row['sample_id']+'.json')
  if not dest.exists():run.write(dest,row)
run.EXP=EXP;run.OUT=EXP/'results';run.PAIRS=PAIRS;run.repair=repair
run.EXTRA_CACHE_DIRS=[OLD/'fixed_pairs',OLD/'gap5_pairs',OLD/'localbc_pairs',HOLD/'gap5',HOLD/'localbc',HEAD/'results',EXP/'pilot_cache']
original_context=run.context
def context(pair):
 ctx=original_context(pair);d=run.read(EXP/(pair+'.json'))
 ctx.shots=d.get('shots',[]);ctx.shots_by_track=d.get('shots_by_track',[[],[]])
 ctx.regions,ctx.source_gaps=repair.role_regions(ctx.people,ctx.limit,ctx.shots)
 return ctx
run.context=context
if __name__=='__main__':run.main()
