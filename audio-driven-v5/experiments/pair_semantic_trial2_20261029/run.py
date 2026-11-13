"""Serial two-source semantic-integrity trial, using existing QC evidence exactly."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';SEVEN=ROOT/'experiments/pair_head_seven_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029'
sys.path[:0]=[str(OLD),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import listen_local_recovery_semantic_trial as repair
PAIRS=['naturalistic_V02_S3610_I00000088','naturalistic_V02_S3579_I00000175']
for pair in PAIRS:
 target=EXP/(pair+'.json')
 if not target.exists():target.symlink_to(SEVEN/(pair+'.json'))
runner.EXP=EXP;runner.OUT=EXP/'results';runner.PAIRS=PAIRS;runner.repair=repair
runner.EXTRA_CACHE_DIRS=[OLD/'fixed_pairs',OLD/'gap5_pairs',OLD/'localbc_pairs',HOLD/'gap5',HOLD/'localbc',ROOT/'experiments/pair_head_attach5_20261029/results',SEVEN/'pilot_cache',SEVEN/'results']
if __name__=='__main__':runner.main()
