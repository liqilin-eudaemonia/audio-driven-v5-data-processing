"""Apply the unchanged user-reviewed J policy to the other five existing sources."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import hashlib,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OLD=ROOT/'experiments/pair_boundary_real_audit_20261029';SEVEN=ROOT/'experiments/pair_head_seven_20261029';HOLD=ROOT/'experiments/pair_boundary_holdout3_20261029';TRIAL=ROOT/'experiments/pair_semantic_trial2_20261029'
sys.path[:0]=[str(OLD),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import listen_local_recovery_semantic_trial as repair
PAIRS=['improvised_V01_S0122_I00000125','naturalistic_V02_S5228_I00000302','naturalistic_V03_S2098_I00000195','improvised_V00_S0862_I00000794','naturalistic_V03_S1909_I00000205']
contract=json.loads((TRIAL/'results/contract.json').read_text())
assert hashlib.sha256(Path(repair.__file__).read_bytes()).hexdigest()==contract['repair_sha256']
for pair in PAIRS:
 target=EXP/(pair+'.json')
 if not target.exists():target.symlink_to(SEVEN/(pair+'.json'))
runner.EXP=EXP;runner.OUT=EXP/'results';runner.PAIRS=PAIRS;runner.repair=repair
runner.EXTRA_CACHE_DIRS=[OLD/'fixed_pairs',OLD/'gap5_pairs',OLD/'localbc_pairs',HOLD/'gap5',HOLD/'localbc',ROOT/'experiments/pair_head_attach5_20261029/results',SEVEN/'pilot_cache',SEVEN/'results',TRIAL/'results']
if __name__=='__main__':runner.main()
