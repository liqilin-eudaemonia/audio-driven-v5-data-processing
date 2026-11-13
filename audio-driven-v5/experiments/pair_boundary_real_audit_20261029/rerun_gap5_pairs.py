"""Apply the five-second pause rule to the same two sources only."""
import rerun_fixed_pairs as run
import listen_local_recovery_gap5 as gap5

if __name__=='__main__':
    run.EXTRA_CACHE_DIRS=[run.EXP/'fixed_pairs']
    run.OUT=run.EXP/'gap5_pairs'
    run.repair=gap5
    run.main()
