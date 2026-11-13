"""Try local-feedback classification and sentence-end priority on two sources."""
import rerun_fixed_pairs as run
import listen_local_recovery_localbc as localbc
if __name__=='__main__':
    run.EXTRA_CACHE_DIRS=[run.EXP/'fixed_pairs',run.EXP/'gap5_pairs']
    run.OUT=run.EXP/'localbc_pairs'
    run.repair=localbc
    run.main()
