import argparse
from pathlib import Path
import filter_full_response_sft as base
import recover_voiced_response_sft as voiced
p=argparse.ArgumentParser()
p.add_argument('action', choices=['worker','supervise','finalize'])
p.add_argument('--work',type=Path,required=True)
a=p.parse_args()
cfg=base.read(a.work/'config.json')
if cfg.get('policy')=='relaxed_original_voiced_recovery':
    voiced.configure_policy()
dict(worker=base.worker,supervise=base.supervise,finalize=base.finalize)[a.action](a.work)
