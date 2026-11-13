"""Same source snapshots and unchanged QC, H and then J, serial CPU execution."""
import ctypes
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import hashlib,json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
sys.path[:0]=[str(ROOT/'experiments/pair_boundary_real_audit_20261029'),str(ROOT/'scripts')]
import rerun_fixed_pairs as runner
import listen_local_recovery_semantic_trial as j
import listen_local_recovery_head_attach as h
contract=json.loads((ROOT/'experiments/pair_semantic_trial2_20261029/results/contract.json').read_text())
assert hashlib.sha256(Path(j.__file__).read_bytes()).hexdigest()==contract['repair_sha256']
variant=sys.argv[1];assert variant in ('h','j')
runner.EXP=EXP;runner.OUT=EXP/('h' if variant=='h' else 'results')
runner.PAIRS=[r['pair'] for r in json.loads((EXP/'selection.json').read_text())['items']]
runner.repair=h if variant=='h' else j
runner.EXTRA_CACHE_DIRS=[EXP/'h'] if variant=='j' else []
runner.main()
