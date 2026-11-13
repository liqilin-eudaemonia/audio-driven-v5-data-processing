"""Validate the final frozen real-CPU run, reuse, comparison, and untouched trainer."""
import json,subprocess,sys,time,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
sys.path.insert(0,str(ROOT/'src'))
from p2_omni_eval.common import read,atomic_json,sha,code_identity,digest
work=OUT/'verified_end_to_end'
while not (work/'results.json').exists():time.sleep(5)
p=read(work/'protocol.json');r=read(work/'results.json')
assert p['code']==code_identity() and r['protocol_sha256']==digest(p) and r['status']=='complete'
assert r['per_seed']['137']['fvd']['full_value']==0
assert abs(r['per_seed']['137']['fd']['full_value'])<1e-8
for metric in ('utmos','sync'):
 assert r['per_seed']['137'][metric]['gt']['full_value']==r['per_seed']['137'][metric]['generated']['full_value']
for row in r['rows']:
 for side in ('gt','generated'):
  assert row['metrics']['sync'][side]['status']=='success'
  assert all(t['aligned_frames']==row['geometry']['frame_count']==50 for t in row['metrics']['sync'][side]['tracks'])
assert r['per_seed']['137']['text']['full_value']['bertscore_f1']>.99
before={str(f):dict(mtime_ns=f.stat().st_mtime_ns,sha256=sha(f)) for f in work.glob('rows/*/complete.json')}
start=time.monotonic()
with (OUT/'verified_reuse.log').open('w') as log:
 subprocess.run([str(PY),str(ROOT/'scripts/p2_omni_eval.py'),'evaluate','--work',str(work),'--device','cpu'],stdout=log,stderr=subprocess.STDOUT,check=True)
reuse_seconds=time.monotonic()-start
assert read(work/'results.json')==r
assert before=={str(f):dict(mtime_ns=f.stat().st_mtime_ns,sha256=sha(f)) for f in work.glob('rows/*/complete.json')}
with (OUT/'verified_compare.log').open('w') as log:
 subprocess.run([str(PY),str(ROOT/'scripts/p2_omni_eval.py'),'compare','--left',str(work),'--right',str(work),'--output',str(OUT/'verified_comparison.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
c=read(OUT/'verified_comparison.json')
assert c['status']=='compared' and all(v['137']['common_count']==2 and v['137']['left']==v['137']['right'] for v in c['metrics'].values())
sys.path.insert(0,str(ROOT/'experiments/p2_preflight_fix_20261026'))
from control import verify_release
verify_release();training=read(ROOT/'experiments/p2_preflight_fix_20261026/status.json')
assert training['state'] in ('training','complete')
assert '71 passed' in (OUT/'regression.log').read_text()
report=dict(status='passed',tests=71,real_cpu_metrics=['I3D FVD-32','UTMOSv2','S3FD/SyncNet LSE-D/LSE-C','METEOR','ROUGE-L','RoBERTa BERTScore','corpus Distinct'],
 fd='reference formula tested with known synthetic T x C features',samples=2,formal_evaluation=False,
 self_fvd=r['per_seed']['137']['fvd'],self_fd=r['per_seed']['137']['fd']['full_value'],
 gt_generated_audio_metrics_exactly_equal=True,sync_preserves_original_50_frames=True,
 reused_receipts_unchanged=True,reused_results_identical=True,reuse_seconds=reuse_seconds,
 common_support_comparison_verified=True,original_training_release_unchanged=True,
 training_state=training['state'],training_step=training.get('step'),code=code_identity(),runtime_sha256=sha(ROOT/'configs/p2_omni_runtime.lock.json'))
atomic_json(OUT/'validation.json',report);print(json.dumps(report,indent=2))
