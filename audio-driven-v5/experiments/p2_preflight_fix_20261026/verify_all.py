"""Require full data, distributed reuse and exact GPU evidence before launch."""
import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'releases/p2_preflight_v1/src'))
from p2_listen.config import load,activate,source_identity
from p2_listen.preflight import metadata
from p2_listen.contracts import digest,read,atomic_json,sha
from control import approved_source
c=load(OUT/'production/config.json');activate(c)
e1=read(ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1/run.json')
meta=metadata(c);summary=read(OUT/'production/audit_summary.json')
assert summary['signatures']==e1['labels']
assert summary['statistics']==e1['formal_statistics']
assert digest(meta['rows'])==e1['observations'] and digest(meta['targets'])==e1['targets']
assert len(meta['rows'])==30415
assert source_identity(c)==approved_source(e1['source'])
gpu=read(OUT/'gpu_comparison.json')
assert len(gpu)==3 and all(v['status']=='passed' and v['bitwise_equal'] for v in gpu.values())
# Bind GPU evidence to current overlay source, not an older test result.
for kind,arm in [('fast','E0'),('fast','E1'),('resume','E1')]:
 r=read(OUT/kind/'runs/s0_20261023'/arm/'run.json')
 assert r['source']==source_identity(c)
source=read(OUT/'source_change_verification.json');assert source['status']=='passed'
assert all(sha(p)==h for p,h in source['overrides'].items())
assert '167 passed' in (OUT/'regression.log').read_text()
assert '14 passed' in (OUT/'control_tests.log').read_text()
audits=[]
for job in (OUT/'audit_jobs').iterdir():
 if not (job/'status.json').exists():continue
 status=read(job/'status.json');assert status['returncodes']==[0,0,0]
 folder=OUT/'production/preflight_sessions'/status['session']
 reports=[read(folder/f'rank_{rank}_report.json') for rank in range(24)]
 assert all(r['samples']==30415 and r['statistics_sha256']==digest(summary['statistics']) for r in reports)
 assert [r['rank'] for r in reports]==list(range(24)) and all(r['world']==24 for r in reports)
 response=read(folder/'result.json');assert response['requests'][0]['binding']==digest(meta['binding'])
 assert response['result']['statistics']==summary['statistics']
 assert response['result']['signatures']==summary['signatures']
 audit=reports[0]['global_audit'];assert sum(r['local_full_audits'] for r in reports)==audit['full_audits']
 audits.append(dict(session=status['session'],cluster_seconds=status['seconds'],rank0_seconds=reports[0]['total_seconds'],**audit))
assert any(a['mode']=='full' and a['full_audits']==1 for a in audits)
assert any(a['mode']=='reused' and a['full_audits']==0 for a in audits)
report=dict(status='passed',cpu_tests=167,control_tests=14,cpu_subtests=18,source_ast_verified=True,
 original_E1_source_files_preserved=70,gpu_comparisons=gpu,samples=30415,
 observations_targets_labels_statistics_exactly_match_E1=True,distributed_full_and_reuse_audits=audits)
atomic_json(OUT/'verification.json',report)
print(json.dumps(report,indent=2))
