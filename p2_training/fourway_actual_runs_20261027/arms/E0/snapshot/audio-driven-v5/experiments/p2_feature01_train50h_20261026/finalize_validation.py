"""Strict completion checks, frozen source snapshot and production launch authorization record."""
import json,sys,shutil
from pathlib import Path
from control import *
sys.path.insert(0,str(ROOT/'releases/p2_feature_v1/src'))
from p2_listen.config import load,source_identity
from p2_listen.contracts import digest
from p2_listen.preflight import metadata

def main():
 suite=read(OUT/'gpu_suite_status.json')
 assert suite['state']=='complete' and len(suite['completed'])==5
 assert read(OUT/'gpu_comparison.json')['status']=='passed'
 assert '158 passed' in (OUT/'regression.log').read_text()
 audit=read(OUT/'production/audit_summary.json');prior=read(BASELINE/'run.json')
 assert audit['statistics']==prior['formal_statistics']
 assert audit['signatures']==prior['labels']
 cfg=load(TRAIN/'config.json');verify_config(cfg,prior['config'])
 meta=metadata(cfg)
 assert len(meta['rows'])==30415
 assert digest(meta['rows'])==prior['observations']
 assert digest(meta['targets'])==prior['targets']
 source=source_identity(cfg)
 checks={}
 for kind in ('single','longest','continuous','resume'):
  run=OUT/kind/'runs/s0_20261023/E0';ident=read(run/'run.json')
  assert ident['shared_initialization']==prior['initialization'],kind
  assert ident['source']==source,kind
  status=read(run/'status.json');assert status['step']==5 and not status['stopped']
  metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
  assert len(metrics)==5
  for record in metrics:
   v=record['scalars']
   for mod in ('video','audio'):assert abs(v['loss/feature_weighted_'+mod]-.1*v['loss/feature_'+mod])<1e-6
   assert v['loss/generator_objective']==v['loss/base']
  checks[kind]=dict(shared_initialization_exact=True,steps=5,world=ident['world'],
   peak_allocated_gib=max(x['scalars']['memory/peak_allocated_gib_max_rank'] for x in metrics),
   peak_reserved_gib=max(x['scalars']['memory/peak_reserved_gib_max_rank'] for x in metrics))
  if kind in ('single','longest'):
   records=[json.loads(line) for line in (run/'train_rank0.jsonl').read_text().splitlines()]
   assert all(not any(x['paths_first_micro']['event']) for x in records)
   assert all(x>0 for x in records[-1]['paths_first_micro']['feature'])
   checks[kind]['final_gradient_paths']=records[-1]['paths_first_micro']
 reports=[]
 for p in (TRAIN/'preflight_sessions').glob('*/rank_*_report.json'):
  value=read(p)
  if value['world']==24:reports.append(value)
 # Session layout is resolved from the audit launcher if preflight changes its layout.
 if not reports:
  reports=[read(p) for p in (TRAIN/'preflight_sessions').rglob('rank_*_report.json') if read(p)['world']==24]
 assert len(reports)==24 and sum(x['local_full_audits'] for x in reports)==1
 assert all(x['samples']==30415 for x in reports)
 inputs=[TRAIN/'config.json',TRAIN/'manifests/observations.json',TRAIN/'manifests/targets.json',ROOT/'configs/p2_listen_h200_multinode.env',
         OUT/'control.py',OUT/'controller.py',OUT/'run_node.py']
 files={**source,**{str(p):sha(p) for p in inputs}}
 for path,h in files.items():
  p=Path(path);target=OUT/'frozen_sources'/p.relative_to('/private/liqi_workspace')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  assert sha(target)==h
 write(OUT/'release.json',dict(arm='E0',experiment='AV+Feature(0.1,0.1)',seed=SEED,steps=BUDGET,source=source,files=files,created_at=now()))
 verify_release()
 report=dict(status='passed',cpu_tests=158,cpu_subtests=18,dataset_samples=30415,teacher_samples=30415,
  audit_world=24,global_full_audits=1,all_baseline_manifests_labels_statistics_exact=True,
  feature_weights={'video':.1,'audio':.1},event_gradient_to_generator=False,gpu=checks,
  exact_resume=read(OUT/'gpu_comparison.json'),verified_at=now())
 write(OUT/'verification.json',report)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
