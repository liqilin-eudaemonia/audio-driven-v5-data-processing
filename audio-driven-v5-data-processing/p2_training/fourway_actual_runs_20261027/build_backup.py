"""Build four checkpoint-matched source snapshots; never copy model/media/cache data."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent
REPO=PROJECT/'audio-driven-v5-data-processing';BASE=REPO/'p2_training/av_event_lora_tensorboard_20261026'
DEST=REPO/'p2_training/fourway_actual_runs_20261027';OUT=Path(__file__).resolve().parent
ARMS=['E0','E1','E0F','E1F']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
old=read(BASE/'BACKUP_MANIFEST.json');known={}
for entry in old['files']:
 p=BASE/entry['path'];assert sha(p)==entry['sha256'],str(p)
 known.setdefault(entry['sha256'],p)
external=[e for e in old['files'] if e['path'].startswith('snapshot/Omni-LTX-stage2/')]
external_changed=[e['source'] for e in external if not Path(e['source']).is_file() or sha(e['source'])!=e['sha256']]
assert not external_changed,external_changed
exps=read(ROOT/'experiments/p2_train24_fourway_eval_20261027/experiments.json')
restored=read(ROOT/'experiments/p2_train24_fourway_eval_20261027/source_restoration.json')
existing_index={}
for name in subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0'):
 p=REPO/name
 if name and p.is_file() and p.suffix=='.py':existing_index.setdefault(sha(p),[]).append(name)
assert not DEST.exists(),'Choose fresh destination rather than overwrite a published backup'
DEST.mkdir(parents=True)
write(DEST/'evidence/experiments.json',exps)
write(DEST/'evidence/prior_backup_audit.json',{a:dict(expected=len(read(Path(exps[a]['run'])/'run.json')['source']),matches=sum(h in existing_index for h in read(Path(exps[a]['run'])/'run.json')['source'].values()),missing=[p for p,h in read(Path(exps[a]['run'])/'run.json')['source'].items() if h not in existing_index]) for a in ARMS})
write(DEST/'evidence/external_source_check.json',dict(checked=len(external),all_match=True,baseline_backup=str(BASE),baseline_commit='4f877175d8f8f510633455b078aeeb97125adf29'))
entries=[]
for arm in ARMS:
 e=exps[arm];run=read(Path(e['run'])/'run.json');snapshot=DEST/'arms'/arm/'snapshot'
 shutil.copytree(BASE/'snapshot',snapshot)
 v5=snapshot/'audio-driven-v5'
 release='p2_feature_v1' if arm.endswith('F') else ('p2_preflight_v1' if arm=='E0' else None)
 if release:
  for folder in [release,*( ['p2_preflight_v1'] if arm.endswith('F') else [])]:
   for p in (ROOT/'releases'/folder).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md','.sh'):
     target=v5/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 bindings=[]
 for name,h in run['source'].items():
  original=Path(name);candidates=[original]
  for values in restored.values():
   if name in values:candidates.append(Path(values[name]['path']))
  if h in known:candidates.append(known[h])
  source=next((p for p in candidates if p.is_file() and sha(p)==h),None)
  if source is None:raise RuntimeError('Missing original source '+name)
  target=snapshot/original.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  bindings.append(dict(original=name,sha256=h,backup=str(target.relative_to(DEST)),copied_from=str(source)))
 evidence=DEST/'arms'/arm/'evidence';write(evidence/'run.json',run);write(evidence/'config.json',run['config'])
 marker=Path(e['checkpoint']).with_suffix('.complete.json');assert read(marker)['sha256']==e['sha256']
 shutil.copyfile(marker,evidence/'step_001268.complete.json')
 write(evidence/'checkpoint_reference.json',dict(path=e['checkpoint'],sha256=e['sha256'],checkpoint_included=False,marker_matches_evaluation_manifest=True))
 write(evidence/'source_bindings.json',bindings)
 # Preserve execution/validation scripts and lightweight config receipts, excluding SSH and media paths as files.
 names=['p2_e1_train50h_lora_tb_20261026','p2_preflight_fix_20261026','p2_feature01_train50h_20261026','p2_e1_feature01_train50h_20261026']
 for name in names:
  folder=ROOT/'experiments'/name
  for p in folder.glob('*'):
   if p.is_file() and p.suffix in ('.py','.json','.md') and p.stat().st_size<4_000_000:
    target=v5/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
  for p in (folder/'tests').glob('*.py'):
   target=v5/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
  config=folder/'production/config.json'
  if config.exists():
   target=v5/config.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(config,target)
 # Portable copies for CPU-only checks; exact production config stays in evidence/config.json and run.json.
 derived=[]
 for path in (v5/'experiments').rglob('*.json'):
  if path.name!='config.json':continue
  value=read(path)
  if isinstance(value,dict) and 'reuse_root' in value:
   before=value['reuse_root'];value['reuse_root']='../audio-driven-v4';write(path,value)
   derived.append(dict(path=str(path.relative_to(DEST)),field='reuse_root',original=before,portable='../audio-driven-v4'))
 write(evidence/'portable_config_changes.json',derived)
 for name in ['P2_显式回应规划双通路实验方案草案.md','P2_OmniResponse评测实现与运行.md','P2_GLARE四项反应评测实现与运行.md']:
  p=ROOT/'related'/name
  if p.exists():shutil.copyfile(p,v5/'related'/name)
 shutil.copyfile(ROOT/'scripts/p2_eval_generate.py',v5/'scripts/p2_eval_generate.py')
 cli=v5/('releases/'+release+'/scripts/p2_listen.py' if release else 'scripts/p2_listen.py')
 entries.append(dict(arm=arm,label={'E0':'AV','E1':'AV+Event','E0F':'AV+Feature','E1F':'AV+Event+Feature'}[arm],release=release,source_count=len(bindings),run=e['run'],checkpoint=e['checkpoint'],checkpoint_sha256=e['sha256'],cli=str(cli.relative_to(DEST))))
write(DEST/'EXPERIMENTS.json',entries)
write(OUT/'build_result.json',dict(backup=str(DEST),arms=entries,external_files_verified=len(external)))
print(json.dumps(dict(backup=str(DEST),source_counts={x['arm']:x['source_count'] for x in entries},external_files_verified=len(external))))
