import sys,json,random,hashlib,shutil,subprocess,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
OUT=Path(__file__).resolve().parent;PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
runs={'E0':ROOT/'experiments/p2_preflight_fix_20261026/production/runs/20261023/E0','E1':ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/runs/20261023/E1','E0F':ROOT/'experiments/p2_feature01_train50h_20261026/production/runs/20261023/E0','E1F':ROOT/'experiments/p2_feature01_train50h_20261026/production/runs/20261023/E1'}
backup=ROOT.parent/'audio-driven-v5-data-processing/p2_training/av_event_lora_tensorboard_20261026/snapshot/audio-driven-v5';restores={};identities={}
for name,run in runs.items():
 d=read(run/'run.json');identities[name]=d;restores[name]={}
 for original,h in d['source'].items():
  if Path(original).exists() and sha(original)==h:continue
  relative=Path(original).relative_to(ROOT);source=backup/relative
  assert source.is_file() and sha(source)==h,original
  dest=OUT/'restored'/name/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
  restores[name][original]={'path':str(dest),'sha256':h}
assert len({d['observations'] for d in identities.values()})==1
base=Path(identities['E0']['config']['output']);observations=read(base/'manifests/observations.json');targets=read(base/'manifests/targets.json');byid={t['sample_id']:t for t in targets}
selected=random.Random(20261027).sample(sorted(observations,key=lambda o:o['sample_id']),24)
counts=collections.Counter();durations=[]
for o in selected:
 label=read(base/'labels/canonical'/f'{o["sample_id"]}.json');counts.update({e['type'] for e in label['events']});durations.append(o['requested_duration_s'])
atomic_json(OUT/'observations.json',selected,immutable=True);atomic_json(OUT/'targets.json',[byid[o['sample_id']] for o in selected],immutable=True)
atomic_json(OUT/'source_restoration.json',restores,immutable=True)
manifest={name:{'run':str(run),'checkpoint':str(run/'step_001268.pt'),'sha256':read(run/'step_001268.complete.json')['sha256'],'config':identities[name]['config'],'initialization':identities[name]['initialization'],'shared_initialization':identities[name].get('shared_initialization')} for name,run in runs.items()}
atomic_json(OUT/'experiments.json',manifest,immutable=True)
atomic_json(OUT/'sampling.json',{'seed':20261027,'population':len(observations),'sample_count':24,'method':'Python Random.sample over sample_id sorted original common training observations, without replacement; no event enrichment','generation_seeds':[137,251],'generation_jobs':192,'sample_ids':[o['sample_id'] for o in selected],'seconds':sum(durations),'class_sample_counts':dict(counts),'evaluation_split':'train_diagnostic','training_data_postprocess':'original training media; full-audio-postprocess targets are NOT substituted'},immutable=True)
for name,run in runs.items():
 subprocess.run([str(PY),str(OUT/'runner.py'),name,'prepare-generation','--run',str(run),'--observations',str(OUT/'observations.json'),'--targets',str(OUT/'targets.json'),'--output',str(OUT/name/'generation'),'--train-diagnostic'],check=True)
print(json.dumps(read(OUT/'sampling.json'),indent=2))
