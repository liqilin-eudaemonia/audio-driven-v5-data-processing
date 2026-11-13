import sys,copy,json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path[:0]=[str(ROOT/'releases/p2_audio_postprocess_v1/src'),str(ROOT/'src')]
from p2_listen.config import load
from p2_listen.contracts import read,atomic_json
from p2_listen.preflight import metadata,get_audit,dependencies
from p2_listen.annotations import class_statistics
from p2_listen.training_data import load_training_sample
reports=[]
for name in ['p2_train50h_audio_postprocess_20261026','p2_full_audio_postprocess_crop384_20261026']:
 prior=read(OUT/(name+'.json'));ids={r['sample_id'] for r in prior['rows']};source=ROOT/'experiments'/name/'view';view=OUT/(name+'_view');(view/'manifests').mkdir(parents=True,exist_ok=True);(view/'cache').mkdir(exist_ok=True);(view/'labels').mkdir(exist_ok=True)
 for path in ['audio_postprocess','labels/canonical','cache/input','cache/text','cache/events','cache/audio_postprocess']:
  dst=view/path
  if not dst.exists():dst.symlink_to(source/path,target_is_directory=True)
 for manifest in ['observations','targets']:
  if name.startswith('p2_full'):
   selected=[]
   with (source/'manifests'/(manifest+'.json')).open() as f:
    for line in f:
     line=line.strip().rstrip(',')
     if not line.startswith('{'):continue
     row=json.loads(line)
     if row['sample_id'] in ids:selected.append(row)
     if len(selected)==len(ids):break
  else:selected=[r for r in read(source/'manifests'/(manifest+'.json')) if r['sample_id'] in ids]
  assert len(selected)==32;atomic_json(view/'manifests'/(manifest+'.json'),selected,immutable=True)
 c=load(source/'config.json');c['output']=str(view);atomic_json(view/'config.json',c,immutable=True)
 meta=metadata(c);a,first=get_audit(c,meta);b,second=get_audit(c,meta);assert a==b and second['mode']=='reused'
 labels=[load_training_sample(c,o,meta['targets'][o['sample_id']])[4] for o in meta['rows']]
 assert a['statistics']==class_statistics(labels,c['events']['pos_weight_cap'],c['events']['K_train'])
 deps=dependencies(c,meta);assert sum('/cache/audio_postprocess/' in p for p in deps)==64;assert sum('/audio_postprocess/recipes/' in p for p in deps)==32;assert not any('/teacher/' in p for p in deps)
 reports.append(dict(dataset=name,samples=len(meta['rows']),first=first,second=second,statistics_match_actual_new_events=True,patch_and_recipe_dependencies_verified=True));atomic_json(OUT/'preflight_validation.json',dict(status='passed',datasets=reports));print(name,first['mode'],second['mode'],flush=True)
