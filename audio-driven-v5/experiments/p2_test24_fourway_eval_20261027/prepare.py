import sys,json,shutil,subprocess
from pathlib import Path
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from p2_listen.contracts import read,sha,atomic_json
from listen_audio_postprocess.recipes import read_listener_audio,implementation_identity
DATA=ROOT/'experiments/p2_test24_backchannel_face_20261027';OLD=ROOT/'experiments/p2_train24_fourway_eval_20261027';PY=ROOT.parent/'Omni-LTX-stage2/.venv/bin/python'
for f,h in read(DATA/'COMPLETE.json')['files'].items():assert sha(DATA/f)==h,f
obs=read(DATA/'manifests/observations.json');targets=read(DATA/'manifests/targets.json');exps=read(OLD/'experiments.json');assert len(obs)==len(targets)==24
atomic_json(OUT/'observations.json',obs,immutable=True);atomic_json(OUT/'targets.json',targets,immutable=True);atomic_json(OUT/'experiments.json',exps,immutable=True)
restores=read(OLD/'source_restoration.json')
for arm,items in restores.items():
 for original,item in items.items():assert sha(item['path'])==item['sha256']
atomic_json(OUT/'source_restoration.json',restores,immutable=True)
audit={}
for arm,v in exps.items():
 train=read(Path(v['config']['output'])/'manifests/observations.json');ids={x['sample_id'] for x in train};groups={g for x in train for g in [x['group_id'],*x['grouping_ids']]}
 overlaps=[o['sample_id'] for o in obs if o['sample_id'] in ids or groups.intersection([o['group_id'],*o['grouping_ids']])];assert not overlaps,(arm,overlaps)
 audit[arm]=dict(train_samples=len(train),sample_group_person_session_overlap=0,checkpoint=v['checkpoint'],checkpoint_sha256=v['sha256'])
atomic_json(OUT/'isolation_audit.json',audit,immutable=True)
processed=[];receipts=[];(OUT/'reference_audio').mkdir(exist_ok=True)
for t in targets:
 recipe=DATA/'audio_postprocess/recipes'/f'{t["sample_id"]}.json';wave,sr,meta=read_listener_audio(read(recipe));dest=OUT/'reference_audio'/f'{t["sample_id"]}.wav';sf.write(dest,wave,sr,subtype='FLOAT')
 r=dict(t,target_audio=str(dest),media_hashes=dict(t['media_hashes'],target_audio=sha(dest)));processed.append(r);receipts.append(dict(sample_id=t['sample_id'],recipe=str(recipe),recipe_file_sha256=sha(recipe),output=str(dest),output_sha256=sha(dest),**meta))
atomic_json(OUT/'processed_targets.json',processed,immutable=True);atomic_json(OUT/'audio_reference_provenance.json',dict(implementation=implementation_identity(),samples=receipts),immutable=True)
atomic_json(OUT/'sampling.json',dict(evaluation_split='held_out_test',sample_count=24,sample_ids=[o['sample_id'] for o in obs],groups=len({o['group_id'] for o in obs}),generation_seeds=[137,251],generation_jobs=192,seconds=sum(o['requested_duration_s'] for o in obs),dataset=str(DATA),selection='fixed user-selected event-rich large-face independent test subset; not natural distribution',labels_are_weak=True,human_ground_truth=False),immutable=True)
for arm,v in exps.items():subprocess.run([str(PY),str(OUT/'runner.py'),arm,'prepare-generation','--run',v['run'],'--observations',str(OUT/'observations.json'),'--targets',str(OUT/'targets.json'),'--output',str(OUT/arm/'generation')],check=True)
print('prepared four held-out protocols; zero identity overlap with all actual training runs',flush=True)
