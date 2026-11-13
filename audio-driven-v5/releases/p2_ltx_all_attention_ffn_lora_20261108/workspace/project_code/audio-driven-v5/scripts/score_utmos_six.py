"""UTMOSv2 six-case screening probe; original listener inputs, no data exclusion."""
import os,json,time,sys,hashlib,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/listen_utmos6_20261026';SOURCE=ROOT/'experiments/listen_audio_cleanup_6_20261026'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['UTMOSV2_CHACHE']=str(RUN/'model_cache');os.environ.setdefault('OMP_NUM_THREADS','4')
sys.path.insert(0,'/private/liqi_workspace/project_code/UTMOSv2')
import numpy as np,torch,soundfile as sf
import utmosv2

torch.set_num_threads(4);torch.set_num_interop_threads(2)
cases=json.loads((SOURCE/'cases.json').read_text());folder=RUN/'inputs';folder.mkdir(parents=True,exist_ok=True)
for c in cases:
 p=SOURCE/'cases'/c['id']/'original.wav';link=folder/(c['id']+'.wav')
 if not link.exists():link.symlink_to(p)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==c['source_hashes']['original.wav']
print('loading fold0 fusion_stage3 CPU',flush=True);started=time.time()
model=utmosv2.create_model(pretrained=True,fold=0,device='cpu')
print('loaded',time.time()-started,flush=True)
scores={c['id']:[] for c in cases};seeds=[42,43,44,45,46]
for seed in seeds:
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 result=model.predict(input_dir=str(folder),device='cpu',batch_size=2,num_workers=0,num_repetitions=1,remove_silent_section=False,predict_dataset='sarulab',verbose=False)
 for x in result:scores[Path(x['file_path']).stem].append(float(x['predicted_mos']))
 print('seed',seed,'scores',result,flush=True)
 (RUN/'partial.json').write_text(json.dumps(scores,indent=2))
rows=[]
for c in cases:
 v=scores[c['id']];rows.append(dict(sample_id=c['id'],short_id=c['id'].split('_')[2],human_group=c['user_feedback'],duration_s=c['duration'],mos_mean=float(np.mean(v)),crop_sd=float(np.std(v,ddof=1)),mos_each_repeat=v,source_sha256=c['source_hashes']['original.wav']))
noisy=[r['mos_mean'] for r in rows if r['human_group']=='noisy'];clean=[r['mos_mean'] for r in rows if r['human_group']=='clean_control'];wins=sum((a>b)+.5*(a==b) for a in clean for b in noisy)
summary=dict(clean_mean=float(np.mean(clean)),noisy_mean=float(np.mean(noisy)),clean_vs_noisy_pairwise_concordance=wins/8,threshold_perfect_separation=min(clean)>max(noisy),clean_min=min(clean),noisy_max=max(noisy))
out=dict(schema='utmos-six-probe-v1',configuration=dict(model='fusion_stage3',fold=0,predict_dataset='sarulab',seeds=seeds,repetitions=5,remove_silent_section=False,device='cpu',batch_size=2,internal_sr=16000),elapsed_s=time.time()-started,rows=rows,summary=summary,training_data_modified=False)
(RUN/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
import csv
with (RUN/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['sample_id','human_group','duration_s','mos_mean','crop_sd']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
