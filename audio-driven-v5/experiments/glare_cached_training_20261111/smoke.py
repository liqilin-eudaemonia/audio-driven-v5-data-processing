import pathlib,json,sys,sqlite3,copy
ROOT=pathlib.Path('/private/liqi_workspace/project_code/audio-driven-v5')
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import glare_cache_cluster as c
from assemble_glare_training_caches import collect
from baseline_training.glare_cached import load_row,CachedFLOAT,save_checkpoint
from baseline_training.training import build,seed,rng,restore_rng,set_train
out=pathlib.Path(__file__).parent
cfg=c.config(ROOT/'experiments/glare_cache_28gpu_20261110/production_gpu_fer_config.json');m=c.load_run(cfg)
p=ROOT/'experiments/glare_current500h_prosody_subset_20261110/full/official_exports/1794354452308478039/retained_all.jsonl'
db=sqlite3.connect(str(pathlib.Path(m['parent'])/'jobs.sqlite'));db.row_factory=sqlite3.Row
values=[]
with p.open() as f:
 for _ in range(3):
  row=json.loads(next(f));job=dict(db.execute('SELECT * FROM jobs WHERE sid=?',(row['sample_id'],)).fetchone())
  row['glare_caches']=collect(cfg,m,job,row,False);values.append(load_row(row,True))
torch.set_num_threads(2);seed(137)
model,meta=build('glare','cuda:0');model.fmt.initialize_weights();wrap=CachedFLOAT(model)
params=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5)
results=[]
for v in values:
 set_train(model,'glare');opt.zero_grad(set_to_none=True)
 losses=wrap(v);losses['loss'].backward()
 groups={k:any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(k)) for k in ['fmt','audio_encoder.audio_projection','intensity_embedder','channel_projection','reaction_classifier']}
 torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
 results.append(dict(sample_id=v['sample_id'],frames=len(v['motion']),loss=float(losses['loss']),gradients=groups))
 print(results[-1],flush=True)
assert all(any(r['gradients'][k] for r in results) for k in results[0]['gradients']), results
saved=dict(trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},optimizer=copy.deepcopy(opt.state_dict()),rng=rng())
save_checkpoint(out/'smoke.pt',saved)
state=copy.deepcopy(saved['rng']);restore_rng(state)
expected=float(wrap(values[0])['loss'].detach())
loaded=torch.load(out/'smoke.pt',map_location='cpu',weights_only=False)
model.load_state_dict(loaded['trainable'],strict=False);opt.load_state_dict(loaded['optimizer']);restore_rng(loaded['rng'])
actual=float(wrap(values[0])['loss'].detach());assert expected==actual,(expected,actual)
result=dict(samples=results,checkpoint_reload_loss_exact=True,formal_training_started=False,scope='single GPU native forward/backward and in-process reload; not multi-rank/cold-process validation',implementation=meta)
(out/'SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',flush=True)
