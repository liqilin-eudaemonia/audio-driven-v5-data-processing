"""CPU audit of existing public-model outputs; no new generation or AF access."""
import csv, hashlib, json, sys, platform
from pathlib import Path
import numpy as np
import scipy, sklearn
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'compare/deployment'))
from vico_literature_metrics import evaluate, parity

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def variance(a):
    x=np.concatenate(a)
    return {g:float(np.var(x[:,s],axis=0,ddof=0).mean()) for g,s in [('exp',slice(6,56)),('pose',slice(0,6))]}
def score(g,p,s,space):
    row=evaluate(g,p,s)
    row['released_flatten_variance']=row['metrics']['Var'].copy()
    row['released_flatten_variance_gt']=row['metrics']['Var_GT'].copy()
    row['metrics']['Var']=variance(p);row['metrics']['Var_GT']=variance(g)
    row['units']=space
    row['aggregation']='FD/P-FD/MSE: equal clips; SID: GT KMeans40/20; Var: pooled per-coefficient ddof0 mean; rPCC: pooled flattened correlation difference'
    return row

base=ROOT/'experiments/p2_public_weights_vico_20261110'
p=read(base/'protocol.json');old=read(base/'RESULTS.json')
records=p['reused_feature_records']+read(base/'EXTRACTION_COMPLETE.json')['records']
features={};files={};checks=[]
for r in records:
    if r['role'] not in ['gt','speaker','dystream']:continue
    path=Path(r['path']);actual=sha(path);assert actual==r['sha256'],path
    with np.load(path) as z:
        a=z['motion'].copy();assert a.shape==(r['frames'],56)
        assert np.array_equal(z['valid'],np.isfinite(a).all(1))
    features[r['id'],r['role']]=a;files[str(path)]=actual
assert len(features)==129
ids=p['ids'];assert len(ids)==43
byid={x['id']:x for x in old['coverage']}
arrays={r:[] for r in ['gt','speaker','dystream']}
full={r:[] for r in arrays}
coverage=[]
for uid in ids:
    idx=np.array(byid[uid]['listener_indices']);si=np.array(byid[uid]['speaker_indices']);assert np.array_equal(idx,si+1)
    for role in arrays:arrays[role].append(features[uid,role][si if role=='speaker' else idx])
    mask=np.isfinite(features[uid,'speaker'][:-1]).all(1)&np.isfinite(features[uid,'gt'][1:]).all(1)&np.isfinite(features[uid,'dystream'][1:]).all(1)
    for role in full:full[role].append(features[uid,role][:-1 if role=='speaker' else None][mask] if role=='speaker' else features[uid,role][1:][mask])
    coverage.append({'id':uid,'sealed_common_frames':len(idx),'dyadic_valid_frames':int(mask.sum())})
results={}
results['dystream_common']=score(arrays['gt'],arrays['dystream'],arrays['speaker'],'raw EMOCA pose6 then exp50')
errors=[]
for key in ['FD','P_FD','MSE','SID','Var','rPCC']:
    for group in ['exp','pose']:
        a=results['dystream_common']['metrics'][key][group];b=old['methods']['dystream']['metrics'][key][group]
        errors.append(abs(a-b))
assert max(errors)<1e-8,max(errors)
checks.append({'name':'dystream_vs_sealed_result_12_scalars','passed':True,'max_abs':max(errors)})
save('AUTHOR_FUNCTION_PARITY.json',parity(arrays['gt'],arrays['dystream'],arrays['speaker']))
results['dystream_pairwise']=score(full['gt'],full['dystream'],full['speaker'],'raw EMOCA pose6 then exp50')
selfcheck=score(full['gt'],full['gt'],full['speaker'],'raw EMOCA pose6 then exp50')
assert all(abs(v)<1e-5 for k in ['FD','P_FD','MSE','rPCC'] for v in selfcheck['metrics'][k].values())
checks.append({'name':'gt_selfcheck','passed':True})
save('GT_SELF_CHECK.json',selfcheck)
print('DyStream rescored; maximum sealed-result difference',max(errors),flush=True)

l2l=ROOT/'experiments/p2_compare_vico43_20261106/learning2listen'
lp=read(l2l/'protocol.json');assert {x['id'] for x in lp['samples']}==set(ids)
native={r:[] for r in ['gt','speaker']}
for role in native:
    for uid in ids:
        path=l2l/'features'/role/(uid+'.npz');files[str(path)]=sha(path)
        a=np.load(path)['motion'];assert a.shape==features[uid,'gt'].shape and np.isfinite(a).all()
        native[role].append(np.concatenate([a[:,50:],a[:,:50]],axis=1))
for name in ['conan','fallon','stephen','trevor']:
    preds=[]
    for uid in ids:
        path=l2l/'generated'/name/(uid+'.npz');receipt=read(path.with_suffix('.json'))
        assert receipt['feature_sha256']==sha(path);files[str(path)]=sha(path)
        a=np.load(path)['motion'];assert a.shape==features[uid,'gt'].shape and np.isfinite(a).all()
        preds.append(np.concatenate([a[:,50:],a[:,:50]],axis=1))
    g=[a[1:] for a in native['gt']];s=[a[:-1] for a in native['speaker']];pr=[a[1:] for a in preds]
    result=score(g,pr,s,'NATIVE DECA: head3+jaw3 then exp50; NOT EMOCA')
    result['comparable_to_dim_table1']=False
    result['reason']='Original person-specific L2L checkpoint, native DECA output; DIM retrained L2L on ViCo in EMOCA. Same metric formulas do not imply same representation.'
    results['l2l_'+name+'_native']=result
    print('L2L native metric diagnostic complete:',name,flush=True)

save('RESULTS.json',{'schema':'vico-report-alignment-audit-v1','new_generation':False,'new_cpu_rescoring':True,'avatar_forcing_evaluated':False,'methods':results,'coverage':coverage,'paper_reproduction_established':False})
cols=[(k,g) for k in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]
with (OUT/'rescored_metrics.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['method','space','clips','frames']+[k+'_'+g for k,g in cols])
    for name,row in results.items():w.writerow([name,row['units'],row['clips'],row['frames']]+[row['metrics'][k][g] for k,g in cols])
policy=ROOT/'configs/p2_vico_dim_variance_per_coefficient_v1.json'
for path in [policy,Path(__file__),ROOT/'compare/deployment/vico_literature_metrics.py',ROOT/'compare/Dyadic-Interaction-Modeling/code/metrics/eval_utils.py',base/'protocol.json',base/'RESULTS.json',l2l/'protocol.json',l2l/'DECA_PROTOCOL.json']:
    files[str(path)]=sha(path)
save('AUDIT.json',{'passed':True,'checks':checks,'verified_emoca_feature_files':129,'verified_l2l_prediction_files':172,'l2l_gt_speaker_files':86,'files':files,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'resource':'CPU only, BLAS/OMP capped at 2; no GPU job launched','limits':'Cached inference outputs re-scored, no stochastic reruns; source availability is a separate audit'})
print('All numerical audits passed',flush=True)
