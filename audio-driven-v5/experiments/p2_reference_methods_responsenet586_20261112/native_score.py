"""Native DECA motion transfer diagnostics, explicitly distinct from paired RGB metrics."""
import numpy as np
from runtime import *
from vico_literature_metrics import evaluate
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['responsenet']);args=parser.parse_args()
for dataset in ([args.dataset] if args.dataset else ['responsenet']):
    ss=[s for s in selected() if s['dataset']==dataset]
    gt=[];speaker=[]
    for s in ss:
        d=WORK/'evaluation_gt'/dataset/s['id'];r=read(d/'complete.json');assert sha(r['path'])==r['sha256']
        gt.append(np.load(r['path'])['motion'])
        speaker.append(np.load(WORK/'features'/dataset/s['id']/'speaker_deca.npz')['motion'])
    for method in ['dualtalk','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor']:
        if (WORK/'results'/f'{method}_{dataset}_native.json').exists():continue
        preds=[]
        for s,g in zip(ss,gt):
            d=folder(method,s);r=read(d/'MOTION_COMPLETE.json');path=d/'motion.npz';assert sha(path)==r['files'][str(path)]
            with np.load(path) as z:
                if method=='dualtalk':
                    v=z['deca_proxy'];ix=np.minimum(np.floor(np.arange(len(g))*25/30+.5).astype(int),len(v)-1);v=v[ix]
                else:v=z['motion']
            assert v.shape==g.shape;preds.append(v)
        order=np.r_[50:56,0:50];g=[x[1:,order] for x in gt];s=[x[:-1,order] for x in speaker];p=[x[1:,order] for x in preds]
        r=evaluate(g,p,s);r['released_flatten_variance']=r['metrics']['Var'];r['released_flatten_variance_gt']=r['metrics']['Var_GT']
        pooled=np.concatenate(p);r['metrics']['Var']={k:float(pooled[:,sl].var(axis=0,ddof=0).mean()) for k,sl in [('exp',slice(6,56)),('pose',slice(0,6))]}
        pooled_gt=np.concatenate(g);r['metrics']['Var_GT']={k:float(pooled_gt[:,sl].var(axis=0,ddof=0).mean()) for k,sl in [('exp',slice(6,56)),('pose',slice(0,6))]}
        r['units']='unscaled raw DECA/proxy coefficients; pose6 then exp50'
        r['aggregation']='FD/P-FD/MSE equal clips; SID GT-fitted KMeans40/20; Var/Var_GT mean of per-coefficient pooled population variance; rPCC pooled flattened values'
        r.update(protocol_sha256=protocol_sha(),scope_sha256=sha(WORK/'RUN_SCOPE.json'),model_manifest_sha256=sha(WORK/'MODEL_MANIFEST.json'),dataset=dataset+' full586',method=method,representation='DECA native/proxy; NOT unified EMOCA or paired RGB8',limitations='DualTalk neck/global-head mapping unverified; public pretrained cross-dataset transfer, not paper reproduction',fps=30,speaker_offset=-1)
        save(WORK/'results'/f'{method}_{dataset}_native.json',r)
        print(dataset,method,r['metrics']['FD'],flush=True)
