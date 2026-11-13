"""Structured DIM released-metric readout; no substitution of DECA for EMOCA.

Imported author functions remain unmodified. This module deliberately separates
mathematical parity from evidence of the published table's complete protocol.
"""
import contextlib
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIM = ROOT / 'compare/Dyadic-Interaction-Modeling/code'


def load_author():
    assert hashlib.sha256((DIM/'metrics/eval_utils.py').read_bytes()).hexdigest() == '0438a541fb632b82585522d7af169ef4f60708f243bf43a2aeb07cebf876e3c5'
    spec = importlib.util.spec_from_file_location('dim_pinned_eval_utils', DIM/'metrics/eval_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays_valid(gt, pred, speaker):
    assert len(gt) == len(pred) == len(speaker) > 0
    for g, p, x in zip(gt, pred, speaker):
        assert g.shape == p.shape == x.shape and g.ndim == 2 and g.shape[1] == 56
        assert len(g) >= 2
        assert all(np.isfinite(a).all() for a in [g,p,x])


def evaluate(gt, pred, speaker):
    arrays_valid(gt,pred,speaker)
    author=load_author()
    groups={'pose':slice(0,6),'exp':slice(6,56)}
    result={key:{} for key in ['FD','P_FD','MSE','SID','SID_GT','Var','Var_GT','rPCC']}
    records=[]
    for g,p,x in zip(gt,pred,speaker):
        metrics={key:{} for key in ['FD','P_FD','MSE']}
        for name,sl in groups.items():
            metrics['FD'][name]=float(author.calculate_frechet_distance(
                *author.calculate_activation_statistics(g[:,sl]),
                *author.calculate_activation_statistics(p[:,sl])))
            metrics['P_FD'][name]=float(author.calculate_frechet_distance(
                *author.calculate_activation_statistics(np.concatenate([x[:,sl],g[:,sl]],axis=-1)),
                *author.calculate_activation_statistics(np.concatenate([x[:,sl],p[:,sl]],axis=-1))))
            metrics['MSE'][name]=float(np.mean((g[:,sl]-p[:,sl])**2))
        records.append(metrics)
    for key in ['FD','P_FD','MSE']:
        for name in groups:
            # Original MSE aggregates NumPy float32 scalars, unlike FD's
            # float64 covariance output. Preserve that arithmetic as well.
            dtype=gt[0].dtype if key=='MSE' else np.float64
            result[key][name]=float(np.mean(np.asarray([r[key][name] for r in records],dtype=dtype)))
    g,p,x=[np.concatenate(a,axis=0) for a in [gt,pred,speaker]]
    for name,sl in groups.items():
        result['SID'][name]=float(author.calcuate_sid(gt,pred,type=name))
        result['SID_GT'][name]=float(author.calcuate_sid(gt,gt,type=name))
        result['Var'][name]=float(np.var(p[:,sl].reshape(-1)))
        result['Var_GT'][name]=float(np.var(g[:,sl].reshape(-1)))
        a=float(np.corrcoef(g[:,sl].reshape(-1),x[:,sl].reshape(-1))[0,1])
        b=float(np.corrcoef(p[:,sl].reshape(-1),x[:,sl].reshape(-1))[0,1])
        result['rPCC'][name]=float(abs(a-b)) if np.isfinite(a) and np.isfinite(b) else None
    return dict(metrics=result,per_clip=records,clips=len(gt),frames=sum(map(len,gt)),
        units='unscaled raw EMOCA coefficients; pose6 then exp50',
        aggregation='FD/P-FD/MSE equal clips; SID GT-fitted global KMeans40/20; Var/rPCC pooled flattened values',
        unavailable='rPCC is null for constant speaker/prediction; never replaced by0')


def author_stdout(gt,pred,speaker):
    """Run original print_metrics in an isolated import namespace."""
    original=sys.modules.get('metrics');utils=sys.modules.get('metrics.eval_utils')
    import types
    package=types.ModuleType('metrics');package.__path__=[str(DIM/'metrics')]
    sys.modules['metrics']=package
    sys.modules['metrics.eval_utils']=load_author()
    try:
        spec=importlib.util.spec_from_file_location('dim_pinned_print_metrics',DIM/'mymetrics.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        assert hashlib.sha256((DIM/'mymetrics.py').read_bytes()).hexdigest() == '96844e66b846a6fafe98e90b3eb0701b5d62cd34a69b623614361557acf5d8f0'
        out=io.StringIO();fields=[]
        def capture(*args,**kwargs):
            fields.append(args)
            builtins.print(*args,**kwargs)
        module.print=capture
        with contextlib.redirect_stdout(out):module.print_metrics(gt,pred,speaker)
        return out.getvalue(),fields
    finally:
        for key,prior in [('metrics',original),('metrics.eval_utils',utils)]:
            if prior is None:sys.modules.pop(key,None)
            else:sys.modules[key]=prior


def parity(gt,pred,speaker):
    """Compare every reported metric, including author GT SID and variance."""
    text,fields=author_stdout(gt,pred,speaker)
    ours=evaluate(gt,pred,speaker)['metrics'];errors=[]
    names={'fid':'FD','pfid':'P_FD','mse':'MSE','sid':'SID','var':'Var','rpcc':'rPCC'}
    for field in fields:
        label=field[0].strip().rstrip(':')
        words=label.strip().replace('_',' ').split()
        if len(words)!=2 or words[0] not in names:continue
        kind,group=names[words[0]],words[1]
        values=[float(v) for v in field[1:]]
        expected=ours[kind][group]
        if kind in ['SID','Var']:
            # Official Var prints GT first; SID prints prediction first.
            order=[ours['Var_GT'][group],ours['Var'][group]] if kind=='Var' else [ours['SID'][group],ours['SID_GT'][group]]
            errors.extend(abs(a-b) for a,b in zip(values,order))
        elif expected is None:assert np.isnan(values[0])
        else:errors.append(abs(values[0]-expected))
    assert len(errors)==16,(len(errors),text)
    assert max(errors)<1e-10,max(errors)
    return dict(passed=True,compared_scalars=len(errors),max_abs=max(errors),author_stdout=text,
        mathematical_parity_only=True,published_protocol_equivalence=False)
