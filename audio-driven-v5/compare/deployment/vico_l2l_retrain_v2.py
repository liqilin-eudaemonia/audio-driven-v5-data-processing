"""Repeat the approved training budget without prematurely stopping VQ warmup.

All features, splits, architectures, losses and learning rates are reused.
Only the stopping rule changes: run the originally budgeted20000 steps/stage.
Validation-only learning checks must pass before any new test evaluation.
"""
import copy
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import vico_l2l_retrain as engine
from vico_common_deca import read,save,sha

ORIGINAL=engine.WORK
WORK=ORIGINAL.with_name('l2l_retrained_v2')
CACHE={}


def prepare():
    assert (ORIGINAL/'FEATURES_COMPLETE.json').exists()
    protocol=copy.deepcopy(read(ORIGINAL/'protocol.json'))
    protocol.update(schema='l2l-vico-local-retrain-v2',parent_protocol_sha256=sha(ORIGINAL/'protocol.json'),
        wrapper_sha256=sha(__file__),change='remove early stopping during the original20000-step/stage budget; select validation minimum, then require learning gate before test',
        motivation='v1 VQ validation best at step1 and loss still descending when stopped; validation-only diagnosis, no test-score tuning',
        features_reused_from=str(ORIGINAL/'features'))
    protocol['train_recipe']['minimum_steps']=protocol['train_recipe']['max_steps_per_stage']
    if (WORK/'protocol.json').exists():assert read(WORK/'protocol.json')==protocol
    else:save(WORK/'protocol.json',protocol)
    save(WORK/'split.json',read(ORIGINAL/'split.json'))
    link=WORK/'features'
    if link.is_symlink():assert link.resolve()==(ORIGINAL/'features').resolve()
    elif link.exists():raise FileExistsError(link)
    else:link.symlink_to(ORIGINAL/'features',target_is_directory=True)
    save(WORK/'FEATURES_COMPLETE.json',dict(pairs=397,streams=794,reused=True,original_completion_sha256=sha(ORIGINAL/'FEATURES_COMPLETE.json')))
    engine.WORK=WORK


def codec_gate():
    import torch
    from vqgan.vqmodules.gan_models import setup_vq_transformer
    import types
    selected=read(WORK/'vq_COMPLETE.json')
    existing=read(WORK/'QUALITY_GATE.json') if (WORK/'QUALITY_GATE.json').exists() else {}
    if existing.get('vq_passed') and existing.get('vq_checkpoint_sha256')==selected['checkpoint_sha256']:
        return existing
    data=CACHE['windows']['validation']['listener']
    values=np.concatenate([data[:,:32],data[:,32:]],axis=0)
    config=read(engine.REPO/'vqgan/configs/l2_32_smoothSS.json')
    model,_,_=setup_vq_transformer(types.SimpleNamespace(),config,load_path=None)
    model.load_state_dict(torch.load(WORK/'vq_best.pth',map_location='cpu',weights_only=False)['state_dict'],strict=True)
    model.eval();reconstruction=baseline=0.;count=0;codes=[]
    with torch.no_grad():
        for start in range(0,len(values),64):
            x=torch.from_numpy(values[start:start+64]).cuda();pred,_=model(x,None)
            group_slices=[slice(0,50),slice(50,53),slice(53,56)]
            reconstruction+=sum((pred[:,:,s]-x[:,:,s]).abs().mean().item() for s in group_slices)*len(x)
            baseline+=sum(x[:,:,s].abs().mean().item() for s in group_slices)*len(x)
            _,indices=model.module.get_quant(x);codes.extend(indices.cpu().reshape(-1).tolist());count+=len(x)
    history=read(WORK/'vq_history.json')
    checks=dict(selected_after_initialization=selected['selected_step']>1,
        validation_total_improved=selected['validation_loss']<history[0]['validation_loss'],
        validation_reconstruction_beats_train_mean=reconstruction<baseline,
        noncollapsed_codebook=len(set(codes))>=2)
    result=dict(passed=False,vq_passed=all(checks.values()),eligible_for_comparison=False,
        status='vq_learning_checked',checks=checks,validation_only=True,
        selected_vq_step=selected['selected_step'],validation_reconstruction_l1=reconstruction/count,
        validation_train_mean_l1=baseline/count,validation_unique_codes=len(set(codes)),
        vq_checkpoint_sha256=selected['checkpoint_sha256'])
    save(WORK/'QUALITY_GATE.json',result)
    del model;torch.cuda.empty_cache()
    if not result['vq_passed']:raise RuntimeError('VQ learning gate failed; no predictor/test result will be treated as a baseline')
    return result


def run():
    prepare();started=time.time();sys.path.insert(0,str(engine.REPO))
    import modules.fact_model as fact
    original_setup=fact.setup_model
    original_windows=engine.make_windows
    def capture_windows():
        data=original_windows();CACHE['windows']=data;return data
    def guarded_predictor(*args,**kwargs):
        import torch
        # Diagnostics must not alter the predictor's initialization/sampling RNG.
        with torch.random.fork_rng(devices=[0]):codec_gate()
        return original_setup(*args,**kwargs)
    engine.make_windows=capture_windows
    fact.setup_model=guarded_predictor
    engine.fit()
    gate=codec_gate();selected=read(WORK/'predictor_COMPLETE.json');history=read(WORK/'predictor_history.json')
    gate['predictor_checks']=dict(selected_after_initialization=selected['selected_step']>1,
        validation_loss_improved=selected['validation_loss']<history[0]['validation_loss'])
    gate['passed']=gate['vq_passed'] and all(gate['predictor_checks'].values())
    gate['eligible_for_comparison']=gate['passed'];gate['status']='learning_gate_passed' if gate['passed'] else 'learning_gate_failed'
    save(WORK/'QUALITY_GATE.json',gate)
    if not gate['passed']:raise RuntimeError('Predictor validation learning gate failed; test not run')
    engine.test();engine.audit()
    assert sha(__file__)==read(WORK/'protocol.json')['wrapper_sha256']
    save(WORK/'COMPLETE.json',dict(status='complete',learning_gate_passed=True,seconds=time.time()-started,results_sha256=sha(WORK/'RESULTS.json')))
    import subprocess
    for script in ['build_common_method_comparison.py','build_vico_three_method_review.py']:
        subprocess.run([sys.executable,str(engine.ROOT/'scripts'/script)],cwd=engine.ROOT,check=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:run()
