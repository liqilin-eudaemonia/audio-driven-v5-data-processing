"""Reuse completed L2L+GPAvatar renders and fix isolated protobuf startup."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from vico_more_common import ROOT,read,sha,save
WORK=ROOT/'experiments/p2_l2l_public_renderer_20261110'
OUT=ROOT/'experiments/p2_vico_additional_optimization_20261110'


def render_complete():
    marker=WORK/'RENDER_COMPLETE.json'
    if not marker.exists():return False
    p=read(WORK/'protocol.json');data=read(marker);digest=sha(WORK/'protocol.json')
    expected={(name,s['id']) for name in p['methods'] for s in p['samples']}
    records=data['records']
    if not data.get('passed') or len(records)!=len(expected) or {(r['model'],r['id']) for r in records}!=expected:return False
    if data['protocol_sha256']!=digest:return False
    predictions={(name,s['id']):s['predictions'][name] for s in p['samples'] for name in p['methods']}
    for r in records:
        pred=predictions[r['model'],r['id']]
        if r['protocol_sha256']!=digest or r['motion_sha256']!=pred['sha256'] or sha(pred['path'])!=pred['sha256']:return False
        if any(not Path(f).is_file() or sha(f)!=h for f,h in r['files'].items()):return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['status','render','smoke','extract','probe'])
    p.add_argument('--check-only',action='store_true');args=p.parse_args();start=time.perf_counter()
    if args.stage=='render':
        # The streaming adapter verifies/reuses existing clips before GPU admission.
        command=[sys.executable,str(ROOT/'compare/deployment/vico_gpavatar_render_fast.py')]
        if args.check_only:command.append('--check-only')
        os.execv(sys.executable,command)
    if args.stage in ('status','render'):
        done=render_complete()
        result=dict(stage='render',state='reused_verified' if done else 'pending',outputs=172 if done else None,
            gpu_requested=False,seconds=time.perf_counter()-start,
            emoca_complete=(WORK/'emoca_eval/COMPLETE.json').exists(),
            image_metrics_complete=all((WORK/'image_eval'/n/'COMPLETE.json').exists() for n in ['conan','fallon','stephen','trevor']))
        save(OUT/'GPAVATAR_FAST_STATUS.json',result);print(json.dumps(result));return
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
    if args.stage=='probe':
        import torch
        import wandb
        models=ROOT/'experiments/p2_vico_literature_protocol_20261108/assets/EMOCA_v2_lr_mse_20'
        checkpoints=list((models/'detail').rglob('*.ckpt')) or list(models.rglob('*.ckpt'))
        def loss(path):
            try:return float(path.stem.rsplit('=',1)[1])
            except (ValueError,IndexError):return float('inf')
        checkpoint=min(sorted(checkpoints),key=loss)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
        counts={name:sum(k.startswith('deca.'+name+'.') for k in state) for name in ['E_flame','E_expression']}
        assert all(counts.values())
        report=dict(passed=True,scope='Actual EMOCA checkpoint CPU load; no GPU encoder parity claim',
            encoder_tensors=counts,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            wandb_version=wandb.__version__,protobuf_implementation=os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'],
            cuda_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-start)
        save(OUT/'GPAVATAR_DEPENDENCY_PROBE.json',report);print(json.dumps(report));return
    if args.check_only:
        print(json.dumps(dict(stage=args.stage,environment_fix='PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python',gpu_requested=False)))
        return
    # Original stage keeps its source locks, SHA checks, batching and GPU admission.
    python=ROOT/'compare/.envs/vico_more/bin/python'
    os.execve(str(python),[str(python),str(WORK/'evaluate.py'),args.stage],dict(os.environ))


if __name__=='__main__':main()
