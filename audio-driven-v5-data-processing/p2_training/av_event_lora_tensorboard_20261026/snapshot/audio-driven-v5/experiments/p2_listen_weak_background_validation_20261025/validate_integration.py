"""Reproduce label import/cache/statistics/gradient checks without loading models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from p2_listen.config import load,activate
from p2_listen.contracts import EVENTS,read,rows,sha,atomic_json,digest
from p2_listen.dataset import load_observations
from p2_listen.features import load_input,folder,identity
from p2_listen.cache import load_part
from p2_listen.annotations import class_statistics,mask_classes
from p2_listen.events import EventHeads,event_sums,normalized_event_loss
from p2_listen.training import supervision_policy
from p2_listen.weak_labels import export_weak_labels

torch.set_num_threads(1)

out=Path(__file__).resolve().parent
cfg=load(out/'config.json');activate(cfg)
original=ROOT/'experiments/listen_debug50_fullraw_20261025'
old_runtime=ROOT/'experiments/p2_listen_gpu_debug50_20261025'
protected=[original/'labels.debug-positive.jsonl',original/'labels.canonical.candidate.jsonl',old_runtime/'summary.json']
protected+=list((old_runtime/'cache/events').glob('*/*'))
before={str(p):sha(p) for p in protected if p.is_file()}

# Re-derive with final code in a temporary directory and require identical labels.
with tempfile.TemporaryDirectory(prefix='p2-weak-replay-') as temp:
    export_weak_labels(cfg, original/'integrated/labels.candidate.jsonl', original/'integrated/review.json',
        original/'integrated/observations.jsonl', original/'raw/visual.jsonl', original/'raw/audio.jsonl',
        original/'media_manifest.json', original/'summary.json', temp, replace_placeholder=True)
    for name in ('labels.json','decisions.json','audit.json','summary.json'):
        assert digest(read(Path(temp)/name)) == digest(read(out/'derived'/name)), name

def command(*args):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/p2_listen.py'),*args,'--config',str(out/'config.json')],cwd=ROOT,capture_output=True,text=True)
    (out/(args[0]+'.log')).write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError(result.stderr)

command('import-labels','--labels',str(out/'derived/labels.admitted.json'))
cache=Path(cfg['output'])/'cache';cache.mkdir(exist_ok=True)
input_link=cache/'input'
if not input_link.exists():input_link.symlink_to(old_runtime/'cache/input',target_is_directory=True)
assert input_link.resolve()==(old_runtime/'cache/input').resolve()
command('cache','--part','events','--device','cpu')
obs=load_observations(cfg)
labels=rows(out/'derived/labels.admitted.json')
by_id={l['sample_id']:l for l in labels}
all_r=[]
for o in obs:
    all_r.append(load_part(folder(cfg,'events',o['sample_id']),identity(cfg,'events',o,label=by_id[o['sample_id']]),'cpu'))
stats=class_statistics(all_r,cap=cfg['events']['pos_weight_cap'])
effective=[k for k in EVENTS if stats[k]['enabled']]
# S0 can now use ordinary supervision policy, without positive-only override.
policy=supervision_policy(stats,EVENTS,labels,s0=True)
try:
    supervision_policy(stats,EVENTS,labels)
    raise AssertionError('Debug data unexpectedly accepted for formal training')
except ValueError:
    pass

# Actual cached temporal grids, fresh small CPU heads; no generated quality claim.
gradients={}
for mode in ('probe_only','joint_event'):
    torch.manual_seed(137)
    heads=EventHeads(12,7,8)
    opt=torch.optim.SGD(heads.parameters(),lr=.01)
    norms=[];losses=[]
    for o,r in zip(obs,all_r):
        _,layout=load_input(cfg,o,'cpu')
        u=SimpleNamespace(video=torch.randn(1,layout['video_valid'].shape[1],12,requires_grad=True),
                          audio=torch.randn(1,layout['audio_valid'].shape[1],7,requires_grad=True),
                          video_valid=layout['video_valid'],audio_valid=layout['audio_valid'])
        opt.zero_grad()
        sums,counts=event_sums(heads(u,layout,mode),mask_classes(r,effective),{k:stats[k]['pos_weight'] for k in EVENTS})
        loss=normalized_event_loss(sums,counts)
        assert torch.isfinite(loss)
        loss.backward()
        norm={m:0. if getattr(u,m).grad is None else float(getattr(u,m).grad.abs().sum()) for m in ('video','audio')}
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in heads.parameters())
        if mode=='probe_only':assert norm=={'video':0.,'audio':0.}
        opt.step();norms.append(norm);losses.append(float(loss.detach()))
    gradients[mode]=dict(samples=len(norms),summed_u_gradients={m:sum(n[m] for n in norms) for m in ('video','audio')},loss_min=min(losses),loss_max=max(losses))
assert all(v>0 for v in gradients['joint_event']['summed_u_gradients'].values())
assert before=={str(p):sha(p) for p in protected if p.is_file()}
report=dict(passed=True,samples=len(obs),effective_classes=effective,class_statistics=stats,supervision_policy=policy,
            gradients=gradients,derivation_replayed_identically=True,original_artifacts_unchanged=True,protected_sha256=before,
            input_cache_reused_with_identity_checks=str(input_link.resolve()),
            new_event_cache=str(cache/'events'),pretrained_models_loaded=False,formal_training_started=False)
atomic_json(out/'integration.json',report)
print({k:v for k,v in report.items() if k!='protected_sha256'})
