"""Data-free CPU checks, separate from the future real-model S0."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from .config import activate, feature_loss_enabled
from .layout import make_layout
from .model import ListeningExtractor
from .events import normalized_event_loss
from .contracts import EVENTS, TYPE_TO_MODALITY
from .training import optimization_step, save_checkpoint, checked_checkpoint, restore_checkpoint, rng_state


def fixture(c):
    activate(c)
    from omni_av_p2.config import Config
    from omni_av_p2.connector import ConditionInputs
    config=Config(hidden_size=8,hidden_layers=(1,2),target_dim=8,video_dim=16,audio_dim=8,
        fusion_width=8,fusion_depth=1,fusion_heads=2,decoder_width=16,decoder_depth=1,diffusion_batch_mul=1,
        vision_fps=8,audio_hz=8)
    layout=make_layout(1.,(1,1))
    condition=ConditionInputs(hidden=torch.randn(1,2,3,8),hidden_valid=torch.ones(1,3,dtype=torch.bool),
                              layer_ids=torch.tensor([1,2]),**{k:layout[k] for k in ("video_coordinates","audio_coordinates","video_valid","audio_valid")})
    targets={}
    for mod in ("video","audio"):
        targets[mod]=torch.randn(1,layout[mod+"_valid"].shape[1],8)
        targets[mod+"_valid"]=layout[mod+"_valid"]
    labels={k: dict(y=torch.tensor([[0.,1.,0.,1.] if TYPE_TO_MODALITY[k] == "video" else [0.,1.,0.,1.,0.,0.,1.,0.]]),
                    known=torch.ones(1, 4 if TYPE_TO_MODALITY[k] == "video" else 8, dtype=torch.bool)) for k in EVENTS}
    return config,(condition,layout,targets,None,labels)


def cpu_audit(c):
    torch.set_num_threads(2)
    torch.manual_seed(723)
    cfg,sample=fixture(c)
    enabled=feature_loss_enabled(c)
    if not enabled:
        sample=(*sample[:2],None,*sample[3:])
    initial=ListeningExtractor(cfg,8,feature_loss_enabled=enabled)
    t=dict(connector_lr=1e-3,feature_lr=1e-3,event_lr=1e-3,clip_norm=1.)
    weights={k:1. for k in EVENTS}
    callback=lambda u,_: (u.video-0.2).square().mean()+(u.audio+0.1).square().mean()
    evidence={"feature_loss_enabled":enabled}
    models={}
    for mode in ("probe_only","joint_event"):
        model=copy.deepcopy(initial)
        opt=torch.optim.AdamW(model.optimizer_groups(t),weight_decay=0)
        for step in range(5):
            optimization_step(model,opt,[sample],callback,mode=mode,pos_weights=weights,event_weight=.05,
                seed=91,ordinals=[step],config=t,audit=True)
        result=model(sample[0],sample[1],sample[2],sample[4],mode,weights,torch.Generator().manual_seed(1))
        loss=normalized_event_loss(result["event_sums"],result["event_counts"])
        grad=torch.autograd.grad(loss,[result["u"].video,result["u"].audio],allow_unused=True,retain_graph=True)
        norms=[0. if g is None else float(g.norm()) for g in grad]
        if mode=="probe_only" and any(norms): raise AssertionError("Probe gradient leaked")
        if mode=="joint_event" and not all(x>0 for x in norms): raise AssertionError("Event gradient missing")
        per_class={}
        for k in EVENTS:
            grad_k=torch.autograd.grad(result["event_sums"][k],[result["u"].video,result["u"].audio],allow_unused=True,retain_graph=True)
            per_class[k]=[0. if g is None else float(g.norm()) for g in grad_k]
        evidence[mode]=dict(event_to_u_norms=norms,event_to_u_per_class=per_class)
        models[mode]=model
        if mode=="joint_event":
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"checkpoint.pt"
                ident={"test":"cpu-resume"}
                save_checkpoint(path,model,opt,5,ident,[rng_state()])
                restored=copy.deepcopy(initial)
                opt2=torch.optim.AdamW(restored.optimizer_groups(t),weight_decay=0)
                restore_checkpoint(checked_checkpoint(path),restored,opt2,ident)
                for m,o in ((model,opt),(restored,opt2)):
                    optimization_step(m,o,[sample],callback,mode=mode,pos_weights=weights,event_weight=.05,
                        seed=91,ordinals=[5],config=t)
                if any(not torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()):
                    raise AssertionError("CPU checkpoint resume differs")
                evidence["resume_exact"]=True
    evidence.update(cpu_only=True,real_s0_completed=False,training_data_used=False)
    return evidence
