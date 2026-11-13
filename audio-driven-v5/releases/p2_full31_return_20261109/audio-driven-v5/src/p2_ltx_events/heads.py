"""Keep temporal positions; pool only video space and valid generated tokens."""
import torch
from torch import nn
from torch.nn import functional as F


def names(c):
    return tuple(task+'/'+n for task,classes in active_classes(c).items()
                 for m in ('video','audio') for n in classes[m])


def active_classes(c):
    result={}
    if c['hidden_event_supervision']['clip_weight']>0:result['clip']=c['event_supervision']['classes']
    if c['hidden_event_supervision']['temporal_weight']>0:result['temporal']=c['hidden_event_supervision']['classes']
    return result


def pool_time(x, groups, valid, bins):
    if x.ndim != 3 or groups.shape != x.shape[:2] or valid.shape != groups.shape:
        raise ValueError('Hidden/token layout mismatch')
    if groups.dtype != torch.long or valid.dtype != torch.bool or bins<1 or x.shape[1]<1 or x.shape[1]%bins:
        raise ValueError('Invalid hidden temporal groups')
    repeat=x.shape[1]//bins
    expected=torch.arange(bins,device=groups.device).repeat_interleave(repeat)[None].expand_as(groups)
    if not torch.equal(groups,expected):raise ValueError('LTX time/space token order changed')
    # Native LTX has a contiguous fixed spatial grid at each time. Reduce that
    # axis directly in FP32: no CUDA atomic scatter or duplicate index tensor.
    count=valid.reshape(x.shape[0],bins,repeat).float().sum(2)
    pooled=x.float().masked_fill(~valid[...,None],0).reshape(x.shape[0],bins,repeat,x.shape[-1]).sum(2)
    return pooled/count.clamp_min(1)[...,None],count>0


class HiddenEventHeads(nn.Module):
    def __init__(self, video_dim, audio_dim, c, width=128):
        super().__init__()
        self.class_names=active_classes(c)
        self.heads=nn.ModuleDict()
        for task in self.class_names:
            for mod,dim in [('video',video_dim),('audio',audio_dim)]:
                size=len(self.class_names[task][mod])
                if size:self.heads[task+'_'+mod]=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,width),
                    nn.GELU(),nn.Linear(width,size))

    def forward(self,hidden,layout,masks):
        logits={}
        for mod,x in hidden.items():
            g=layout[mod+'_groups'];intervals=layout[mod+'_intervals']
            valid=masks[mod] & layout[mod+'_token_valid']
            with torch.autocast(x.device.type,enabled=False):
                timed,valid_time=pool_time(x,g,valid,intervals.shape[1])
                pooled=timed.masked_fill(~valid_time[...,None],0).sum(1)/valid_time.sum(1).clamp_min(1)[:,None]
                for task,value in [('clip',pooled),('temporal',timed)]:
                    key=task+'_'+mod
                    if key in self.heads:
                        z=self.heads[key](value)
                        logits.update({task+'/'+name:z[...,i] for i,name in enumerate(self.class_names[task][mod])})
        return logits


def event_sums(logits,labels,pos_weights):
    if set(logits)!=set(labels) or set(logits)!=set(pos_weights):
        raise ValueError('Hidden Event vocabulary mismatch')
    sums={};counts={}
    for name,z in logits.items():
        row=labels[name];y=row['y'];known=row['known'];weight=row['weight']
        if z.shape!=y.shape or known.shape!=y.shape or weight.shape!=y.shape:
            raise ValueError('Hidden Event target shape mismatch: '+name)
        if (known.dtype!=torch.bool or not torch.isfinite(y).all() or not torch.isfinite(weight).all()
                or ((y<0)|(y>1)).any() or (weight<0).any()):
            raise ValueError('Invalid occupancy/known/weight')
        raw=F.binary_cross_entropy_with_logits(z.float(),y.float(),reduction='none',
            pos_weight=torch.as_tensor(pos_weights[name],device=z.device,dtype=torch.float32))
        sums[name]=(raw*weight.masked_fill(~known,0)).sum()
        counts[name]=known.sum()
    return sums,counts


def objective(values,c,counts,accumulation=1,world_size=1):
    from p2_clip_events.model import normalized_loss
    losses={}
    for task in ('clip','temporal'):
        if c['hidden_event_supervision'][task+'_weight']==0:
            losses[task]=values['av']*0
            continue
        subset={k:v for k,v in values['event_sums'].items() if k.startswith(task+'/')}
        # The inherited executor SUM-reduces gradients, rather than DDP's mean.
        losses[task]=normalized_loss(subset,{k:counts[k] for k in subset}) if subset else values['av']*0
    spec=c['hidden_event_supervision']
    event=spec['clip_weight']*losses['clip']+spec['temporal_weight']*losses['temporal']
    w=c['loss_weights']
    total=w['av']*values['av']/(accumulation*world_size)+w['event']*event
    total+=sum(w[m+'_feature']*v/(accumulation*world_size) for m,v in values['features'].items())
    return total,losses,event
