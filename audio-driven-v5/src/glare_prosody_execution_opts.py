"""Reversible inference-only overlays; adoption requires raw/token paired acceptance."""
import inspect
from glare_prosody_subset import IntegrityError


def install_last_logits(runtime):
    lm=runtime.analyzer.model.language_model
    if 'num_logits_to_keep' not in inspect.signature(lm.forward).parameters:
        raise IntegrityError('Installed LM lacks last-token generation option')
    original=lm.forward
    def forward(*args,**kwargs):
        if kwargs.get('labels') is not None:raise IntegrityError('Last-logits overlay is generation-only')
        kwargs['num_logits_to_keep']=1
        return original(*args,**kwargs)
    lm.forward=forward
    return lambda:setattr(lm,'forward',original)


def install_masked_audio_prefix(runtime):
    import torch
    tower=runtime.analyzer.model.audio_tower;original=tower.forward
    if tower.training:raise IntegrityError('Padding removal is inference-only')
    original_positions=tower.embed_positions
    class PrefixPositions(torch.nn.Module):
        def __init__(self,weight,length):
            super().__init__();self.register_buffer('weight',weight[:length],persistent=False)
    def forward(input_features,*args,**kwargs):
        if args:raise IntegrityError('Expected keyword attention mask')
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=4 or mask.shape[1]!=1:
            raise IntegrityError('Cannot establish masked audio tail')
        keys=torch.isfinite(mask[:,0,0,:]);lengths=keys.sum(-1)
        prefix=torch.arange(keys.shape[-1],device=keys.device)[None,:]<lengths[:,None]
        if not torch.equal(keys,prefix):raise IntegrityError('Audio valid keys are not contiguous prefixes')
        # Two extra convolution-output positions preserve conv1+conv2 boundary context.
        positions=min(keys.shape[-1],int(lengths.max().item())+2)
        mel_frames=positions*2
        old_max=tower.config.max_source_positions;old_positions=tower.embed_positions
        tower.config.max_source_positions=positions
        tower.embed_positions=PrefixPositions(original_positions.weight,positions)
        try:
            kwargs=dict(kwargs,attention_mask=mask[:,:,:positions,:positions].contiguous())
            return original(input_features[:,:,:mel_frames].contiguous(),**kwargs)
        finally:
            tower.embed_positions=old_positions;tower.config.max_source_positions=old_max
    tower.forward=forward
    return lambda:setattr(tower,'forward',original)


def install(runtime,mode):
    restores=[]
    if mode in ('last_logits','trim_audio'):restores.append(install_last_logits(runtime))
    if mode=='trim_audio':restores.append(install_masked_audio_prefix(runtime))
    def restore():
        for call in reversed(restores):call()
    return restore
