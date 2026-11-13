"""Use the renderer's own causal position generators, in physical seconds."""
from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def _position_strategy():
    from p2_whisper_av.bootstrap import activate_ltx
    activate_ltx()
    from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig,TextToVideoStrategy
    return TextToVideoStrategy(TextToVideoConfig(with_audio=True,i2av_only=True,first_frame_conditioning_p=1))


def make_layout(av,support,duration):
    if duration<=0:raise ValueError('Positive real duration required')
    strategy=_position_strategy();z=av['latents']['latents'];az=av['audio_latents']['latents']
    b,_,f,h,w=z.shape
    if b!=1 or az.shape[0]!=b:raise ValueError('Per-sample hidden Event layout requires batch=1')
    fps=float(av['latents']['fps'].flatten()[0]);device=z.device
    physical=dict(video=strategy._get_video_positions(f,h,w,b,fps,device,torch.float32),
        audio=strategy._get_audio_positions(az.shape[2],b,device,torch.float32))
    # The native strategy computes RoPE coordinates in the latent dtype.
    # BF16 can round frame indices before division by fps. Reconstruct that
    # exact native computation for checking; keep FP32 seconds for labels.
    positions=dict(video=strategy._get_video_positions(f,h,w,b,fps,device,z.dtype),
        audio=strategy._get_audio_positions(az.shape[2],b,device,az.dtype))
    result={}
    for mod,p in positions.items():
        repeat=h*w if mod=='video' else 1
        count=f if mod=='video' else az.shape[2]
        groups=torch.arange(count,device=device).repeat_interleave(repeat)[None]
        precise=physical[mod]
        intervals=precise[:,0,::repeat,:]
        valid=support[mod].to(device).clone()
        valid &= (precise[:,0,:,1]<=duration+1e-6) & (precise[:,0,:,1]>precise[:,0,:,0])
        if mod=='video':valid &= groups>0  # reference image is fixed, not generated
        result.update({mod+'_positions':p,mod+'_groups':groups,mod+'_intervals':intervals,
                       mod+'_token_valid':valid})
    return result


def check_prepared(layout,prepared):
    for mod in ('video','audio'):
        actual=getattr(prepared,mod).positions
        expected=layout[mod+'_positions']
        if actual.shape!=expected.shape or not torch.allclose(actual.float(),expected.float(),rtol=1e-5,atol=1e-5):
            raise ValueError('Native renderer / Event timestamp mismatch: '+mod)


def rasterize(row,layout,classes,background_weight=.1):
    """Fractional occupancy; unknown support stays masked, including partial bins."""
    from p2_listen.annotations import intersection_length,union
    result={}
    for mod in ('video','audio'):
        intervals=layout[mod+'_intervals'];groups=layout[mod+'_groups']
        supported=torch.zeros(intervals.shape[:2],device=groups.device)
        supported.scatter_add_(1,groups,layout[mod+'_token_valid'].float())
        for name in classes[mod]:
            data=row['classes'].get(name,{'spans':[],'coverage':[],'unknown':[]})
            spans=union(data['spans']);coverage=union(data['coverage']);unknown=union(data.get('unknown',[]))
            y=torch.zeros(intervals.shape[:2],device=intervals.device)
            known=torch.zeros_like(y,dtype=torch.bool);weight=torch.zeros_like(y)
            for i,(a,b) in enumerate(intervals[0].tolist()):
                if (b<=a or not supported[0,i] or intersection_length(coverage,a,b)<b-a-1e-6
                        or intersection_length(unknown,a,b)>1e-8):continue
                y[0,i]=min(1.,intersection_length(spans,a,b)/(b-a))
                known[0,i]=True;weight[0,i]=1. if y[0,i]>0 else background_weight
            result['temporal/'+name]=dict(y=y,known=known,weight=weight)
    return result
