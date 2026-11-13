"""Native differentiable hidden and answer-only CE, retaining the planning PEFT adapter."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from p2_listen.contracts import read


def adapter_parameters(thinker):
    names,params=[],[]
    for name,p in thinker.named_parameters():
        active='lora_' in name
        if active and '.model.layers.' not in '.'+name:
            raise ValueError('Only language-layer planning adapters are supported')
        p.requires_grad_(active)
        if active:names.append(name);params.append(p)
    if not params:raise ValueError('Missing original planning LoRA')
    return names,params


def load_online(cfg,adapter,device):
    from omni_av_p2.data import load_omni
    from peft import PeftModel
    spec=read(Path(adapter)/'adapter_config.json')
    if (spec['r'],spec['lora_alpha'],spec.get('lora_dropout',0))!=(32,64,0):
        raise ValueError('Expected existing r32/alpha64/dropout0 SFT adapter')
    if spec.get('modules_to_save') or spec.get('bias','none')!='none':
        raise ValueError('Unexpected trainable non-LoRA modules')
    if Path(spec['base_model_name_or_path']).resolve()!=Path(cfg.omni_path).resolve():
        raise ValueError('Planning adapter belongs to a different Qwen base')
    runtime,processor=load_omni(cfg,device)
    runtime.thinker.requires_grad_(False)
    peft=PeftModel.from_pretrained(runtime.thinker,adapter,is_trainable=True)
    runtime.thinker=peft.get_base_model()
    adapter_parameters(runtime.thinker)
    runtime.thinker.config.use_cache=False
    runtime.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return OnlineQwen(runtime,processor),peft


def text_forward(thinker,full,hidden_states=True):
    return thinker.model(inputs_embeds=full['prefix'],position_ids=full['prefix_positions'],
        attention_mask=full['prefix_valid'].long(),output_hidden_states=hidden_states,
        use_cache=False,return_dict=True)


def answer_loss(thinker,full,chunk_size=128):
    """Predict response including EOS; never allocate prefix-token vocabulary logits."""
    out=text_forward(thinker,full,hidden_states=False)
    n=full['prefix_tokens'];count=full['plan_tokens']+1
    if n<1 or chunk_size<1:raise ValueError('Invalid supervised suffix')
    # Position n-1 predicts the first answer token, S-2 predicts native im_end.
    hidden=out.last_hidden_state[:,n-1:n-1+count]
    eos=full['terminal_token_id']
    targets=torch.cat([full['plan_token_ids'],torch.full((1,1),eos,device=hidden.device,dtype=torch.long)],1)
    if hidden.shape[:2]!=targets.shape:raise ValueError('Shifted response alignment mismatch')
    total=hidden.sum()*0
    for start in range(0,count,chunk_size):
        logits=thinker.lm_head(hidden[:,start:start+chunk_size]).float()
        total=total+F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets[:,start:start+chunk_size].reshape(-1),reduction='sum')
    return total,count


class OnlineQwen(nn.Module):
    """Register original adapter parameters only; frozen3B weights remain external."""
    def __init__(self,runtime,processor):
        super().__init__()
        object.__setattr__(self,'runtime',runtime);object.__setattr__(self,'processor',processor)
        self.parameter_names,params=adapter_parameters(runtime.thinker)
        self.adapters=nn.ParameterList(params)
        self.train()

    def train(self,mode=True):
        super().train(mode)
        self.runtime.thinker.eval()
        self.runtime.thinker.model.train(mode)
        return self

    def assert_ownership(self):
        actual={id(p)for p in self.runtime.thinker.parameters()if p.requires_grad}
        if actual!={id(p)for p in self.adapters}:raise ValueError('Qwen backbone became trainable or adapter missing')

    def _full(self,prepared,prompt,maximum):
        from p3_listen.features import prefix
        from p2_plan_cache.encoding import append_plan
        pref=prefix(self.runtime,self.processor,None,prepared=prepared)
        model=self.runtime.thinker
        full=append_plan(pref,self.processor.tokenizer,model.get_input_embeddings(),prompt,
                         maximum,model.config.text_config.max_position_embeddings)
        full['terminal_token_id']=self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        return full

    def condition(self,prepared,prompt,layout,maximum=32768):
        from omni_av_p2.connector import ConditionInputs
        full=self._full(prepared,prompt,maximum);out=text_forward(self.runtime.thinker,full)
        layers=self.runtime.config.hidden_layers
        if max(layers)>=len(out.hidden_states):raise ValueError('Qwen hidden-layer contract changed')
        h=torch.stack([out.hidden_states[i]for i in layers],1)
        fields={k:v.to(h.device)for k,v in layout.items()if k in (
            'video_coordinates','audio_coordinates','video_valid','audio_valid')}
        result=ConditionInputs(h,full['prefix_valid'],layer_ids=torch.tensor(layers,device=h.device),**fields)
        result.validate();return result

    def planning_loss(self,prepared,target,maximum=32768):
        return answer_loss(self.runtime.thinker,self._full(prepared,target,maximum))
