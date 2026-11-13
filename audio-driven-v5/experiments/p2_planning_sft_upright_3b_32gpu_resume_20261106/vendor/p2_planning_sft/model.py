"""Language-only PEFT adapters and target-token causal loss."""
import torch
import torch.nn.functional as F


def build_model(config, adapter=None, trainable=True):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        config['model'], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa')
    base.requires_grad_(False)
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    else:
        spec = config['lora']
        # Anchor the regex at the Thinker language layers: no visual/audio adapters.
        targets = r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=spec['rank'],
            lora_alpha=spec['alpha'], lora_dropout=spec['dropout'], bias='none', target_modules=targets))
    model.config.use_cache = not trainable
    if trainable:
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not names or any('lora_' not in n or '.model.layers.' not in n for n in names):
            raise ValueError('Unexpected trainable parameters outside language LoRA')
    return model


def response_loss(outputs, labels, num_items_in_batch=None):
    # Discard ignored positions before the FP32 CE allocation. Qwen's native
    # forward in transformers 4.57.6 ignores num_items_in_batch in its own loss.
    targets = labels[:, 1:]
    keep = targets.ne(-100)
    logits = outputs.logits[:, :-1][keep].float()
    total = F.cross_entropy(logits, targets[keep], reduction='sum')
    denominator = keep.sum() if num_items_in_batch is None else num_items_in_batch
    return total / denominator
