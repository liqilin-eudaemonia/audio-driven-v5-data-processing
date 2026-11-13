"""Same P3 objective with resident features, prepared masks and cached geometry.

Inputs in this module are owned, validated, immutable experiment data. Model weights,
loss reduction, mask/noise seeds and optimizer updates retain the pilot convention.
"""
from dataclasses import dataclass
from copy import deepcopy

import torch

from omni_av.masking import hybrid_attention
from .model import Layout, Planner, content_mask


@dataclass
class Resident:
    layout: Layout
    targets: torch.Tensor
    cpu_layout: Layout
    sample_id: str


def resident(row, device):
    cpu = Layout(**row["layout"])
    cpu.validate()
    if row["targets"].requires_grad or not torch.isfinite(row["targets"]).all():
        raise ValueError("Invalid teacher")
    if row["targets"].shape != (*cpu.valid.shape, cpu.prefix.shape[-1]):
        raise ValueError("Teacher/layout shape mismatch")
    return Resident(cpu.to(device), row["targets"].to(device), cpu, row["sample_id"])


def prepare_mask(data, cfg, seed, all_mask=False):
    masked = content_mask(data.cpu_layout, cfg, seed, all_mask)
    ids = [torch.where((masked & (data.cpu_layout.modalities == mod)).flatten())[0] for mod in (0, 1)]
    if any(len(x) == 0 for x in ids):
        raise ValueError("Both modalities require supervision")
    device = data.targets.device
    return masked.to(device), [x.to(device) for x in ids]


class FastPlanner(Planner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._geometry = {}

    def geometry(self, layout, dtype):
        key = (id(layout), dtype)
        if key not in self._geometry:
            layout.validate()
            coords = layout.coordinates.clone()
            coords[..., 0] = (coords[..., 0] * self.cfg.time_ticks_per_second).round()
            offset = layout.prefix_positions.amax(dim=(0, 2)) + 1
            positions = torch.cat([layout.prefix_positions, coords.long().permute(2, 0, 1)
                                   + offset[None, :, None]], -1)
            attention = hybrid_attention(layout.prefix_valid, layout.valid, dtype)
            # Retain the object, preventing id reuse. Clear after long evaluation suites.
            self._geometry[key] = (layout, positions, attention)
        return self._geometry[key][1:]

    def hidden(self, layout, masked, values):
        positions, attention = self.geometry(layout, values.dtype)
        mod = layout.modalities
        target = torch.where(masked[..., None], self.mask_token[mod].to(values.dtype), values)
        target = (target + self.modality_embedding[mod].to(values.dtype)) * layout.valid[..., None]
        embeds = torch.cat([layout.prefix.detach(), target], 1)
        out = self.thinker.model(inputs_embeds=embeds, attention_mask=attention,
            position_ids=positions, use_cache=False, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-2][:, layout.prefix.shape[1]:]

    def prepared_loss(self, data, masked, ids, seed, sigma_override=None):
        h = self.hidden(data.layout, masked, data.targets).reshape(-1, self.cfg.hidden_size)
        return self.loss_from_hidden(h,data,ids,seed,sigma_override)

    def prepared_batch_loss(self, data, masks, ids, seeds):
        """One batched Thinker forward; keep each example's mean loss and RNG unchanged."""
        b=len(data)
        p=max(x.layout.prefix.shape[1] for x in data)
        n=max(x.targets.shape[1] for x in data)
        d=self.cfg.hidden_size
        first=data[0]
        prefix=first.layout.prefix.new_zeros((b,p,d))
        positions=first.layout.prefix_positions.new_zeros((3,b,p))
        prefix_valid=first.layout.prefix_valid.new_zeros((b,p))
        coordinates=first.layout.coordinates.new_zeros((b,n,3))
        modalities=first.layout.modalities.new_zeros((b,n))
        valid=first.layout.valid.new_zeros((b,n))
        target=first.targets.new_zeros((b,n,d))
        masked=first.layout.valid.new_zeros((b,n))
        for j,item in enumerate(data):
            pi,ni=item.layout.prefix.shape[1],item.targets.shape[1]
            prefix[j,:pi]=item.layout.prefix[0]
            positions[:,j,:pi]=item.layout.prefix_positions[:,0]
            prefix_valid[j,:pi]=item.layout.prefix_valid[0]
            coordinates[j,:ni]=item.layout.coordinates[0]
            modalities[j,:ni]=item.layout.modalities[0]
            valid[j,:ni]=item.layout.valid[0]
            target[j,:ni]=item.targets[0]
            masked[j,:ni]=masks[j][0]
        layout=Layout(prefix,positions,prefix_valid,coordinates,modalities,valid)
        hidden=self.hidden(layout,masked,target)
        losses=[self.loss_from_hidden(hidden[j],data[j],ids[j],seeds[j]) for j in range(b)]
        return {name:torch.stack([loss[name] for loss in losses]).mean() for name in ('video','audio')}

    def loss_from_hidden(self, h, data, ids, seed, sigma_override=None):
        targets = data.targets.reshape(-1, self.cfg.hidden_size)
        losses = {}
        for mod, name in enumerate(("video", "audio")):
            c = self.connectors[mod](h.index_select(0, ids[mod]).float())
            z = targets.index_select(0, ids[mod]).detach().float().repeat_interleave(self.cfg.diffusion_batch_mul, 0)
            c = c.repeat_interleave(self.cfg.diffusion_batch_mul, 0)
            generator = torch.Generator(device=z.device).manual_seed(seed + mod + 100)
            noise = torch.randn(z.shape, device=z.device, generator=generator)
            sigma = torch.rand(z.shape[0], device=z.device, generator=generator)
            if sigma_override is not None:
                sigma.fill_(sigma_override)
            x = z * (1 - sigma[:, None]) + noise * sigma[:, None]
            pred = self.decoders[mod].net(x.to(c.dtype), (1000 * sigma).to(c.dtype), c).float()
            losses[name] = (pred - (noise - z)).square().mean()
        return losses


def execution_mode(model, fast):
    if fast:
        model.thinker.model.gradient_checkpointing_disable()
    else:
        model.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for decoder in model.decoders:
        decoder.net.grad_checkpointing = not fast


def optimizer_for(model, checkpoint=None, fused=True):
    heads, adapters = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (adapters if name.startswith("thinker.") else heads).append(p)
    opt = torch.optim.AdamW([dict(params=heads, lr=model.cfg.heads_lr),
                            dict(params=adapters, lr=model.cfg.lora_lr)], weight_decay=0,
                           fused=fused)
    if checkpoint is not None:
        # AdamW may keep CPU step tensors by reference; never mutate the source checkpoint.
        opt.load_state_dict(deepcopy(checkpoint["optimizer"]))
        # Old checkpoints recorded fused=None. Explicitly select the measured execution mode.
        for group in opt.param_groups:
            group["fused"], group["foreach"] = fused, False if fused else None
        if fused:
            for p, state in opt.state.items():
                if "step" in state:
                    state["step"] = state["step"].to(p.device)
    return opt, heads + adapters


def audit_gradients(model):
    maxima = {"mask": [], "connector": [], "decoder": [], "lora": []}
    for name, p in model.named_parameters():
        if not p.requires_grad and p.grad is not None:
            raise RuntimeError(f"Frozen gradient: {name}")
        if p.grad is not None:
            group = "lora" if name.startswith("thinker.") else "decoder" if name.startswith("decoders.") else "connector" if name.startswith("connectors.") else "mask"
            maxima[group].append(p.grad.detach().abs().amax())
    # One device-to-host transfer, instead of hundreds of serial scalar synchronizations.
    device = model.mask_token.device
    packed = torch.stack([torch.stack(v).max() if v else torch.zeros((), device=device)
                          for v in maxima.values()]).cpu()
    if not torch.isfinite(packed).all():
        raise RuntimeError("Nonfinite audited gradient")
    return dict(zip(maxima, packed.tolist()))
