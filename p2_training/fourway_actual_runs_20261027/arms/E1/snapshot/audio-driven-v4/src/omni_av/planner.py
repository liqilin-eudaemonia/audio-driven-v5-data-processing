from dataclasses import dataclass

import torch
from torch import nn

from .flow import FeatureFlow
from .masking import fill_schedule, hybrid_attention, sample_target_mask


@dataclass
class PlanningBatch:
    prefix: torch.Tensor  # B,P,D, frozen perception + text embeddings
    prefix_positions: torch.Tensor  # 3,B,P
    prefix_valid: torch.Tensor  # B,P
    text_mask: torch.Tensor  # B,P, only response-plan text
    targets: torch.Tensor  # B,N,D, teacher features (training only)
    coordinates: torch.Tensor  # B,N,3, seconds / spatial row / spatial column
    modalities: torch.Tensor  # B,N, 0=video, 1=audio
    valid: torch.Tensor  # B,N

    def validate(self):
        b, n, d = self.targets.shape
        p = self.prefix.shape[1]
        if self.prefix.shape != (b, p, d) or self.prefix_positions.shape != (3, b, p):
            raise ValueError("Prefix dimensions mismatch")
        if self.coordinates.shape != (b, n, 3) or self.modalities.shape != (b, n):
            raise ValueError("Target layout mismatch")
        for x, shape in ((self.valid, (b, n)), (self.prefix_valid, (b, p)), (self.text_mask, (b, p))):
            if x.shape != shape or x.dtype != torch.bool:
                raise ValueError("Invalid validity mask")
        if not self.valid.any(-1).all() or not self.prefix_valid.any(-1).all():
            raise ValueError("Empty sample")
        if not torch.isfinite(self.targets).all() or not torch.isfinite(self.prefix).all():
            raise ValueError("Nonfinite feature data")
        if ((self.modalities[self.valid] < 0) | (self.modalities[self.valid] > 1)).any():
            raise ValueError("Unknown modality")
        if (self.text_mask & ~self.prefix_valid).any():
            raise ValueError("Text mask selects padding")

    def to(self, device, dtype=torch.bfloat16):
        values = {}
        for key, value in vars(self).items():
            target_dtype = dtype if key in {"prefix", "targets"} else value.dtype
            values[key] = value.to(device=device, dtype=target_dtype)
        return PlanningBatch(**values)


class AVPlanner(nn.Module):
    def __init__(self, thinker, config):
        super().__init__()
        self.thinker, self.config = thinker, config
        d = config.hidden_size
        self.mask_token = nn.Parameter(torch.zeros(2, d))
        self.modality_embedding = nn.Parameter(torch.randn(2, d) * 0.01)
        self.video_flow = FeatureFlow(d, d, config.flow_width, config.flow_depth)
        self.audio_flow = FeatureFlow(d, d, config.flow_width, config.flow_depth)

    def train(self, mode=True):
        super().train(mode)
        self.thinker.visual.eval()
        self.thinker.audio_tower.eval()
        return self

    def hidden(self, batch, masked, values=None):
        batch.validate()
        if masked.shape != batch.valid.shape or (masked & ~batch.valid).any():
            raise ValueError("Invalid content mask")
        values = batch.targets if values is None else values
        ids = batch.modalities.clamp(0, 1)
        target = torch.where(masked[..., None], self.mask_token[ids].to(values.dtype), values)
        target = target + self.modality_embedding[ids].to(values.dtype)
        target = target * batch.valid[..., None]
        embeddings = torch.cat([batch.prefix.detach(), target], 1)
        coords = batch.coordinates.clone()
        coords[..., 0] = (coords[..., 0] * self.config.time_ticks_per_second).round()
        offset = batch.prefix_positions.amax(dim=(0, 2)) + 1
        positions = coords.long().permute(2, 0, 1) + offset[None, :, None]
        positions = torch.cat([batch.prefix_positions, positions], -1)
        mask = hybrid_attention(batch.prefix_valid, batch.valid, embeddings.dtype)
        outputs = self.thinker.model(inputs_embeds=embeddings, attention_mask=mask,
                                     position_ids=positions, use_cache=False,
                                     output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[-2]

    def forward(self, batch, masked=None):
        if masked is None:
            masked = sample_target_mask(batch.valid, batch.coordinates[..., 0], batch.modalities,
                                        self.config.mask_min, self.config.all_mask_probability)
        hidden = self.hidden(batch, masked)
        target_hidden = hidden[:, batch.prefix.shape[1]:]
        losses = {}
        for mod, name, decoder in ((0, "plan_video", self.video_flow), (1, "plan_audio", self.audio_flow)):
            selected = masked & batch.valid & (batch.modalities == mod)
            losses[name] = decoder.loss(batch.targets[selected], target_hidden[selected],
                                        self.config.diffusion_batch_mul)
        visible = torch.cat([batch.text_mask, batch.valid & ~masked], 1)
        return {"loss": sum(losses.values()), "losses": losses, "hidden": hidden,
                "renderer_mask": visible, "masked": masked}

    @torch.no_grad()
    def plan(self, layout, generator=None, steps=None, denoising_steps=None):
        """Only layout/prefix are consumed. Caller must supply zero target storage.

        Reject nonzero targets so this API cannot silently become an Oracle sampler.
        """
        if torch.count_nonzero(layout.targets):
            raise ValueError("Inference requires an empty target layout, not GT features")
        values = torch.zeros_like(layout.targets)
        masked = layout.valid.clone()
        steps = steps or self.config.planning_steps
        orders = []
        schedules = []
        for i in range(masked.shape[0]):
            idx = torch.where(masked[i])[0]
            order = idx[torch.randperm(len(idx), device=idx.device, generator=generator)]
            orders.append(order)
            schedules.append(list(fill_schedule(len(idx), steps)))
        history = []
        for k in range(steps):
            select = torch.zeros_like(masked)
            for i, order in enumerate(orders):
                begin, end = schedules[i][k]
                select[i, order[begin:end]] = True
            if select.any():
                hidden = self.hidden(layout, masked, values)[:, layout.prefix.shape[1]:]
                for mod, decoder in ((0, self.video_flow), (1, self.audio_flow)):
                    positions = select & (layout.modalities == mod)
                    if positions.any():
                        values[positions] = decoder.sample(hidden[positions],
                            denoising_steps or self.config.denoising_steps, generator)
                masked[select] = False
            history.append(int(masked.sum()))
        if masked.any():
            raise RuntimeError("Unfilled planning positions")
        hidden = self.hidden(layout, masked, values)
        return {"features": values, "hidden": hidden,
                "renderer_mask": torch.cat([layout.text_mask, layout.valid], 1), "remaining": history}
