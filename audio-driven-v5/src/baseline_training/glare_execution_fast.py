"""Execution-only window reuse; no model, sampling or loss changes."""
from collections import OrderedDict
import torch
from .glare_cached import CachedFLOAT, configure_window


class FastCachedFLOAT(CachedFLOAT):
    def __init__(self, model, capacity=256):
        super().__init__(model)
        self.window_cache = OrderedDict()
        self.capacity = capacity

    def set_window(self, n, device):
        model = self.model
        key = (n, str(device))
        if key not in self.window_cache:
            configure_window(model, n, device)
            self.window_cache[key] = (model.fmt.pos_embed.data, model.fmt.alignment_mask)
            if len(self.window_cache) > self.capacity:
                self.window_cache.popitem(last=False)
        else:
            pos, mask = self.window_cache[key]
            model.num_frames_for_clip = model.fmt.num_frames_for_clip = n - 1
            model.num_total_frames = model.fmt.num_total_frames = n
            model.opt.wav2vec_sec = (n - 1) / 25
            model.fmt.pos_embed.data = pos
            model.fmt.alignment_mask = mask
            self.window_cache.move_to_end(key)

    def forward(self, value):
        model = self.model; device = next(model.fmt.parameters()).device
        n = len(value['motion']); self.set_window(n, device)
        self._length = n - 1
        self._wav2vec = value['wav2vec'][None].to(device, non_blocking=True)
        self._emotion = value['emotion'][None].to(device, non_blocking=True)
        model.reaction_loss.mask = value['reaction_valid'][1:][None].to(device, non_blocking=True)
        def reorder(x): return torch.cat([x[1:], x[:1]])[None].to(device, non_blocking=True)
        audio = torch.cat([value['audio'][640:], torch.zeros(640)])[None].to(device, non_blocking=True)
        stats = {k: v[None].to(device, non_blocking=True) for k, v in value['statistics'].items()}
        try:
            return model(reorder(value['motion']), audio, stats,
                         reorder(value['intensity']), reorder(value['reaction']))
        finally:
            self._wav2vec = self._emotion = None
