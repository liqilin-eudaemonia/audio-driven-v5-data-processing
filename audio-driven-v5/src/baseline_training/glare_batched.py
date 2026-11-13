"""Exact-length batches within an unchanged effective update; no temporal padding."""
from collections import OrderedDict
import torch
from .glare_execution_fast import FastCachedFLOAT


def length_batches(values, limit):
    groups = OrderedDict()
    for value in values:
        groups.setdefault(len(value['motion']), []).append(value)
    for group in groups.values():
        for start in range(0, len(group), limit):
            yield group[start:start+limit]


class PairMaskedReactionLoss(torch.nn.Module):
    def forward(self, prediction, target):
        loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none')
        mask = self.mask.to(loss.dtype)
        # Preserve equal pair weighting, including zero contribution from unknown pairs.
        return ((loss * mask).sum((1,2)) / mask.sum((1,2)).clamp_min(1)).mean()


class BatchedCachedFLOAT(FastCachedFLOAT):
    def __init__(self, model):
        super().__init__(model)
        model.reaction_loss = PairMaskedReactionLoss()

    def forward(self, values):
        if isinstance(values, dict): values = [values]
        n = len(values[0]['motion'])
        assert all(len(v['motion']) == n for v in values)
        model = self.model; device = next(model.fmt.parameters()).device
        self.set_window(n, device); self._length = n-1
        def stack(key): return torch.stack([v[key] for v in values]).to(device,non_blocking=True)
        self._wav2vec = stack('wav2vec');self._emotion = stack('emotion')
        model.reaction_loss.mask = stack('reaction_valid')[:,1:]
        def reordered(key):
            x=stack(key);return torch.cat([x[:,1:],x[:,:1]],dim=1)
        audio=stack('audio');audio=torch.cat([audio[:,640:],audio.new_zeros(len(values),640)],dim=1)
        stats={k:torch.stack([v['statistics'][k] for v in values]).to(device,non_blocking=True)
               for k in values[0]['statistics']}
        try:
            return model(reordered('motion'),audio,stats,reordered('intensity'),reordered('reaction'))
        finally:
            self._wav2vec=self._emotion=None


def bucket_order(lengths, seed, limit):
    """Shuffle within exact lengths, then shuffle batches; retain every index once."""
    g=torch.Generator().manual_seed(seed)
    groups=OrderedDict()
    for i in torch.randperm(len(lengths),generator=g).tolist():
        groups.setdefault(int(lengths[i]),[]).append(i)
    batches=[v[start:start+limit] for v in groups.values() for start in range(0,len(v),limit)]
    return [i for j in torch.randperm(len(batches),generator=g).tolist() for i in batches[j]]
