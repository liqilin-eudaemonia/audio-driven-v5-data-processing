"""Bounded two-group CUDA transfer prefetch for native FLOAT training."""
import torch
from baseline_training.glare_batched import BatchedCachedFLOAT


def prepare(values, device, stream):
    def stack(key):
        return torch.stack([v[key] for v in values]).pin_memory()
    host = {k: stack(k) for k in ('motion', 'audio', 'wav2vec', 'emotion', 'reaction', 'reaction_valid', 'intensity')}
    host['statistics'] = {k: torch.stack([v['statistics'][k] for v in values]).pin_memory()
                          for k in values[0]['statistics']}
    with torch.cuda.stream(stream):
        result = {k: v.to(device, non_blocking=True) for k, v in host.items() if k != 'statistics'}
        result['statistics'] = {k: v.to(device, non_blocking=True) for k, v in host['statistics'].items()}
        ready = torch.cuda.Event()
        ready.record(stream)
    return result, ready, host


def prefetched(groups, device):
    stream = torch.cuda.Stream(device=device)
    current = None
    for group in groups:
        upcoming = prepare(group, device, stream)
        if current is not None:
            tensors, ready, host = current
            torch.cuda.current_stream(device).wait_event(ready)
            for key, value in tensors.items():
                if key == 'statistics':
                    for tensor in value.values(): tensor.record_stream(torch.cuda.current_stream(device))
                else: value.record_stream(torch.cuda.current_stream(device))
            yield tensors
        current = upcoming
    if current is not None:
        tensors, ready, host = current
        torch.cuda.current_stream(device).wait_event(ready)
        for key, value in tensors.items():
            if key == 'statistics':
                for tensor in value.values(): tensor.record_stream(torch.cuda.current_stream(device))
            else: value.record_stream(torch.cuda.current_stream(device))
        yield tensors


class PreparedFLOAT(BatchedCachedFLOAT):
    def forward(self, values):
        if not isinstance(values, dict) or 'motion' not in values or values['motion'].ndim != 3:
            return super().forward(values)
        model = self.model
        motion = values['motion']
        n = motion.shape[1]
        self.set_window(n, motion.device)
        self._length = n - 1
        self._wav2vec = values['wav2vec']
        self._emotion = values['emotion']
        model.reaction_loss.mask = values['reaction_valid'][:, 1:]
        def reordered(x): return torch.cat([x[:, 1:], x[:, :1]], dim=1)
        audio = values['audio']
        audio = torch.cat([audio[:, 640:], audio.new_zeros(len(audio), 640)], dim=1)
        try:
            return model(reordered(motion), audio, values['statistics'],
                         reordered(values['intensity']), reordered(values['reaction']))
        finally:
            self._wav2vec = self._emotion = None
