from torch import nn
from .events import EventHeads, event_sums


class ListeningExtractor(nn.Module):
    def __init__(self, config, head_width=128):
        super().__init__()
        from omni_av_p2.model import Extractor
        self.extractor = Extractor(config)
        self.event_heads = EventHeads(config.video_dim, config.audio_dim, head_width)

    def forward(self, condition, layout, targets=None, labels=None,
                event_mode="probe_only", pos_weights=None, generator=None):
        u = self.extractor.connector(condition)
        logits = self.event_heads(u, layout, event_mode)
        features = {}
        if targets is not None:
            for mod in ("video", "audio"):
                valid = targets[mod + "_valid"] & getattr(u, mod + "_valid")
                if valid.any():
                    features[mod] = getattr(self.extractor, mod + "_decoder").loss(targets[mod], getattr(u, mod), valid, generator)
                else:
                    # Keep every parameter in the DDP graph even for unknown tails.
                    features[mod] = sum(p.sum() * 0 for p in getattr(self.extractor, mod + "_decoder").parameters()) + getattr(u, mod).sum() * 0
        sums, counts = ({}, {}) if labels is None else event_sums(logits, labels, pos_weights)
        return dict(u=u, logits=logits, features=features, event_sums=sums, event_counts=counts)

    def optimizer_groups(self, training):
        return [dict(name="connector", params=list(self.extractor.connector.parameters()), lr=training["connector_lr"]),
                dict(name="features", params=list(self.extractor.video_decoder.parameters()) + list(self.extractor.audio_decoder.parameters()), lr=training["feature_lr"]),
                dict(name="events", params=list(self.event_heads.parameters()), lr=training["event_lr"])]
