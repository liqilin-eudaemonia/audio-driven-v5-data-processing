from torch import nn

from .connector import FusionConnector
from .decoder import EmbeddingDecoder


class Extractor(nn.Module):
    def __init__(self, config, auxiliary=True):
        super().__init__()
        self.connector = FusionConnector(config)
        if auxiliary:
            self.video_decoder = EmbeddingDecoder(config.target_dim, config.video_dim,
                config.decoder_width, config.decoder_depth, config.diffusion_batch_mul)
            self.audio_decoder = EmbeddingDecoder(config.target_dim, config.audio_dim,
                config.decoder_width, config.decoder_depth, config.diffusion_batch_mul)

    def forward(self, conditions, targets=None, generator=None):
        u = self.connector(conditions)
        # Keep auxiliary parameters inside DDP's forward/backward transaction.
        return u if targets is None else (u, self.feature_losses(u, targets, generator))

    def feature_losses(self, u, targets, generator=None):
        return {mod: getattr(self, mod + "_decoder").loss(targets[mod], getattr(u, mod),
                targets[mod + "_valid"] & getattr(u, mod + "_valid"), generator)
                for mod in ("video", "audio")}
