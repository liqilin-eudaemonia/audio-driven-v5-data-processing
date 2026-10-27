"""A single DDP forward owns Connector, auxiliary and renderer-LoRA gradients."""
from torch import nn

from omni_av_p2.ltx import append_conditions, av_loss
from omni_av_p2.model import Extractor
from .lora import inject_text_lora


class TrainingModel(nn.Module):
    def __init__(self, cfg, lora, renderer, strategy, auxiliary=True, extractor=None):
        super().__init__()
        self.extractor = extractor if extractor is not None else Extractor(cfg, auxiliary=auxiliary)
        self.adapters, self.targets = inject_text_lora(renderer, lora)
        self.cfg, self.lora = cfg, lora
        # Register the shared adapter objects, not the 22B frozen base. The same
        # tensors are used by renderer forward and synchronized by this DDP root.
        object.__setattr__(self, "renderer", renderer)
        object.__setattr__(self, "strategy", strategy)
        self.assert_parameter_ownership()

    def assert_parameter_ownership(self):
        actual = {id(p) for p in self.renderer.parameters() if p.requires_grad}
        expected = {id(p) for p in self.adapters.parameters()}
        if actual != expected:
            raise RuntimeError("Renderer trainables must be exactly registered text LoRA")
        if {id(p) for p in self.parameters()} != expected | {id(p) for p in self.extractor.parameters()}:
            raise RuntimeError("Incomplete DDP parameter ownership")

    def train(self, mode=True):
        super().train(mode)
        self.renderer.train(mode)
        return self

    def forward(self, condition, targets, prepared, generator=None):
        if targets is None:
            u, feature = self.extractor(condition), {}
        else:
            u, feature = self.extractor(condition, targets, generator)
        lav, parts = av_loss(self.renderer, self.strategy, append_conditions(prepared, u))
        lv, la = feature.get("video", lav.new_zeros(())), feature.get("audio", lav.new_zeros(()))
        total = self.cfg.av_weight * lav + self.cfg.feature_video_weight * lv + self.cfg.feature_audio_weight * la
        return total, lav, lv, la, parts["video_loss"], parts["audio_loss"]

    def optimizer_groups(self):
        return [dict(params=list(self.extractor.parameters()), lr=self.cfg.learning_rate, name="extractor"),
                dict(params=list(self.adapters.parameters()), lr=self.lora.learning_rate, name="text_lora")]
