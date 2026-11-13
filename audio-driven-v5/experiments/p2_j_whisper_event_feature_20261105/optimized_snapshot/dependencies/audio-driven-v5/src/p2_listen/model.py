from torch import nn
from itertools import chain
from .events import EventHeads, event_sums


class ListeningExtractor(nn.Module):
    def __init__(self, config, head_width=128, *, feature_loss_enabled=True,
                 text_lora=None, renderer=None):
        super().__init__()
        from omni_av_p2.model import Extractor
        if type(feature_loss_enabled) is not bool:
            raise ValueError("feature_loss_enabled must be boolean")
        self.feature_loss_enabled = feature_loss_enabled
        self.extractor = Extractor(config, auxiliary=feature_loss_enabled)
        self.event_heads = EventHeads(config.video_dim, config.audio_dim, head_width)
        self.text_lora = nn.ModuleList()
        self.text_lora_targets = []
        self.text_lora_spec = text_lora
        if text_lora is not None:
            if renderer is None:
                raise ValueError("Text LoRA requires the renderer before model/optimizer/checkpoint construction")
            from omni_av_p21.config import LoRAConfig
            from omni_av_p21.lora import inject_text_lora
            self.text_lora, self.text_lora_targets = inject_text_lora(renderer, LoRAConfig(**text_lora))
            # Register only adapter objects, not the frozen 22B renderer. Both
            # forwards share these exact objects; DDP/checkpoints own them here.
            object.__setattr__(self, "_renderer", renderer)
            self.assert_parameter_ownership()

    @property
    def text_lora_enabled(self):
        return self.text_lora_spec is not None

    def assert_parameter_ownership(self):
        if not self.text_lora_enabled:
            return
        actual = {id(p) for p in self._renderer.parameters() if p.requires_grad}
        expected = {id(p) for p in self.text_lora.parameters()}
        if actual != expected:
            raise RuntimeError("Renderer trainables must be exactly the registered text LoRA parameters")
        owned = {id(p) for p in chain(self.extractor.parameters(), self.event_heads.parameters())} | expected
        if {id(p) for p in self.parameters()} != owned:
            raise RuntimeError("Incomplete or duplicate renderer parameter ownership")

    def main_parameters(self):
        return chain(self.extractor.parameters(), self.text_lora.parameters())

    def train(self, mode=True):
        super().train(mode)
        if self.text_lora_enabled:
            self._renderer.train(mode)
        return self

    def forward(self, condition, layout, targets=None, labels=None,
                event_mode="probe_only", pos_weights=None, generator=None,
                *, av_callback=None, av_pack=None):
        if targets is not None and not self.feature_loss_enabled:
            raise ValueError("Feature supervision is disabled; omit teacher targets")
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
        result = dict(u=u, logits=logits, features=features, event_sums=sums, event_counts=counts)
        if av_callback is not None:
            # The renderer uses adapters registered under this DDP forward.
            objective = av_callback(u, av_pack)
            if isinstance(objective, tuple):
                result["av"], result["av_components"] = objective
            else:
                result["av"] = objective
        return result

    def optimizer_groups(self, training):
        groups = [dict(name="connector", params=list(self.extractor.connector.parameters()), lr=training["connector_lr"])]
        if self.feature_loss_enabled:
            groups.append(dict(name="features", params=list(self.extractor.video_decoder.parameters()) +
                               list(self.extractor.audio_decoder.parameters()), lr=training["feature_lr"]))
        groups.append(dict(name="events", params=list(self.event_heads.parameters()), lr=training["event_lr"]))
        if self.text_lora_enabled:
            self.assert_parameter_ownership()
            if training.get("text_lora") != dict(enabled=True, **self.text_lora_spec):
                raise ValueError("Optimizer text LoRA settings differ from model")
            groups.append(dict(name="text_lora", params=list(self.text_lora.parameters()),
                               lr=self.text_lora_spec["learning_rate"]))
        return groups


def build_model(c, renderer=None):
    """Training, inference and replay must construct the same optional architecture."""
    from .config import extractor_config, feature_loss_enabled, text_lora_config, performance_config
    if renderer is not None and "performance" in c["training"]:
        from .performance import configure_renderer
        configure_renderer(renderer, performance_config(c)["gradient_checkpointing"])
    return ListeningExtractor(extractor_config(c), c["events"]["head_width"],
                              feature_loss_enabled=feature_loss_enabled(c),
                              text_lora=text_lora_config(c), renderer=renderer)
