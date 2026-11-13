"""Same full AV/Feature/clip-Event forward; expanded native LTX LoRA only."""
from p2_ltx_events.model import LTXHiddenEvents
from .lora import extend


class Model(LTXHiddenEvents):
    def __init__(self, c, cfg, renderer):
        super().__init__(c, cfg, renderer)
        extend(self)

    def groups(self):
        groups = super().groups()
        for group in groups:
            if group['name'] == 'text_lora':
                group['name'] = 'ltx_lora'
        return groups

