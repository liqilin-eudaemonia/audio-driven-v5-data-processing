"""Separate architecture contract; old concatenation checkpoints remain legacy."""
from dataclasses import asdict, dataclass
import json
import math
from omni_av_p2.config import Config
from .config import LoRAConfig

VERSION = 'p21_qwen_residual_three_stage_v1'
STAGES = {1: 'connector_features', 2: 'connector_qwen_av', 3: 'connector_qwen_dit_lora'}

@dataclass(frozen=True)
class ResidualConfig:
    learning_rate: float = 1e-4
    shard_optimizer: bool = True

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Invalid Qwen attention learning rate')
        if type(self.shard_optimizer) is not bool:
            raise ValueError('shard_optimizer must be boolean')

    def to_dict(self):
        return asdict(self)


def read_config(path=None):
    data = json.loads(path.read_text()) if path else {}
    if set(data) - {'model', 'lora', 'residual'}:
        raise ValueError('Expected model/lora/residual configuration')
    cfg = Config(**data.get('model', {}))
    if min(cfg.feature_video_weight, cfg.feature_audio_weight) <= 0:
        raise ValueError('Three-stage recipe requires both auxiliary feature objectives')
    return cfg, LoRAConfig(**data.get('lora', {})), ResidualConfig(**data.get('residual', {}))
