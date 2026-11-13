from dataclasses import asdict, dataclass
import json
import math

from omni_av_p2.config import Config


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 1e-5

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (self.alpha, self.learning_rate)):
            raise ValueError("Invalid LoRA scale or learning rate")

    def to_dict(self):
        return asdict(self)


def read_config(path=None):
    value = json.loads(path.read_text()) if path else {}
    if set(value) - {"model", "lora"}:
        raise ValueError("P2.1 config accepts only model and lora sections")
    # LoRA settings must not enter the frozen-perception P2 cache contract.
    return Config(**value.get("model", {})), LoRAConfig(**value.get("lora", {}))
