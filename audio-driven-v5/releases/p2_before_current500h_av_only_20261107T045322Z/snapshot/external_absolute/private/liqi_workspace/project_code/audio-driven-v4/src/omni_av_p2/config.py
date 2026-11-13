from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    omni_path: str = "/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B"
    ltx_path: str = "/public/2_CKPT/2_AVATAR/LTX23/LTX_23/ltx-2.3-22b-dev.safetensors"
    gemma_path: str = "/public/2_CKPT/0_WM/LTX2.0_TEXT_ENCODER"
    hidden_size: int = 2048
    hidden_layers: tuple[int, ...] = (8, 18, 28, 35)
    layer_fusion: str = "attention"
    target_dim: int = 2048
    video_dim: int = 4096
    audio_dim: int = 2048
    fusion_width: int = 512
    fusion_depth: int = 2
    fusion_heads: int = 8
    decoder_width: int = 1024
    decoder_depth: int = 16
    diffusion_batch_mul: int = 16
    zero_init_connector: bool = True
    vision_fps: float = 2.0
    audio_hz: float = 4.0
    vision_max_pixels: int = 65536
    feature_video_weight: float = 1.0
    feature_audio_weight: float = 1.0
    av_weight: float = 1.0
    learning_rate: float = 1e-4

    def __post_init__(self):
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        if (not self.hidden_layers or any(type(n) is not int or n <= 0 for n in self.hidden_layers)
                or tuple(sorted(set(self.hidden_layers))) != self.hidden_layers):
            raise ValueError("hidden_layers must be positive, unique, ascending block-output indices")
        if self.layer_fusion not in {"attention", "weighted_sum", "single"}:
            raise ValueError("Unknown layer_fusion")
        if self.layer_fusion == "single" and len(self.hidden_layers) != 1:
            raise ValueError("Single-layer baseline requires exactly one hidden layer")
        for key in ("hidden_size", "target_dim", "video_dim", "audio_dim", "fusion_width",
                    "fusion_depth", "fusion_heads", "decoder_width", "decoder_depth", "diffusion_batch_mul"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        if self.fusion_width % self.fusion_heads or self.decoder_width % 2:
            raise ValueError("Invalid model width/head divisibility")
        for key in ("vision_fps", "audio_hz", "learning_rate", "av_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        for key in ("feature_video_weight", "feature_audio_weight"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"Invalid {key}")

    def to_dict(self):
        result = asdict(self)
        result["hidden_layers"] = list(self.hidden_layers)
        return result
