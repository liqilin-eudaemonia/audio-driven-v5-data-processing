from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlanConfig:
    omni_path: str = "/public/2_CKPT/3_PreTRAINED/Qwen2.5-Omni-3B"
    ltx_path: str = "/public/2_CKPT/2_AVATAR/LTX23/LTX_23/ltx-2.3-22b-dev.safetensors"
    gemma_path: str = "/public/2_CKPT/0_WM/LTX2.0_TEXT_ENCODER"
    hidden_size: int = 2048
    lora_rank: int = 16
    lora_alpha: float = 32.0
    flow_width: int = 512
    flow_depth: int = 3
    diffusion_batch_mul: int = 4
    vision_fps: float = 2.0
    audio_hz: float = 4.0
    vision_max_pixels: int = 65536
    time_ticks_per_second: int = 25
    planning_steps: int = 8
    denoising_steps: int = 5
    mask_min: float = 0.7
    all_mask_probability: float = 0.2
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for key in ("hidden_size", "lora_rank", "flow_width", "flow_depth", "diffusion_batch_mul",
                    "vision_max_pixels", "time_ticks_per_second", "planning_steps", "denoising_steps"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not (0 < self.mask_min <= 1 and 0 <= self.all_mask_probability <= 1):
            raise ValueError("Invalid masking probabilities")
        if self.vision_fps <= 0 or self.audio_hz <= 0 or self.lora_alpha <= 0:
            raise ValueError("Invalid rate or LoRA alpha")

    def to_dict(self):
        return asdict(self)
