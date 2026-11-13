from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ltx_trainer.quantization import QuantizationOptions
from ltx_trainer.training_strategies.base_strategy import TrainingStrategyConfigBase
from ltx_trainer.training_strategies.text_to_video import (
    TextToVideoConfig,
    uses_online_qwen_conditioning,
    uses_online_text_conditioning,
)
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_path: str | Path = Field(
        ...,
        description="Model path - local path to safetensors checkpoint file",
    )

    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Path to text encoder (required for LTX-2/Gemma models, optional for LTXV/T5 models)",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str | Path) -> str | Path:
        """Validate that model_path is either a valid URL or an existing local path."""
        is_url = str(v).startswith(("http://", "https://"))

        if is_url:
            raise ValueError(f"Model path cannot be a URL: {v}")

        if not Path(v).exists():
            raise ValueError(f"Model path does not exist: {v}")

        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="List of modules to target with LoRA",
    )

    exclude_modules: list[str] | str | None = Field(
        default=None,
        description="Optional PEFT module names or regex to exclude from LoRA targeting. "
        "Use a regex when excluding a whole module subtree such as Qwen cross-attention.",
    )

    modules_to_save: list[str] = Field(
        default_factory=list,
        description="Transformer submodules to train fully and persist alongside LoRA weights. "
        "These modules are manually unfrozen and explicitly added to the checkpoint instead of "
        "using PEFT modules_to_save, which would duplicate large injection branches.",
    )


class Stage2LoraConfig(ConfigBaseModel):
    """Second-stage LoRA on top of a learned Stage-1 conditioning checkpoint."""

    enabled: bool = False
    injection_lora_enabled: bool = True
    injection_rank: int = Field(default=32, ge=1)
    injection_alpha: int = Field(default=32, ge=1)
    injection_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    injection_modules: list[str] = Field(
        default_factory=lambda: ["qwen_attn2", "audio_qwen_attn2"]
    )
    adapter_modules: list[str] = Field(
        default_factory=lambda: ["video_qwen_layer_adapter", "audio_qwen_layer_adapter"]
    )
    full_trainable_modules: list[str] = Field(default_factory=list)
    train_adapter_norm_and_logits: bool = True
    qwen_training_mode: Literal["frozen", "lora"] = "lora"
    qwen_rank: int = Field(default=32, ge=1)
    qwen_alpha: int = Field(default=64, ge=1)
    qwen_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    qwen_last_n_layers: int = Field(default=36, ge=1)
    qwen_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    qwen_gradient_checkpointing: bool = True
    qwen_fsdp: bool = False
    injection_learning_rate: float = Field(default=5e-5, gt=0.0)
    qwen_learning_rate: float = Field(default=1e-5, gt=0.0)
    qwen_checkpoint: str | Path | None = None


def _get_strategy_discriminator(v: dict | TrainingStrategyConfigBase) -> str:
    """Discriminator function for strategy config union."""
    if isinstance(v, dict):
        return v.get("name", "text_to_video")
    return v.name


# Union type for all strategy configs with discriminator
TrainingStrategyConfig = Annotated[
    Annotated[TextToVideoConfig, Tag("text_to_video")]
    | Annotated[VideoToVideoConfig, Tag("video_to_video")],
    Discriminator(_get_strategy_discriminator),
]


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="AdamW decay applied to matrix weights; biases, normalization/"
        "scalar parameters, layer logits, and learned Resampler queries use zero decay.",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "step",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    offload_optimizer_during_validation: bool = Field(
        default=False,
        description="Offload optimizer state to CPU before validation video sampling and reload "
        "it afterwards, to free VRAM for inference. Useful when optimizer state is large "
        "(e.g. AdamW for full fine-tuning or high-rank LoRA) and validation OOMs because the "
        "VAE decoder + transformer + optimizer state cannot coexist on the GPU. Has no effect "
        "for FSDP (sharded state). Disabled by default.",
    )

    ddp_find_unused_parameters: bool = Field(
        default=True,
        description="Enable DDP unused-parameter graph traversal. Disable only when every "
        "trainable transformer branch participates in every forward pass.",
    )

    ddp_gradient_as_bucket_view: bool = Field(
        default=False,
        description="Make DDP gradients views into communication buckets to reduce copies and memory.",
    )

    context_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Degree of sequence parallelism / context parallelism (e.g. 1, 2, 4, 8).",
    )


class LengthBucketConfig(ConfigBaseModel):
    """Opt-in UniAV-style length-bucket sampling for multi-length precomputed latents.

    When ``enabled=true``, the trainer uses ``LengthBucketOnlineDataset`` +
    ``LengthBucketSampler`` instead of the legacy single-root shuffled DataLoader.
    Existing configs leave this disabled and are unchanged.
    """

    enabled: bool = Field(
        default=False,
        description="Enable length-bucket multi-root online dataset + synced bucket sampler.",
    )

    sync_bucket_across_ranks: bool = Field(
        default=True,
        description="If True, every rank selects the same length bucket on a given step "
        "(UniAV default). Prevents FSDP stragglers from mismatched sequence lengths.",
    )

    max_batches_per_epoch: int | None = Field(
        default=None,
        description="Optional cap on global batches per sampler epoch. None = no cap "
        "(sum of ceil(bucket_size / (world_size * batch_size)) across buckets).",
        gt=0,
    )

    allowed_frames: list[int] | None = Field(
        default=None,
        description="Optional allow-list of pixel frame counts (e.g. [73, 121, 177]). "
        "None keeps every root whose path contains ltxNNN.",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        default="",
        description="Path to folder containing preprocessed training data "
        "(legacy single-root path; required when length_bucket.enabled is false).",
    )

    preprocessed_data_roots: list[str] = Field(
        default_factory=list,
        description="Multi-root precomputed latent directories for length-bucket training "
        "(e.g. .../precomputed_1088_640_sd_ltx73_av, ..._ltx121_av). Used only when "
        "length_bucket.enabled is true. Each path must contain an 'ltxNNN' segment.",
    )

    precomputed_sample_index_files: list[str] = Field(
        default_factory=list,
        description="Optional prevalidated per-root JSONL/TXT sample indexes. When set, "
        "PrecomputedDataset loads relative cache paths from these files and skips recursive "
        "source-directory discovery. Must align one-to-one with preprocessed_data_roots.",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )

    metadata_jsonls: list[str] = Field(
        default_factory=list,
        description="Metadata JSONL files used to join cached latents/conditions to raw captions and videos. "
        "Required whenever Gemma or Qwen conditioning is online and for length-bucket training.",
    )

    caption_field: str = Field(
        default="sd_text",
        description="JSONL column holding the full raw caption for online Gemma and/or Qwen encoding. "
        "The text is used verbatim (no truncation of trailing audio/music analysis).",
    )

    video_field: str = Field(
        default="video",
        description="JSONL column holding the source video path (used to read the first frame "
        "for online Qwen conditioning).",
    )

    path_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Optional prefix remapping applied to JSONL video paths, e.g. "
        '{"/mnt/data04/480634/public": "/public"} when metadata was written on another cluster.',
    )

    max_training_frames: int | None = Field(
        default=None,
        ge=1,
        description="Optional pixel-frame cap for prefix-cropping paired video/audio latents at load time.",
    )

    length_bucket: LengthBucketConfig = Field(
        default_factory=LengthBucketConfig,
        description="Opt-in multi-length bucket sampler (default disabled; legacy path unchanged).",
    )

    @model_validator(mode="after")
    def validate_data_roots(self) -> "DataConfig":
        if self.max_training_frames is not None and self.max_training_frames % 8 != 1:
            raise ValueError("data.max_training_frames must satisfy F % 8 == 1")
        if self.length_bucket.enabled:
            if not self.preprocessed_data_roots and not self.preprocessed_data_root:
                raise ValueError(
                    "length_bucket.enabled=true requires data.preprocessed_data_roots "
                    "(or a fallback data.preprocessed_data_root)"
                )
            root_count = len(self.preprocessed_data_roots) or 1
            if (
                self.precomputed_sample_index_files
                and len(self.precomputed_sample_index_files) != root_count
            ):
                raise ValueError(
                    "data.precomputed_sample_index_files must align one-to-one with the "
                    f"resolved data roots: {len(self.precomputed_sample_index_files)} indexes "
                    f"for {root_count} roots"
                )
        elif not self.preprocessed_data_root:
            raise ValueError(
                "data.preprocessed_data_root is required when length_bucket.enabled is false"
            )
        return self


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    metadata_jsonl: str | None = Field(
        default=None,
        description="Optional JSONL file providing validation samples (caption + video path per row). "
        "When set, prompts are taken from jsonl_caption_field and I2V condition images are the "
        "first frames of jsonl_video_field videos. Mutually exclusive with prompts/images.",
    )

    from_train: bool = Field(
        default=False,
        description="Build a deterministic fixed validation JSONL from the training metadata. "
        "The metadata file matching validation.video_dims frames is preferred.",
    )

    from_train_seed: int = Field(
        default=20260806,
        description="Seed used to select the fixed validation rows from training metadata.",
    )

    from_train_metadata_jsonl: str | None = Field(
        default=None,
        description="Optional training metadata JSONL used for validation selection. "
        "When omitted, a matching entry is selected from data.metadata_jsonls.",
    )

    from_train_wait_timeout_seconds: int = Field(
        default=600,
        description="How long nonzero ranks wait for rank 0 to materialize the shared validation JSONL.",
        gt=0,
    )

    jsonl_caption_field: str = Field(
        default="sd_text",
        description="JSONL column holding the validation prompt text (used verbatim).",
    )

    jsonl_video_field: str = Field(
        default="video",
        description="JSONL column holding the video path whose first frame conditions I2V validation.",
    )

    jsonl_max_samples: int | None = Field(
        default=None,
        description="Optional cap on the number of validation samples taken from metadata_jsonl.",
        ge=1,
    )

    omni_condition_version: Literal["teacher_forcing_raw", "generated_raw"] | None = Field(
        default=None,
        description="Optional strict Omni package version for validation. "
        "Generated-Plan end-to-end evaluation sets generated_raw explicitly; "
        "ordinary Oracle validation derives teacher_forcing_raw from the training strategy.",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    qwen_condition_files: list[str] | None = Field(
        default=None,
        description="Optional list of precomputed Qwen/Qwen-VL condition .pt files to use during validation. "
        "One file path must be provided for each validation prompt when set.",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    reference_downscale_factor: int = Field(
        default=1,
        description="Downscale factor for reference videos in IC-LoRA validation. "
        "When > 1, reference videos are processed at 1/n resolution (e.g., 2 means half resolution). "
        "Must match the factor used during dataset preprocessing.",
        ge=1,
    )

    video_dims: tuple[int, int, int] = Field(
        default=(960, 544, 97),
        description="Dimensions of validation videos (width, height, frames). "
        "Width and height must be divisible by 32. Frames must satisfy frames % 8 == 1 for LTX-2.",
    )

    @field_validator("video_dims")
    @classmethod
    def validate_video_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate video dimensions for LTX-2 compatibility."""
        width, height, frames = v

        if width % 32 != 0:
            raise ValueError(f"Width ({width}) must be divisible by 32")
        if height % 32 != 0:
            raise ValueError(f"Height ({height}) must be divisible by 32")
        if frames % 8 != 1:
            raise ValueError(
                f"Frames ({frames}) must satisfy frames % 8 == 1 for LTX-2 (e.g., 1, 9, 17, 25, ...)"
            )

        return v

    frame_rate: float = Field(
        default=25.0,
        description="Frame rate for validation videos",
        gt=0,
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    extended_metadata_jsonl: str | None = Field(
        default=None,
        description="Optional larger validation JSONL evaluated at extended_interval.",
    )

    extended_interval: int | None = Field(
        default=None,
        description="Step interval for the optional larger validation profile.",
        gt=0,
    )

    extended_jsonl_caption_field: str = Field(
        default="prompt",
        description="Caption field used by extended_metadata_jsonl.",
    )

    extended_jsonl_video_field: str = Field(
        default="video",
        description="Reference video field used by extended_metadata_jsonl.",
    )

    extended_max_samples: int = Field(
        default=40,
        description="Maximum samples loaded from extended_metadata_jsonl.",
        ge=1,
    )

    guidance_scale: float = Field(
        default=4.0,
        description="CFG guidance scale to use during validation",
        ge=1.0,
    )

    stg_scale: float = Field(
        default=1.0,
        description="STG (Spatio-Temporal Guidance) scale. 0.0 disables STG. "
        "Recommended value is 1.0. STG is combined with CFG for improved video quality.",
        ge=0.0,
    )

    stg_blocks: list[int] | None = Field(
        default=[29],
        description="Which transformer blocks to perturb for STG. "
        "None means all blocks are perturbed. Recommended for LTX-2: [29].",
    )

    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_av",
        description="STG mode: 'stg_av' skips both audio and video self-attention, "
        "'stg_v' skips only video self-attention.",
    )

    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio in validation samples. "
        "Independent of training strategy setting - you can generate audio "
        "in validation even when not training the audio branch.",
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    run_only: bool = Field(
        default=False,
        description="Initialize the model, run validation at the current checkpoint step, and exit without training.",
    )

    metrics: list[Literal["id_similarity", "fid", "fvd", "sync_c"]] = Field(
        default_factory=list,
        description="Optional benchmark metrics. The production protocol enables all four supported metrics.",
    )
    allow_small_sample_metrics: bool = Field(
        default=False,
        description=(
            "Allow diagnostic four-metric scoring below eight samples. "
            "FID/FVD remain reported but are not statistically stable."
        ),
    )

    metric_prefix: str = Field(
        default="validation",
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="TensorBoard/W&B namespace for primary validation benchmark metrics.",
    )
    extended_metric_prefix: str = Field(
        default="validation_extended",
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="TensorBoard/W&B namespace for extended validation benchmark metrics.",
    )
    plain_prompt_is_transcript: bool = Field(
        default=False,
        description="Treat an unstructured validation prompt as the spoken transcript for SyncNet eligibility.",
    )

    id_python: str = Field(default="/private/software/conda/envs/ovi/bin/python")
    id_model_root: str = Field(
        default="/private/codes/exp/boxian_workspace/models/benchmark/insightface"
    )
    id_model_name: str = Field(default="buffalo_l")
    id_sample_frames: int = Field(default=8, gt=0)
    id_cpu: bool = Field(default=True)
    benchmark_root: str = Field(
        default="/private/codes/exp/boxian_workspace/codes/benchmark"
    )
    avbench_python: str = Field(
        default="/private/software/conda/envs/avbench/bin/python"
    )
    vocal_sync_python: str = Field(
        default="/private/software/conda/envs/vocal_sync/bin/python"
    )
    syncnet_root: str = Field(
        default="/private/codes/joint_gen/complex_motion_train/syncnet_python"
    )
    metric_device: str = Field(default="cuda:0")
    syncnet_gpus: str = Field(default="0")
    fid_frames: int = Field(default=4, gt=0)
    fvd_frames: int = Field(default=16, gt=1)
    metric_timeout_seconds: int = Field(default=3600, gt=0)
    metric_failure_policy: Literal["warn", "error"] = Field(
        default="warn",
        description="Whether metric-process failures continue training or raise. run_only always raises.",
    )

    include_reference_in_output: bool = Field(
        default=False,
        description="For video-to-video training: concatenate the original reference video side-by-side "
        "with the generated output. The reference comes from the input video, not from the model's output.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(
                f"Number of images ({len(v)}) must match number of prompts ({num_prompts})"
            )

        for image_path in v:
            if not Path(image_path).exists():
                raise ValueError(f"Image path '{image_path}' does not exist")

        return v

    @field_validator("qwen_condition_files")
    @classmethod
    def validate_qwen_condition_files(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Validate that number of Qwen condition files (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if len(v) != num_prompts:
            raise ValueError(
                f"Number of qwen_condition_files ({len(v)}) must match number of prompts ({num_prompts})"
            )

        for condition_path in v:
            if not Path(condition_path).exists():
                raise ValueError(
                    f"Qwen condition file '{condition_path}' does not exist"
                )

        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_reference_videos(
        cls, v: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        if v is None:
            return None

        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(
                f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})"
            )

        for video_path in v:
            if not Path(video_path).exists():
                raise ValueError(f"Reference video path '{video_path}' does not exist")

        return v

    @model_validator(mode="after")
    def validate_scaled_reference_dimensions(self) -> "ValidationConfig":
        """Validate that scaled reference dimensions are valid when reference_downscale_factor > 1."""
        if self.reference_downscale_factor > 1:
            width, height, _frames = self.video_dims

            # Validate that downscale factor evenly divides the target dimensions
            if width % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Width {width} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {width} evenly."
                )
            if height % self.reference_downscale_factor != 0:
                raise ValueError(
                    f"Height {height} is not evenly divisible by reference_downscale_factor "
                    f"{self.reference_downscale_factor}. Choose a downscale factor that divides {height} evenly."
                )

            scaled_width = width // self.reference_downscale_factor
            scaled_height = height // self.reference_downscale_factor

            # Validate scaled dimensions are divisible by 32
            if scaled_width % 32 != 0:
                raise ValueError(
                    f"Scaled reference width {scaled_width} (from {width} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )
            if scaled_height % 32 != 0:
                raise ValueError(
                    f"Scaled reference height {scaled_height} (from {height} / {self.reference_downscale_factor}) "
                    f"is not divisible by 32. Choose a different downscale factor or adjust video_dims."
                )

        if self.from_train and (self.metadata_jsonl or self.prompts or self.images):
            raise ValueError(
                "validation.from_train is mutually exclusive with metadata_jsonl/prompts/images"
            )
        if self.run_only and (not self.interval or self.skip_initial_validation):
            raise ValueError(
                "validation.run_only requires interval and skip_initial_validation=false"
            )
        if self.metrics:
            required = {"id_similarity", "fid", "fvd", "sync_c"}
            if set(self.metrics) != required or len(self.metrics) != len(required):
                raise ValueError(
                    f"validation.metrics must contain exactly {sorted(required)}"
                )
            if (
                not self.allow_small_sample_metrics
                and (self.jsonl_max_samples is None or self.jsonl_max_samples < 8)
            ):
                raise ValueError(
                    "the four-metric validation protocol requires at least 8 primary samples"
                )
        if bool(self.extended_metadata_jsonl) != bool(self.extended_interval):
            raise ValueError(
                "validation.extended_metadata_jsonl and extended_interval must be set together"
            )
        if self.extended_metadata_jsonl and self.extended_max_samples < 8:
            raise ValueError("extended validation requires at least 8 samples")
        if self.extended_interval and (
            not self.interval or self.extended_interval % self.interval != 0
        ):
            raise ValueError(
                "validation.extended_interval must be a multiple of validation.interval"
            )

        return self


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )

    precision: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="Precision to use when saving checkpoint weights. Options: 'bfloat16' or 'float32'.",
    )

    no_resume: bool = Field(
        default=False,
        description="When True, ignore any saved training state and start from step 0. "
        "Model weights from load_checkpoint are still loaded, but optimizer/scheduler "
        "state and step counter are reset.",
    )

    save_training_state: Literal["full", "minimal", "off"] = Field(
        default="minimal",
        description="Save training state alongside checkpoints for resume. "
        "'full': optimizer + scheduler + RNG + step; FSDP optimizer state is saved per rank and requires "
        "the same world size when resuming. "
        "'minimal': scheduler + RNG + step only and is not an exact optimizer resume. "
        "'off': nothing saved, resume not possible.",
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(
        default=False,
        description="Whether to push the model weights to the Hugging Face Hub",
    )
    hub_model_id: str | None = Field(
        default=None,
        description="Hugging Face Hub repository ID (e.g., 'username/repo-name')",
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )

    resume_run: bool = Field(
        default=True,
        description=(
            "When resuming from a checkpoint with saved training state, resume the original W&B run. "
            "Set to False to keep checkpoint step/scheduler/RNG resume but start a new W&B run."
        ),
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class ClusterNode(ConfigBaseModel):
    """One node in a multi-machine training cluster (consumed by the bx_*_mm.sh launchers)."""

    name: str = Field(description="Human-readable node name (e.g. avgen-2)")
    host: str = Field(description="SSH-reachable IP or hostname")
    ssh_user: str = Field(default="root")
    ssh_port: int = Field(default=22)
    role: Literal["master", "slave"] = Field(default="slave")


class ClusterConfig(ConfigBaseModel):
    """Multi-machine cluster description. Ignored by the trainer itself; the

    bx_train_*_mm.sh launcher reads it to SSH into slave nodes and build torchrun
    rendezvous args (mirrors the UniAV cluster block convention)."""

    master_port: int = Field(default=29517)
    nproc_per_node: int = Field(default=8)
    nodes: list[ClusterNode] = Field(default_factory=list)
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Restricted NCCL/Gloo and communication-library overrides propagated to every node.",
    )


class LtxTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    stage2: Stage2LoraConfig = Field(default_factory=Stage2LoraConfig)
    training_strategy: TrainingStrategyConfig = Field(
        default_factory=TextToVideoConfig,
        description="Training strategy configuration. Determines the training mode and its parameters.",
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    cluster: ClusterConfig | None = Field(
        default=None,
        description="Optional multi-machine cluster description used by the bx_*_mm.sh launchers. "
        "The trainer itself ignores this section.",
    )

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_strategy_compatibility(self) -> "LtxTrainerConfig":
        """Validate that training strategy and other configurations are compatible."""

        # Check that reference videos are provided when using video_to_video strategy
        if (
            self.training_strategy.name == "video_to_video"
            and self.validation.interval
            and not self.validation.reference_videos
        ):
            raise ValueError(
                "reference_videos must be provided in validation config when using video_to_video strategy"
            )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError(
                "LoRA configuration must be provided when training_mode is 'lora'"
            )

        if self.stage2.enabled:
            if (
                self.model.training_mode == "full"
                and self.stage2.injection_lora_enabled
            ):
                raise ValueError(
                    "Full-SFT Stage-2 must set stage2.injection_lora_enabled=false; "
                    "the full DiT remains trainable and Stage-2 only adds Qwen LoRA"
                )
            if not self.model.load_checkpoint:
                raise ValueError(
                    "stage2.enabled=true requires a Stage-1 model.load_checkpoint"
                )
            if not getattr(self.training_strategy, "use_qwen_cond", False):
                raise ValueError(
                    "stage2.enabled=true requires training_strategy.use_qwen_cond=true"
                )
            if not uses_online_qwen_conditioning(self.training_strategy):
                raise ValueError(
                    "Qwen Stage-2 LoRA requires online_qwen_conditioning=true"
                )

        # Check that LoRA config is provided when using video_to_video strategy
        if (
            self.training_strategy.name == "video_to_video"
            and self.model.training_mode != "lora"
        ):
            raise ValueError(
                "Training mode must be 'lora' when using video_to_video strategy"
            )

        if self.data.length_bucket.enabled:
            online_text = uses_online_text_conditioning(self.training_strategy)
            online_qwen = getattr(
                self.training_strategy, "use_qwen_cond", False
            ) and uses_online_qwen_conditioning(self.training_strategy)
            if not (online_text or online_qwen):
                raise ValueError(
                    "data.length_bucket.enabled=true requires online Gemma or online Qwen conditioning"
                )
            if self.optimization.batch_size != 1:
                raise ValueError(
                    "data.length_bucket.enabled=true requires optimization.batch_size=1 "
                    f"(got {self.optimization.batch_size})"
                )

        if (
            getattr(self.training_strategy, "use_omni_cond", False)
            and getattr(
                self.training_strategy, "omni_layer_adapter_num_layers", 1
            )
            == 4
            and self.data.max_training_frames is not None
        ):
            raise ValueError(
                "Oracle/teacher-forcing Omni requires data.max_training_frames=null; "
                "GT Plan, video latent, and audio latent must cover the same full duration"
            )

        return self
