import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType
from accelerate.utils import InitProcessGroupKwargs, gather_object, send_to_device, set_seed
from ltx_core.distributed.parallel_state import (
    get_dp_rank,
    get_dp_world_size,
    get_sp_rank,
    init_sequence_parallel,
    suspend_sequence_parallel,
)
from ltx_trainer.cp_sampler import ContextParallelDistributedSampler
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812

import wandb
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import (
    OfflineOmniConditionDataset,
    OnlineConditionDataset,
    PrecomputedDataset,
    collate_online_condition_batch,
    collate_precomputed_batch,
)
from ltx_trainer.gpu_utils import (
    free_gpu_memory,
    free_gpu_memory_context,
    get_gpu_memory_gb,
)
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.length_bucket_dataset import LengthBucketOnlineDataset
from ltx_trainer.length_bucket_sampler import LengthBucketSampler
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.omni_condition_utils import (
    OmniConditionBundle,
    load_omni_condition_file,
)
from ltx_trainer.online_conditioning import (
    OnlineQwenEncoder,
    encode_modality_text_batch_online,
    encode_text_batch_online,
)
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.sigma_tracker import SigmaBucketTracker
from ltx_trainer.stage2_lora import (
    apply_stage2_injection_lora,
    named_optimizer_group_grad_norms,
    save_qwen_lora,
)
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_state import ConfigFingerprint, RngStates, TrainingState
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.training_strategies.text_to_video import (
    uses_online_qwen_conditioning,
    uses_online_text_conditioning,
)
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_benchmark import ValidationBenchmark, validation_collective_timeout_seconds
from ltx_trainer.validation_sampler import (
    CachedPromptEmbeddings,
    GenerationConfig,
    ValidationSampler,
)
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore",
    message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
)

# Disable progress bars if not main process
# Global main process across all nodes (torchrun/accelerate set RANK; fall back to
# LOCAL_RANK for single-node launchers). Using LOCAL_RANK here would make every
# node's local rank 0 write checkpoints/wandb concurrently in multi-node runs.
IS_MAIN_PROCESS = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[
    [int, int, list[Path] | None], None
]  # (step, total, sampled paths or None) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


@dataclass(frozen=True)
class CachedQwenCondition:
    """Pre-computed Qwen/Qwen-VL condition tensors for one validation prompt."""

    video_context: Tensor
    video_context_mask: Tensor
    audio_context: Tensor | None = None
    audio_context_mask: Tensor | None = None


@dataclass(frozen=True)
class TrainingStepOutput:
    """Output from a single training step."""

    loss: Tensor  # [B,] per-element loss (unreduced)
    sigma: Tensor  # [B,] sampled sigma, detached from computational graph
    loss_components: dict[str, Tensor]  # [B,] per-element component losses for logging


class LtxvTrainer:
    _VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

    def _setup_file_logging(self) -> None:
        """Mirror all trainer logs to a per-rank file under output_dir/logs.

        The Rich console handler only shows the live progress bar in interactive runs,
        which makes post-hoc debugging (especially multi-node) painful. Every rank gets
        its own plain-text log file with timestamps; rank 0 carries the full story
        (loss lines, validation, checkpoints), other ranks log warnings and errors.
        """
        import logging

        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
        log_dir = Path(self._config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self._run_stamp = (
            run_stamp  # reused by TensorBoard so file log and TB run share a name
        )
        log_path = log_dir / f"train_{run_stamp}_rank{rank}.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [rank {rank}] %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        # A file-only logger for periodic loss lines: keeps the console progress bar
        # clean while the file always carries step/loss/lr history for debugging.
        self._file_logger = logging.getLogger("ltxv_trainer.file")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False
        self._file_logger.addHandler(file_handler)
        # Non-main ranks keep WARNING+ so real errors from any node land in a file.
        logger.info(f"File logging enabled: {log_path}")
        self._log_file_path = log_path

    def _setup_tensorboard(self) -> None:
        """Create a TensorBoard SummaryWriter under output_dir/tensorboard (rank 0 only).

        Mirrors the pattern from UniAV's train_double_tower_stage2_tts_audio.py: scalars go
        to both W&B and TensorBoard so runs remain inspectable locally (e.g. offline W&B mode)
        via `tensorboard --logdir <output_dir>/tensorboard`. One subdir per launch (shares the
        file-log timestamp) so restarts show up as separate curves.
        """
        self._tb_writer = None
        if not IS_MAIN_PROCESS:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("tensorboard is not installed; skipping TensorBoard logging")
            return
        tb_dir = (
            Path(self._config.output_dir) / "tensorboard" / f"train_{self._run_stamp}"
        )
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logging enabled: {tb_dir}")

    def _materialize_train_validation_jsonl(self) -> Path:
        """Select a fixed eight-case validation set from the actual training metadata."""
        val_config = self._config.validation
        data_config = self._config.data
        source = val_config.from_train_metadata_jsonl
        if source is None:
            frame_token = f"ltx{val_config.video_dims[2]}"
            matching = [
                path
                for path in data_config.metadata_jsonls
                if frame_token in Path(path).name
            ]
            if not matching:
                raise ValueError(
                    f"validation.from_train could not find {frame_token} in data.metadata_jsonls; "
                    "set validation.from_train_metadata_jsonl explicitly"
                )
            source = matching[0]

        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"training validation source not found: {source_path}"
            )

        target_dir = Path(self._config.output_dir) / "validation"
        target_path = (
            target_dir
            / f"validation_from_train_seed_{val_config.from_train_seed}.jsonl"
        )
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        if rank == 0:
            target_dir.mkdir(parents=True, exist_ok=True)
            reuse = False
            if target_path.is_file():
                try:
                    rows = [
                        json.loads(line)
                        for line in target_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    reuse = len(rows) == val_config.jsonl_max_samples and all(
                        row.get("_validation_source") == str(source_path)
                        for row in rows
                    )
                except (OSError, json.JSONDecodeError):
                    reuse = False

            if not reuse:
                count = int(val_config.jsonl_max_samples or 8)
                rng = random.Random(val_config.from_train_seed)
                selected: list[dict[str, Any]] = []
                selected_videos: set[str] = set()
                path_remap = dict(data_config.path_remap or {})
                file_size = source_path.stat().st_size

                def remap_path(path: str) -> str:
                    for src, dst in path_remap.items():
                        if path.startswith(src):
                            return dst + path[len(src) :]
                    return path

                with source_path.open("rb") as handle:
                    for _attempt in range(max(1000, count * 500)):
                        handle.seek(rng.randrange(max(1, file_size)))
                        if handle.tell() > 0:
                            handle.readline()
                        raw = handle.readline()
                        if not raw:
                            continue
                        try:
                            row = json.loads(raw)
                            caption = row.get(val_config.jsonl_caption_field)
                            video = row.get(val_config.jsonl_video_field)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(caption, str):
                            continue
                        has_dialogue = "[WORDS]" in caption or (
                            "<dialogue>" in caption and "<words:" in caption
                        )
                        if not has_dialogue or not isinstance(video, str):
                            continue
                        face = row.get("human_face_quality") or {}
                        labels = face.get("labels") or {}
                        try:
                            single_person = float(labels.get("avg_persons", 0.0)) == 1.0
                        except (TypeError, ValueError):
                            single_person = False
                        if face.get("status") != "ok" or not single_person:
                            continue
                        video = remap_path(video)
                        if video in selected_videos or not Path(video).is_file():
                            continue
                        row[val_config.jsonl_video_field] = video
                        row["_validation_source"] = str(source_path)
                        row["_validation_seed"] = val_config.from_train_seed
                        selected.append(row)
                        selected_videos.add(video)
                        if len(selected) == count:
                            break

                if len(selected) != count:
                    raise RuntimeError(
                        f"selected only {len(selected)}/{count} usable validation rows from {source_path}"
                    )
                tmp_path = target_path.with_name(
                    f".{target_path.name}.{os.getpid()}.tmp"
                )
                with tmp_path.open("w", encoding="utf-8") as handle:
                    for row in selected:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, target_path)
                logger.info(
                    f"Validation: materialized {count} fixed training rows at {target_path}"
                )

        deadline = time.monotonic() + val_config.from_train_wait_timeout_seconds
        while not target_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.25)
        if not target_path.is_file():
            raise TimeoutError(f"timed out waiting for rank 0 to create {target_path}")
        return target_path

    def _resolve_validation_jsonl(self) -> None:
        """Expand validation.metadata_jsonl into prompts + condition media paths.

        Each row contributes its caption (verbatim, full sd_text by default) as the
        prompt and its video path as the I2V condition — the first frame is read at
        sampling time. Must run before strategy/text-encoder setup so downstream
        caching sees the resolved prompts.
        """
        val_config = self._config.validation
        if val_config.from_train:
            val_config.metadata_jsonl = str(self._materialize_train_validation_jsonl())
        if not val_config.metadata_jsonl:
            return
        if val_config.prompts or val_config.images:
            raise ValueError(
                "validation.metadata_jsonl is mutually exclusive with validation.prompts/images"
            )

        jsonl_path = Path(val_config.metadata_jsonl).expanduser()
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"validation.metadata_jsonl not found: {jsonl_path}"
            )

        path_remap = dict(getattr(self._config.data, "path_remap", {}) or {})

        def _remap(path: str) -> str:
            for src, dst in path_remap.items():
                if path.startswith(src):
                    return dst + path[len(src) :]
            return path

        prompts: list[str] = []
        media_paths: list[str] = []
        metadata_rows: list[dict] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                caption = row.get(val_config.jsonl_caption_field)
                video_path = row.get(val_config.jsonl_video_field)
                if not isinstance(caption, str) or not caption.strip():
                    logger.warning(
                        f"validation jsonl {jsonl_path.name}:{line_number}: empty caption, skipping"
                    )
                    continue
                if not isinstance(video_path, str) or not video_path.strip():
                    logger.warning(
                        f"validation jsonl {jsonl_path.name}:{line_number}: missing video path, skipping"
                    )
                    continue
                video_path = _remap(video_path)
                if not Path(video_path).exists():
                    logger.warning(
                        f"validation jsonl {jsonl_path.name}:{line_number}: {video_path} not found, skipping"
                    )
                    continue
                prompts.append(caption)
                media_paths.append(video_path)
                resolved_row = dict(row)
                resolved_row[val_config.jsonl_video_field] = video_path
                metadata_rows.append(resolved_row)
                if (
                    val_config.jsonl_max_samples is not None
                    and len(prompts) >= val_config.jsonl_max_samples
                ):
                    break

        if not prompts:
            raise ValueError(
                f"validation.metadata_jsonl produced no usable samples: {jsonl_path}"
            )
        if (
            val_config.metrics
            and val_config.jsonl_max_samples is not None
            and len(prompts) != val_config.jsonl_max_samples
        ):
            raise ValueError(
                "validation metrics require an exact fixed set of "
                f"{val_config.jsonl_max_samples} usable samples, but "
                f"{jsonl_path} produced only {len(prompts)} after filtering "
                "missing/invalid captions and video paths"
            )

        val_config.prompts = prompts
        val_config.images = media_paths
        self._validation_metadata_rows = metadata_rows
        logger.info(
            f"Validation: {len(prompts)} samples resolved from {jsonl_path.name}"
        )

    def _resolve_extended_validation_jsonl(self) -> None:
        """Resolve the optional larger validation profile without replacing fixed8."""
        val_config = self._config.validation
        if not val_config.extended_metadata_jsonl:
            return

        jsonl_path = Path(val_config.extended_metadata_jsonl).expanduser()
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"validation.extended_metadata_jsonl not found: {jsonl_path}"
            )

        path_remap = dict(getattr(self._config.data, "path_remap", {}) or {})

        def remap(path: str) -> str:
            for source, destination in path_remap.items():
                if path.startswith(source):
                    return destination + path[len(source) :]
            return path

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if (
                    len(self._extended_validation_prompts)
                    >= val_config.extended_max_samples
                ):
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get(val_config.extended_jsonl_caption_field)
                video = row.get(val_config.extended_jsonl_video_field)
                if not isinstance(prompt, str) or not prompt.strip():
                    logger.warning(
                        f"extended validation {jsonl_path.name}:{line_number}: "
                        "empty prompt, skipping"
                    )
                    continue
                if not isinstance(video, str) or not video.strip():
                    logger.warning(
                        f"extended validation {jsonl_path.name}:{line_number}: "
                        "missing video, skipping"
                    )
                    continue
                video = remap(video)
                if not Path(video).is_file():
                    logger.warning(
                        f"extended validation {jsonl_path.name}:{line_number}: "
                        f"{video} not found, skipping"
                    )
                    continue
                resolved_row = dict(row)
                resolved_row[val_config.extended_jsonl_video_field] = video
                self._extended_validation_prompts.append(prompt)
                self._extended_validation_images.append(video)
                self._extended_validation_metadata_rows.append(resolved_row)

        expected = val_config.extended_max_samples
        if len(self._extended_validation_prompts) != expected:
            raise ValueError(
                f"extended validation expected {expected} usable samples from "
                f"{jsonl_path}, got {len(self._extended_validation_prompts)}"
            )
        logger.info(
            f"Extended validation: {expected} samples resolved from {jsonl_path.name}"
        )

    def _load_validation_omni_conditions(self) -> None:
        """Load Route-A packages for both validation profiles by sample ID."""
        strategy_config = self._config.training_strategy
        if not getattr(strategy_config, "use_omni_cond", False):
            return

        condition_root = Path(strategy_config.omni_conditions_dir).expanduser()
        expected_version = self._config.validation.omni_condition_version
        if expected_version is None:
            expected_version = (
                "teacher_forcing_raw"
                if strategy_config.use_omni_field_resampler
                else
                "teacher_forcing"
                if strategy_config.omni_layer_adapter_num_layers == 4
                else "generated"
                if strategy_config.omni_layer_adapter_num_layers == 1
                else None
            )

        def load_profile(
            rows: list[dict], label: str
        ) -> list[OmniConditionBundle] | None:
            if not rows:
                return None
            bundles: list[OmniConditionBundle] = []
            for index, row in enumerate(rows):
                sample_id = row.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(
                        f"{label} validation row {index} requires a non-empty sample_id "
                        "when use_omni_cond=true"
                    )
                bundles.append(
                    load_omni_condition_file(
                        condition_root / f"{sample_id}.pt",
                        expected_version=expected_version,
                        expected_sample_id=sample_id,
                        expected_route=getattr(
                            strategy_config,
                            "omni_expected_route",
                            "route_a_no_context",
                        ),
                    )
                )
            logger.info(
                f"Validation: loaded {len(bundles)} offline Omni packages for {label}"
            )
            return bundles

        self._validation_omni_conditions = load_profile(
            self._validation_metadata_rows, "primary"
        )
        self._extended_validation_omni_conditions = load_profile(
            self._extended_validation_metadata_rows, "extended"
        )

    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        self._validation_metadata_rows: list[dict] = []
        self._extended_validation_prompts: list[str] = []
        self._extended_validation_images: list[str] = []
        self._extended_validation_metadata_rows: list[dict] = []
        self._validation_omni_conditions: list[OmniConditionBundle] | None = None
        self._extended_validation_omni_conditions: (
            list[OmniConditionBundle] | None
        ) = None
        self._cached_extended_validation_embeddings: (
            list[CachedPromptEmbeddings] | None
        ) = None
        self._setup_file_logging()
        self._setup_tensorboard()
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._resolve_validation_jsonl()
        self._resolve_extended_validation_jsonl()
        self._load_validation_omni_conditions()
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._text_encoder = None
        self._online_qwen_encoder: OnlineQwenEncoder | None = None
        self._pending_stage2_state: dict[str, Tensor] = {}
        self._stage2_injection_targets: list[str] = []
        self._cached_validation_embeddings = (
            self._load_text_encoder_and_cache_embeddings()
        )
        self._cached_validation_qwen_conditions = (
            self._load_validation_qwen_conditions()
        )
        self._load_models()
        self._setup_accelerator()
        self._training_data_fingerprint = self._compute_training_data_fingerprint()
        self._collect_trainable_params()
        self._loaded_checkpoint_path: Path | None = None
        self._resolved_qwen_checkpoint = self._config.stage2.qwen_checkpoint
        self._load_checkpoint()
        if self._config.stage2.enabled:
            self._setup_stage2_training()
        self._refresh_trainable_params()
        self._prepare_models_for_training()
        self._dataset = None
        self._length_bucket_sampler: LengthBucketSampler | None = None
        self._cp_sampler: ContextParallelDistributedSampler | None = None
        self._length_bucket_place_on_device = False
        self._length_bucket_epoch = 0
        self._cp_epoch = 0
        self._global_step = -1
        self._checkpoint_paths: list[Path] = []
        self._last_saved_step: int | None = None
        self._training_state_paths: list[Path] = []
        self._training_state_size_warned = False
        self._wandb_run = None
        self._sigma_tracker = SigmaBucketTracker()

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Args:
            disable_progress_bars: Disable Rich progress bars (useful for multi-process runs).
            step_callback: Optional callback invoked after each optimization step.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        initial_step, training_state = self._resume_state
        resuming = training_state is not None

        set_seed(cfg.seed)
        logger.debug(
            f"Process {self._accelerator.process_index} using seed: {cfg.seed}"
        )

        self._init_optimizer()

        if training_state is not None and not self._restore_training_state(
            training_state
        ):
            initial_step = 0
            resuming = False

        # Initialize W&B after restore so we only resume the run when state restore succeeds.
        resume_run_id = (
            training_state.wandb_run_id
            if resuming and training_state is not None and self._config.wandb.resume_run
            else None
        )
        self._init_wandb(resume_run_id=resume_run_id)

        data_iter = None
        if not cfg.validation.run_only:
            self._init_dataloader()
            data_iter = iter(self._dataloader)
            self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        if cfg.validation.run_only:
            self._global_step = initial_step
            progress = TrainingProgress(enabled=False, total_steps=1)
            with progress, self._offloaded_optimizer_state():
                self._run_distributed_validation(progress)
            self._accelerator.wait_for_everyone()
            if IS_MAIN_PROCESS:
                if self._wandb_run is not None:
                    self._wandb_run.finish()
                if self._tb_writer is not None:
                    self._tb_writer.close()
            stats = TrainingStats(
                total_time_seconds=time.time() - train_start_time,
                steps_per_second=0.0,
                samples_per_second=0.0,
                peak_gpu_memory_gb=max(start_mem, get_gpu_memory_gb(device)),
                global_batch_size=0,
                num_processes=self._accelerator.num_processes,
            )
            self._accelerator.end_training()
            return Path(cfg.output_dir) / "samples", stats

        remaining_steps = cfg.optimization.steps - initial_step
        if remaining_steps <= 0:
            raise ValueError(
                f"No remaining training steps: initial_step={initial_step} >= "
                f"target_steps={cfg.optimization.steps}. Nothing to train."
            )

        if resuming:
            logger.info(
                f"🚀 Resuming training from step {initial_step} → {cfg.optimization.steps}"
            )
        else:
            logger.info("🚀 Starting training...")

        # Create progress tracking (disabled for non-main processes or when explicitly disabled)
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=remaining_steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning(
                "Progress bars disabled. Intermediate status messages will be logged instead."
            )

        self._transformer.train()
        if self._online_qwen_encoder is not None and self._config.stage2.enabled:
            self._online_qwen_encoder.model.train()
            self._online_qwen_encoder.base_model.visual.eval()
        self._global_step = initial_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        with progress:
            if cfg.validation.interval and not cfg.validation.skip_initial_validation:
                with self._offloaded_optimizer_state():
                    sampled_videos_paths = self._run_distributed_validation(progress)

            self._accelerator.wait_for_everyone()

            accumulated_loss = 0.0
            accumulated_components: dict[str, float] = {}
            accumulated_sigmas: list[float] = []
            accumulated_element_losses: list[float] = []
            accumulated_train_time = 0.0
            accumulated_micro_batches = 0

            for step in range(
                remaining_steps * cfg.optimization.gradient_accumulation_steps
            ):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self._length_bucket_sampler is not None:
                        self._length_bucket_epoch += 1
                        self._length_bucket_sampler.set_epoch(self._length_bucket_epoch)
                    if self._cp_sampler is not None:
                        self._cp_epoch += 1
                        self._cp_sampler.set_epoch(self._cp_epoch)
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                if self._length_bucket_place_on_device:
                    batch = send_to_device(batch, self._accelerator.device)

                step_start_time = time.time()
                omni_monitor_metrics: dict[str, float] = {}
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (
                        step + 1
                    ) % cfg.optimization.gradient_accumulation_steps == 0
                    monitor_omni = self._should_monitor_omni(
                        is_optimization_step
                    )
                    if monitor_omni:
                        self._set_omni_residual_monitoring(True)

                    output = self._training_step(batch)
                    self._accelerator.backward(output.loss.mean())

                    if is_optimization_step:
                        self._global_step += 1

                    if self._accelerator.sync_gradients:
                        self._sync_replicated_qwen_grads()
                        if (
                            os.environ.get("LTX_AUDIT_OMNI_GRADS") == "1"
                            and self._global_step
                            == (
                                2
                                if getattr(
                                    self._config.training_strategy,
                                    "use_omni_field_resampler",
                                    False,
                                )
                                else 1
                            )
                        ):
                            self._audit_oracle_omni_gradients()
                        if monitor_omni:
                            omni_monitor_metrics = self._collect_omni_monitor_metrics()

                    group_grad_norms: dict[str, float] = {}
                    if (
                        self._accelerator.sync_gradients
                        and IS_MAIN_PROCESS
                        and self._accelerator.distributed_type != DistributedType.FSDP
                    ):
                        group_grad_norms = named_optimizer_group_grad_norms(
                            self._optimizer
                        )

                    grad_norm = None
                    if (
                        self._accelerator.sync_gradients
                        and cfg.optimization.max_grad_norm > 0
                    ):
                        grad_norm = self._clip_training_gradients(cfg.optimization.max_grad_norm)

                    skip_optimizer_step = bool(
                        getattr(self, "_skip_optimizer_step", False)
                    )
                    if skip_optimizer_step:
                        self._skip_optimizer_step = False
                    else:
                        self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None and not skip_optimizer_step:
                        self._lr_scheduler.step()

                    accumulated_train_time += time.time() - step_start_time
                    accumulated_micro_batches += 1
                    accumulated_loss += output.loss.detach().mean().item()
                    for name, value in output.loss_components.items():
                        accumulated_components[name] = (
                            accumulated_components.get(name, 0.0)
                            + value.detach().mean().item()
                        )
                    if IS_MAIN_PROCESS:
                        accumulated_sigmas.extend(output.sigma.detach().cpu().tolist())
                        accumulated_element_losses.extend(
                            output.loss.detach().cpu().tolist()
                        )

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        with self._offloaded_optimizer_state():
                            sampled_videos_paths = self._run_distributed_validation(
                                progress
                            )
                            extended_interval = cfg.validation.extended_interval
                            if (
                                extended_interval
                                and self._global_step % extended_interval == 0
                            ):
                                self._run_distributed_validation(
                                    progress, extended=True
                                )

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                    ):
                        self._save_checkpoint()

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(
                            self._global_step,
                            cfg.optimization.steps,
                            sampled_videos_paths,
                        )

                    self._accelerator.wait_for_everyone()

                    # Log the mean over all micro-batches that contributed to this optimizer step.
                    if is_optimization_step:
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        step_loss = accumulated_loss / accumulated_micro_batches
                        step_time = accumulated_train_time
                        component_metrics = {
                            name: value / accumulated_micro_batches
                            for name, value in accumulated_components.items()
                        }

                        progress.update_training(
                            loss=step_loss,
                            lr=current_lr,
                            step_time=step_time,
                            advance=True,
                        )

                        if IS_MAIN_PROCESS:
                            self._sigma_tracker.update(
                                accumulated_sigmas, accumulated_element_losses
                            )
                            metrics = {
                                "train/loss": step_loss,
                                "train/learning_rate": current_lr,
                                "train/step_time": step_time,
                                "train/global_step": self._global_step,
                            }
                            if grad_norm is not None:
                                metrics["train/grad_norm"] = float(grad_norm)
                            metrics.update(
                                {
                                    f"train/grad_norm_{name}": value
                                    for name, value in group_grad_norms.items()
                                }
                            )
                            metrics.update(
                                {
                                    f"train/{name}": value
                                    for name, value in component_metrics.items()
                                }
                            )
                            metrics.update(
                                {
                                    f"train/{name}": value
                                    for name, value in omni_monitor_metrics.items()
                                }
                            )
                            for group in self._optimizer.param_groups:
                                if group.get("name"):
                                    metrics[f"train/lr_{group['name']}"] = group["lr"]
                            metrics.update(self._sigma_tracker.get_metrics())
                            self._log_metrics(metrics)

                        if (
                            disable_progress_bars
                            and IS_MAIN_PROCESS
                            and self._global_step % 20 == 0
                        ):
                            elapsed = time.time() - train_start_time
                            steps_done = self._global_step - initial_step
                            if steps_done > 0:
                                total_estimated = elapsed / steps_done * remaining_steps
                                total_time = (
                                    f"{total_estimated // 3600:.0f}h "
                                    f"{(total_estimated % 3600) // 60:.0f}m"
                                )
                            else:
                                total_time = "calculating..."
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} - "
                                f"Loss: {step_loss:.4f}, LR: {current_lr:.2e}, "
                                f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                            )

                        if IS_MAIN_PROCESS and hasattr(self, "_file_logger"):
                            component_text = " ".join(
                                f"{name}={value:.4f}"
                                for name, value in sorted(component_metrics.items())
                            )
                            omni_monitor_text = " ".join(
                                f"{name}={value:.4f}"
                                for name, value in sorted(omni_monitor_metrics.items())
                            )
                            extra_text = " ".join(
                                text
                                for text in (component_text, omni_monitor_text)
                                if text
                            )
                            self._file_logger.info(
                                f"step={self._global_step}/{cfg.optimization.steps} "
                                f"loss={step_loss:.4f} lr={current_lr:.2e} "
                                f"step_time={step_time:.2f}s"
                                + (f" {extra_text}" if extra_text else "")
                            )

                        accumulated_loss = 0.0
                        accumulated_components.clear()
                        accumulated_sigmas.clear()
                        accumulated_element_losses.clear()
                        accumulated_train_time = 0.0
                        accumulated_micro_batches = 0

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(
                            peak_mem_during_training, current_mem
                        )

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        # Calculate steps/second over entire training
        total_time_seconds = train_end_time - train_start_time
        steps_per_second = remaining_steps / total_time_seconds

        effective_global_batch = (
            cfg.optimization.batch_size
            * self._accelerator.num_processes
            * cfg.optimization.gradient_accumulation_steps
            // cfg.acceleration.context_parallel_size
        )
        samples_per_second = steps_per_second * effective_global_batch

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=effective_global_batch,
        )

        final_prefix = "lora" if self._config.model.training_mode == "lora" else "model"
        final_checkpoint = (
            Path(self._config.output_dir)
            / "checkpoints"
            / f"{final_prefix}_weights_step_{self._global_step:05d}.safetensors"
        )
        if self._last_saved_step == self._global_step and final_checkpoint.is_file():
            logger.info(
                f"Reusing checkpoint already saved at final step {self._global_step}"
            )
            saved_path = final_checkpoint if IS_MAIN_PROCESS else None
        else:
            saved_path = self._save_checkpoint()

        if IS_MAIN_PROCESS:
            # Log the training statistics
            self._log_training_stats(stats)

            # Upload artifacts to hub if enabled
            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            # Log final stats to W&B
            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

            if self._tb_writer is not None:
                self._tb_writer.close()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> TrainingStepOutput:
        """Perform a single training step using the configured strategy."""
        if uses_online_text_conditioning(self._config.training_strategy) or (
            getattr(self._config.training_strategy, "use_qwen_cond", False)
            and uses_online_qwen_conditioning(self._config.training_strategy)
        ):
            self._encode_conditions_online(batch)

        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]

        if conditions.get("online_encoded", False):
            # Online path: embeddings already went through the full pipeline
            # (feature extractor + connectors) in _encode_conditions_online.
            pass
        else:
            if "video_prompt_embeds" in conditions:
                # New format: separate video/audio features from precompute()
                video_features = conditions["video_prompt_embeds"]
                audio_features = conditions.get("audio_prompt_embeds")
            else:
                # Legacy format: single prompt_embeds tensor — duplicate for both modalities
                video_features = conditions["prompt_embeds"]
                audio_features = conditions["prompt_embeds"]

            mask = conditions["prompt_attention_mask"]
            additive_mask = convert_to_additive_mask(mask, video_features.dtype)
            video_embeds, audio_embeds, attention_mask = (
                self._embeddings_processor.create_embeddings(
                    video_features, audio_features, additive_mask
                )
            )

            conditions["video_prompt_embeds"] = video_embeds
            conditions["audio_prompt_embeds"] = audio_embeds
            conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._training_strategy.prepare_training_inputs(
            batch, self._timestep_sampler
        )

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss_breakdown = self._training_strategy.compute_loss(
            video_pred, audio_pred, model_inputs
        )
        sigma = (
            model_inputs.video.sigma.detach()
            if model_inputs.video.enabled
            else model_inputs.audio.sigma.detach()
        )

        return TrainingStepOutput(
            loss=loss_breakdown.loss,
            sigma=sigma,
            loss_components=loss_breakdown.components,
        )

    def _get_online_qwen_encoder(self) -> "OnlineQwenEncoder":
        """Lazily build the frozen Stage-1 or trainable Stage-2 Qwen encoder."""
        if self._online_qwen_encoder is None:
            strategy_config = self._config.training_strategy
            if not strategy_config.qwen_model_path:
                raise ValueError(
                    "online_qwen_conditioning requires training_strategy.qwen_model_path"
                )
            if (
                strategy_config.use_qwen_layer_adapters
                and strategy_config.qwen_feature_layer_mode != "all"
            ):
                raise ValueError(
                    "use_qwen_layer_adapters=True requires qwen_feature_layer_mode='all'"
                )
            self._online_qwen_encoder = OnlineQwenEncoder(
                model_path=strategy_config.qwen_model_path,
                device=self._accelerator.device,
                max_length=strategy_config.qwen_max_length,
                image_width=strategy_config.qwen_image_width,
                image_height=strategy_config.qwen_image_height,
                dtype=torch.bfloat16,
                feature_layer_mode=strategy_config.qwen_feature_layer_mode,
                train_lora=self._config.stage2.enabled
                and self._config.stage2.qwen_training_mode == "lora",
                lora_rank=self._config.stage2.qwen_rank,
                lora_alpha=self._config.stage2.qwen_alpha,
                lora_dropout=self._config.stage2.qwen_dropout,
                lora_last_n_layers=self._config.stage2.qwen_last_n_layers,
                lora_target_modules=self._config.stage2.qwen_target_modules,
                gradient_checkpointing=self._config.stage2.qwen_gradient_checkpointing,
                dynamic_length=self._config.stage2.enabled,
                lora_checkpoint=self._resolved_qwen_checkpoint,
            )
        return self._online_qwen_encoder

    def _encode_conditions_online(self, batch: dict[str, Any]) -> None:
        """Encode only the condition branches configured for online execution."""
        strategy_config = self._config.training_strategy
        online_text = uses_online_text_conditioning(strategy_config)
        online_qwen = getattr(
            strategy_config, "use_qwen_cond", False
        ) and uses_online_qwen_conditioning(strategy_config)
        texts = batch.pop("sd_text", None)
        first_frames = batch.pop("first_frame", None)
        omni_conditions = batch.get("omni_conditions")
        use_omni = bool(getattr(strategy_config, "use_omni_cond", False))
        video_prompts = None
        audio_prompts = None
        if use_omni and isinstance(omni_conditions, dict):
            video_prompts = omni_conditions.get("video_prompt")
            audio_prompts = omni_conditions.get("audio_prompt")

        if (online_text or online_qwen) and texts is None and not (
            online_text and video_prompts is not None and audio_prompts is not None
        ):
            raise ValueError(
                "Online Gemma/Qwen conditioning requires raw sd_text in the batch"
            )

        if online_text:
            if self._text_encoder is None:
                raise RuntimeError(
                    "Online Gemma conditioning requested but the text encoder is not loaded"
                )
            if use_omni:
                if not isinstance(video_prompts, list) or not isinstance(
                    audio_prompts, list
                ):
                    raise ValueError(
                        "Omni text conditioning requires batched video_prompt and audio_prompt strings"
                    )
                encoded = encode_modality_text_batch_online(
                    self._text_encoder,
                    self._embeddings_processor,
                    video_prompts,
                    audio_prompts,
                )
            else:
                encoded = encode_text_batch_online(
                    self._text_encoder, self._embeddings_processor, texts
                )
            batch["conditions"] = {
                "video_prompt_embeds": encoded["video_prompt_embeds"],
                "audio_prompt_embeds": encoded["audio_prompt_embeds"],
                "prompt_attention_mask": encoded["prompt_attention_mask"],
                **(
                    {
                        "video_prompt_attention_mask": encoded[
                            "video_prompt_attention_mask"
                        ],
                        "audio_prompt_attention_mask": encoded[
                            "audio_prompt_attention_mask"
                        ],
                    }
                    if use_omni
                    else {}
                ),
                "online_encoded": True,
            }
        elif "conditions" not in batch:
            raise ValueError(
                "Offline Gemma conditioning requires a matching conditions/ cache file"
            )

        if not online_qwen:
            return
        if first_frames is None:
            raise ValueError(
                "Online Qwen conditioning requires first_frame in the batch"
            )

        qwen_dropout = float(getattr(strategy_config, "qwen_dropout", 0.0))
        if qwen_dropout > 0 and torch.rand(1).item() < qwen_dropout:
            batch.pop("qwen_conditions", None)
            return

        qwen_encoder = self._get_online_qwen_encoder()
        qwen_encoded = qwen_encoder.encode_batch(list(first_frames), texts)
        batch["qwen_conditions"] = {
            "qwen_video_context": qwen_encoded.context,
            "qwen_attention_mask": qwen_encoded.attention_mask,
        }

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(
        self,
    ) -> list[CachedPromptEmbeddings] | None:
        """Load text encoder + embeddings processor, compute and cache validation embeddings."""

        # This method:
        #   1. Loads the pure Gemma text encoder on GPU
        #   2. Loads the embeddings processor (feature extractor + connectors)
        #   3. If validation prompts are configured, computes and caches their embeddings
        #   4. Unloads the Gemma model entirely, keeps the embeddings processor for training

        # Load text encoder (pure Gemma LLM) on GPU — LOCAL_RANK before Accelerator exists
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        init_device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        logger.debug("Loading text encoder...")
        text_encoder = load_text_encoder(
            gemma_model_path=self._config.model.text_encoder_path,
            device=init_device,
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        # Load embeddings processor (feature extractor + connectors)
        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device=init_device,
            dtype=torch.bfloat16,
        )

        # These are condition encoders, not optimizer-owned trainable modules.
        # They are loaded independently, so freezing the transformer/VAE below
        # does not freeze Gemma or its feature extractor/connectors.
        text_encoder.requires_grad_(False).eval()
        self._embeddings_processor.requires_grad_(False).eval()

        # Cache validation embeddings if prompts are configured
        cached_embeddings = None
        primary_prompt_count = len(self._config.validation.prompts)
        validation_prompts = (
            list(self._config.validation.prompts) + self._extended_validation_prompts
        )
        validation_omni_conditions: list[OmniConditionBundle | None] = [
            *(self._validation_omni_conditions or [None] * primary_prompt_count),
            *(
                self._extended_validation_omni_conditions
                or [None] * len(self._extended_validation_prompts)
            ),
        ]
        if validation_prompts:
            logger.info(
                f"Pre-computing embeddings for {len(validation_prompts)} validation prompts..."
            )
            cached_embeddings = []
            # no_grad, NOT inference_mode: FSDP forwards choke on inference tensors
            # (and inference tensors cannot be saved for backward if reused).
            with torch.no_grad():
                for prompt, omni_bundle in zip(
                    validation_prompts, validation_omni_conditions, strict=True
                ):
                    if omni_bundle is None:
                        pos_hs, pos_mask = text_encoder.encode(prompt)
                        pos_out = self._embeddings_processor.process_hidden_states(
                            pos_hs, pos_mask
                        )
                        video_context_positive = pos_out.video_encoding
                        audio_context_positive = pos_out.audio_encoding
                    else:
                        video_hs, video_mask = text_encoder.encode(
                            omni_bundle.video_prompt
                        )
                        video_out = self._embeddings_processor.process_hidden_states(
                            video_hs, video_mask
                        )
                        audio_hs, audio_mask = text_encoder.encode(
                            omni_bundle.audio_prompt
                        )
                        audio_out = self._embeddings_processor.process_hidden_states(
                            audio_hs, audio_mask
                        )
                        if audio_out.audio_encoding is None:
                            raise ValueError(
                                "Omni validation requires the Gemma audio connector"
                            )
                        video_context_positive = video_out.video_encoding
                        audio_context_positive = audio_out.audio_encoding

                    neg_hs, neg_mask = text_encoder.encode(
                        self._config.validation.negative_prompt
                    )
                    neg_out = self._embeddings_processor.process_hidden_states(
                        neg_hs, neg_mask
                    )

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=video_context_positive.cpu(),
                            audio_context_positive=(
                                audio_context_positive.cpu()
                                if audio_context_positive is not None
                                else None
                            ),
                            video_context_negative=neg_out.video_encoding.cpu(),
                            audio_context_negative=(
                                neg_out.audio_encoding.cpu()
                                if neg_out.audio_encoding is not None
                                else None
                            ),
                        )
                    )

        if cached_embeddings is not None:
            extended = cached_embeddings[primary_prompt_count:]
            self._cached_extended_validation_embeddings = extended or None
            primary = cached_embeddings[:primary_prompt_count]
            cached_embeddings = primary or None

        if uses_online_text_conditioning(self._config.training_strategy):
            # Online text mode: keep the Gemma model (and the feature extractor inside the
            # embeddings processor) resident for per-step text encoding.
            self._text_encoder = text_encoder
            logger.info(
                "Online text conditioning: keeping Gemma text encoder resident for training"
            )
        else:
            self._text_encoder = None
            # Unload Gemma model and feature extractor, keep only connectors for training
            del text_encoder
            self._embeddings_processor.feature_extractor = None

            logger.debug("Validation prompt embeddings cached. Gemma model unloaded")
        return cached_embeddings

    def _load_validation_qwen_conditions(self) -> list[CachedQwenCondition] | None:
        """Load optional validation-time Qwen condition tensors from precomputed .pt files."""
        condition_files = self._config.validation.qwen_condition_files
        if not condition_files:
            if (
                getattr(self._config.training_strategy, "use_qwen_cond", False)
                and self._config.validation.prompts
            ):
                if (
                    uses_online_qwen_conditioning(self._config.training_strategy)
                    and self._config.validation.images
                ):
                    logger.info(
                        "Validation Qwen conditions will be encoded online from the condition first frames"
                    )
                else:
                    logger.warning(
                        "training_strategy.use_qwen_cond=True but validation.qwen_condition_files is not set; "
                        "validation has no online image path, so the Qwen condition branch is disabled"
                    )
            return None

        cached_conditions = []
        for condition_file in condition_files:
            data = torch.load(condition_file, map_location="cpu", weights_only=True)
            video_context = self._select_qwen_tensor(
                data, ("qwen_video_context", "qwen_context", "qwen_hidden_states")
            )
            if video_context is None:
                raise ValueError(
                    f"Qwen condition file {condition_file} must contain one of: "
                    "qwen_video_context, qwen_context, qwen_hidden_states"
                )

            video_context = self._normalize_validation_qwen_context(
                video_context, "qwen_video_context"
            )
            video_context_mask = self._normalize_validation_qwen_mask(
                self._select_qwen_tensor(
                    data, ("qwen_attention_mask", "qwen_context_mask")
                ),
                seq_len=video_context.shape[1],
                name="qwen_attention_mask",
            )

            audio_context = self._select_qwen_tensor(data, ("qwen_audio_context",))
            audio_context_mask = None
            if audio_context is not None:
                audio_context = self._normalize_validation_qwen_context(
                    audio_context, "qwen_audio_context"
                )
                audio_context_mask = self._normalize_validation_qwen_mask(
                    self._select_qwen_tensor(
                        data, ("qwen_audio_attention_mask", "qwen_audio_context_mask")
                    ),
                    seq_len=audio_context.shape[1],
                    name="qwen_audio_attention_mask",
                )

            cached_conditions.append(
                CachedQwenCondition(
                    video_context=video_context,
                    video_context_mask=video_context_mask,
                    audio_context=audio_context,
                    audio_context_mask=audio_context_mask,
                )
            )

        logger.info(f"Loaded {len(cached_conditions)} validation Qwen condition files")
        return cached_conditions

    def _encode_validation_qwen_online(
        self, prompt_idx: int, prompt: str
    ) -> CachedQwenCondition:
        """Online-encode (and memoize) the Qwen condition for one validation prompt.

        Uses the validation image as the first frame and the validation prompt as the
        text, through the exact same encoder as training — keeping validation on the
        training distribution without qwen_condition_files.
        """
        if not hasattr(self, "_online_validation_qwen_cache"):
            self._online_validation_qwen_cache: dict[int, CachedQwenCondition] = {}
        if prompt_idx in self._online_validation_qwen_cache:
            return self._online_validation_qwen_cache[prompt_idx]

        from ltx_trainer.online_conditioning import load_first_frame_for_qwen

        strategy_config = self._config.training_strategy
        image_path = self._config.validation.images[prompt_idx]
        first_frame = load_first_frame_for_qwen(
            image_path,
            target_width=strategy_config.qwen_image_width,
            target_height=strategy_config.qwen_image_height,
        )
        encoded = self._get_online_qwen_encoder().encode_batch(
            [first_frame], [prompt], require_grad=False
        )
        condition = CachedQwenCondition(
            video_context=encoded.context.to("cpu"),
            video_context_mask=encoded.attention_mask.to("cpu"),
        )
        self._online_validation_qwen_cache[prompt_idx] = condition
        return condition

    @staticmethod
    def _select_qwen_tensor(
        data: dict[str, Any], keys: tuple[str, ...]
    ) -> Tensor | None:
        for key in keys:
            value = data.get(key)
            if isinstance(value, Tensor):
                return value
        return None

    @staticmethod
    def _normalize_validation_qwen_context(context: Tensor, name: str) -> Tensor:
        if context.ndim == 4:
            # [B, layers, seq, hidden] -> use the final stored layer for validation sampling.
            context = context[:, -1]
        if context.ndim == 3:
            # [layers, seq, hidden] -> use the final stored layer for validation sampling.
            context = context[-1]
        if context.ndim != 2:
            raise ValueError(
                f"{name} must have shape [seq, hidden] or [layers, seq, hidden], got {context.shape}"
            )
        return context.unsqueeze(0).contiguous()

    @staticmethod
    def _normalize_validation_qwen_mask(
        mask: Tensor | None, seq_len: int, name: str
    ) -> Tensor:
        if mask is None:
            return torch.ones(1, seq_len, dtype=torch.long)
        if mask.ndim == 3:
            mask = mask[:, -1]
        if mask.ndim == 2:
            mask = mask[-1]
        if mask.ndim != 1 or mask.shape[0] != seq_len:
            raise ValueError(
                f"{name} must have shape [seq] or [layers, seq], got {mask.shape}; expected seq={seq_len}"
            )
        return mask.to(dtype=torch.long).unsqueeze(0).contiguous()

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = (
            self._training_strategy.requires_audio
            or self._config.validation.generate_audio
        )

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None
            or self._config.validation.reference_videos is not None
        )

        # Load all model components (except text encoder - already handled)
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._embeddings_processor was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = (
            torch.bfloat16
            if self._config.model.training_mode == "lora"
            else torch.float32
        )
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if getattr(self._config.training_strategy, "use_qwen_cond", False):
            strategy_config = self._config.training_strategy
            self._transformer.enable_qwen_conditioning(
                qwen_context_dim=strategy_config.qwen_cross_attention_dim,
                audio_qwen_context_dim=strategy_config.audio_qwen_cross_attention_dim,
                enable_audio=bool(
                    strategy_config.with_audio and strategy_config.qwen_cond_for_audio
                ),
            )
            if getattr(strategy_config, "use_qwen_layer_adapters", False):
                if strategy_config.qwen_feature_layer_mode != "all":
                    raise ValueError(
                        "use_qwen_layer_adapters=True requires qwen_feature_layer_mode='all'"
                    )
                self._transformer.enable_qwen_layer_adapters(
                    num_layers=strategy_config.qwen_layer_adapter_num_layers,
                    input_dim=strategy_config.qwen_layer_adapter_input_dim
                    or strategy_config.qwen_cross_attention_dim,
                    video_output_dim=strategy_config.qwen_cross_attention_dim,
                    audio_output_dim=strategy_config.audio_qwen_cross_attention_dim,
                    enable_audio=bool(
                        strategy_config.with_audio
                        and strategy_config.qwen_cond_for_audio
                    ),
                )

        if getattr(self._config.training_strategy, "use_omni_cond", False):
            strategy_config = self._config.training_strategy
            self._transformer.enable_omni_conditioning(
                omni_context_dim=strategy_config.omni_cross_attention_dim,
                audio_omni_context_dim=strategy_config.audio_omni_cross_attention_dim,
                enable_audio=bool(
                    strategy_config.with_audio and strategy_config.omni_cond_for_audio
                ),
            )
            if strategy_config.use_omni_field_resampler:
                self._transformer.enable_omni_field_resampler(
                    num_layers=strategy_config.omni_layer_adapter_num_layers,
                    input_dim=strategy_config.omni_field_resampler_input_dim,
                    output_dim=strategy_config.omni_layer_adapter_input_dim,
                    num_heads=strategy_config.omni_field_resampler_num_heads,
                )
            if strategy_config.use_omni_layer_adapters:
                self._transformer.enable_omni_layer_adapters(
                    num_layers=strategy_config.omni_layer_adapter_num_layers,
                    input_dim=strategy_config.omni_layer_adapter_input_dim,
                    video_output_dim=strategy_config.omni_cross_attention_dim,
                    audio_output_dim=strategy_config.audio_omni_cross_attention_dim,
                    enable_audio=bool(
                        strategy_config.with_audio
                        and strategy_config.omni_cond_for_audio
                    ),
                )

        if getattr(
            self._config.training_strategy, "use_listener_timbre_cond", False
        ):
            strategy_config = self._config.training_strategy
            if not strategy_config.with_audio:
                raise ValueError(
                    "listener timbre conditioning requires an audio-enabled strategy"
                )
            self._transformer.enable_listener_timbre_conditioning(
                listener_timbre_context_dim=strategy_config.listener_timbre_dim
            )

        if getattr(self._config.training_strategy, "use_first_frame_audio_cond", False):
            self._transformer.enable_first_frame_audio_conditioning()

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(
                f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...'
            )
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(
                f"Unknown training mode: {self._config.model.training_mode}"
            )

        self._trainable_params = [
            p for p in self._transformer.parameters() if p.requires_grad
        ]
        logger.debug(
            f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}"
        )

    def _refresh_trainable_params(self) -> None:
        self._transformer_trainable_params = [
            parameter
            for parameter in self._transformer.parameters()
            if parameter.requires_grad
        ]
        injection_ids: set[int] = set()
        if self._config.stage2.enabled and self._config.stage2.injection_lora_enabled:
            stage2_cfg = self._config.stage2
            stage2_names = [
                *stage2_cfg.injection_modules,
                *stage2_cfg.adapter_modules,
                *stage2_cfg.full_trainable_modules,
            ]
            for name, parameter in self._transformer.named_parameters():
                if parameter.requires_grad and (
                    "stage2_lora_" in name
                    or any(
                        self._state_key_belongs_to_module(name, module_name)
                        for module_name in stage2_names
                    )
                ):
                    injection_ids.add(id(parameter))
        self._stage2_injection_trainable_params = [
            parameter
            for parameter in self._transformer_trainable_params
            if id(parameter) in injection_ids
        ]
        self._stage1_dit_trainable_params = [
            parameter
            for parameter in self._transformer_trainable_params
            if id(parameter) not in injection_ids
        ]
        self._stage2_injection_trainable_numel = sum(
            parameter.numel() for parameter in self._stage2_injection_trainable_params
        )
        self._stage1_dit_trainable_numel = sum(
            parameter.numel() for parameter in self._stage1_dit_trainable_params
        )
        self._qwen_trainable_params = (
            self._online_qwen_encoder.trainable_parameters()
            if self._online_qwen_encoder is not None and self._config.stage2.enabled
            else []
        )
        self._optimizer_param_names = {
            id(parameter): f"transformer.{name}"
            for name, parameter in self._transformer.named_parameters()
            if parameter.requires_grad
        }
        if self._online_qwen_encoder is not None and self._config.stage2.enabled:
            self._optimizer_param_names.update(
                {
                    id(parameter): f"qwen.{name}"
                    for name, parameter in self._online_qwen_encoder.model.named_parameters()
                    if parameter.requires_grad
                }
            )
        all_named_parameters = [
            *(
                (f"transformer.{name}", parameter)
                for name, parameter in self._transformer.named_parameters()
                if parameter.requires_grad
            ),
            *(
                (
                    (f"qwen.{name}", parameter)
                    for name, parameter in self._online_qwen_encoder.model.named_parameters()
                    if parameter.requires_grad
                )
                if self._online_qwen_encoder is not None and self._config.stage2.enabled
                else ()
            ),
        ]
        self._optimizer_param_uses_decay = {
            id(parameter): self._parameter_uses_weight_decay(name, parameter)
            for name, parameter in all_named_parameters
        }
        self._trainable_params = [
            *self._transformer_trainable_params,
            *self._qwen_trainable_params,
        ]
        logger.info(
            f"Trainable parameters: transformer={sum(p.numel() for p in self._transformer_trainable_params):,}, "
            f"qwen={sum(p.numel() for p in self._qwen_trainable_params):,}, "
            f"total={sum(p.numel() for p in self._trainable_params):,}"
        )

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(
            **self._config.flow_matching.timestep_sampling_params
        )

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            exclude_modules=self._config.lora.exclude_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)
        self._unfreeze_lora_full_modules()

    def _lora_full_module_names(self) -> list[str]:
        """Return newly initialized modules trained in full alongside LoRA."""
        names = list(self._config.lora.modules_to_save)
        if getattr(self._config.training_strategy, "use_omni_cond", False):
            names.extend(("omni_attn2", "video_omni_layer_adapter"))
            strategy_config = self._config.training_strategy
            if strategy_config.use_omni_field_resampler:
                names.append("omni_field_resampler")
            if strategy_config.with_audio and strategy_config.omni_cond_for_audio:
                names.extend(
                    ("audio_omni_attn2", "audio_omni_layer_adapter")
                )
        if getattr(
            self._config.training_strategy, "use_listener_timbre_cond", False
        ):
            names.append("audio_timbre_attn2")
        if getattr(self._config.training_strategy, "use_first_frame_audio_cond", False):
            names.extend(("audio_first_frame_patch_embedding", "audio_ff_attn2"))
        return list(dict.fromkeys(names))

    def _lora_persistent_module_names(self) -> list[str]:
        """Return full modules required to reconstruct a LoRA checkpoint."""
        return self._lora_full_module_names()

    @staticmethod
    def _state_key_belongs_to_module(key: str, module_name: str) -> bool:
        normalized = key.replace("base_model.model.", "", 1)
        return (
            normalized.startswith(f"{module_name}.") or f".{module_name}." in normalized
        )

    def _unfreeze_lora_full_modules(self) -> None:
        """Train new injection modules directly, matching the OVI LoRA recipe."""
        base_model = self._transformer.get_base_model()
        requested = self._lora_full_module_names()
        if self._config.stage2.enabled:
            matched = set()
            for module_path, _module in base_model.named_modules():
                for name in requested:
                    if module_path == name or module_path.endswith(f".{name}"):
                        matched.add(name)
            missing = sorted(set(requested) - matched)
            if missing:
                raise ValueError(
                    f"Stage-2 modules to persist were not found: {missing}"
                )
            logger.info(
                "Stage-2: preserving Stage-1 full injection tensors as frozen learned bases"
            )
            return

        overlapping_lora = [
            name
            for name, _param in base_model.named_parameters()
            if "lora_" in name
            and any(
                self._state_key_belongs_to_module(name, full_name)
                for full_name in requested
            )
        ]
        if overlapping_lora:
            raise ValueError(
                "Fully trained LoRA modules must be excluded from PEFT target_modules; "
                f"overlapping LoRA parameters: {overlapping_lora[:5]}"
            )

        matched: dict[str, set[int]] = {name: set() for name in requested}
        for module_path, module in base_model.named_modules():
            for name in requested:
                if module_path == name or module_path.endswith(f".{name}"):
                    module.requires_grad_(True)
                    matched[name].update(id(param) for param in module.parameters())

        missing = [name for name, param_ids in matched.items() if not param_ids]
        if missing:
            raise ValueError(
                f"LoRA full modules were requested but not found: {missing}"
            )

        param_by_id = {id(param): param for param in base_model.parameters()}
        for name, param_ids in matched.items():
            count = sum(param_by_id[param_id].numel() for param_id in param_ids)
            logger.info(f"LoRA full module '{name}': {count:,} trainable parameters")

    def _setup_stage2_training(self) -> None:
        cfg = self._config.stage2
        if cfg.injection_lora_enabled:
            self._stage2_injection_targets = apply_stage2_injection_lora(
                self._transformer,
                injection_modules=cfg.injection_modules,
                adapter_modules=cfg.adapter_modules,
                rank=cfg.injection_rank,
                alpha=cfg.injection_alpha,
                dropout=cfg.injection_dropout,
                train_adapter_norm_and_logits=cfg.train_adapter_norm_and_logits,
                full_trainable_modules=cfg.full_trainable_modules,
            )
        else:
            self._stage2_injection_targets = []
        if self._pending_stage2_state:
            if not cfg.injection_lora_enabled:
                raise ValueError(
                    "Checkpoint contains Stage-2 injection LoRA tensors but "
                    "stage2.injection_lora_enabled=false"
                )
            base = self._transformer.get_base_model()
            incompatible = base.load_state_dict(
                self._pending_stage2_state, strict=False
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"Unexpected Stage-2 injection LoRA keys: {incompatible.unexpected_keys[:8]}"
                )
            logger.info(
                f"Loaded {len(self._pending_stage2_state)} Stage-2 injection LoRA tensors"
            )
        if cfg.qwen_training_mode == "lora":
            if cfg.qwen_fsdp:
                raise NotImplementedError(
                    "qwen_fsdp is enabled but replicated-Qwen must be tested first"
                )
            encoder = self._get_online_qwen_encoder()
            logger.info(
                f"Qwen Stage-2 LoRA targets={len(encoder.qwen_lora_targets)}, "
                f"trainable={sum(p.numel() for p in encoder.trainable_parameters()):,}"
            )
        if cfg.injection_lora_enabled:
            logger.info(
                f"Stage-2 injection LoRA rank={cfg.injection_rank} "
                f"matched {len(self._stage2_injection_targets)} Linear modules"
            )
        else:
            logger.info(
                "Stage-2 injection LoRA disabled; continuing to optimize the full DiT"
            )

    def _clip_training_gradients(self, max_norm: float) -> Tensor:
        """Default behavior; joint Stage 2 overrides mixed FSDP/replica clipping."""
        return self._accelerator.clip_grad_norm_(self._trainable_params, max_norm)

    def _sync_replicated_qwen_grads(self) -> None:
        if not self._qwen_trainable_params:
            return
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return
        world_size = torch.distributed.get_world_size()
        grads = [
            parameter.grad
            if parameter.grad is not None
            else torch.zeros_like(parameter)
            for parameter in self._qwen_trainable_params
        ]
        flat_grad = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(flat_grad, op=torch.distributed.ReduceOp.SUM)
        flat_grad.div_(world_size)
        for parameter, synced_grad in zip(
            self._qwen_trainable_params,
            torch._utils._unflatten_dense_tensors(flat_grad, grads),
            strict=True,
        ):
            if parameter.grad is None:
                parameter.grad = synced_grad
            else:
                parameter.grad.copy_(synced_grad)

    def _omni_monitor_model(self) -> Any:
        model = self._accelerator.unwrap_model(
            self._transformer, keep_torch_compile=False
        )
        return model.get_base_model() if hasattr(model, "get_base_model") else model

    def _should_monitor_omni(self, is_optimization_step: bool) -> bool:
        strategy = self._config.training_strategy
        interval = int(getattr(strategy, "omni_monitor_interval", 0))
        # When called at the beginning of an optimization step, _global_step has
        # not yet been incremented for this step.
        target_step = self._global_step + 1
        return bool(
            is_optimization_step
            and getattr(strategy, "use_omni_cond", False)
            and interval > 0
            and target_step > 0
            and target_step % interval == 0
        )

    def _set_omni_residual_monitoring(self, enabled: bool) -> None:
        if not getattr(self._config.training_strategy, "use_omni_cond", False):
            return
        model = self._omni_monitor_model()
        if not hasattr(model, "set_omni_residual_monitoring"):
            if enabled:
                raise RuntimeError("LTX model does not support Omni residual monitoring")
            return
        block_indices = set(
            getattr(
                self._config.training_strategy,
                "omni_residual_monitor_blocks",
                (),
            )
        )
        model.set_omni_residual_monitoring(enabled, block_indices)

    def _collect_omni_monitor_metrics(self) -> dict[str, float]:
        """Collect low-frequency layer weights, target gradients, and residual ratios."""
        device = self._accelerator.device
        metrics: dict[str, float] = {}
        categories = (
            ("grad_norm_video_omni_adapter", "video_omni_layer_adapter"),
            ("grad_norm_audio_omni_adapter", "audio_omni_layer_adapter"),
            ("grad_norm_omni_resampler", "omni_field_resampler"),
        )
        squared = torch.zeros(len(categories), dtype=torch.float32, device=device)
        for name, parameter in self._transformer.named_parameters():
            if parameter.grad is None:
                continue
            for index, (_metric_name, fragment) in enumerate(categories):
                if fragment in name:
                    squared[index] += parameter.grad.detach().float().square().sum()
                    break
        if self._accelerator.distributed_type == DistributedType.FSDP:
            squared = self._accelerator.reduce(squared, reduction="sum")
        for index, (metric_name, _fragment) in enumerate(categories):
            metrics[metric_name] = squared[index].sqrt().item()

        model = self._omni_monitor_model()
        layer_numbers = (12, 24, 36, 48)
        for stream, adapter_name in (
            ("video", "video_omni_layer_adapter"),
            ("audio", "audio_omni_layer_adapter"),
        ):
            adapter = getattr(model, adapter_name, None)
            if adapter is None:
                continue
            expected = len(layer_numbers)
            packed = torch.zeros(expected + 1, dtype=torch.float32, device=device)
            weights = getattr(adapter, "_last_layer_weights", None)
            if isinstance(weights, Tensor) and weights.numel() == expected:
                packed[:expected] = weights.float().reshape(expected)
                packed[-1] = 1
            packed = self._accelerator.reduce(packed, reduction="sum")
            if packed[-1] <= 0:
                raise RuntimeError(f"No rank recorded {adapter_name} layer weights")
            averaged_weights = packed[:expected] / packed[-1]
            for layer, weight in zip(layer_numbers, averaged_weights, strict=True):
                metrics[f"{stream}_omni_layer_weight_{layer}"] = weight.item()

        raw_residuals = model.pop_omni_residual_metrics()
        block_indices = getattr(
            self._config.training_strategy,
            "omni_residual_monitor_blocks",
            (),
        )
        streams = ("video", "audio") if getattr(
            self._config.training_strategy, "omni_cond_for_audio", False
        ) else ("video",)
        for block_index in block_indices:
            for stream in streams:
                name = f"{stream}_omni_residual_ratio/block_{block_index}"
                value = raw_residuals.get(name)
                packed = torch.zeros(3, dtype=torch.float32, device=device)
                if value is not None:
                    packed[:2] = value
                    packed[2] = 1
                packed = self._accelerator.reduce(packed, reduction="sum")
                if packed[2] > 0:
                    metrics[name] = torch.sqrt(
                        packed[0]
                        / packed[1].clamp_min(torch.finfo(torch.float32).eps)
                    ).item()
        return metrics

    def _audit_oracle_omni_gradients(self) -> None:
        """Fail fast if either independent Stage-1 Omni branch is disconnected."""
        device = self._accelerator.device
        audit_resampler = bool(
            getattr(
                self._config.training_strategy,
                "use_omni_field_resampler",
                False,
            )
        )
        totals = torch.zeros(
            3 if audit_resampler else 2,
            dtype=torch.float32,
            device=device,
        )
        matched = [0] * len(totals)
        for name, parameter in self._transformer.named_parameters():
            if parameter.grad is None:
                continue
            if "video_omni_layer_adapter" in name:
                index = 0
            elif "audio_omni_layer_adapter" in name:
                index = 1
            elif audit_resampler and "omni_field_resampler" in name:
                index = 2
            else:
                continue
            matched[index] += 1
            totals[index] += parameter.grad.detach().float().square().sum()

        totals = self._accelerator.reduce(totals, reduction="sum")
        if not torch.isfinite(totals).all() or (totals <= 0).any():
            raise RuntimeError(
                "Oracle Omni gradient audit failed: "
                f"video_sq_norm={totals[0].item()}, "
                f"audio_sq_norm={totals[1].item()}, local_matches={matched}"
            )
        if self._accelerator.is_main_process:
            logger.info(
                "Oracle Omni gradient audit passed: "
                f"video_norm={totals[0].sqrt().item():.6g}, "
                f"audio_norm={totals[1].sqrt().item():.6g}"
                + (
                    f", resampler_norm={totals[2].sqrt().item():.6g}"
                    if audit_resampler
                    else ""
                )
            )

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config, then resolve resume state."""
        if not self._config.model.load_checkpoint:
            self._resume_state: tuple[int, TrainingState | None] = (0, None)
            return

        automatic = self._config.model.load_checkpoint == "auto"
        if automatic:
            from ltx_trainer.checkpoint_resume import select_resume_checkpoint

            if self._config.checkpoints.no_resume or self._config.checkpoints.save_training_state != "full":
                raise ValueError("load_checkpoint=auto requires no_resume=false and full training state")
            # Rank 0 selects once, so every rank loads the same completed save.
            selected = [None, None]
            if self._accelerator.is_main_process:
                try:
                    path = select_resume_checkpoint(Path(self._config.output_dir), self._config.model.training_mode)
                    selected[0] = str(path) if path else None
                except Exception as exc:
                    selected[1] = str(exc)
            if self._accelerator.num_processes > 1:
                torch.distributed.broadcast_object_list(selected, src=0)
            if selected[1]:
                raise ValueError(selected[1])
            checkpoint_path = Path(selected[0]) if selected[0] else None
            if checkpoint_path is None:
                logger.info("No previous checkpoint; starting the first run")
                self._resume_state = (0, None)
                return
        else:
            checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(
                f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}"
            )
            self._resume_state = (0, None)
            return

        self._loaded_checkpoint_path = checkpoint_path
        if (
            self._config.stage2.enabled
            and self._config.stage2.qwen_training_mode == "lora"
        ):
            from ltx_trainer.lora_checkpoint import resolve_qwen_lora_companion

            self._resolved_qwen_checkpoint = resolve_qwen_lora_companion(
                checkpoint_path, self._config.stage2.qwen_checkpoint
            )
            if self._resolved_qwen_checkpoint is not None:
                logger.info(
                    f"Resolved Stage-2 Qwen LoRA companion: {self._resolved_qwen_checkpoint}"
                )
        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

        self._resume_state = self._resolve_resume_state()

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = self._load_checkpoint_state_dict(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    @staticmethod
    def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, Tensor]:
        """Load a checkpoint state dict from either safetensors or torch serialization.

        Older full fine-tune checkpoints in this trainer were written with
        `accelerator.save()` while using a `.safetensors` suffix. Detect the
        actual file format so resume works for both old and new checkpoints.
        """
        with checkpoint_path.open("rb") as file:
            magic = file.read(4)

        if magic == b"PK\x03\x04":
            state_dict = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        else:
            state_dict = load_file(str(checkpoint_path), device="cpu")

        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Expected checkpoint to contain a state dict, got {type(state_dict).__name__}"
            )

        return state_dict

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        from ltx_trainer.lora_checkpoint import (
            _require_complete_module_state,
            load_movie_lora_tensors,
            split_lora_and_full_module_state,
        )

        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {
            k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()
        }

        lora_state, full_module_state = split_lora_and_full_module_state(state_dict)

        if self._config.stage2.enabled:
            self._pending_stage2_state = {
                key: value
                for key, value in full_module_state.items()
                if "stage2_lora_" in key
            }
            full_module_state = {
                key.replace(".base_layer.", "."): value
                for key, value in full_module_state.items()
                if "stage2_lora_" not in key
            }

        # Restore PEFT A/B tensors and the manually trained injection branches.
        base_model = self._transformer.get_base_model()
        load_movie_lora_tensors(self._transformer, lora_state)
        _require_complete_module_state(
            base_model,
            full_module_state,
            tuple(self._lora_persistent_module_names()),
        )
        if full_module_state:
            incompatible = base_model.load_state_dict(full_module_state, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    "Full-weight LoRA tensors do not match the configured injection modules: "
                    f"{incompatible.unexpected_keys[:5]}"
                )
            logger.info(
                f"Loaded {len(full_module_state)} full-weight injection tensors"
            )

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _resolve_resume_state(self) -> tuple[int, TrainingState | None]:
        """Determine resume state by looking for a training state file next to the loaded checkpoint.
        Returns (initial_step, TrainingState or None).
        If no_resume config is set, no checkpoint loaded, or no state file found: returns (0, None).
        """
        if self._config.checkpoints.no_resume or self._loaded_checkpoint_path is None:
            return 0, None

        state = self._load_training_state(self._loaded_checkpoint_path)
        if self._config.model.load_checkpoint == "auto":
            from ltx_trainer.checkpoint_resume import require_full_resume_state

            if state is None:
                raise ValueError("Automatic resume state is missing or unreadable")
            require_full_resume_state(state, self._loaded_checkpoint_path, self._accelerator.num_processes)
        if state is None:
            return 0, None

        fp = state.config_fingerprint
        cfg = self._config
        mismatches: list[str] = []
        if fp.optimizer_type != cfg.optimization.optimizer_type:
            mismatches.append(
                f"optimizer_type: {fp.optimizer_type} → {cfg.optimization.optimizer_type}"
            )
        if fp.scheduler_type != cfg.optimization.scheduler_type:
            mismatches.append(
                f"scheduler_type: {fp.scheduler_type} → {cfg.optimization.scheduler_type}"
            )
        if fp.training_mode != cfg.model.training_mode:
            mismatches.append(
                f"training_mode: {fp.training_mode} → {cfg.model.training_mode}"
            )
        if (
            cfg.model.training_mode == "lora"
            and cfg.lora is not None
            and fp.lora_rank is not None
            and fp.lora_rank != cfg.lora.rank
        ):
            mismatches.append(f"lora_rank: {fp.lora_rank} → {cfg.lora.rank}")
        stage2 = cfg.stage2
        fingerprint_checks = (
            ("stage2_enabled", fp.stage2_enabled, stage2.enabled),
            (
                "stage2_injection_lora_enabled",
                fp.stage2_injection_lora_enabled,
                stage2.injection_lora_enabled,
            ),
            ("stage2_injection_rank", fp.stage2_injection_rank, stage2.injection_rank),
            ("qwen_training_mode", fp.qwen_training_mode, stage2.qwen_training_mode),
            ("qwen_rank", fp.qwen_rank, stage2.qwen_rank),
            (
                "gradient_accumulation_steps",
                fp.gradient_accumulation_steps,
                cfg.optimization.gradient_accumulation_steps,
            ),
            ("weight_decay", fp.weight_decay, cfg.optimization.weight_decay),
            ("batch_size", fp.batch_size, cfg.optimization.batch_size),
            ("seed", fp.seed, cfg.seed),
            ("world_size", fp.world_size, self._accelerator.num_processes),
            (
                "context_parallel_size",
                fp.context_parallel_size,
                cfg.acceleration.context_parallel_size,
            ),
            (
                "training_data_fingerprint",
                fp.training_data_fingerprint,
                self._training_data_fingerprint,
            ),
        )
        for name, saved, current in fingerprint_checks:
            if saved is not None and saved != current:
                mismatches.append(f"{name}: {saved} → {current}")
        required_resume_fields = (
            "weight_decay",
            "batch_size",
            "seed",
            "world_size",
            "context_parallel_size",
            "training_data_fingerprint",
        )
        for name in required_resume_fields:
            if getattr(fp, name) is None:
                mismatches.append(f"{name}: missing from legacy checkpoint")
        if mismatches:
            if self._config.model.load_checkpoint == "auto":
                raise ValueError("Automatic resume fingerprint mismatch: " + "; ".join(mismatches))
            logger.warning(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "🚨 CHECKPOINT RESUME REJECTED — TRAINING WILL RESTART FROM STEP 0 🚨\n"
                "The saved data/topology fingerprint does not match this run:\n"
                f"  - {'; '.join(mismatches)}\n"
                "Model weights were loaded, but optimizer/scheduler/RNG and the saved data "
                "cursor will NOT be resumed. Set checkpoints.no_resume=true only when this "
                "weight-only restart is intentional.\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            return 0, None

        if state.global_step < 0:
            logger.warning(
                f"⚠️ Training state has invalid global_step={state.global_step!r}. Starting from step 0."
            )
            return 0, None
        logger.info(f"📌 Resuming from step {state.global_step}")
        return state.global_step, state

    def _compute_training_data_fingerprint(self) -> str:
        """Hash the training data contract and manifest/index contents used for sampling."""
        data = self._config.data
        strategy = self._config.training_strategy
        contract = {
            "preprocessed_data_root": data.preprocessed_data_root,
            "preprocessed_data_roots": data.preprocessed_data_roots,
            "metadata_jsonls": data.metadata_jsonls,
            "precomputed_sample_index_files": data.precomputed_sample_index_files,
            "caption_field": data.caption_field,
            "video_field": data.video_field,
            "path_remap": data.path_remap,
            "max_training_frames": data.max_training_frames,
            "length_bucket": data.length_bucket.model_dump(mode="json"),
            "omni_conditions_dir": getattr(strategy, "omni_conditions_dir", None),
            "omni_expected_route": getattr(strategy, "omni_expected_route", None),
        }
        if getattr(strategy, "use_omni_field_resampler", False) and getattr(strategy, "use_omni_cond", False):
            audit_path = Path(data.preprocessed_data_root) / "full_training_contract_audit.json"
            if audit_path.is_file():
                contract["stage1_audit_fingerprint"] = json.loads(audit_path.read_text()).get("fingerprint")
        digest = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        )
        manifest_paths = [
            *data.metadata_jsonls,
            *data.precomputed_sample_index_files,
        ]
        for configured_path in manifest_paths:
            path = Path(configured_path).expanduser().resolve()
            digest.update(b"\0path\0")
            digest.update(str(path).encode())
            if not path.is_file():
                digest.update(b"\0missing\0")
                continue
            digest.update(b"\0content\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _load_training_state(checkpoint_path: Path) -> TrainingState | None:
        """Load training state file that corresponds to a checkpoint weights file."""
        match = re.search(r"step_(\d+)", checkpoint_path.name)
        if not match:
            return None

        step_str = match.group(1)
        state_path = checkpoint_path.parent / f"training_state_step_{step_str}.pt"

        if not state_path.exists():
            return None

        try:
            raw: dict = torch.load(state_path, map_location="cpu", weights_only=False)
            state = TrainingState.from_save_dict(raw)
            logger.info(f"📥 Loaded training state from {state_path}")
            return state
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load training state from {state_path}: {e}"
            )
            return None

    def _restore_training_state(self, training_state: TrainingState) -> bool:
        """Restore optimizer, scheduler, and RNG states from a loaded TrainingState.
        Must be called after _init_optimizer() (which calls accelerator.prepare).
        Returns True if restore succeeded, False if it failed (caller should fall back to step 0).
        """
        restore_error: str | None = None
        optimizer_restored = False
        try:
            if training_state.optimizer_state_path is not None:
                self._restore_fsdp_optimizer_state(training_state)
                optimizer_restored = True
            elif training_state.optimizer_state_dict is not None:
                self._optimizer.load_state_dict(training_state.optimizer_state_dict)
                logger.debug("Restored optimizer state (full mode)")
                optimizer_restored = True
            elif self._config.checkpoints.save_training_state == "full":
                raise RuntimeError(
                    "Training state has no optimizer state, but the current config requires "
                    "save_training_state='full'"
                )

            if (
                training_state.lr_scheduler_state_dict is not None
                and self._lr_scheduler is not None
            ):
                self._lr_scheduler.load_state_dict(
                    training_state.lr_scheduler_state_dict
                )
                logger.debug("Restored LR scheduler state")
        except Exception as e:
            restore_error = (
                f"rank {self._accelerator.process_index}: {type(e).__name__}: {e}"
            )

        restore_errors = (
            gather_object([restore_error])
            if self._accelerator.num_processes > 1
            else [restore_error]
        )
        failures = [error for error in restore_errors if error is not None]
        if failures:
            if self._config.model.load_checkpoint == "auto":
                raise RuntimeError("Automatic state restore failed: " + "; ".join(failures))
            logger.warning(
                "⚠️ Failed to restore training state on one or more ranks; resetting optimizer/scheduler "
                f"and starting from step 0. Errors: {'; '.join(failures)}"
            )
            # Some ranks may already have loaded their local shard. Recreate the optimizer
            # everywhere so the step-0 fallback cannot start with divergent states.
            self._init_optimizer()
            return False

        if not optimizer_restored:
            logger.warning(
                "Resuming from minimal training state: step, scheduler, and RNG were "
                "restored, but optimizer moments were not saved. The optimizer starts "
                "fresh, so this is not a numerically continuous resume."
            )

        rng = training_state.rng_states
        if training_state.rng_states_by_rank is not None:
            if (
                len(training_state.rng_states_by_rank)
                == self._accelerator.num_processes
            ):
                rng = training_state.rng_states_by_rank[self._accelerator.process_index]
            else:
                logger.warning(
                    "Saved per-rank RNG world size does not match; using the main-rank RNG "
                    "as a non-bitwise-compatible fallback"
                )
        elif self._accelerator.num_processes > 1:
            logger.warning(
                "Legacy checkpoint has no per-rank RNG states; resume is not bitwise exact"
            )
        if rng.torch_state is not None:
            torch.random.set_rng_state(rng.torch_state)
        if rng.cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng.cuda_state, device=self._accelerator.device)
        logger.debug("Restored RNG states")

        return True

    def _restore_fsdp_optimizer_state(self, training_state: TrainingState) -> None:
        if self._accelerator.distributed_type != DistributedType.FSDP:
            raise RuntimeError(
                "FSDP optimizer checkpoint cannot be loaded outside FSDP training"
            )
        if training_state.optimizer_state_format != "fsdp_local_optimizer_v1":
            raise ValueError(
                f"Unsupported FSDP optimizer state format: {training_state.optimizer_state_format!r}"
            )
        if self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "Cannot resolve FSDP optimizer state without a loaded model checkpoint"
            )

        optimizer_dir = (
            self._loaded_checkpoint_path.parent / training_state.optimizer_state_path
        )
        if not optimizer_dir.is_dir():
            raise FileNotFoundError(
                f"FSDP optimizer state directory not found: {optimizer_dir}"
            )
        expected_rank_files = {
            f"rank_{rank:05d}.pt" for rank in range(self._accelerator.num_processes)
        }
        actual_rank_files = {path.name for path in optimizer_dir.glob("rank_*.pt")}
        if actual_rank_files != expected_rank_files:
            missing = sorted(expected_rank_files - actual_rank_files)
            unexpected = sorted(actual_rank_files - expected_rank_files)
            raise RuntimeError(
                "Incomplete FSDP optimizer checkpoint: "
                f"missing={missing}, unexpected={unexpected}"
            )
        manifest_path = optimizer_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != training_state.optimizer_state_format:
            raise RuntimeError(
                "FSDP optimizer state format mismatch: "
                f"training_state={training_state.optimizer_state_format!r}, "
                f"manifest={manifest.get('format')!r}"
            )
        if int(manifest.get("global_step", -1)) != training_state.global_step:
            raise RuntimeError(
                "FSDP optimizer state step mismatch: "
                f"training_state={training_state.global_step}, "
                f"manifest={manifest.get('global_step')!r}"
            )
        saved_world_size = int(manifest["world_size"])
        if saved_world_size != self._accelerator.num_processes:
            raise RuntimeError(
                "FSDP local optimizer state requires the same world size: "
                f"checkpoint={saved_world_size}, current={self._accelerator.num_processes}"
            )
        if manifest.get("optimizer_type") != self._config.optimization.optimizer_type:
            raise RuntimeError(
                "FSDP optimizer type mismatch: "
                f"checkpoint={manifest.get('optimizer_type')!r}, "
                f"current={self._config.optimization.optimizer_type!r}"
            )
        current_topology = self._fsdp_optimizer_topology()
        for key, current_value in current_topology.items():
            if manifest.get(key) != current_value:
                raise RuntimeError(
                    f"FSDP optimizer topology mismatch for {key}: "
                    f"checkpoint={manifest.get(key)!r}, current={current_value!r}"
                )

        rank_file = optimizer_dir / f"rank_{self._accelerator.process_index:05d}.pt"
        if not rank_file.is_file():
            raise FileNotFoundError(f"FSDP optimizer rank state not found: {rank_file}")
        optimizer_state = torch.load(rank_file, map_location="cpu", weights_only=True)
        self._optimizer.load_state_dict(optimizer_state)
        logger.info(f"Restored rank-local FSDP optimizer state from {rank_file}")

    def _fsdp_optimizer_topology(self) -> dict[str, str | bool]:
        plugin = self._accelerator.state.fsdp_plugin
        if plugin is None:
            raise RuntimeError("FSDP plugin is unavailable")
        return {
            "fsdp_sharding_strategy": str(plugin.sharding_strategy),
            "fsdp_use_orig_params": bool(plugin.use_orig_params),
        }

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if (
            self._accelerator.distributed_type == DistributedType.FSDP
            and self._config.model.training_mode == "lora"
        ):
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model()
            if hasattr(self._transformer, "get_base_model")
            else self._transformer
        )

        transformer.set_gradient_checkpointing(
            self._config.optimization.enable_gradient_checkpointing
        )

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(
            f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB"
        )

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(
                    f"Checkpoint file must have a .safetensors extension: {checkpoint_path}"
                )
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = [
                path
                for path in checkpoint_path.rglob("*step_*.safetensors")
                if path.name.startswith(("lora_weights_step_", "model_weights_step_"))
            ]

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(
                f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory."
            )

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        strategy_config = self._config.training_strategy
        if (
            getattr(strategy_config, "use_omni_cond", False)
            and getattr(strategy_config, "omni_layer_adapter_num_layers", 1) == 4
            and self._config.data.max_training_frames is not None
        ):
            raise ValueError(
                "Oracle/teacher-forcing Omni conditions describe the complete target; "
                "data.max_training_frames must be null so video, audio, and GT Plan "
                "cover exactly the same time range."
            )
        online_text = uses_online_text_conditioning(strategy_config)
        online_qwen = getattr(
            strategy_config, "use_qwen_cond", False
        ) and uses_online_qwen_conditioning(strategy_config)
        online = online_text or online_qwen
        length_bucket_cfg = self._config.data.length_bucket
        use_length_bucket = bool(length_bucket_cfg.enabled)
        if (
            use_length_bucket
            and self._config.acceleration.context_parallel_size > 1
        ):
            raise ValueError(
                "Context parallelism does not yet support the length-bucket dataset path"
            )

        if self._dataset is None:
            if use_length_bucket:
                self._dataset = self._build_length_bucket_dataset()
            elif online:
                if not self._config.data.metadata_jsonls:
                    raise ValueError(
                        "Online Gemma/Qwen conditioning requires data.metadata_jsonls"
                    )
                self._dataset = OnlineConditionDataset(
                    self._config.data.preprocessed_data_root,
                    metadata_jsonls=self._config.data.metadata_jsonls,
                    with_audio=getattr(strategy_config, "with_audio", False),
                    audio_latents_dir=getattr(
                        strategy_config, "audio_latents_dir", "audio_latents"
                    ),
                    include_conditions=not online_text,
                    caption_field=self._config.data.caption_field,
                    video_field=self._config.data.video_field,
                    path_remap=self._config.data.path_remap,
                    qwen_image_width=getattr(strategy_config, "qwen_image_width", 1088),
                    qwen_image_height=getattr(
                        strategy_config, "qwen_image_height", 640
                    ),
                    max_training_frames=self._config.data.max_training_frames,
                )
                logger.info(
                    f"Loaded mixed-conditioning dataset with {len(self._dataset):,} samples "
                    f"(Gemma_online={online_text}, Qwen_online={online_qwen})"
                )
            else:
                # Get data sources from the training strategy
                data_sources = self._training_strategy.get_data_sources()

                self._dataset = PrecomputedDataset(
                    self._config.data.preprocessed_data_root, data_sources=data_sources
                )
                logger.debug(
                    f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}"
                )

        if (
            getattr(strategy_config, "use_omni_cond", False)
            and not isinstance(self._dataset, OfflineOmniConditionDataset)
        ):
            if use_length_bucket:
                raise ValueError(
                    "Offline Omni interface currently requires the non-length-bucket "
                    "dataset path; add bucket-key condition mapping before enabling both"
                )
            sample_ids = getattr(self._dataset, "sample_ids", None)
            if sample_ids is None:
                raise ValueError(
                    "Omni conditioning requires a dataset exposing stable sample_ids"
                )
            expected_version = (
                "teacher_forcing_raw"
                if strategy_config.use_omni_field_resampler
                else
                "teacher_forcing"
                if strategy_config.omni_layer_adapter_num_layers == 4
                else "generated"
                if strategy_config.omni_layer_adapter_num_layers == 1
                else None
            )
            self._dataset = OfflineOmniConditionDataset(
                self._dataset,
                omni_condition_root=strategy_config.omni_conditions_dir,
                sample_ids=sample_ids,
                listener_timbre_root=(
                    strategy_config.listener_timbre_embeddings_dir
                    if strategy_config.use_listener_timbre_cond
                    else None
                ),
                expected_version=expected_version,
                expected_route=getattr(
                    strategy_config, "omni_expected_route", "route_a_no_context"
                ),
            )
            logger.info(
                "Attached offline Omni conditions "
                f"(version={expected_version or 'auto'}, timbre="
                f"{strategy_config.use_listener_timbre_cond})"
            )

        num_workers = self._config.data.num_dataloader_workers
        if getattr(self._config.acceleration, "context_parallel_size", 1) > 1:
            sampler = ContextParallelDistributedSampler(
                self._dataset,
                num_replicas=self._accelerator.num_processes,
                rank=self._accelerator.process_index,
                sp_size=self._config.acceleration.context_parallel_size,
                shuffle=True,
                seed=self._config.seed,
                drop_last=True,
            )
            dataloader = DataLoader(
                self._dataset,
                batch_size=self._config.optimization.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0,
                collate_fn=collate_online_condition_batch
                if online
                else collate_precomputed_batch,
            )
            self._dataloader = dataloader
            self._cp_sampler = sampler
            self._length_bucket_place_on_device = True
            initial_step, _training_state = self._resume_state
            batches_per_epoch = sampler.num_samples // self._config.optimization.batch_size
            if initial_step > 0 and batches_per_epoch > 0:
                global_micro = (
                    initial_step * self._config.optimization.gradient_accumulation_steps
                )
                epoch = global_micro // batches_per_epoch
                start_batch = global_micro % batches_per_epoch
                sampler.set_epoch(
                    epoch,
                    start_index=start_batch * self._config.optimization.batch_size,
                )
                self._cp_epoch = epoch
                logger.info(
                    f"ContextParallelDistributedSampler resume: epoch={epoch}, "
                    f"start_batch={start_batch} (from global_step={initial_step})"
                )
        elif use_length_bucket:
            self._init_length_bucket_dataloader(num_workers=num_workers)
        else:
            dataloader = DataLoader(
                self._dataset,
                batch_size=self._config.optimization.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0,
                collate_fn=collate_online_condition_batch
                if online
                else collate_precomputed_batch,
            )
            self._dataloader = self._accelerator.prepare(dataloader)

    def _resolve_length_bucket_roots(self) -> list[str]:
        roots = list(self._config.data.preprocessed_data_roots)
        if not roots and self._config.data.preprocessed_data_root:
            roots = [self._config.data.preprocessed_data_root]
        return roots

    def _build_length_bucket_dataset(self) -> LengthBucketOnlineDataset:
        strategy_config = self._config.training_strategy
        if not self._config.data.metadata_jsonls:
            raise ValueError("length_bucket.enabled=true requires data.metadata_jsonls")
        roots = self._resolve_length_bucket_roots()
        dataset = LengthBucketOnlineDataset(
            roots,
            metadata_jsonls=self._config.data.metadata_jsonls,
            sample_index_files=self._config.data.precomputed_sample_index_files,
            with_audio=getattr(strategy_config, "with_audio", False),
            audio_latents_dir=getattr(
                strategy_config, "audio_latents_dir", "audio_latents"
            ),
            include_conditions=not uses_online_text_conditioning(strategy_config),
            caption_field=self._config.data.caption_field,
            video_field=self._config.data.video_field,
            path_remap=self._config.data.path_remap,
            qwen_image_width=getattr(strategy_config, "qwen_image_width", 1088),
            qwen_image_height=getattr(strategy_config, "qwen_image_height", 640),
            allowed_frames=self._config.data.length_bucket.allowed_frames,
        )
        logger.info(
            f"Loaded length-bucket mixed-conditioning dataset with {len(dataset):,} samples"
        )
        return dataset

    def _init_length_bucket_dataloader(self, *, num_workers: int) -> None:
        """Build UniAV-style bucket sampler DataLoader without Accelerate re-sharding.

        The sampler already shards by rank and syncs bucket choice across ranks. We therefore
        skip ``accelerator.prepare(dataloader)`` and place batches on device manually.
        """
        if self._config.optimization.batch_size != 1:
            raise ValueError("length_bucket path requires optimization.batch_size=1")
        if not isinstance(self._dataset, LengthBucketOnlineDataset):
            raise TypeError(
                f"length_bucket path expects LengthBucketOnlineDataset, got {type(self._dataset).__name__}"
            )

        batch_size = self._config.optimization.batch_size
        sampler = LengthBucketSampler(
            self._dataset,
            batch_size=batch_size,
            num_replicas=self._accelerator.num_processes,
            rank=self._accelerator.process_index,
            seed=self._config.seed,
            sync_bucket_across_ranks=self._config.data.length_bucket.sync_bucket_across_ranks,
            max_batches_per_epoch=self._config.data.length_bucket.max_batches_per_epoch,
        )

        # Resume into the deterministic step stream when continuing from a checkpoint.
        initial_step, _training_state = self._resume_state
        if initial_step > 0 and sampler.num_batches_for_this_rank > 0:
            global_micro = (
                initial_step * self._config.optimization.gradient_accumulation_steps
            )
            epoch = global_micro // sampler.num_batches_for_this_rank
            start_batch = global_micro % sampler.num_batches_for_this_rank
            sampler.set_epoch(epoch, start_step_index=start_batch)
            self._length_bucket_epoch = epoch
            logger.info(
                f"LengthBucketSampler resume: epoch={epoch}, start_step_index={start_batch} "
                f"(from global_step={initial_step})"
            )
        else:
            self._length_bucket_epoch = 0

        self._length_bucket_sampler = sampler
        self._length_bucket_place_on_device = True
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            collate_fn=collate_online_condition_batch,
        )

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(
                    adapter_name="default", init_lora_weights=True
                )

    @staticmethod
    def _parameter_uses_weight_decay(name: str, parameter: Tensor) -> bool:
        """Apply decay only to matrix weights, excluding learned query tokens."""
        if parameter.ndim <= 1 or name.endswith(".bias"):
            return False
        if "layer_logits" in name:
            return False
        if "field_queries" in name or "global_queries" in name:
            return False
        return True

    def _add_weight_decay_groups(
        self,
        groups: list[dict[str, Any]],
        *,
        params: list[Tensor],
        lr: float,
        name: str,
    ) -> None:
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for parameter in params:
            use_decay = self._optimizer_param_uses_decay.get(id(parameter))
            if use_decay is None:
                parameter_name = self._optimizer_param_names.get(
                    id(parameter), "<unnamed>"
                )
                raise RuntimeError(
                    "Optimizer parameter was not classified before distributed wrapping: "
                    f"{parameter_name}"
                )
            (decay if use_decay else no_decay).append(parameter)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": lr,
                    "weight_decay": self._config.optimization.weight_decay,
                    "name": f"{name}_decay",
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{name}_no_decay",
                }
            )

    def _init_optimizer(self) -> None:
        """Initialize optimizer groups for DiT, Stage-2 injection, and replicated Qwen."""
        opt_cfg = self._config.optimization
        param_groups: list[dict[str, Any]] = []
        if self._config.stage2.enabled:
            stage2_cfg = self._config.stage2
            # FSDP(use_orig_params=True) shards the original Parameter objects in
            # place, but its wrapped names are no longer suitable for classification.
            # The partition must therefore be captured before accelerator.prepare().
            dit_params = self._stage1_dit_trainable_params
            injection_params = self._stage2_injection_trainable_params
            dit_group_name = (
                "dit_full_sft"
                if self._config.model.training_mode == "full"
                else "dit_stage1_lora"
            )
            if dit_params:
                self._add_weight_decay_groups(
                    param_groups,
                    params=dit_params,
                    lr=opt_cfg.learning_rate,
                    name=dit_group_name,
                )
            if injection_params:
                self._add_weight_decay_groups(
                    param_groups,
                    params=injection_params,
                    lr=stage2_cfg.injection_learning_rate,
                    name="stage2_injection_lora",
                )
            if self._qwen_trainable_params:
                self._add_weight_decay_groups(
                    param_groups,
                    params=self._qwen_trainable_params,
                    lr=stage2_cfg.qwen_learning_rate,
                    name="qwen_stage2_lora",
                )
        else:
            self._add_weight_decay_groups(
                param_groups,
                params=self._trainable_params,
                lr=opt_cfg.learning_rate,
                name="stage1_trainable",
            )

        parameter_ids = [
            id(parameter)
            for group in param_groups
            for parameter in group["params"]
        ]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("A trainable parameter appears in multiple optimizer groups")
        if set(parameter_ids) != {id(parameter) for parameter in self._trainable_params}:
            raise RuntimeError("Optimizer groups do not cover every trainable parameter exactly once")
        logger.info(
            "Optimizer groups: "
            + ", ".join(
                f"{group['name']}=rank_shard:{sum(p.numel() for p in group['params']):,},"
                f"tensors:{len(group['params'])}@lr={group['lr']:.2e},"
                f"wd={group['weight_decay']:.2e}"
                for group in param_groups
            )
        )

        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(param_groups)
        elif opt_cfg.optimizer_type == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit  # noqa: PLC0415

            optimizer = AdamW8bit(param_groups)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(
            optimizer, lr_scheduler
        )

    @contextlib.contextmanager
    def _offloaded_optimizer_state(self) -> Iterator[None]:
        """Context manager that offloads optimizer state to CPU during validation.
        Opt-in via `acceleration.offload_optimizer_during_validation`. Frees VRAM for
        validation video generation when optimizer state is large (e.g. full fine-tune
        AdamW, high-rank LoRA). No-op for FSDP (sharded state -- manual `.cpu()` breaks
        metadata).
        """
        enabled = (
            self._config.acceleration.offload_optimizer_during_validation
            and self._accelerator.distributed_type != DistributedType.FSDP
        )

        # Track exactly which tensors we move so we don't promote ones that were
        # intentionally on CPU (e.g. AdamW's `step` scalar on recent PyTorch).
        offloaded: list[tuple[dict, str]] = []
        if enabled:
            offloaded_bytes = 0
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        offloaded.append((state, k))
                        offloaded_bytes += v.nbytes
            if offloaded:
                logger.info(
                    f"Offloading optimizer state to CPU ({offloaded_bytes / 1e9:.1f} GB)"
                )
                for state, k in offloaded:
                    state[k] = state[k].cpu()

        try:
            yield
        finally:
            device = self._accelerator.device
            for state, k in offloaded:
                state[k] = state[k].to(device)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        # Accelerate's AcceleratedScheduler advances the wrapped scheduler
        # num_processes times per optimization step (compensating for each process
        # stepping independently). Scale total iters accordingly so the decay
        # horizon matches optimization.steps in GLOBAL steps — otherwise the LR
        # decays num_processes× too fast in multi-GPU runs.
        steps = self._config.optimization.steps * max(
            1, self._accelerator.num_processes
        )
        params = self._config.optimization.scheduler_params or {}
        params = dict(params)  # don't mutate the config

        # Optional linear warmup (scheduler_params.warmup_steps, in GLOBAL optimization
        # steps). LR ramps 0 -> learning_rate over warmup_steps, then hands off to the
        # base scheduler. Scaled by num_processes for the same AcceleratedScheduler
        # reason as `steps` above.
        warmup_steps = int(params.pop("warmup_steps", 0)) * max(
            1, self._accelerator.num_processes
        )
        # SequentialLR resets the main scheduler's counter at the milestone, so the
        # decay phase spans the REMAINING steps after warmup.
        steps = max(1, steps - warmup_steps)

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),
                T_mult=params.pop("T_mult", 1),
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-8,  # LinearLR requires > 0; effectively starts at ~0
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            if scheduler is None:
                scheduler = warmup
            else:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, scheduler],
                    milestones=[warmup_steps],
                )

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # Keep the conservative default for legacy configs; fixed I2AV Stage-2 runs can
        # disable graph traversal after verifying that every transformer parameter is used.
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self._config.acceleration.ddp_find_unused_parameters,
            gradient_as_bucket_view=self._config.acceleration.ddp_gradient_as_bucket_view,
        )

        # Idle validation ranks may wait while full-length samples are generated.
        # Match collectives to the configured validation budget, rather than NCCL's
        # 600-second default (which is shorter than measured 720p sampling).
        collective_timeout = validation_collective_timeout_seconds(self._config.validation)
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=collective_timeout))
        logger.info(f"Distributed collective timeout: {collective_timeout}s")

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            accumulation = self._config.optimization.gradient_accumulation_steps
            cp_size = max(1, self._config.acceleration.context_parallel_size)
            global_batch = (
                local_batch * (self._accelerator.num_processes // cp_size) * accumulation
            )
            logger.info(
                f"Local batch size: {local_batch}, gradient accumulation: {accumulation}, "
                f"effective global batch size: {global_batch}"
            )

        if getattr(self._config.acceleration, "context_parallel_size", 1) > 1:
            init_sequence_parallel(self._config.acceleration.context_parallel_size)
            logger.info(
                f"🚀 Context Parallelism enabled: sp_size={self._config.acceleration.context_parallel_size}, "
                f"sp_rank={get_sp_rank()}, dp_rank={get_dp_rank()}, dp_world_size={get_dp_world_size()}"
            )

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin")
            and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(
                f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}"
            )

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if (
            self._accelerator.distributed_type == DistributedType.FSDP
            and self._config.acceleration.quantization
        ):
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    def _run_distributed_validation(
        self, progress: TrainingProgress, *, extended: bool = False
    ) -> list[Path]:
        """Generate one validation profile and score it on global rank 0."""
        val_config = self._config.validation
        original_profile = None
        label = "extended" if extended else None
        if extended:
            if not self._extended_validation_prompts:
                raise RuntimeError(
                    "extended validation requested without resolved prompts"
                )
            original_profile = (
                val_config.prompts,
                val_config.images,
                self._validation_metadata_rows,
                self._cached_validation_embeddings,
                self._cached_validation_qwen_conditions,
                self._validation_omni_conditions,
            )
            val_config.prompts = self._extended_validation_prompts
            val_config.images = self._extended_validation_images
            self._validation_metadata_rows = self._extended_validation_metadata_rows
            self._cached_validation_embeddings = (
                self._cached_extended_validation_embeddings
            )
            self._cached_validation_qwen_conditions = None
            self._validation_omni_conditions = (
                self._extended_validation_omni_conditions
            )

        # Qwen LoRA changes every optimizer step. Reusing conditions encoded by an
        # earlier checkpoint would make Stage-2 validation blind to those updates.
        self._online_validation_qwen_cache = {}

        try:
            benchmark = ValidationBenchmark(
                val_config,
                Path(self._config.output_dir) / "samples",
                self._global_step,
                expected_sample_count=(
                    val_config.extended_max_samples
                    if extended
                    else int(val_config.jsonl_max_samples or len(val_config.prompts))
                ),
                metadata_rows=self._validation_metadata_rows,
                label=label,
                metric_prefix=(
                    val_config.extended_metric_prefix
                    if extended
                    else val_config.metric_prefix
                ),
            )
            benchmark.prepare(self._accelerator.is_main_process)
            self._accelerator.wait_for_everyone()

            was_training = self._transformer.training
            qwen_model = (
                self._online_qwen_encoder.model
                if self._online_qwen_encoder is not None
                else None
            )
            qwen_was_training = bool(qwen_model is not None and qwen_model.training)
            self._transformer.eval()
            if qwen_model is not None:
                qwen_model.eval()
            try:
                with suspend_sequence_parallel():
                    sampled = self._sample_videos(progress, output_label=label)
            finally:
                if was_training:
                    self._transformer.train()
                if qwen_model is not None and qwen_was_training:
                    qwen_model.train()
                    # PEFT's ``base_model`` is a LoraModel wrapper, while the
                    # Qwen vision tower lives under the backbone's ``model``.
                    # OnlineQwenEncoder resolves and validates that backbone
                    # once in ``_bind_backbone_refs``; use the stable reference
                    # instead of depending on PEFT/Transformers wrapper depth.
                    self._online_qwen_encoder.base_model.visual.eval()

            if self._accelerator.num_processes > 1:
                sampled = sorted(gather_object(sampled), key=lambda x: x[0])

            paths = [path for _, path in sampled]
            if self._accelerator.is_main_process and paths:
                wandb_key = (
                    "validation_extended_samples" if extended else "validation_samples"
                )
                self._log_validation_samples(
                    paths[:8], val_config.prompts[:8], key=wandb_key
                )

            references = val_config.images or []
            benchmark.run_or_wait(
                is_main_process=self._accelerator.is_main_process,
                paths=paths,
                prompts=val_config.prompts,
                references=references,
                log_metrics=self._log_metrics,
            )
            if not benchmark.enabled:
                self._accelerator.wait_for_everyone()
            return paths
        finally:
            if original_profile is not None:
                (
                    val_config.prompts,
                    val_config.images,
                    self._validation_metadata_rows,
                    self._cached_validation_embeddings,
                    self._cached_validation_qwen_conditions,
                    self._validation_omni_conditions,
                ) = original_profile

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(
        self, progress: TrainingProgress, *, output_label: str | None = None
    ) -> list[tuple[int, Path]]:
        """Run validation by generating videos from this rank's share of the validation prompts.
        Prompts are split round-robin across ranks via `process_index` / `num_processes`,
        which collapses to "all prompts" when running on a single GPU. Returns
        (prompt_idx, path) tuples so the caller can reconstruct global order without
        relying on filename conventions.
        Under FSDP with multiple processes, ranks pad with extra generate passes (same prompt,
        no disk write) so every rank runs the same number of forwards — avoids collective mismatch.
        """
        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        # Zero gradients and free GPU memory to reclaim memory before validation sampling
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        prompts = self._config.validation.prompts
        rank = self._accelerator.process_index
        world_size = self._accelerator.num_processes
        rank_indices = list(range(rank, len(prompts), world_size))

        # FSDP: every rank must run the same number of forwards; pad with duplicate generates (no save).
        work: list[tuple[int, bool]] = [(i, True) for i in rank_indices]
        if (
            self._accelerator.distributed_type == DistributedType.FSDP
            and world_size > 1
        ):
            max_per_rank = math.ceil(len(prompts) / world_size)
            pad_seed = rank_indices[-1] if rank_indices else 0
            work += [(pad_seed, False)] * (max_per_rank - len(work))

        sampling_ctx = progress.start_sampling(
            num_prompts=len(work),
            num_steps=inference_steps,
        )

        # Create a validation sampler with loaded models and progress tracking
        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
            move_transformer_to_device=False,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        results: list[tuple[int, Path]] = []
        width, height, num_frames = self._config.validation.video_dims

        for local_i, (prompt_idx, save_output) in enumerate(work):
            prompt = prompts[prompt_idx]
            metadata = (
                self._validation_metadata_rows[prompt_idx]
                if prompt_idx < len(self._validation_metadata_rows)
                else {}
            )
            sample_width = int(metadata.get("target_width", width))
            sample_height = int(metadata.get("target_height", height))
            sample_num_frames = int(metadata.get("target_frames", num_frames))
            sample_frame_rate = float(
                metadata.get("fps", self._config.validation.frame_rate)
            )
            sampling_ctx.start_video(local_i)

            # Load conditioning image if provided
            condition_image = None
            if use_images:
                image_path = self._config.validation.images[prompt_idx]
                if Path(image_path).suffix.lower() in self._VIDEO_SUFFIXES:
                    # Validation sample from metadata_jsonl: condition on the video's first frame
                    from ltx_trainer.online_conditioning import read_first_frame

                    image = read_first_frame(image_path)
                else:
                    image = open_image_as_srgb(image_path)
                # Convert PIL image to tensor [C, H, W] in [0, 1]
                condition_image = F.to_tensor(image)

            # Load reference video if provided (for IC-LoRA)
            reference_video = None
            if use_reference_videos:
                ref_video_path = self._config.validation.reference_videos[prompt_idx]
                # read_video returns [F, C, H, W] in [0, 1]
                reference_video, _ = read_video(
                    ref_video_path, max_frames=sample_num_frames
                )

            # Get cached embeddings for this prompt if available
            cached_embeddings = (
                self._cached_validation_embeddings[prompt_idx]
                if self._cached_validation_embeddings is not None
                else None
            )
            cached_qwen = (
                self._cached_validation_qwen_conditions[prompt_idx]
                if self._cached_validation_qwen_conditions is not None
                else None
            )
            if (
                cached_qwen is None
                and uses_online_qwen_conditioning(self._config.training_strategy)
                and getattr(self._config.training_strategy, "use_qwen_cond", False)
                and use_images
            ):
                cached_qwen = self._encode_validation_qwen_online(prompt_idx, prompt)
            omni_bundle = (
                self._validation_omni_conditions[prompt_idx]
                if self._validation_omni_conditions is not None
                else None
            )
            omni_context = (
                omni_bundle.plan_latent.unsqueeze(0)
                if omni_bundle is not None
                else None
            )
            omni_context_mask = (
                omni_bundle.omni_attention_mask[-1].unsqueeze(0)
                if omni_bundle is not None
                else None
            )
            omni_field_mask = (
                omni_bundle.plan_field_mask.unsqueeze(0)
                if omni_bundle is not None
                and omni_bundle.plan_field_mask is not None
                else None
            )
            use_audio_omni = bool(
                omni_bundle is not None
                and generate_audio
                and getattr(
                    self._config.training_strategy, "omni_cond_for_audio", True
                )
            )

            # Create generation config
            gen_config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=sample_height,
                width=sample_width,
                num_frames=sample_num_frames,
                frame_rate=sample_frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                qwen_context=cached_qwen.video_context
                if cached_qwen is not None
                else None,
                qwen_context_mask=cached_qwen.video_context_mask
                if cached_qwen is not None
                else None,
                audio_qwen_context=(
                    cached_qwen.audio_context
                    if cached_qwen is not None
                    and getattr(
                        self._config.training_strategy, "qwen_cond_for_audio", True
                    )
                    else None
                ),
                audio_qwen_context_mask=(
                    cached_qwen.audio_context_mask
                    if cached_qwen is not None
                    and getattr(
                        self._config.training_strategy, "qwen_cond_for_audio", True
                    )
                    else None
                ),
                audio_qwen_fallback_to_video=getattr(
                    self._config.training_strategy, "qwen_cond_for_audio", True
                ),
                omni_context=omni_context,
                omni_context_mask=omni_context_mask,
                omni_field_mask=omni_field_mask,
                audio_omni_context=omni_context if use_audio_omni else None,
                audio_omni_context_mask=(
                    omni_context_mask if use_audio_omni else None
                ),
                audio_omni_field_mask=omni_field_mask if use_audio_omni else None,
                audio_omni_fallback_to_video=use_audio_omni,
                first_frame_audio_cond=getattr(
                    self._config.training_strategy, "use_first_frame_audio_cond", False
                ),
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
            )

            # Generate sample
            video, audio = sampler.generate(
                config=gen_config,
                device=self._accelerator.device,
            )

            if not save_output:
                continue

            # Save output (image for single frame, video otherwise)
            ext = "png" if sample_num_frames == 1 else "mp4"
            label_suffix = f"_{output_label}" if output_label else ""
            output_path = (
                output_dir
                / f"step_{self._global_step:06d}{label_suffix}_{prompt_idx + 1:02d}.{ext}"
            )
            # Save the input prompt alongside the sample so outputs stay interpretable
            # without digging through the config/JSONL.
            prompt_lines = [prompt]
            if use_images:
                prompt_lines.append(
                    f"\n[condition]: {self._config.validation.images[prompt_idx]}"
                )
            if omni_bundle is not None:
                prompt_lines.extend(
                    [
                        f"\n[omni_video_prompt]: {omni_bundle.video_prompt}",
                        f"\n[omni_audio_prompt]: {omni_bundle.audio_prompt}",
                    ]
                )
            output_path.with_suffix(".txt").write_text(
                "\n".join(prompt_lines) + "\n", encoding="utf-8"
            )
            if sample_num_frames == 1:
                save_image(video, output_path)
            else:
                save_video(
                    video_tensor=video,
                    output_path=output_path,
                    fps=sample_frame_rate,
                    audio=audio,
                    audio_sample_rate=self._vocoder.output_sampling_rate
                    if audio is not None
                    else None,
                )
            results.append((prompt_idx, output_path))

        # Clean up progress tasks
        sampling_ctx.cleanup()

        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        profile = f" {output_label}" if output_label else ""
        logger.info(
            f"🎥 Validation{profile} samples for step {self._global_step} "
            f"saved in {rel_outputs_path}"
        )
        return results

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_checkpoint(self) -> Path | None:
        """Save the model weights."""
        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        # Prepare paths
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Get state dict (collective operation - all processes must participate)
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if IS_MAIN_PROCESS:
            save_dir.mkdir(exist_ok=True, parents=True)

            # Determine save precision
            save_dtype = (
                torch.bfloat16
                if self._config.checkpoints.precision == "bfloat16"
                else torch.float32
            )

            # For LoRA: extract only adapter weights; for full: use as-is
            if is_lora:
                unwrapped = self._accelerator.unwrap_model(
                    self._transformer, keep_torch_compile=False
                )
                # For FSDP, pass full_state_dict since model params aren't directly accessible
                state_dict = get_peft_model_state_dict(
                    unwrapped, state_dict=full_state_dict if is_fsdp else None
                )

                # OVI-style injection modules are manually unfrozen instead of wrapped in
                # PEFT modules_to_save. Persist their full tensors in the same checkpoint.
                full_module_names = self._lora_persistent_module_names()
                full_module_state = {
                    key: value
                    for key, value in full_state_dict.items()
                    if any(
                        self._state_key_belongs_to_module(key, name)
                        for name in full_module_names
                    )
                }
                if not full_module_state:
                    raise RuntimeError(
                        "LoRA checkpoint would omit all configured full injection modules"
                    )
                state_dict.update(full_module_state)
                logger.info(
                    f"Saving {len(full_module_state)} full-weight injection tensors with LoRA"
                )

                # Remove "base_model.model." prefix added by PEFT
                state_dict = {
                    k.replace("base_model.model.", "", 1): v
                    for k, v in state_dict.items()
                }

                # Convert to ComfyUI-compatible format (add "diffusion_model." prefix)
                state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}

                # Cast to configured precision
                state_dict = {
                    k: v.to(save_dtype) if isinstance(v, Tensor) else v
                    for k, v in state_dict.items()
                }

                # Build metadata for safetensors file
                metadata = self._build_checkpoint_metadata()

                # Save to disk with metadata
                weights_tmp_path = saved_weights_path.with_suffix(".safetensors.tmp")
                save_file(state_dict, weights_tmp_path, metadata=metadata)
                os.replace(weights_tmp_path, saved_weights_path)

            else:
                # Cast to configured precision
                full_state_dict = {
                    k: v.to(save_dtype) if isinstance(v, Tensor) else v
                    for k, v in full_state_dict.items()
                }

                # Save to disk as real safetensors. Older checkpoints used accelerator.save()
                # with a .safetensors suffix, so the loader still accepts both formats.
                weights_tmp_path = saved_weights_path.with_suffix(".safetensors.tmp")
                save_file(
                    full_state_dict,
                    weights_tmp_path,
                    metadata=self._build_checkpoint_metadata(),
                )
                os.replace(weights_tmp_path, saved_weights_path)

            if (
                self._config.stage2.enabled
                and self._config.stage2.qwen_training_mode == "lora"
                and self._online_qwen_encoder is not None
            ):
                qwen_path = (
                    save_dir / f"qwen_weights_step_{self._global_step:05d}.safetensors"
                )
                qwen_tmp_path = qwen_path.with_suffix(".safetensors.tmp")
                save_qwen_lora(
                    self._online_qwen_encoder.model,
                    str(qwen_tmp_path),
                    metadata={
                        "format": "ltx_stage2_qwen_lora_v1",
                        "rank": str(self._config.stage2.qwen_rank),
                        "alpha": str(self._config.stage2.qwen_alpha),
                        "last_n_layers": str(self._config.stage2.qwen_last_n_layers),
                        "ltx_checkpoint": filename,
                    },
                )
                os.replace(qwen_tmp_path, qwen_path)
                logger.info(
                    f"Saved Qwen Stage-2 LoRA for step {self._global_step} to {qwen_path.name}"
                )

            rel_path = saved_weights_path.relative_to(self._config.output_dir)
            logger.info(
                f"💾 {prefix.capitalize()} weights for step {self._global_step} saved in {rel_path}"
            )

            self._checkpoint_paths.append(saved_weights_path)
            self._cleanup_checkpoints()

        # Full FSDP optimizer checkpointing is collective, so every rank must enter.
        self._accelerator.wait_for_everyone()
        self._save_training_state(save_dir)
        self._accelerator.wait_for_everyone()

        self._last_saved_step = self._global_step
        return saved_weights_path if IS_MAIN_PROCESS else None

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints, including files inherited from a resumed run."""
        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if self._config.model.training_mode == "lora" else "model"

        def checkpoint_step(path: Path) -> int:
            match = re.search(r"step_(\d+)", path.name)
            return int(match.group(1)) if match else -1

        checkpoints = sorted(
            save_dir.glob(f"{prefix}_weights_step_*.safetensors"), key=checkpoint_step
        )
        keep_n = self._config.checkpoints.keep_last_n
        checkpoints_to_remove = checkpoints[:-keep_n] if keep_n > 0 else []
        for old_checkpoint in checkpoints_to_remove:
            old_checkpoint.unlink(missing_ok=True)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
            step_suffix = old_checkpoint.stem.rsplit("_step_", 1)[-1]
            associated_paths = (
                old_checkpoint.parent / f"qwen_weights_step_{step_suffix}.safetensors",
                old_checkpoint.parent / f"training_state_step_{step_suffix}.pt",
            )
            for associated_path in associated_paths:
                if associated_path.exists():
                    associated_path.unlink()
                    logger.info(f"Removed old checkpoint companion: {associated_path}")
            optimizer_state_dir = (
                old_checkpoint.parent / f"optimizer_state_step_{step_suffix}"
            )
            if optimizer_state_dir.is_dir():
                shutil.rmtree(optimizer_state_dir)
                logger.info(f"Removed old checkpoint companion: {optimizer_state_dir}")
        self._checkpoint_paths = checkpoints[-keep_n:] if keep_n > 0 else checkpoints

    def _save_training_state(self, save_dir: Path) -> None:
        """Save training state alongside checkpoint for resume.
        Respects checkpoints.save_training_state config:
        - "full": optimizer + scheduler + RNG + step + wandb_run_id
        - "minimal": scheduler + RNG + step + wandb_run_id
        - "off": skip entirely
        """
        mode = self._config.checkpoints.save_training_state
        if mode == "off":
            return

        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
        save_fsdp_optimizer = mode == "full" and is_fsdp
        local_rng = RngStates(
            torch_state=torch.random.get_rng_state(),
            cuda_state=torch.cuda.get_rng_state(self._accelerator.device)
            if torch.cuda.is_available()
            else None,
        )
        rng_states_by_rank = None
        if self._accelerator.num_processes > 1:
            gathered_rng = gather_object([local_rng.model_dump()])
            rng_states_by_rank = [RngStates(**state) for state in gathered_rng]

        if not IS_MAIN_PROCESS and not save_fsdp_optimizer:
            return

        optimizer_state = None
        optimizer_state_path = None
        optimizer_state_format = None
        if mode == "full":
            if is_fsdp:
                optimizer_dir = self._save_fsdp_optimizer_state(save_dir)
                optimizer_state_path = optimizer_dir.name
                optimizer_state_format = "fsdp_local_optimizer_v1"
            else:
                optimizer_state = self._optimizer.state_dict()

        if not IS_MAIN_PROCESS:
            return

        state = TrainingState(
            global_step=self._global_step,
            config_fingerprint=ConfigFingerprint(
                optimizer_type=self._config.optimization.optimizer_type,
                scheduler_type=self._config.optimization.scheduler_type,
                training_mode=self._config.model.training_mode,
                lora_rank=self._config.lora.rank
                if self._config.lora is not None
                else None,
                stage2_enabled=self._config.stage2.enabled,
                stage2_injection_lora_enabled=(
                    self._config.stage2.injection_lora_enabled
                ),
                stage2_injection_rank=self._config.stage2.injection_rank,
                qwen_training_mode=self._config.stage2.qwen_training_mode,
                qwen_rank=self._config.stage2.qwen_rank,
                gradient_accumulation_steps=(
                    self._config.optimization.gradient_accumulation_steps
                ),
                weight_decay=self._config.optimization.weight_decay,
                batch_size=self._config.optimization.batch_size,
                seed=self._config.seed,
                world_size=self._accelerator.num_processes,
                context_parallel_size=(
                    self._config.acceleration.context_parallel_size
                ),
                training_data_fingerprint=self._training_data_fingerprint,
            ),
            rng_states=local_rng,
            rng_states_by_rank=rng_states_by_rank,
            lr_scheduler_state_dict=self._lr_scheduler.state_dict()
            if self._lr_scheduler is not None
            else None,
            optimizer_state_dict=optimizer_state,
            optimizer_state_path=optimizer_state_path,
            optimizer_state_format=optimizer_state_format,
            wandb_run_id=self._wandb_run.id if self._wandb_run is not None else None,
        )

        state_path = save_dir / f"training_state_step_{self._global_step:05d}.pt"
        tmp_path = state_path.with_suffix(".pt.tmp")
        try:
            torch.save(state.to_save_dict(), tmp_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        tmp_path.rename(state_path)

        file_size_gb = state_path.stat().st_size / (1024**3)
        if file_size_gb > 1.0 and not self._training_state_size_warned:
            self._training_state_size_warned = True
            logger.warning(
                f"⚠️ Training state file is {file_size_gb:.1f} GB (full mode includes optimizer state). "
                f'Set checkpoints.save_training_state="minimal" to save only scheduler/RNG/step (~few KB), '
                f'or "off" to disable entirely.'
            )

        if (
            not self._training_state_paths
            or self._training_state_paths[-1] != state_path
        ):
            self._training_state_paths.append(state_path)
        self._cleanup_training_states()

        rel_path = state_path.relative_to(self._config.output_dir)
        logger.debug(f"Training state saved to {rel_path}")

    def _save_fsdp_optimizer_state(self, save_dir: Path) -> Path:
        final_dir = save_dir / f"optimizer_state_step_{self._global_step:05d}"
        tmp_dir = save_dir / f".optimizer_state_step_{self._global_step:05d}.tmp"

        # A completed directory was atomically renamed and is safe to reuse when the
        # final training save repeats an interval checkpoint at the same step. All
        # ranks must agree on visibility or continuing would deadlock at a later barrier.
        reuse_flags = gather_object([final_dir.is_dir()])
        if any(reuse_flags):
            if not all(reuse_flags):
                raise RuntimeError(
                    f"Inconsistent shared-filesystem view for FSDP optimizer checkpoint: {reuse_flags}"
                )
            logger.info(f"Reusing existing sharded FSDP optimizer state at {final_dir}")
            return final_dir

        if IS_MAIN_PROCESS:
            save_dir.mkdir(exist_ok=True, parents=True)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
        self._accelerator.wait_for_everyone()

        rank_file = tmp_dir / f"rank_{self._accelerator.process_index:05d}.pt"
        rank_tmp = rank_file.with_suffix(".pt.tmp")
        torch.save(self._optimizer.state_dict(), rank_tmp)
        os.replace(rank_tmp, rank_file)
        self._accelerator.wait_for_everyone()

        if IS_MAIN_PROCESS:
            rank_files = sorted(tmp_dir.glob("rank_*.pt"))
            if len(rank_files) != self._accelerator.num_processes:
                raise RuntimeError(
                    f"Incomplete FSDP optimizer checkpoint: {len(rank_files)}/"
                    f"{self._accelerator.num_processes} rank files"
                )
            manifest = {
                "format": "fsdp_local_optimizer_v1",
                "global_step": self._global_step,
                "world_size": self._accelerator.num_processes,
                "optimizer_type": self._config.optimization.optimizer_type,
                **self._fsdp_optimizer_topology(),
            }
            manifest_path = tmp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(tmp_dir, final_dir)
            logger.info(
                f"Saved rank-local FSDP optimizer state to {final_dir} "
                f"({self._accelerator.num_processes} shards)"
            )
        self._accelerator.wait_for_everyone()
        return final_dir

    def _cleanup_training_states(self) -> None:
        """Remove orphaned training states, including those from earlier process runs."""
        save_dir = Path(self._config.output_dir) / "checkpoints"

        def state_step(path: Path) -> int:
            match = re.search(r"step_(\d+)", path.name)
            return int(match.group(1)) if match else -1

        states = sorted(save_dir.glob("training_state_step_*.pt"), key=state_step)
        keep_n = self._config.checkpoints.keep_last_n
        to_remove = states[:-keep_n] if keep_n > 0 else []
        for old_state in to_remove:
            old_state.unlink(missing_ok=True)
            match = re.search(r"step_(\d+)", old_state.name)
            if match:
                optimizer_dir = (
                    old_state.parent / f"optimizer_state_step_{match.group(1)}"
                )
                if optimizer_dir.is_dir():
                    shutil.rmtree(optimizer_dir)
        self._training_state_paths = states[-keep_n:] if keep_n > 0 else states

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if self._config.model.training_mode == "lora":
            modules_to_save = self._lora_full_module_names()
            persistent_modules = self._lora_persistent_module_names()
            metadata.update(
                {
                    "lora_rank": str(self._config.lora.rank),
                    "lora_alpha": str(self._config.lora.alpha),
                    "lora_modules_to_save": json.dumps(
                        list(dict.fromkeys(modules_to_save))
                    ),
                    "lora_persistent_modules": json.dumps(
                        list(dict.fromkeys(persistent_modules))
                    ),
                }
            )

        strategy_config = self._config.training_strategy
        if getattr(strategy_config, "use_omni_cond", False):
            metadata.update(
                {
                    "omni_conditioning": "true",
                    "omni_condition_source": "offline",
                    "omni_layer_adapter_num_layers": str(
                        strategy_config.omni_layer_adapter_num_layers
                    ),
                    "omni_layer_adapter_input_dim": str(
                        strategy_config.omni_layer_adapter_input_dim
                    ),
                    "omni_cross_attention_dim": str(
                        strategy_config.omni_cross_attention_dim
                    ),
                    "audio_omni_cross_attention_dim": str(
                        strategy_config.audio_omni_cross_attention_dim
                        or strategy_config.omni_cross_attention_dim
                    ),
                    "omni_field_resampler": str(
                        strategy_config.use_omni_field_resampler
                    ).lower(),
                    "omni_field_resampler_input_dim": str(
                        strategy_config.omni_field_resampler_input_dim
                    ),
                    "omni_field_resampler_num_heads": str(
                        strategy_config.omni_field_resampler_num_heads
                    ),
                }
            )
        if getattr(strategy_config, "use_listener_timbre_cond", False):
            metadata.update(
                {
                    "listener_timbre_conditioning": "true",
                    "listener_timbre_dim": str(strategy_config.listener_timbre_dim),
                }
            )

        if self._config.stage2.enabled:
            metadata.update(
                {
                    "stage2_enabled": "true",
                    "stage2_injection_lora_enabled": str(
                        self._config.stage2.injection_lora_enabled
                    ).lower(),
                    "stage2_injection_rank": str(self._config.stage2.injection_rank),
                    "stage2_injection_alpha": str(self._config.stage2.injection_alpha),
                    "stage2_injection_targets": json.dumps(
                        self._stage2_injection_targets
                    ),
                    "qwen_training_mode": self._config.stage2.qwen_training_mode,
                    "qwen_lora_rank": str(self._config.stage2.qwen_rank),
                    "qwen_lora_alpha": str(self._config.stage2.qwen_alpha),
                }
            )
        if metadata:
            logger.info(f"Saving checkpoint metadata keys: {sorted(metadata)}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(
            f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}"
        )

    def _init_wandb(self, resume_run_id: str | None = None) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        init_kwargs: dict[str, Any] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "name": Path(self._config.output_dir).name,
            "tags": wandb_config.tags,
            "config": self._config.model_dump(),
        }
        if resume_run_id is not None:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases and TensorBoard."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
        if self._tb_writer is not None:
            step = max(self._global_step, 0)
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(name, value, step)
            self._tb_writer.flush()

    def _log_validation_samples(
        self,
        sample_paths: list[Path],
        prompts: list[str],
        *,
        key: str = "validation_samples",
    ) -> None:
        """Log validation samples (videos or images) to Weights & Biases."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".heic",
            ".webp",
        )

        if is_image:
            samples = [
                wandb.Image(str(path), caption=prompt)
                for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        else:
            samples = [
                wandb.Video(
                    str(path), caption=prompt, format=path.suffix.lower().lstrip(".")
                )
                for path, prompt in zip(sample_paths, prompts, strict=True)
            ]
        self._wandb_run.log({key: samples}, step=self._global_step)
