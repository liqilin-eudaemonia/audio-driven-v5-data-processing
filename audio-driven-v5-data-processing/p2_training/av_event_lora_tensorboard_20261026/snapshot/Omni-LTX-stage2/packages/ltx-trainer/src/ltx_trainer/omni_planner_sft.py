"""Standalone Route-A Qwen3-Omni Planner LoRA supervised fine-tuning."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ltx_trainer.omni_condition_utils import load_omni_condition_file
from ltx_trainer.omni_planner_checkpoint import (
    PLANNER_SFT_FORMAT,
    load_planner_adapter,
    save_planner_adapter,
)
from ltx_trainer.omni_stage2 import OmniJointConfig, OnlineOmniPlanner
from ltx_trainer.stage1_readiness import read_rows, row_id, write_json_atomic


class PlannerSftClusterNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    host: str
    ssh_user: str = "root"
    ssh_port: int = Field(default=22, gt=0, lt=65536)
    role: str


class PlannerSftClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    master_port: int = Field(default=29872, gt=0, lt=65536)
    nproc_per_node: int = Field(default=8, gt=0)
    nodes: list[PlannerSftClusterNode] = Field(default_factory=list)


class PlannerSftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner: OmniJointConfig
    metadata_jsonl: str
    conditions_dir: str
    output_dir: str
    expected_train_samples: int = Field(default=91009, gt=0)
    steps: int = Field(default=1000, gt=0)
    per_device_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=2e-6, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    save_interval: int = Field(default=100, gt=0)
    seed: int = 45
    num_dataloader_workers: int = Field(default=2, ge=0)
    resume_checkpoint: str | None = None
    validation_metadata_jsonls: list[str] = Field(default_factory=list)
    validation_interval: int | None = None
    validation_max_samples: int = Field(default=24, gt=0)
    validation_generate_samples: int = Field(default=0, ge=0)
    cluster: PlannerSftClusterConfig | None = None


class PlannerSftDataset(Dataset[str]):
    def __init__(self, metadata_jsonl: Path, conditions_dir: Path, expected_count: int) -> None:
        rows = read_rows(metadata_jsonl)
        self.sample_ids = [row_id(row) for row in rows]
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Planner SFT metadata contains duplicate sample IDs")
        if len(self.sample_ids) != expected_count:
            raise ValueError(
                f"Expected {expected_count} Planner SFT samples, found {len(self.sample_ids)}"
            )
        self.conditions_dir = conditions_dir

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> str:
        return self.sample_ids[index]

    def condition(self, sample_id: str) -> Any:
        return load_omni_condition_file(
            self.conditions_dir / f"{sample_id}.pt",
            expected_version="teacher_forcing_raw",
            expected_sample_id=sample_id,
            expected_route="route_a_no_context",
        )


def validate_sft_data(
    config: PlannerSftConfig, *, check_condition_files: bool = True
) -> PlannerSftDataset:
    if config.planner.planner_input_mode != "audio_timing":
        raise ValueError(
            "Standalone Planner SFT requires planner_input_mode=audio_timing"
        )
    required = [
        Path(config.planner.model_path),
        Path(config.metadata_jsonl),
        Path(config.conditions_dir),
        *(Path(path) for path in config.planner.manifests),
        *(Path(path) for path in config.validation_metadata_jsonls),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Planner SFT inputs: " + ", ".join(missing))
    dataset = PlannerSftDataset(
        Path(config.metadata_jsonl),
        Path(config.conditions_dir),
        config.expected_train_samples,
    )
    manifest_ids = {
        row_id(row)
        for path in config.planner.manifests
        for row in read_rows(Path(path))
    }
    absent = set(dataset.sample_ids) - manifest_ids
    if absent:
        raise ValueError(f"{len(absent)} SFT sample IDs are absent from causal manifests")
    validation_ids = {
        row_id(row)
        for path in config.validation_metadata_jsonls
        for row in read_rows(Path(path))
    }
    overlap = set(dataset.sample_ids) & validation_ids
    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} overlapping sample IDs")
    # Full tensor reads belong to the existing dataset audit. Check file presence here.
    if check_condition_files:
        missing_conditions = [
            sid
            for sid in dataset.sample_ids
            if not (Path(config.conditions_dir) / f"{sid}.pt").is_file()
        ]
        if missing_conditions:
            raise FileNotFoundError(
                f"Missing {len(missing_conditions)} qualified conditions; "
                f"first={missing_conditions[0]}"
            )
    return dataset


def _distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world, local_rank


def _schedule(config: PlannerSftConfig):
    def factor(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-8, step / config.warmup_steps)
        span = max(1, config.steps - config.warmup_steps)
        return max(0.1, 1.0 - 0.9 * (step - config.warmup_steps) / span)

    return factor


def rescale_resume_batch_offset(
    batch_in_epoch: int,
    saved_world_size: int,
    current_world_size: int,
    saved_per_device_batch_size: int = 1,
    current_per_device_batch_size: int = 1,
) -> int:
    """Preserve the approximate number of globally consumed samples."""
    if min(
        saved_world_size,
        current_world_size,
        saved_per_device_batch_size,
        current_per_device_batch_size,
    ) <= 0:
        raise ValueError("Distributed world and batch sizes must be positive")
    consumed = batch_in_epoch * saved_world_size * saved_per_device_batch_size
    current_global_batch = current_world_size * current_per_device_batch_size
    return consumed // current_global_batch


class PlannerSftTrainer:
    def __init__(self, config: PlannerSftConfig) -> None:
        self.config = config
        self.rank, self.world, self.local_rank = _distributed()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        random.seed(config.seed + self.rank)
        torch.manual_seed(config.seed)
        validation_error: list[str | None] = [None]
        self.dataset: PlannerSftDataset | None = None
        if self.rank == 0:
            try:
                self.dataset = validate_sft_data(config)
            except Exception as exc:
                validation_error[0] = f"{type(exc).__name__}: {exc}"
        if self.world > 1:
            dist.broadcast_object_list(validation_error, src=0)
        if validation_error[0]:
            raise RuntimeError("Planner SFT data preflight failed: " + validation_error[0])
        if self.dataset is None:
            self.dataset = validate_sft_data(config, check_condition_files=False)
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.planner = OnlineOmniPlanner(
            config.planner,
            self.device,
            self.output / "causal_audio" / f"rank_{self.rank}",
        )
        if self.world > 1:
            self.planner.model = DistributedDataParallel(
                self.planner.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        self.optimizer = AdamW(
            [parameter for parameter in self.planner.model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, _schedule(config))
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        if config.resume_checkpoint:
            self._resume(Path(config.resume_checkpoint))

    @property
    def raw_model(self) -> Any:
        return (
            self.planner.model.module
            if isinstance(self.planner.model, DistributedDataParallel)
            else self.planner.model
        )

    def _resume(self, checkpoint: Path) -> None:
        metadata = load_planner_adapter(
            self.raw_model,
            checkpoint,
            self.config.planner,
            accepted_formats=(PLANNER_SFT_FORMAT,),
        )
        saved_world = int(metadata.get("world_size", "1"))
        saved_batch_size = int(metadata.get("per_device_batch_size", "1"))
        same_data_parallel_shape = (
            saved_world == self.world
            and saved_batch_size == self.config.per_device_batch_size
        )
        state_rank = self.rank if same_data_parallel_shape else 0
        state_path = checkpoint.with_name(
            checkpoint.stem + f"_training_state_rank_{state_rank:05d}.pt"
        )
        if not state_path.is_file():
            raise FileNotFoundError(f"Planner SFT resume state is missing: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])
        self.epoch = int(state["epoch"])
        saved_batch = int(state["batch_in_epoch"])
        if same_data_parallel_shape:
            self.batch_in_epoch = saved_batch
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        else:
            self.batch_in_epoch = rescale_resume_batch_offset(
                saved_batch,
                saved_world,
                self.world,
                saved_batch_size,
                self.config.per_device_batch_size,
            )
            resume_seed = self.config.seed + self.step * self.world + self.rank
            torch.manual_seed(resume_seed)
            torch.cuda.manual_seed(resume_seed)
            if self.rank == 0:
                print(
                    json.dumps(
                        {
                            "resume_world_size_change": {
                                "saved_world_size": saved_world,
                                "current_world_size": self.world,
                                "saved_per_device_batch_size": saved_batch_size,
                                "current_per_device_batch_size": self.config.per_device_batch_size,
                                "saved_batch_in_epoch": saved_batch,
                                "current_batch_in_epoch": self.batch_in_epoch,
                            }
                        }
                    ),
                    flush=True,
                )

    def _save(self) -> None:
        checkpoint = self.output / "checkpoints" / f"omni_planner_sft_step_{self.step:05d}.safetensors"
        if self.rank == 0:
            save_planner_adapter(
                self.raw_model,
                checkpoint,
                self.config.planner,
                checkpoint_format=PLANNER_SFT_FORMAT,
                extra_metadata={
                    "step": str(self.step),
                    "training_samples": str(len(self.dataset)),
                    "world_size": str(self.world),
                    "per_device_batch_size": str(
                        self.config.per_device_batch_size
                    ),
                },
            )
        torch.save(
            {
                "step": self.step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device),
            },
            checkpoint.with_name(
                checkpoint.stem + f"_training_state_rank_{self.rank:05d}.pt"
            ),
        )

    def _evaluate(self) -> None:
        if not self.config.validation_metadata_jsonls:
            return
        condition_root = self.config.planner.validation_conditions_dir
        if not condition_root:
            raise ValueError("Planner validation requires planner.validation_conditions_dir")
        sample_ids = [
            row_id(row)
            for path in self.config.validation_metadata_jsonls
            for row in read_rows(Path(path))
        ][: self.config.validation_max_samples]
        indexed_ids = list(enumerate(sample_ids))
        local_ids = indexed_ids[self.rank :: self.world]
        wrapped_model = self.planner.model
        raw_model = self.raw_model
        was_training = wrapped_model.training
        # Validation has no backward pass. Bypass DDP so a 24-item protocol can
        # be sharded over 32 ranks without requiring equal forward counts.
        self.planner.model = raw_model
        raw_model.eval()
        total_loss = 0.0
        total_tokens = 0
        valid_plans = 0
        try:
            with torch.no_grad():
                for index, sample_id in local_ids:
                    bundle = load_omni_condition_file(
                        Path(condition_root) / f"{sample_id}.pt",
                        expected_version="teacher_forcing_raw",
                        expected_sample_id=sample_id,
                        expected_route="route_a_no_context",
                    )
                    loss, tokens = self.planner.supervised_loss(
                        sample_id, bundle.raw_plan_json
                    )
                    total_loss += float(loss) * tokens
                    total_tokens += tokens
                    if index < self.config.validation_generate_samples:
                        inputs = self.planner.prepare_prompt(sample_id)
                        generated = raw_model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            do_sample=False,
                            use_cache=True,
                        )
                        text = self.planner.processor.tokenizer.decode(
                            generated[0, inputs["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        ).strip()
                        try:
                            parsed = json.loads(text)
                            self.planner.canonical_plan(sample_id, parsed)
                            valid_plans += 1
                        except (TypeError, ValueError, json.JSONDecodeError):
                            pass
        finally:
            self.planner.model = wrapped_model
            if was_training:
                wrapped_model.train()
                raw_model.get_base_model().visual.eval()
                raw_model.get_base_model().audio_tower.eval()
        totals = torch.tensor(
            [total_loss, total_tokens, valid_plans],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_tokens, valid_plans = totals.tolist()
        if self.rank == 0:
            write_json_atomic(
                self.output / "validation" / f"step_{self.step:05d}.json",
                {
                    "step": self.step,
                    "samples": len(sample_ids),
                    "plan_ce": total_loss / max(1, total_tokens),
                    "generated_samples": min(
                        len(sample_ids), self.config.validation_generate_samples
                    ),
                    "valid_generated_plans": int(valid_plans),
                },
            )

    def train(self) -> None:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world,
            rank=self.rank,
            shuffle=True,
            seed=self.config.seed,
            drop_last=False,
        )
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.per_device_batch_size,
            sampler=sampler,
            num_workers=self.config.num_dataloader_workers,
            collate_fn=list,
        )
        self.optimizer.zero_grad(set_to_none=True)
        while self.step < self.config.steps:
            sampler.set_epoch(self.epoch)
            for batch_index, sample_ids in enumerate(loader):
                if batch_index < self.batch_in_epoch:
                    continue
                bundles = [
                    self.dataset.condition(sample_id) for sample_id in sample_ids
                ]
                loss, token_count = self.planner.supervised_batch_loss(
                    sample_ids,
                    [bundle.raw_plan_json for bundle in bundles],
                )
                (loss / self.config.gradient_accumulation_steps).backward()
                self.batch_in_epoch = batch_index + 1
                if self.batch_in_epoch % self.config.gradient_accumulation_steps:
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.planner.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if self.rank == 0:
                    print(
                        json.dumps(
                            {
                                "step": self.step,
                                "epoch": self.epoch,
                                "plan_ce": float(loss.detach()),
                                "plan_tokens": token_count,
                                "lr": self.scheduler.get_last_lr()[0],
                            }
                        ),
                        flush=True,
                    )
                if self.step % self.config.save_interval == 0 or self.step == self.config.steps:
                    if self.world > 1:
                        dist.barrier()
                    self._save()
                    if self.world > 1:
                        dist.barrier()
                if (
                    self.config.validation_interval
                    and self.step % self.config.validation_interval == 0
                ):
                    self._evaluate()
                if self.step >= self.config.steps:
                    break
            self.epoch += 1
            self.batch_in_epoch = 0
        if self.rank == 0:
            write_json_atomic(
                self.output / "completed.json",
                {"status": "completed", "step": self.step, "samples": len(self.dataset)},
            )
        if self.world > 1:
            dist.barrier()
            dist.destroy_process_group()
