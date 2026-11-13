"""Stage-2-only trainer. The Stage-1 trainer and entry point remain unchanged."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import DataLoader

from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.datasets import collate_online_condition_batch
from ltx_trainer.omni_planner_checkpoint import (
    PLANNER_JOINT_FORMAT,
    PLANNER_SFT_FORMAT,
    file_sha256,
    load_planner_adapter,
    save_planner_adapter,
)
from ltx_trainer.omni_stage2 import OmniJointConfig, OnlineOmniPlanner
from ltx_trainer.omni_stage2_async_evaluation import publish_checkpoint
from ltx_trainer.omni_stage2_distributed import joint_clip_coefficient, resumed_dp_sampler
from ltx_trainer.trainer import LtxvTrainer, TrainingStepOutput

OMNI_STAGE2_ARCHITECTURE = "route_a_joint_v2_moe_resampler_lora"


def companion_path(checkpoint: Path) -> Path:
    match = re.search(r"step_(\d+)", checkpoint.name)
    if not match:
        raise ValueError(f"Checkpoint has no step: {checkpoint}")
    return checkpoint.with_name(f"omni_weights_step_{match[1]}.safetensors")


def checkpoint_omni_config_matches(serialized: str, requested: OmniJointConfig) -> bool:
    """Compare trainable Planner topology while ignoring evaluation-only paths."""
    checkpoint_config = OmniJointConfig.model_validate_json(serialized)
    runtime_fields = {"validation_conditions_dir", "planner_init_checkpoint"}
    return checkpoint_config.model_dump(exclude=runtime_fields) == requested.model_dump(
        exclude=runtime_fields
    )


def validate_joint_configuration(trainer_config: LtxTrainerConfig, omni_config: OmniJointConfig) -> None:
    if trainer_config.stage2.enabled:
        raise ValueError("Legacy Qwen-VL stage2 must be disabled for Omni joint training")
    strategy = trainer_config.training_strategy
    if not (
        strategy.use_omni_cond
        and strategy.use_omni_field_resampler
        and strategy.omni_expected_route == "route_a_no_context"
    ):
        raise ValueError("Joint Stage 2 requires Route-A raw Omni + trainable Resampler")
    if tuple(omni_config.layers) != (12, 24, 36, 48) or omni_config.hidden_size != 2048:
        raise ValueError("First implementation preserves the Stage-1 four-layer raw contract")
    if omni_config.planner_input_mode != "audio_frame_timing":
        raise ValueError(
            "Joint Stage 2 preserves Speaker audio + Listener frame + timing input"
        )
    if (
        omni_config.attention_rank,
        omni_config.attention_alpha,
        omni_config.expert_rank,
        omni_config.expert_alpha,
        omni_config.lora_dropout,
    ) != (32, 64, 2, 4, 0.0):
        raise ValueError("Omni joint v2 requires attention 32/64, experts 2/4, and dropout 0")
    if (
        omni_config.resampler_rank,
        omni_config.resampler_alpha,
        omni_config.resampler_learning_rate,
    ) != (32, 32, 2e-5):
        raise ValueError("Omni joint v2 requires Resampler rank/alpha 32/32 and LR 2e-5")
    if omni_config.learning_rate != 2e-6 or omni_config.token_loss_weight != 1.0:
        raise ValueError("Omni joint v2 requires Planner LR 2e-6 and Plan CE weight 1")
    if trainer_config.optimization.learning_rate != 1e-5:
        raise ValueError("Omni joint v2 requires LTX joint LR 1e-5")
    if trainer_config.acceleration.context_parallel_size not in {1, 4}:
        raise ValueError("Joint Omni v2 requires CP=1 or the Stage-1-compatible CP=4 topology")
    if trainer_config.acceleration.mixed_precision_mode != "bf16":
        raise ValueError("Omni joint v2 requires bf16")
    if not trainer_config.model.load_checkpoint or trainer_config.model.load_checkpoint == "auto":
        raise ValueError("Select an explicit Stage-1 initialization or Stage-2 resume checkpoint")
    if trainer_config.checkpoints.keep_last_n != -1:
        raise ValueError("Omni joint v2 keeps paired checkpoints: set keep_last_n=-1")
    if trainer_config.checkpoints.save_training_state != "full":
        raise ValueError("Joint training requires full optimizer/RNG state")
    if trainer_config.checkpoints.precision != "float32":
        logger.warning("BF16 checkpoint quantizes FP32 trainable weights; use float32 for optimizer resume")
    if strategy.omni_dropout != 0.1:
        raise ValueError("Omni joint v2 transfers the reference condition dropout: set omni_dropout=0.1")
    if not trainer_config.acceleration.ddp_find_unused_parameters:
        raise ValueError("omni_dropout>0 requires ddp_find_unused_parameters=true")
    output = Path(trainer_config.output_dir).resolve()
    stage2_root = Path(__file__).resolve().parents[4]
    if not output.is_relative_to(stage2_root / "outputs"):
        raise ValueError("Stage-2 outputs must live under this worktree's outputs directory")
    initial = Path(trainer_config.model.load_checkpoint).resolve()
    if initial.is_relative_to(output) and trainer_config.checkpoints.no_resume:
        raise ValueError("A fresh Stage 2 output must not contain its Stage-1 initialization")


class OmniStage2Trainer(LtxvTrainer):
    def __init__(self, trainer_config: LtxTrainerConfig, omni_config: OmniJointConfig) -> None:
        self.omni_config = omni_config
        self.planner: OnlineOmniPlanner | None = None
        validate_joint_configuration(trainer_config, omni_config)
        super().__init__(trainer_config)

    def _load_validation_omni_conditions(self) -> None:
        strategy = self._config.training_strategy
        original_root = strategy.omni_conditions_dir
        try:
            if self.omni_config.validation_conditions_dir:
                strategy.omni_conditions_dir = self.omni_config.validation_conditions_dir
            super()._load_validation_omni_conditions()
        finally:
            strategy.omni_conditions_dir = original_root

    def _setup_accelerator(self) -> None:
        super()._setup_accelerator()
        if self._accelerator.distributed_type.name not in {"NO", "MULTI_GPU", "FSDP"}:
            raise ValueError("Omni joint supports single GPU, DDP, or FSDP1")
        if self._accelerator.distributed_type.name == "FSDP":
            plugin = self._accelerator.state.fsdp_plugin
            if getattr(plugin, "fsdp_version", 1) != 1 or not plugin.use_orig_params:
                raise ValueError("Joint FSDP requires FSDP1 with use_orig_params=true")

    def _load_checkpoint(self) -> None:
        checkpoint = self._find_checkpoint(self._config.model.load_checkpoint)
        if checkpoint is None:
            raise FileNotFoundError(self._config.model.load_checkpoint)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        joint = metadata.get("omni_joint_stage2") == "true"
        architecture = metadata.get("omni_stage2_architecture")
        if joint and architecture != OMNI_STAGE2_ARCHITECTURE:
            raise ValueError(
                "Stage-2 v2 refuses v1 checkpoints; initialize from the published Stage-1 checkpoint"
            )
        # Keep trainable destinations in FP32 before copy_: loading an FP32
        # checkpoint into BF16 modules would otherwise quantize it before FSDP.
        with torch.no_grad():
            for parameter in self._trainable_params:
                parameter.data = parameter.data.float()
            self._omni_resampler().float()
        if joint:
            self._enable_resampler_stage2_lora()
        # Load LTX weights first; allocate Omni only after FSDP shards LTX.
        super()._load_checkpoint()
        if not joint:
            self._enable_resampler_stage2_lora()

    def _omni_resampler(self) -> torch.nn.Module:
        base = self._transformer.get_base_model()
        resampler = getattr(base, "omni_field_resampler", None)
        if resampler is None:
            raise ValueError("Joint Stage 2 requires an initialized Omni FieldAware Resampler")
        return resampler

    def _enable_resampler_stage2_lora(self) -> None:
        resampler = self._omni_resampler()
        if getattr(resampler, "stage2_lora_enabled", False):
            return
        # Unified seed for FSDP replica consistency
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._config.seed)
            resampler.enable_stage2_lora(
                rank=self.omni_config.resampler_rank,
                alpha=self.omni_config.resampler_alpha,
            )
        logger.info(f"Resampler Stage 2 LoRA initialized with seed={self._config.seed}")

        self._resampler_trainable_params = [
            parameter for parameter in resampler.parameters() if parameter.requires_grad
        ]
        if not self._resampler_trainable_params:
            raise RuntimeError("Omni Resampler Stage-2 adapter has no trainable parameters")
        logger.info(
            "Omni joint Stage 2 v2: "
            f"{sum(p.numel() for p in self._resampler_trainable_params):,} "
            "Resampler LoRA/query/norm parameters"
        )

    def _lora_full_module_names(self) -> list[str]:
        # Stage 2 v2 trains only Resampler LoRA/query/norm, never its dense base.
        return [
            name
            for name in super()._lora_full_module_names()
            if name != "omni_field_resampler"
        ]

    def _lora_persistent_module_names(self) -> list[str]:
        # The learned Stage-1 dense Resampler base is still required for inference.
        return list(dict.fromkeys([*self._lora_full_module_names(), "omni_field_resampler"]))

    def _prepare_models_for_training(self) -> None:
        super()._prepare_models_for_training()
        self._initialize_planner()
        self._append_planner_parameters()

    def _initialize_planner(self) -> None:
        checkpoint = self._loaded_checkpoint_path
        if checkpoint is None:
            raise ValueError("Stage-2 initialization checkpoint was not loaded")
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        joint = metadata.get("omni_joint_stage2") == "true"
        if not joint and not self._config.checkpoints.no_resume:
            raise ValueError("Stage-1 initialization requires no_resume=true; do not restore its optimizer")
        # Identical LoRA initialization on all ranks without changing caller RNG state.
        device = self._accelerator.device
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(self._config.seed)
            self.planner = OnlineOmniPlanner(
                self.omni_config,
                device,
                Path(self._config.output_dir) / "causal_audio" / f"rank_{self._accelerator.process_index}",
            )
        if joint:
            mode = "fsdp" if self._accelerator.distributed_type.name == "FSDP" else "replicated"
            if not self._config.checkpoints.no_resume and metadata.get("omni_parallel_mode", "replicated") != mode:
                raise RuntimeError("Stage-2 optimizer resume cannot change DDP/FSDP mode")
            if not self._config.checkpoints.no_resume and self._resume_state[1] is None:
                raise RuntimeError("Stage-2 resume state is missing or incompatible; refusing a silent restart")
            paired = companion_path(checkpoint)
            with safe_open(str(paired), framework="pt", device="cpu") as handle:
                paired_metadata = handle.metadata() or {}
            if paired_metadata.get("ltx_checkpoint") != checkpoint.name:
                raise ValueError("Omni companion belongs to a different main checkpoint")
            if paired_metadata.get("format") != "omni_joint_stage2_lora_v2":
                raise ValueError("Omni companion is not a Stage-2 v2 Planner adapter")
            if not checkpoint_omni_config_matches(
                paired_metadata.get("omni_config", ""),
                self.omni_config,
            ):
                raise ValueError("Omni companion configuration differs from requested configuration")
            state = load_file(str(paired), device="cpu")
            expected = get_peft_model_state_dict(self.planner.model)
            if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in expected):
                raise ValueError("Omni companion tensor keys/shapes do not match LoRA architecture")
            set_peft_model_state_dict(self.planner.model, state)
        elif self.omni_config.planner_init_checkpoint:
            load_planner_adapter(
                self.planner.model,
                self.omni_config.planner_init_checkpoint,
                self.omni_config,
                accepted_formats=(PLANNER_SFT_FORMAT,),
                accepted_input_modes=("audio_timing",),
            )
            logger.info(
                "Initialized Stage-2 Planner from standalone SFT adapter "
                f"{self.omni_config.planner_init_checkpoint}"
            )

    def _restore_training_state(self, training_state: Any) -> bool:
        restored = super()._restore_training_state(training_state)
        if not restored:
            raise RuntimeError("Stage-2 optimizer restore failed; refusing to restart with a mismatched data position")
        return True

    def _refresh_trainable_params(self) -> None:
        super()._refresh_trainable_params()
        self._resampler_trainable_params = [
            parameter for parameter in self._omni_resampler().parameters() if parameter.requires_grad
        ]
        if self.planner is not None:
            self._append_planner_parameters()

    def _append_planner_parameters(self) -> None:
        assert self.planner is not None
        self._qwen_trainable_params = []
        for name, parameter in self.planner.model.named_parameters():
            if parameter.requires_grad:
                self._qwen_trainable_params.append(parameter)
                self._optimizer_param_names[id(parameter)] = f"omni.{name}"
                self._optimizer_param_uses_decay[id(parameter)] = self._parameter_uses_weight_decay(name, parameter)
        self._trainable_params.extend(self._qwen_trainable_params)
        logger.info(
            f"Omni joint Stage 2: {sum(p.numel() for p in self._qwen_trainable_params):,} Planner LoRA parameters; "
            f"Plan CE weight={self.omni_config.token_loss_weight}; AV gradient path enabled"
        )

    def _add_weight_decay_groups(
        self, groups: list[dict[str, Any]], *, params: list[Tensor], lr: float, name: str
    ) -> None:
        if name != "stage1_trainable":
            return super()._add_weight_decay_groups(groups, params=params, lr=lr, name=name)
        planner_ids = {id(p) for p in self._qwen_trainable_params}
        resampler_ids = {
            id(p) for p in getattr(self, "_resampler_trainable_params", [])
        }
        super()._add_weight_decay_groups(
            groups,
            params=[
                p
                for p in params
                if id(p) not in planner_ids and id(p) not in resampler_ids
            ],
            lr=lr,
            name="ltx_joint",
        )
        super()._add_weight_decay_groups(
            groups,
            params=[p for p in params if id(p) in resampler_ids],
            lr=self.omni_config.resampler_learning_rate,
            name="omni_resampler_stage2",
        )
        super()._add_weight_decay_groups(
            groups,
            params=[p for p in params if id(p) in planner_ids],
            lr=self.omni_config.learning_rate,
            name="omni_plan_lora",
        )

    def _init_dataloader(self) -> None:
        super()._init_dataloader()
        cfg = self._config
        completed_steps, _ = self._resume_state
        sampler = resumed_dp_sampler(
            self._dataset,
            world_size=self._accelerator.num_processes,
            rank=self._accelerator.process_index,
            batch_size=cfg.optimization.batch_size,
            accumulation=cfg.optimization.gradient_accumulation_steps,
            completed_steps=completed_steps,
            seed=cfg.seed,
            sp_size=cfg.acceleration.context_parallel_size,
        )
        self._cp_sampler = sampler  # existing loop advances epoch and clears resume offset
        self._cp_epoch = sampler.epoch
        self._length_bucket_place_on_device = True
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=cfg.optimization.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=cfg.data.num_dataloader_workers,
            pin_memory=cfg.data.num_dataloader_workers > 0,
            persistent_workers=cfg.data.num_dataloader_workers > 0,
            collate_fn=collate_online_condition_batch,
            generator=torch.Generator().manual_seed(cfg.seed + self._accelerator.process_index),
        )
        logger.info(
            f"Joint DP/CP sampler: rank={self._accelerator.process_index}, "
            f"epoch={sampler.epoch}, sample_offset={sampler.start_index}"
        )

    def _clip_training_gradients(self, max_norm: float) -> Tensor:
        if self._accelerator.distributed_type.name != "FSDP":
            return super()._clip_training_gradients(max_norm)
        # The FSDP collective obtains the full LTX norm (including HYBRID_SHARD
        # replicas). Infinity measures without clipping; add the already averaged
        # Omni norm once, then apply one common coefficient to both parameter sets.
        ltx_norm = self._transformer.clip_grad_norm_(math.inf, norm_type=2.0)
        try:
            total, coefficient = joint_clip_coefficient(
                ltx_norm, self._qwen_trainable_params, max_norm
            )
        except FloatingPointError:
            # One pathological sample must not tear down the entire multi-node
            # job. None gradients make AdamW skip both the update and decoupled
            # weight decay; the generic loop also suppresses its scheduler step.
            for parameter in self._trainable_params:
                parameter.grad = None
            self._skip_optimizer_step = True
            logger.warning(
                f"[rank {self._accelerator.process_index}] Skipping optimizer step "
                "because the joint LTX/Omni gradient norm is non-finite"
            )
            return ltx_norm
        for parameter in self._trainable_params:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient.to(parameter.grad.device, parameter.grad.dtype))
        return total

    def _training_step(self, batch: dict[str, Any]) -> TrainingStepOutput:
        assert self.planner is not None
        batch["omni_conditions"], token_loss = self.planner.encode_conditions(batch["omni_conditions"])
        # This is a startup gate, not a per-step monitor. Repeating the full
        # gradient norm and 32-rank filesystem audit needlessly stalls training.
        audit_gradients = (
            os.environ.get("OMNI_STAGE2_AUDIT_GRADS") == "1"
            and self._global_step == 0
        )
        if audit_gradients:
            self._omni_gradient_probe = batch["omni_conditions"]["plan_latent"]
            self._omni_gradient_probe.retain_grad()
        strategy = self._config.training_strategy
        configured_dropout = strategy.omni_dropout
        try:
            # The gradient gate needs an AV-conditioned sample on every rank.
            # Normal training keeps the configured 0.1 condition dropout.
            if audit_gradients:
                strategy.omni_dropout = 0.0
            result = super()._training_step(batch)
        finally:
            strategy.omni_dropout = configured_dropout
        total = result.loss + self.omni_config.token_loss_weight * token_loss
        if audit_gradients:
            if not torch.isfinite(total).all():
                raise FloatingPointError("Non-finite joint loss in GPU smoke")
            path = Path(self._config.output_dir) / f"joint_loss_rank_{self._accelerator.process_index}.jsonl"
            with path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "completed_steps": self._global_step,
                            "total_loss": total.detach().float().mean().item(),
                            "av_loss": result.loss.detach().float().mean().item(),
                            "plan_token_loss": token_loss.detach().float().item(),
                            "plan_tokens": batch["omni_conditions"]["plan_latent"].shape[2],
                        }
                    )
                    + "\n"
                )
        return TrainingStepOutput(
            loss=total,
            sigma=result.sigma,
            loss_components={
                **result.loss_components,
                "plan_token_loss": token_loss.detach(),
                "av_loss": result.loss.detach(),
            },
        )

    def _sync_replicated_qwen_grads(self) -> None:
        # One flat buffer for this ~130M-parameter adapter costs about 500 MiB at
        # backward peak and can OOM. Conversely, one collective per LoRA tensor
        # leaves the GPUs mostly waiting on thousands of small all-reduces.
        # Bounded buckets retain the low peak while amortizing collective latency.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            world_size = torch.distributed.get_world_size()
            max_bucket_bytes = 32 * 1024 * 1024
            bucket: list[Tensor] = []
            bucket_parameters: list[Tensor] = []
            bucket_bytes = 0

            def reduce_bucket() -> None:
                nonlocal bucket, bucket_parameters, bucket_bytes
                if not bucket:
                    return
                flat = torch._utils._flatten_dense_tensors(bucket)
                torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
                flat.div_(world_size)
                synced = torch._utils._unflatten_dense_tensors(flat, bucket)
                for parameter, gradient in zip(bucket_parameters, synced, strict=True):
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.copy_(gradient)
                bucket = []
                bucket_parameters = []
                bucket_bytes = 0

            for parameter in self._qwen_trainable_params:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                gradient_bytes = gradient.numel() * gradient.element_size()
                if bucket and (
                    bucket_bytes + gradient_bytes > max_bucket_bytes
                    or gradient.dtype != bucket[0].dtype
                    or gradient.device != bucket[0].device
                ):
                    reduce_bucket()
                bucket.append(gradient)
                bucket_parameters.append(parameter)
                bucket_bytes += gradient_bytes
            reduce_bucket()
        probe = getattr(self, "_omni_gradient_probe", None)
        if probe is None:
            return
        if probe.grad is None:
            raise RuntimeError("AV loss did not reach online Plan features")
        av_norm = probe.grad.detach().float().norm()
        planner_norm = torch.stack(
            [p.grad.detach().float().norm() for p in self._qwen_trainable_params if p.grad is not None]
        ).norm()
        if not torch.isfinite(av_norm) or not torch.isfinite(planner_norm) or av_norm <= 0 or planner_norm <= 0:
            raise RuntimeError("Non-finite/zero joint Omni gradient in GPU smoke")
        path = Path(self._config.output_dir) / f"joint_gradient_rank_{self._accelerator.process_index}.jsonl"
        with path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": self._global_step,
                        "plan_feature_av_grad_norm": av_norm.item(),
                        "planner_lora_grad_norm": planner_norm.item(),
                    }
                )
                + "\n"
            )
        self._omni_gradient_probe = None

    # The inherited optimizer-boundary hook averages _qwen_trainable_params over DP ranks.
    # These are now Omni LoRA parameters; no second optimizer or duplicate synchronization.

    def _sample_videos(self, progress: Any, *, output_label: str | None = None) -> list[tuple[int, Path]]:
        assert self.planner is not None
        original = self._validation_omni_conditions
        was_training = self.planner.model.training
        self.planner.model.eval()
        try:
            refreshed = []
            with torch.no_grad():
                for bundle in original or []:
                    encoding = self.planner.encode(bundle.sample_id, bundle.raw_plan_json, compute_loss=False)
                    refreshed.append(
                        replace(
                            bundle,
                            plan_latent=encoding.hidden.cpu(),
                            omni_attention_mask=encoding.token_mask[None]
                            .expand(len(self.omni_config.layers), -1)
                            .cpu(),
                            plan_field_mask=encoding.field_mask.cpu(),
                        )
                    )
            self._validation_omni_conditions = refreshed or None
            return super()._sample_videos(progress, output_label=output_label)
        finally:
            self._validation_omni_conditions = original
            if was_training:
                self.planner.train()

    def _compute_training_data_fingerprint(self) -> str:
        digest = hashlib.sha256(super()._compute_training_data_fingerprint().encode())
        digest.update(OMNI_STAGE2_ARCHITECTURE.encode())
        digest.update(
            self.omni_config.model_dump_json(
                exclude={"planner_init_checkpoint"}
            ).encode()
        )
        if self.omni_config.planner_init_checkpoint:
            digest.update(file_sha256(self.omni_config.planner_init_checkpoint).encode())
        for path in self.omni_config.manifests:
            digest.update(Path(path).read_bytes())
        return digest.hexdigest()

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        metadata = super()._build_checkpoint_metadata()
        metadata.update(
            {
                "omni_joint_stage2": "true",
                "omni_stage2_architecture": OMNI_STAGE2_ARCHITECTURE,
                "omni_parallel_mode": "fsdp" if self._accelerator.distributed_type.name == "FSDP" else "replicated",
                "omni_condition_source": "online_gt_teacher_forcing",
                "omni_config": self.omni_config.model_dump_json(),
            }
        )
        if self.omni_config.planner_init_checkpoint:
            initialization = Path(self.omni_config.planner_init_checkpoint)
            with safe_open(str(initialization), framework="pt", device="cpu") as handle:
                initialization_metadata = handle.metadata() or {}
            metadata.update(
                {
                    "planner_init_checkpoint": str(initialization.resolve()),
                    "planner_init_sha256": file_sha256(initialization),
                    "planner_init_step": initialization_metadata.get("step", "unknown"),
                }
            )
        return metadata

    def _save_checkpoint(self) -> Path | None:
        assert self.planner is not None
        prefix = "lora" if self._config.model.training_mode == "lora" else "model"
        main = (
            Path(self._config.output_dir) / "checkpoints" / f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        )
        if self._accelerator.is_main_process:
            main.parent.mkdir(parents=True, exist_ok=True)
            target = companion_path(main)
            save_planner_adapter(
                self.planner.model,
                target,
                self.omni_config,
                checkpoint_format=PLANNER_JOINT_FORMAT,
                extra_metadata={
                    "ltx_checkpoint": main.name,
                    "omni_config": self.omni_config.model_dump_json(),
                },
            )
        self._accelerator.wait_for_everyone()
        saved = super()._save_checkpoint()
        self._accelerator.wait_for_everyone()
        error = [None]
        if self._accelerator.is_main_process:
            try:
                if saved is None:
                    raise RuntimeError("Main rank did not return a checkpoint path")
                publish_checkpoint(
                    saved,
                    companion=companion_path(saved),
                    step=self._global_step,
                    queue=Path(self._config.output_dir) / "evaluation_queue",
                    config={
                        **self._config.model_dump(mode="json"),
                        "omni_joint": self.omni_config.model_dump(mode="json"),
                    },
                    fsdp_world_size=(
                        self._accelerator.num_processes
                        if self._accelerator.distributed_type.name == "FSDP"
                        else None
                    ),
                )
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if self._accelerator.num_processes > 1:
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError("Stage-2 checkpoint publication failed: " + error[0])
        return saved
