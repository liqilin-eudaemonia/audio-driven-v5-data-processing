"""Two-stage sampler installation only in the independent, evaluation-only process."""
from __future__ import annotations

import gc
from functools import partial
from pathlib import Path

import torch
from ltx_core.model.transformer.model import BasicAVTransformerBlock
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from ltx_trainer.lora_checkpoint import apply_movie_lora
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.omni_two_stage import OmniTwoStageSampler


def install_two_stage_sampler(config, options: dict) -> None:
    if not config.validation.run_only:
        raise ValueError("Two-stage async sampling is evaluation-only")
    from inference import fuse_distilled_lora
    from ltx_pipelines.utils.blocks import VideoUpsampler
    import ltx_trainer.trainer as trainer_module

    base = Path(config.model.model_path)
    weights = Path(config.model.load_checkpoint)
    upscaler = Path(options["spatial_upscaler"])
    distilled = Path(options["distilled_lora"])
    strength = float(options.get("distilled_strength", 0.8))
    for path in (base, weights, upscaler, distilled):
        if not path.is_file():
            raise FileNotFoundError(path)

    class IndependentSampler(OmniTwoStageSampler):
        def __init__(self, **kwargs):
            # No shared training model is present: this process only loads weights and evaluates.
            model = kwargs["transformer"]
            # A DDP wrapper can be removed for independent sampling, but FSDP
            # owns parameter materialization and must remain around Stage 1.
            while hasattr(model, "module") and not isinstance(
                model, FullyShardedDataParallel
            ):
                model = model.module
            kwargs["transformer"] = model
            device = next(model.parameters()).device
            self._generated = False
            super().__init__(
                **kwargs,
                upsample=VideoUpsampler(str(base), str(upscaler), torch.bfloat16, device),
                replace_transformer=lambda old: self._load_variant(old, distilled_variant=True),
            )

        def _load_variant(self, old, *, distilled_variant: bool):
            distributed = isinstance(old, FullyShardedDataParallel)
            if not distributed:
                old.to("meta")
                gc.collect()
                torch.cuda.empty_cache()
            fresh = load_transformer(base, device="cpu", dtype=torch.bfloat16)
            if distilled_variant:
                fresh = fuse_distilled_lora(fresh, distilled, strength)
            if config.training_strategy.use_first_frame_audio_cond:
                fresh.enable_first_frame_audio_conditioning()
            fresh = apply_movie_lora(fresh, weights).requires_grad_(False).eval()
            if distributed:
                fresh = FullyShardedDataParallel(
                    fresh,
                    auto_wrap_policy=partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={BasicAVTransformerBlock},
                    ),
                    device_id=torch.cuda.current_device(),
                    use_orig_params=True,
                    sync_module_states=False,
                )
            return fresh

        @torch.no_grad()
        def generate(self, config, device):
            if self._generated:
                self._transformer = self._load_variant(self._transformer, distilled_variant=False)
            self._generated = True
            # Fresh variants need placement; the ordinary trainer sampler owns an already placed model.
            self._transformer.to(device)
            return super().generate(config, device)

    trainer_module.ValidationSampler = IndependentSampler
