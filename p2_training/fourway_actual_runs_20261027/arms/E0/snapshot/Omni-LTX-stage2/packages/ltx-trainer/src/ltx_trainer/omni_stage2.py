"""Differentiable Route-A Plan conditioning and Plan-only next-token supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_trainer.omni_stage2_contract import (
    CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT,
    PlanCompiler,
    build_plan_field_mask,
    derive_causal_timing,
    prepare_causal_speaker_audio,
)
from ltx_trainer.stage1_readiness import index_rows, read_rows


class OmniJointConfig(BaseModel):
    """Dedicated schema; does not change any Stage-1 configuration defaults."""

    model_config = ConfigDict(extra="forbid")
    model_path: str
    manifests: list[str] = Field(min_length=1)
    validation_conditions_dir: str | None = None
    planner_init_checkpoint: str | None = None
    planner_input_mode: Literal["audio_frame_timing", "audio_timing"] = (
        "audio_frame_timing"
    )
    attention_rank: int = Field(default=32, gt=0)
    attention_alpha: int = Field(default=64, gt=0)
    expert_rank: int = Field(default=2, gt=0)
    expert_alpha: int = Field(default=4, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    learning_rate: float = Field(default=2e-6, gt=0)
    resampler_rank: int = Field(default=32, gt=0)
    resampler_alpha: int = Field(default=32, gt=0)
    resampler_learning_rate: float = Field(default=2e-5, gt=0)
    token_loss_weight: float = Field(default=1.0, gt=0)
    gradient_checkpointing: bool = True
    max_tokens: int = Field(default=32768, gt=0)
    audio_tail_ms: int = Field(default=200, ge=0)
    layers: tuple[int, ...] = (12, 24, 36, 48)
    hidden_size: int = Field(default=2048, gt=0)


def thinker_expert_parameter_names(thinker: torch.nn.Module) -> list[str]:
    """Return fused Thinker expert tensors, deliberately excluding router gates."""
    names = {
        name
        for name, _ in thinker.named_parameters()
        if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
    }
    layer_count = len(thinker.model.layers)
    expected = {
        f"model.layers.{index}.mlp.experts.{leaf}"
        for index in range(layer_count)
        for leaf in ("gate_up_proj", "down_proj")
    }
    if names and names != expected:
        missing = sorted(expected - names)
        unexpected = sorted(names - expected)
        raise ValueError(
            "Unexpected Thinker fused-expert layout; "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    return sorted(names)


def thinker_lora_config(
    cfg: OmniJointConfig, thinker: torch.nn.Module
) -> LoraConfig:
    """Build one PEFT adapter with r32 attention and budget-matched r2 experts."""
    attention_modules = sorted(
        name
        for name, module in thinker.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.layers.")
        and name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"))
    )
    expert_modules = sorted(
        name
        for name, module in thinker.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("model.layers.")
        and ".mlp.experts." in name
        and name.endswith((".gate_proj", ".up_proj", ".down_proj"))
    )
    expert_parameters = thinker_expert_parameter_names(thinker)
    layer_count = len(thinker.model.layers)
    if len(attention_modules) != layer_count * 4:
        raise ValueError(
            f"Expected four attention projections per Thinker layer, got {len(attention_modules)}"
        )
    if bool(expert_modules) == bool(expert_parameters):
        raise ValueError("Expected exactly one unfused or fused Thinker expert layout")
    if expert_parameters and cfg.lora_dropout != 0:
        raise ValueError("PEFT target_parameters for fused MoE experts requires lora_dropout=0")
    expert_targets = expert_modules or expert_parameters
    expert_pattern = (
        r"model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)"
    )
    module_targets: str | list[str]
    rank_pattern: dict[str, int]
    alpha_pattern: dict[str, int]
    if expert_modules:
        module_targets = (
            r"model\.layers\.\d+\.(?:self_attn\."
            r"(?:q_proj|k_proj|v_proj|o_proj)|mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj))"
        )
        rank_pattern = {expert_pattern: cfg.expert_rank}
        alpha_pattern = {expert_pattern: cfg.expert_alpha}
    else:
        module_targets = attention_modules
        rank_pattern = dict.fromkeys(expert_targets, cfg.expert_rank)
        alpha_pattern = dict.fromkeys(expert_targets, cfg.expert_alpha)
    return LoraConfig(
        r=cfg.attention_rank,
        lora_alpha=cfg.attention_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=module_targets,
        target_parameters=expert_parameters or None,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )


def thinker_eos_token_id(tokenizer: Any) -> int:
    """Direct Thinker loading does not inherit the full Omni wrapper's text EOS."""
    token = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token is None or token == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no valid assistant end token")
    return int(token)


def plan_labels(input_ids: Tensor, prompt_length: int) -> Tensor:
    """Supervise every Plan token including im_end, never input/prompt tokens."""
    if input_ids.ndim != 2 or not 0 < prompt_length < input_ids.shape[1]:
        raise ValueError("Expected a nonempty prompt followed by Plan tokens")
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    return labels


def plan_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Causal shift: the last prompt position predicts the first Plan token."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


@dataclass
class PlanEncoding:
    hidden: Tensor  # [layers, tokens, hidden], deliberately not detached
    token_mask: Tensor
    field_mask: Tensor
    token_loss: Tensor
    token_count: int


def collate_encodings(encodings: list[PlanEncoding]) -> dict[str, Tensor]:
    """Pad differentiably; padded tokens cannot participate in Resampler attention."""
    length = max(item.hidden.shape[1] for item in encodings)
    hidden = torch.stack([F.pad(item.hidden, (0, 0, 0, length - item.hidden.shape[1])) for item in encodings])
    mask = torch.stack([F.pad(item.token_mask, (0, length - item.token_count)) for item in encodings])
    fields = torch.stack([F.pad(item.field_mask, (0, length - item.token_count)) for item in encodings])
    return {
        "plan_latent": hidden,
        "omni_attention_mask": mask[:, None].expand(-1, hidden.shape[1], -1),
        "plan_field_mask": fields,
    }


class OnlineOmniPlanner:
    def __init__(self, cfg: OmniJointConfig, device: torch.device, cache_dir: Path) -> None:
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        self.cfg = cfg
        self.device = device
        self.cache_dir = cache_dir
        rows = [row for path in cfg.manifests for row in read_rows(Path(path))]
        self.samples = index_rows(rows)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(cfg.model_path, local_files_only=True)
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            cfg.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        thinker.requires_grad_(False)
        thinker.config.use_cache = False
        lora_config = thinker_lora_config(cfg, thinker)
        expected_targets = len(
            [
                name
                for name, module in thinker.named_modules()
                if isinstance(module, torch.nn.Linear)
                and name.startswith("model.layers.")
                and (
                    name.endswith(
                        (
                            ".self_attn.q_proj",
                            ".self_attn.k_proj",
                            ".self_attn.v_proj",
                            ".self_attn.o_proj",
                        )
                    )
                    or (
                        ".mlp.experts." in name
                        and name.endswith((".gate_proj", ".up_proj", ".down_proj"))
                    )
                )
            ]
        ) + len(lora_config.target_parameters or [])
        self.model = get_peft_model(
            thinker,
            lora_config,
        )
        if cfg.gradient_checkpointing:
            thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            thinker.enable_input_require_grads()
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No Omni LoRA parameters matched")
        trainable_names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if sum("lora_A" in name for name in trainable_names) != expected_targets:
            raise ValueError("Not every configured Omni attention/expert target received LoRA A")
        if sum("lora_B" in name for name in trainable_names) != expected_targets:
            raise ValueError("Not every configured Omni attention/expert target received LoRA B")
        if any(".mlp.gate." in name for name in trainable_names):
            raise ValueError("Omni router parameters must remain frozen")
        if not any(".mlp.experts." in name and "lora_" in name for name in trainable_names):
            raise ValueError("No expert MLP LoRA parameters matched")
        self.train()

    def train(self) -> None:
        self.model.train()
        base = self.model.get_base_model()
        base.visual.eval()
        base.audio_tower.eval()

    def _prompt_messages(self, sample_id: str) -> list[dict[str, Any]]:
        sample = self.samples[sample_id]
        _, end_ms = derive_causal_timing(sample)
        files = sample["files"]
        source = files.get("speaker_audio") or files.get("speaker_pure_wav")
        if not source or not Path(source).is_file():
            raise FileNotFoundError(f"{sample_id}: missing causal Speaker audio")
        audio = prepare_causal_speaker_audio(
            source,
            self.cache_dir / f"{sample_id}.wav",
            end_ms,
            self.cfg.audio_tail_ms,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "[SPEAKER_AUDIO]\nSpeaker audio for this interaction turn.",
            },
            {"type": "audio", "audio": audio},
        ]
        if self.cfg.planner_input_mode == "audio_frame_timing":
            image = files.get("first_frame_jpg")
            if not image or not Path(image).is_file():
                raise FileNotFoundError(f"{sample_id}: missing Listener first frame")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "[LISTENER_FRAME0]\n"
                            "Static listener portrait at t=0 (Native Resolution)."
                        ),
                    },
                    {"type": "image", "image": image},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": "[CAUSAL_TIMING]\n"
                + json.dumps(
                    {
                        "speaker_start_ms": 0,
                        "speaker_end_ms": end_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n\nProduce the Avatar Plan JSON now.",
            }
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": CAUSAL_PLANNER_SYSTEM_PROMPT_NO_CONTEXT}]},
            {
                "role": "user",
                "content": content,
            },
        ]

    def prepare_prompts(self, sample_ids: list[str]) -> dict[str, Tensor]:
        """Prepare a padded multimodal batch without changing sample order."""
        from qwen_omni_utils import process_mm_info

        if not sample_ids:
            raise ValueError("At least one Planner sample is required")
        conversations = [self._prompt_messages(sample_id) for sample_id in sample_ids]
        audios, images, _ = process_mm_info(conversations, use_audio_in_video=False)
        texts = [
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            for messages in conversations
        ]
        return dict(
            self.processor(
                text=texts,
                audio=audios,
                images=images,
                padding=True,
                return_tensors="pt",
                use_audio_in_video=False,
            )
            .to(device=self.device)
            .to(dtype=torch.bfloat16)
        )

    def prepare_prompt(self, sample_id: str) -> dict[str, Tensor]:
        return self.prepare_prompts([sample_id])

    def canonical_plan(self, sample_id: str, raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        plan = raw.get("avatar_plan", raw)
        _, end_ms = derive_causal_timing(self.samples[sample_id])
        anchors = plan.get("timing_anchors", {})
        if (
            anchors.get("speaker_start_ms") != 0
            or anchors.get("speaker_end_ms") != end_ms
            or anchors.get("source") != "LOCKED_METADATA"
            or anchors.get("locked") is not True
        ):
            raise ValueError(f"{sample_id}: Plan and causal manifest timing disagree")
        return json.dumps({"avatar_plan": plan}, ensure_ascii=False), plan

    def _supervised_inputs(
        self, sample_id: str, raw: dict[str, Any]
    ) -> tuple[dict[str, Tensor], Tensor, int, int]:
        serialized, _ = self.canonical_plan(sample_id, raw)
        inputs = self.prepare_prompt(sample_id)
        prompt_length = inputs["input_ids"].shape[1]
        tokens = self.processor.tokenizer(
            serialized + "<|im_end|>",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        token_count = tokens.shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens], dim=1)
        if inputs["input_ids"].shape[1] > self.cfg.max_tokens:
            raise ValueError(f"{sample_id}: over max_tokens; never truncate supervised Plans")
        inputs["attention_mask"] = torch.cat(
            [
                inputs["attention_mask"],
                torch.ones_like(tokens, dtype=inputs["attention_mask"].dtype),
            ],
            dim=1,
        )
        return inputs, plan_labels(inputs["input_ids"], prompt_length), prompt_length, token_count

    def _supervised_batch_inputs(
        self, sample_ids: list[str], raws: list[dict[str, Any]]
    ) -> tuple[dict[str, Tensor], Tensor, int]:
        if len(sample_ids) != len(raws) or not sample_ids:
            raise ValueError("Planner batch IDs and targets must be nonempty and aligned")
        serialized = [
            self.canonical_plan(sample_id, raw)[0]
            for sample_id, raw in zip(sample_ids, raws, strict=True)
        ]
        inputs = self.prepare_prompts(sample_ids)
        prompt_width = inputs["input_ids"].shape[1]
        token_rows = [
            self.processor.tokenizer(
                plan + "<|im_end|>",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0].to(self.device)
            for plan in serialized
        ]
        max_plan_tokens = max(row.shape[0] for row in token_rows)
        if prompt_width + max_plan_tokens > self.cfg.max_tokens:
            raise ValueError("Planner batch exceeds max_tokens; never truncate supervised Plans")
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Planner tokenizer must define a pad token")
        plan_ids = torch.full(
            (len(token_rows), max_plan_tokens),
            pad_token_id,
            dtype=inputs["input_ids"].dtype,
            device=self.device,
        )
        plan_mask = torch.zeros(
            (len(token_rows), max_plan_tokens),
            dtype=inputs["attention_mask"].dtype,
            device=self.device,
        )
        labels = torch.full(
            (len(token_rows), prompt_width + max_plan_tokens),
            -100,
            dtype=inputs["input_ids"].dtype,
            device=self.device,
        )
        for row_index, row in enumerate(token_rows):
            length = row.shape[0]
            plan_ids[row_index, :length] = row
            plan_mask[row_index, :length] = 1
            labels[row_index, prompt_width : prompt_width + length] = row
        inputs["input_ids"] = torch.cat([inputs["input_ids"], plan_ids], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], plan_mask], dim=1
        )
        return inputs, labels, sum(row.shape[0] for row in token_rows)

    def supervised_batch_loss(
        self, sample_ids: list[str], raws: list[dict[str, Any]]
    ) -> tuple[Tensor, int]:
        """Compute token-normalized Planner SFT loss for a true micro-batch."""
        inputs, labels, token_count = self._supervised_batch_inputs(sample_ids, raws)
        output = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            output_router_logits=False,
        )
        return plan_cross_entropy(output.logits, labels), token_count

    def supervised_loss(self, sample_id: str, raw: dict[str, Any]) -> tuple[Tensor, int]:
        """Compute Planner-only SFT loss without materializing hidden-state stacks."""
        return self.supervised_batch_loss([sample_id], [raw])

    def encode(self, sample_id: str, raw: dict[str, Any], *, compute_loss: bool = True) -> PlanEncoding:
        serialized, _ = self.canonical_plan(sample_id, raw)
        inputs, labels, prompt_length, token_count = self._supervised_inputs(sample_id, raw)
        output = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            output_router_logits=False,
        )
        hidden = torch.stack([output.hidden_states[layer][0, prompt_length:] for layer in self.cfg.layers])
        if hidden.shape != (len(self.cfg.layers), token_count, self.cfg.hidden_size):
            raise ValueError(f"Unexpected Omni Plan feature shape: {tuple(hidden.shape)}")
        loss = plan_cross_entropy(output.logits, labels) if compute_loss else hidden.new_zeros(())
        return PlanEncoding(
            hidden=hidden,
            token_mask=torch.ones(token_count, device=self.device, dtype=torch.bool),
            field_mask=build_plan_field_mask(serialized, self.processor.tokenizer, token_count).to(self.device),
            token_loss=loss,
            token_count=token_count,
        )

    def encode_conditions(self, conditions: dict[str, Any]) -> tuple[dict[str, Any], Tensor]:
        encodings = []
        for index, (sid, raw) in enumerate(zip(conditions["sample_id"], conditions["raw_plan_json"], strict=True)):
            if conditions["planner_route"][index] != "route_a_no_context":
                raise ValueError("Stage 2 joint training only supports Route A")
            if conditions["version"] != "teacher_forcing_raw":
                raise ValueError("Joint supervised training requires GT teacher_forcing_raw Plans")
            _, plan = self.canonical_plan(sid, raw)
            audio, video = PlanCompiler.render_ltx_prompts(plan)
            if audio != conditions["audio_prompt"][index] or video != conditions["video_prompt"][index]:
                raise ValueError(f"{sid}: cached prompts disagree with GT Plan")
            encodings.append(self.encode(sid, raw))
        total_tokens = sum(item.token_count for item in encodings)
        loss = sum(item.token_loss * item.token_count for item in encodings) / total_tokens
        return {**conditions, **collate_encodings(encodings)}, loss
